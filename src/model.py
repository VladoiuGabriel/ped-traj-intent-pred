import torch
import torch.nn as nn
import math
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLPBridge(nn.Module):
    """
    Projects CLIP patch tokens to Qwen2.5 hidden dimension.
    Input:  (batch, 197, 768)
    Output: (batch, 197, 1536)
    """
    def __init__(self, clip_dim=768, qwen_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim)
        )

    def forward(self, x):
        return self.net(x)


def timestep_embedding(t, dim):
    """
    Sinusoidal timestep embedding.
    t:   (batch,) integer timesteps
    dim: embedding dimension
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FlowDiTBlock(nn.Module):
    """
    DiT block adapted for flow matching with:
    - AdaLN conditioned on continuous timestep t in [0,1]
    - Self-attention over waypoints
    - Cross-attention on context (Qwen2.5 + visual features)
    - FFN
    """
    def __init__(self, hidden_dim=256, context_dim=1536, num_heads=4):
        super().__init__()

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            kdim=context_dim, vdim=context_dim,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, t_emb, context):
        """
        x:       (batch, n_waypoints, hidden_dim)
        t_emb:   (batch, hidden_dim)
        context: (batch, seq_len, context_dim)
        """
        ada = self.adaLN(t_emb).unsqueeze(1)
        s1, b1, s2, b2, s3, b3 = ada.chunk(6, dim=-1)

        x_norm = self.norm1(x) * (1 + s1) + b1
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x) * (1 + s2) + b2
        cross_out, _ = self.cross_attn(x_norm, context, context)
        x = x + cross_out

        x_norm = self.norm3(x) * (1 + s3) + b3
        x = x + self.ffn(x_norm)

        return x


class TrajectoryFlowDiT(nn.Module):
    """
    Flow Matching DiT for trajectory prediction.
    Learns a velocity field v(x_t, t, context) that transports
    noise z ~ N(0,I) to clean trajectory x0 along straight paths:
        x(t) = (1-t)*z + t*x0
        v_target = x0 - z
    """
    def __init__(
        self,
        n_waypoints=6,
        traj_dim=2,
        hidden_dim=256,
        context_dim=1536,
        num_heads=4,
        depth=4
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.traj_proj = nn.Linear(traj_dim, hidden_dim)

        self.t_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.blocks = nn.ModuleList([
            FlowDiTBlock(hidden_dim, context_dim, num_heads)
            for _ in range(depth)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, traj_dim)
        )

    def forward(self, x_t, t, context):
        """
        x_t:     (batch, n_waypoints, 2)  — interpolated trajectory
        t:       (batch,)                 — continuous timestep in [0,1]
        context: (batch, seq_len, 1536)   — visual + language context
        Returns: (batch, n_waypoints, 2)  — predicted velocity field
        """
        x = self.traj_proj(x_t)

        t_emb = timestep_embedding((t * 999).long(), x.shape[-1])
        t_emb = self.t_proj(t_emb)

        for block in self.blocks:
            x = block(x, t_emb, context)

        return self.out_proj(x)


class PedTrajModel(nn.Module):
    """
    Full pipeline:
    CLIP-ViT (frozen) -> MLP Bridge -> concat with obs tokens -> Qwen2.5 -> FlowDiT head
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.clip = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        for param in self.clip.parameters():
            param.requires_grad = False

        self.bridge = MLPBridge(clip_dim=768, qwen_dim=1536)

        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        for param in self.qwen.parameters():
            param.requires_grad = False

        self.flow = TrajectoryFlowDiT(
            n_waypoints=6,
            traj_dim=2,
            hidden_dim=256,
            context_dim=1536,
            num_heads=4,
            depth=4
        )

    def encode_image(self, images):
        """
        images: list of PIL Images (batch_size)
        Returns: (batch_size, 197, 1536) visual tokens in Qwen space
        """
        inputs = self.clip_processor(
            images=images, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            clip_out = self.clip(**inputs)
            patch_tokens = clip_out.last_hidden_state

        return self.bridge(patch_tokens.float())

    def encode_obs_trajectory(self, obs):
        """
        obs: (batch, 4, 2) observed waypoints
        Returns: (batch, seq_len, 1536) Qwen2.5 hidden states
        """
        prompts = []
        for b in range(obs.shape[0]):
            coords = ", ".join(
                [f"({obs[b, i, 0]:.2f}, {obs[b, i, 1]:.2f})"
                 for i in range(obs.shape[1])]
            )
            prompts.append(
                f"A pedestrian was observed at positions: {coords}. "
                f"Predict the future trajectory."
            )

        tokens = self.qwen_tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True
        ).to(self.device)

        with torch.no_grad():
            qwen_out = self.qwen(
                **tokens,
                output_hidden_states=True
            )
            text_features = qwen_out.hidden_states[-1]

        return text_features.float()

    def get_context(self, images, obs):
        """
        images: list of PIL Images
        obs:    (batch, 4, 2) observed trajectory
        Returns: (batch, 197+seq_len, 1536)
        """
        visual_tokens = self.encode_image(images)
        text_features = self.encode_obs_trajectory(obs)

        return torch.cat([visual_tokens, text_features], dim=1)

    def flow_matching_loss(self, context, x0):
        """
        Flow matching training loss.
        x0: (batch, 6, 2) ground truth future trajectory
        """
        b = x0.shape[0]
        z = torch.randn_like(x0)
        t = torch.rand(b, device=self.device)
        t_exp = t[:, None, None]

        x_t = (1 - t_exp) * z + t_exp * x0
        v_target = x0 - z
        v_pred = self.flow(x_t, t, context)

        return nn.functional.mse_loss(v_pred, v_target)

    def forward(self, images, obs, pred_gt):
        """
        Training forward pass.
        images:  list of PIL Images
        obs:     (batch, 4, 2) observed trajectory
        pred_gt: (batch, 6, 2) ground truth future trajectory
        Returns: flow matching loss (scalar)
        """
        context = self.get_context(images, obs)
        return self.flow_matching_loss(context, pred_gt)

    @torch.no_grad()
    def predict(self, images, obs, n_samples=6, steps=50):
        """
        Inference: generate n_samples trajectory predictions via Euler integration.
        images: list of PIL Images
        obs:    (1, 4, 2) observed trajectory
        Returns: (n_samples, 6, 2) predicted trajectories
        """
        context = self.get_context(images, obs)
        ctx = context.expand(n_samples, -1, -1)

        x = torch.randn(n_samples, self.flow.n_waypoints, 2, device=self.device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((n_samples,), i * dt, device=self.device)
            v = self.flow(x, t, ctx)
            x = x + v * dt

        return x