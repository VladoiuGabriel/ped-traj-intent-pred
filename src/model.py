import torch
import torch.nn as nn
import math
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class PlanningTokenProjector(nn.Module):
    """2048 to 256 two layer mlp with relu"""

    def __init__(self, vlm_dim=2048, flow_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vlm_dim, vlm_dim // 2),
            nn.ReLU(),
            nn.Linear(vlm_dim // 2, flow_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)


class FlowDiTBlock(nn.Module):
    """adaln self-attn cross-attn on planning token ffn"""

    def __init__(self, hidden_dim=256, context_dim=256, num_heads=4):
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
    """flow matching dit conditioned on planning token"""

    def __init__(
        self,
        n_waypoints=6,
        traj_dim=2,
        hidden_dim=256,
        context_dim=256,
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
        x = self.traj_proj(x_t)
        t_emb = timestep_embedding(t, x.shape[-1])
        t_emb = self.t_proj(t_emb)

        for block in self.blocks:
            x = block(x, t_emb, context)

        return self.out_proj(x)


class PedTrajModel(nn.Module):
    """qwen2-vl-3b extracts planning token mlp projects to flowdit"""

    def __init__(
        self,
        device='cuda',
        vlm_name='Qwen/Qwen2.5-VL-3B-Instruct',
        sigma=0.1,
        waypoint_dropout=0.15,
        use_lora=False,
        lora_rank=16,
        lora_alpha=16
    ):
        super().__init__()
        self.device = device
        self.sigma = sigma
        self.n_waypoints = 6
        self.waypoint_dropout = waypoint_dropout

        print("Loading Qwen2.5-VL-3B...", flush=True)
        self.processor = AutoProcessor.from_pretrained(vlm_name)

        self.processor.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[PLAN]']}
        )
        self.plan_token_id = self.processor.tokenizer.convert_tokens_to_ids('[PLAN]')

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device}
        )
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))
        self.vlm.gradient_checkpointing_enable()

        for param in self.vlm.parameters():
            param.requires_grad = False

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none"
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            for name, param in self.vlm.named_parameters():
                if 'visual' in name and 'lora' not in name:
                    param.requires_grad = False
            self.vlm.print_trainable_parameters()
            print("Qwen2-VL-3B Lora Training ready", flush=True)
        else:
            print("Qwen2-VL-3B loaded and frozen", flush=True)

        self.projector = PlanningTokenProjector(vlm_dim=2048, flow_dim=256)
        self.plan_norm = nn.LayerNorm(2048)

        self.obs_encoder = nn.Sequential(
            nn.Linear(4 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.flow = TrajectoryFlowDiT(
            n_waypoints=6,
            traj_dim=2,
            hidden_dim=256,
            context_dim=256,
            num_heads=4,
            depth=4
        )

    @staticmethod
    def polar_to_cartesian(x):
        r     = x[..., 0:1]
        theta = x[..., 1:2]
        cart_x = r * torch.cos(theta)
        cart_y = r * torch.sin(theta)
        return torch.cat([cart_x, cart_y], dim=-1)

    def get_planning_token(self, images, obs):
        """runs vlm forward and extracts hidden state at [PLAN] position"""
        batch_size = obs.shape[0]
        planning_tokens = []

        for b in range(batch_size):
            coords = ", ".join(
                [f"({obs[b, i, 0]:.2f}, {obs[b, i, 1]:.2f})"
                 for i in range(obs.shape[1])]
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[b]},
                        {"type": "text", "text":
                            f"A pedestrian was observed at positions: {coords}. "
                            f"Plan the trajectory. [PLAN]"
                        }
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            if not self.vlm.training:
                with torch.no_grad():
                    outputs = self.vlm(**inputs, output_hidden_states=True)
            else:
                outputs = self.vlm(**inputs, output_hidden_states=True)

            input_ids = inputs['input_ids'][0]
            plan_positions = (input_ids == self.plan_token_id).nonzero(as_tuple=True)[0]
            plan_pos = plan_positions[-1].item() if len(plan_positions) > 0 else -1

            hidden_raw = outputs.hidden_states[-1][0, plan_pos, :].to(torch.float32)
            hidden = hidden_raw.float()
            planning_tokens.append(hidden)

        return torch.stack(planning_tokens).float()

    def apply_waypoint_dropout(self, obs):
        """zeros out random waypoints with p=0.15 during training"""
        if not self.training:
            return obs
        mask = torch.bernoulli(
            torch.full((obs.shape[0], obs.shape[1]), 1 - self.waypoint_dropout)
        ).to(obs.device)
        return obs * mask.unsqueeze(-1)

    def flow_matching_loss(self, context, x0):
        """flow matching loss in polar space"""
        b = x0.shape[0]
        z = torch.randn_like(x0) * self.sigma
        t = torch.rand(b, device=self.device)
        t_exp = t[:, None, None]

        x_t = (1 - t_exp) * z + t_exp * x0
        v_target = (x0 - x_t) / (1 - t_exp + 1e-8)
        v_pred = self.flow(x_t, t, context)

        return nn.functional.mse_loss(v_pred, v_target)

    def forward(self, images, obs, pred_gt):
        """training forward pass — obs and pred_gt in polar coords"""
        obs = self.apply_waypoint_dropout(obs)
        planning_token = self.get_planning_token(images, obs)
        context = self.projector(self.plan_norm(planning_token))
        obs_flat = obs.reshape(obs.shape[0], -1)
        obs_enc = self.obs_encoder(obs_flat).unsqueeze(1)
        context = context + obs_enc
        return self.flow_matching_loss(context, pred_gt)

    @torch.no_grad()
    def predict(self, images, obs, n_samples=6, steps=50):
        """generates n_samples trajectories via euler integration, returns cartesian"""
        planning_token = self.get_planning_token(images, obs)
        context = self.projector(self.plan_norm(planning_token))
        obs_flat = obs.reshape(obs.shape[0], -1)
        obs_enc = self.obs_encoder(obs_flat).unsqueeze(1)
        context = context + obs_enc
        ctx = context.expand(n_samples, -1, -1)

        x = torch.randn(n_samples, self.n_waypoints, 2, device=self.device) * self.sigma
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((n_samples,), i * dt, device=self.device)
            v = self.flow(x, t, ctx)
            x = x + v * dt

        return self.polar_to_cartesian(x)