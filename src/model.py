import torch
import torch.nn as nn
import math
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLPBridge(nn.Module):
    """
    projects CLIP patch tokens to Qwen2.5 hidden dimension.
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


def get_ddpm_schedule(T=100, beta_start=1e-4, beta_end=0.02):
    """ddpm schedule"""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def timestep_embedding(t, dim):
    """
    timestep embedding
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  



class DiTBlock(nn.Module):
    """
    single DiT block with:
    - AdaLN conditioned on timestep
    - self-attention over waypoints
    - cross-attention on context (qwen2.5 features)
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
        x:       (batch, n_waypoints, hidden_dim)  — noisy trajectory tokens
        t_emb:   (batch, hidden_dim)               — timestep embedding
        context: (batch, seq_len, context_dim)     — qwen2.5 features
        """
        ada = self.adaLN(t_emb)                         
        ada = ada.unsqueeze(1)                            
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



# class TrajectoryDiT(nn.Module):
#     """
#     DiT denoiser for trajectory prediction=>
#     takes noisy trajectories + context and predicts noise
#     """
#     def __init__(
#         self,
#         n_waypoints=6,
#         traj_dim=2,
#         hidden_dim=256,
#         context_dim=1536,
#         num_heads=4,
#         depth=4,
#         T=100
#     ):
#         super().__init__()
#         self.n_waypoints = n_waypoints
#         self.T = T

#         self.traj_proj = nn.Linear(traj_dim, hidden_dim)

#         self.t_proj = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )

#         self.blocks = nn.ModuleList([
#             DiTBlock(hidden_dim, context_dim, num_heads)
#             for _ in range(depth)
#         ])

#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, traj_dim)
#         )

#         betas, alphas, alphas_cumprod = get_ddpm_schedule(T)
#         self.register_buffer('betas', betas)
#         self.register_buffer('alphas', alphas)
#         self.register_buffer('alphas_cumprod', alphas_cumprod)

#     def forward(self, x_noisy, t, context):
#         """
#         x_noisy: (batch, n_waypoints, 2)   — noisy trajectory
#         t:       (batch,)                  — integer timesteps
#         context: (batch, seq_len, 1536)    — qwen2.5 features
#         returns: (batch, n_waypoints, 2)   — predicted noise
#         """
#         x = self.traj_proj(x_noisy)

#         t_emb = timestep_embedding(t, x.shape[-1])   
#         t_emb = self.t_proj(t_emb)                  

        
#         for block in self.blocks:
#             x = block(x, t_emb, context)

       
#         return self.out_proj(x)  

#     def add_noise(self, x0, t, noise=None):
#         """
#         forward diffusion: add noise to clean trajectory x0 at timestep t
#         x0: (batch, n_waypoints, 2)
#         t:  (batch,) integer timesteps
#         """
#         if noise is None:
#             noise = torch.randn_like(x0)

#         alpha_bar = self.alphas_cumprod[t]              
#         alpha_bar = alpha_bar[:, None, None]            

#         x_noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
#         return x_noisy, noise

#     @torch.no_grad()
#     def ddim_sample(self, context, n_samples=6, ddim_steps=20):
#         """
#         DDIM sampling: generate n_samples trajectories from noise
#         context: (1, seq_len, 1536)
#         returns: (n_samples, n_waypoints, 2)
#         """
#         device = next(self.parameters()).device

#         ctx = context.expand(n_samples, -1, -1)

    
#         x = torch.randn(n_samples, self.n_waypoints, 2, device=device)

        
#         step_size = self.T // ddim_steps
#         timesteps = list(range(0, self.T, step_size))[::-1]

#         for i, t_val in enumerate(timesteps):
#             t_batch = torch.full((n_samples,), t_val, device=device, dtype=torch.long)

#             noise_pred = self.forward(x, t_batch, ctx)

#             alpha_bar_t = self.alphas_cumprod[t_val]
#             alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]] \
#                 if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

#             x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) \
#                       / torch.sqrt(alpha_bar_t)

#             x = torch.sqrt(alpha_bar_prev) * x0_pred \
#                 + torch.sqrt(1 - alpha_bar_prev) * noise_pred

#         return x  



class TrajectoryFlow(nn.Module):
    """
    flow matching variant
    """
    def __init__(
        self,
        n_waypoints=6,
        traj_dim=2,
        hidden_dim=256,
        context_dim=1536
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.traj_proj = nn.Linear(traj_dim, hidden_dim)

        self.t_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.ctx_proj = nn.Linear(context_dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, traj_dim)
        )

    def forward(self, x_t, t, context):
        """
        x_t: (b, n, 2)
        t: (b,)
        context: (b, seq, 1536)
        """

        h_x = self.traj_proj(x_t)

        t = t[:, None].float()
        h_t = self.t_proj(t).unsqueeze(1).expand_as(h_x)

        ctx = context.mean(dim=1)
        h_c = self.ctx_proj(ctx).unsqueeze(1).expand_as(h_x)

        h = torch.cat([h_x, h_t, h_c], dim=-1)

        return self.net(h)


class PedTrajModel(nn.Module):
    """
    Full pipeline:
    CLIP-ViT (frozen) → MLP Bridge → concat with obs tokens → Qwen2.5 → DiT head
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

       
        # self.dit = TrajectoryDiT(
        #     n_waypoints=6,
        #     traj_dim=2,
        #     hidden_dim=256,
        #     context_dim=1536,
        #     num_heads=4,
        #     depth=4,
        #     T=100
        # )
        
        self.flow = TrajectoryFlow(
            n_waypoints=6,
            traj_dim=2,
            hidden_dim=256,
            context_dim=1536
        )
        
        

    def encode_image(self, images):
        """
        images: list of PIL Images (batch_size)
        returns: (batch_size, 197, 1536) visual tokens in Qwen space
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
        encode observed trajectory as text prompt and get qwen2.5 features.
        obs: (batch, 4, 2) observed waypoints
        returns: (batch, seq_len, 1536) qwen2.5 hidden states
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
        combine visual tokens + text features into context for DiT
        images: list of PIL Images
        obs:   (batch, 4, 2) observed trajectory
        returns: (batch, 197+seq_len, 1536)
        """
        visual_tokens = self.encode_image(images)           
        text_features = self.encode_obs_trajectory(obs)    

        context = torch.cat([visual_tokens, text_features], dim=1)
        return context
    
    def flow_matching_loss(self, context, x0):
        b = x0.shape[0]
        z = torch.randn_like(x0)
        t = torch.rand(b, device=self.device)
        t_exp = t[:, None, None]
        x_t = (1 - t_exp) * x0 + t_exp * z
        target = z - x0
        pred = self.flow(x_t, t, context)

        return nn.functional.mse_loss(pred, target)

    def forward(self, images, obs, pred_gt):
        """
        Training forward pass.
        image:   list of PIL Images
        obs:     (batch, 4, 2) observed trajectory
        pred_gt: (batch, 6, 2) ground truth future trajectory
        Returns: diffusion loss (scalar)
        """
        context = self.get_context(images, obs) 
        # batch_size = pred_gt.shape[0]
        # t = torch.randint(0, self.dit.T, (batch_size,), device=self.device)

        # x_noisy, noise = self.dit.add_noise(pred_gt, t)
        # noise_pred = self.dit(x_noisy, t, context)

        # loss = nn.functional.mse_loss(noise_pred, noise)
        # return loss
        
        return self.flow_matching_loss(context, pred_gt)
        

    # @torch.no_grad()
    # def predict(self, images, obs, n_samples=6):
    #     """
    #     inference: generate n_samples trajectory predictions
    #     image: list of PIL Image
    #     obs:   (1, 4, 2) observed trajectory
    #     returns: (n_samples, 6, 2) predicted trajectories
    #     """
    #     context = self.get_context(images, obs) 
    #     return self.dit.ddim_sample(context, n_samples=n_samples, ddim_steps=20)
    
    @torch.no_grad()
    def predict_flow(self, images, obs, n_samples=6, steps=20):
        context = self.get_context(images, obs)
        device = self.device
        x = torch.randn(n_samples, 6, 2, device=device)
        dt = 1.0 / steps
        ctx = context.expand(n_samples, -1, -1)
        for i in range(steps):
            t = torch.full((n_samples,), i / steps, device=device)
            v = self.flow(x, t, ctx)
            x = x - v * dt
        return x