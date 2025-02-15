import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from einops import rearrange, repeat

# Import Mamba operations
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

@dataclass
class MambaConfig:
    d_model: int = 768
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 128
    ngroups: int = 1
    bias: bool = False
    conv_bias: bool = True
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
    activation: str = "swish"
    use_mem_eff_path: bool = False
    chunk_size: int = 256

class BaseMambaBlock(nn.Module):
    """Base class with common components for all Mamba variants"""
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = self.expand * self.d_model
        self.headdim = config.headdim
        self.ngroups = config.ngroups
        self.nheads = self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        
        # Initialize dt
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        dt = torch.clamp(dt, min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Initialize A
        A = torch.randn(self.nheads)
        A_log = torch.log(A.abs())
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # Output components
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

class OriginalMambaBlock(BaseMambaBlock):
    """Original Mamba block with 1D conv + SSM"""
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        
        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=config.bias)
        
        # 1D Convolution
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=config.d_conv,
            groups=conv_dim,
            padding=config.d_conv - 1,
            bias=config.conv_bias
        )
        
    def forward(self, u):
        B, L, D = u.shape
        
        # Input projection
        zxbcdt = self.in_proj(u)
        
        # Use efficient path with fused operations
        if self.config.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                -torch.exp(self.A_log),
                D=self.D,
                chunk_size=self.config.chunk_size,
                activation=self.config.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
            )
            return out
        
        # Standard path
        z, xBC, dt = torch.split(
            zxbcdt, 
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
            dim=-1
        )
        
        dt = F.softplus(dt + self.dt_bias)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))[:, :L, :]
        
        x, B, C = torch.split(
            xBC, 
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], 
            dim=-1
        )
        
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            -torch.exp(self.A_log),
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.config.chunk_size,
            D=self.D,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        
        y = self.norm(y, z)
        return self.out_proj(y)

class Mamba2DConvBlock(BaseMambaBlock):
    """Mamba variant with 2D convolution"""
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        
        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=config.bias)
        
        # 2D Convolution
        self.xBC_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=self.xBC_dim,
            out_channels=self.xBC_dim,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=config.conv_bias,
            groups=self.xBC_dim
        )
        
    def forward(self, u):
        B, L, D = u.shape
        
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, 
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
            dim=-1
        )
        
        # 2D Convolution
        dt = F.softplus(dt + self.dt_bias)

        # image path
        img_width = 16
        img_height = L // img_width

        xBC = xBC.view(B, img_height, img_width, -1).permute(0, 3, 1, 2)  # Add channel dimension
        xBC = F.silu(self.conv2d(xBC).permute(0, 2, 3, 1).view(B, L, self.xBC_dim))
        xBC = xBC.view(B, L, self.xBC_dim)
        
        x, B, C = torch.split(
            xBC, 
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], 
            dim=-1
        )
        
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            -torch.exp(self.A_log),
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.config.chunk_size,
            D=self.D,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        
        y = self.norm(y, z)
        return self.out_proj(y)

class MambaAttentionBlock(BaseMambaBlock):
    """Mamba variant with self-attention using explicit QKV projections"""
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        
        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=config.bias)
        
        # QKV projections
        self.q_proj = nn.Linear(self.d_inner, self.d_inner)
        self.k_proj = nn.Linear(self.d_inner, self.d_inner)
        self.v_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # Multi-head attention parameters
        self.num_heads = 8
        self.head_dim = self.d_inner // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
    def forward(self, u):
        B, L, D = u.shape
        
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, 
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], 
            dim=-1
        )
        
        dt = F.softplus(dt + self.dt_bias)
        
        # Get the part that would have gone through conv
        x = xBC[:, :, :self.d_inner]
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # B, H, L, D
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, self.d_inner)
        
        # Split remaining features for SSM
        _, B, C = torch.split(
            xBC[:, :, self.d_inner:], 
            [0, self.ngroups * self.d_state, self.ngroups * self.d_state], 
            dim=-1
        )
        
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            -torch.exp(self.A_log),
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.config.chunk_size,
            D=self.D,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        
        y = self.norm(y, z)
        return self.out_proj(y)

def benchmark_model(model: nn.Module, batch_size=16, seq_length=512, n_runs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result = {"forward_times": [], "backward_times": []}
    
    for _ in range(n_runs):
        x = torch.randn(batch_size, seq_length, model.d_model, device=device, requires_grad=True)
        
        # Forward pass
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    
        start.record()
        y = model(x)
        end.record()
        
        torch.cuda.synchronize()
        result["forward_times"].append(start.elapsed_time(end))
            
        # Backward pass
        loss = y.sum()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        loss.backward()
        end.record()
        
        torch.cuda.synchronize()
        result["backward_times"].append(start.elapsed_time(end))
 
    return result

if __name__ == "__main__":
    config = MambaConfig()
    models = {
        "Original": OriginalMambaBlock(config),
        "2DConv": Mamba2DConvBlock(config),
        "Attention": MambaAttentionBlock(config)
    }
    
    print("\nBenchmarking Results:")
    print("-" * 60)
    print(f"{'Model':<15} {'Forward (ms)':<20} {'Backward (ms)':<20}")
    print("-" * 60)
    
    for name, model in models.items():
        results = benchmark_model(model)
        fwd_mean = np.mean(results["forward_times"][10:])  # Skip first 10 for warmup
        bwd_mean = np.mean(results["backward_times"][10:])
        print(f"{name:<15} {fwd_mean:>8.2f} ± {np.std(results['forward_times'][10:]):>6.2f} "
              f"{bwd_mean:>8.2f} ± {np.std(results['backward_times'][10:]):>6.2f}")