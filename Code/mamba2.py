"""
A minimal Mamba-2 implementation (Dao & Gu, "Transformers are SSMs", 2024),
sized down to compare against ViterbiNet's per-sample MLP as a drop-in
alternative symbol-detector network at a matched parameter budget.

Unlike Mamba-1 (Code/mamba.py, already in this repo), Mamba-2's core
simplification is that each head's state-transition matrix A is a SCALAR
times identity (not a full per-channel diagonal), which lets the whole
recurrence be written as a structured (semiseparable) matrix and computed
either via the paper's chunked SSD algorithm or -- equivalently, and far
simpler to get right at this tiny scale -- a plain sequential recurrence
over time. This file implements the sequential form only: at ~1-2k
parameters per layer and block_length=120, there is no efficiency reason to
implement the parallel chunked-scan form, only correctness risk.

Reference: https://arxiv.org/abs/2405.21060 and the authors' own reference
implementation (state-spaces/mamba, Mamba2 class); this is a fresh,
from-scratch reimplementation at a much smaller scale, not a port.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from Code.mamba import RMSNorm


@dataclass
class Mamba2Config:
    d_model: int          # D: the sequence embedding dimension
    n_layers: int
    d_state: int = 8       # N: SSM state size per head
    expand_factor: int = 2  # E: d_inner = E * d_model
    head_dim: int = 4       # P: per-head channel width (d_inner must be a multiple of this)
    d_conv: int = 4         # short causal conv kernel width over (x, B, C)
    dt_min: float = 0.001
    dt_max: float = 0.1

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        assert self.d_inner % self.head_dim == 0, \
            f'd_inner={self.d_inner} must be a multiple of head_dim={self.head_dim}'
        self.n_heads = self.d_inner // self.head_dim


class Mamba2Block(nn.Module):
    """One Mamba-2 layer: in_proj -> short causal conv -> selective SSM
    (scalar-per-head A) -> gated RMSNorm -> out_proj. n_groups=1 throughout
    (B and C shared across all heads) -- the standard simplification for a
    model this small; the paper's n_groups>1 option only matters for
    tensor-parallel sharding at real LLM scale."""

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = cfg = config

        # z (gate, d_inner) + xBC (x, B, C concatenated: d_inner + 2*d_state) + dt (n_heads)
        conv_dim = cfg.d_inner + 2 * cfg.d_state
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_inner + conv_dim + cfg.n_heads, bias=False)

        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=cfg.d_conv,
                                 groups=conv_dim, padding=cfg.d_conv - 1, bias=True)

        # A is a per-head SCALAR (Mamba-2's key simplification vs Mamba-1's
        # per-channel diagonal A) -- stored in log-space, always negative
        # after -exp(), matching the original Mamba parameterization.
        self.A_log = nn.Parameter(torch.log(torch.empty(cfg.n_heads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(cfg.n_heads))  # per-head skip-connection scale

        dt = torch.exp(torch.rand(cfg.n_heads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
                        + math.log(cfg.dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)).neg().nan_to_num(0.0))

        self.norm = RMSNorm(cfg.d_inner)
        self.out_proj = nn.Linear(cfg.d_inner, cfg.d_model, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        b, l, _ = x.shape
        cfg = self.config

        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(zxbcdt, [cfg.d_inner, cfg.d_inner + 2 * cfg.d_state, cfg.n_heads], dim=-1)

        xBC = self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2)
        xBC = F.silu(xBC)
        x_ssm, B, C = torch.split(xBC, [cfg.d_inner, cfg.d_state, cfg.d_state], dim=-1)

        dt = F.softplus(dt + self.dt_bias)              # (b, l, n_heads)
        A = -torch.exp(self.A_log)                       # (n_heads,) -- always negative

        x_ssm = x_ssm.view(b, l, cfg.n_heads, cfg.head_dim)
        dA = torch.exp(dt * A)                            # (b, l, n_heads)

        h = x_ssm.new_zeros(b, cfg.n_heads, cfg.head_dim, cfg.d_state)
        ys = []
        for t in range(l):
            # h_t = dA_t * h_{t-1} + dt_t * outer(x_t, B_t), shared B across heads (n_groups=1)
            dBx = torch.einsum('bh,bhp,bn->bhpn', dt[:, t], x_ssm[:, t], B[:, t])
            h = dA[:, t].view(b, cfg.n_heads, 1, 1) * h + dBx
            ys.append(torch.einsum('bhpn,bn->bhp', h, C[:, t]))
        y = torch.stack(ys, dim=1)                        # (b, l, n_heads, head_dim)
        y = y + self.D.view(1, 1, -1, 1) * x_ssm
        y = y.reshape(b, l, cfg.d_inner)

        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba2 = Mamba2Block(config)

    def forward(self, x):
        return self.mamba2(self.norm(x)) + x


class Mamba2(nn.Module):
    """Stack of Mamba-2 residual blocks. Input/output: (batch, seq_len, d_model)."""

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.layers = nn.ModuleList(ResidualBlock(config) for _ in range(config.n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
