from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn


class MultiKernelDWConv2D(nn.Module):
    """Parallel multi-kernel *depthwise* conv for (B,C,H,W) input.

    Requirement mapping:
    - For each 2D plane(channel), apply 1x1/3x3/5x5 conv in parallel
    - Global average pool
    - Concat

    Output: (B, C, K) where K=len(kernel_sizes)
    """

    def __init__(self, channels: int, kernel_sizes: Sequence[int] = (1, 3, 5)):
        super().__init__()
        self.channels = int(channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                    groups=self.channels,
                    bias=True,
                )
                for k in self.kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        pooled = []
        for conv in self.convs:
            y = conv(x)
            # GAP: (B,C)
            pooled.append(y.mean(dim=(2, 3)))
        # (B,C,K)
        return torch.stack(pooled, dim=-1)


class CosinePositionalEncoding(nn.Module):
    """Cosine positional encoding for a sequence (B,L,D)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        b, l, d = x.shape
        device = x.device
        pos = torch.arange(l, device=device, dtype=torch.float32).unsqueeze(1)  # (L,1)
        i = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)  # (1,D)
        div = torch.pow(10000.0, (2.0 * torch.floor(i / 2.0)) / float(d))
        pe = torch.cos(pos / div)  # (L,D)
        return x + pe.unsqueeze(0)


class VarEncoder(nn.Module):
    """FC + cosine PE + MHA(3 heads) + FFN + LSTM + Add&Norm.

    Input/Output: (B, L, D)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        ff_mult: int = 4,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pe = CosinePositionalEncoding(d_model)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.ReLU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, batch_first=True)
        self.lstm_proj = nn.Linear(lstm_hidden, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)

        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.mha_norm(x + attn_out)

        ff_out = self.ffn(x)
        x = self.ffn_norm(x + ff_out)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        x = self.lstm_norm(x + lstm_out)

        return x


class LinearAR(nn.Module):
    """Linear AR-like bypass on the raw (B,C,H,W) input.

    Note: this does NOT use target history; it is a linear regression on current sample features.
    """

    def __init__(self, in_channels: int, h: int, w: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.h = int(h)
        self.w = int(w)
        self.fc = nn.Linear(self.in_channels * self.h * self.w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.fc(x.view(b, -1))


class StarFusion(nn.Module):
    """StarNet/Hadamard fusion module.

    Two FC branches (one GELU, one identity) -> Hadamard product -> FC to scalar.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc_gelu = nn.Linear(in_dim, hidden_dim)
        self.fc_id = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act(self.fc_gelu(x))
        b = self.fc_id(x)
        h = a * b
        return self.out(h)


class TCLNLike(nn.Module):
    """TCLN-like research model for scheme-1 input (B,C=p_vars,H,W).

    - Multi-kernel DWConv per variable plane -> GAP -> tokens per variable
    - Encoder (MHA+LSTM)
    - Linear AR bypass
    - Fusion: plain concat head OR StarFusion(Hadamard)

    Output: (B,)
    """

    def __init__(
        self,
        in_channels: int,
        h: int,
        w: int,
        *,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        d_model: int = 64,
        lstm_hidden: int = 64,
        fusion: str = "plain",  # plain / starnet
        starnet_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.h = int(h)
        self.w = int(w)

        self.mk = MultiKernelDWConv2D(self.in_channels, kernel_sizes=kernel_sizes)
        self.token_proj = nn.Linear(len(kernel_sizes), d_model)

        self.encoder = VarEncoder(d_model=d_model, nhead=4, lstm_hidden=lstm_hidden, dropout=dropout)

        self.ar = LinearAR(in_channels=self.in_channels, h=self.h, w=self.w)

        in_fuse = d_model + 1
        if fusion == "plain":
            self.fusion = nn.Sequential(nn.ReLU(), nn.Linear(in_fuse, 1))
        elif fusion == "starnet":
            self.fusion = StarFusion(in_dim=in_fuse, hidden_dim=starnet_hidden)
        else:
            raise ValueError("fusion must be plain or starnet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        tokens = self.mk(x)  # (B,C,K)
        tokens = self.token_proj(tokens)  # (B,C,D)
        enc = self.encoder(tokens)  # (B,C,D)
        enc_vec = enc.mean(dim=1)  # (B,D)

        ar_out = self.ar(x)  # (B,1)
        z = torch.cat([enc_vec, ar_out], dim=1)

        y = self.fusion(z).squeeze(-1)
        return y
