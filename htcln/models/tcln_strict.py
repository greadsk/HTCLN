from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn

from .starnet import StarHead
from .tcln import CosinePositionalEncoding


class MultiKernelCNN2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 8, kernel_sizes: Sequence[int] = (1, 3, 5)):
        super().__init__()
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=int(in_channels),
                    out_channels=int(out_channels),
                    kernel_size=int(k),
                    padding=(int(k) - 1) // 2,
                    bias=True,
                )
                for k in self.kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for conv in self.convs:
            y = conv(x)
            feats.append(y.mean(dim=(2, 3)))
        return torch.cat(feats, dim=1)


class TCLNEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        nhead: int = 3,
        ff_mult: int = 4,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=int(d_model), num_heads=int(nhead), dropout=float(dropout), batch_first=True)
        self.mha_norm = nn.LayerNorm(int(d_model))

        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) * int(ff_mult)),
            nn.ReLU(),
            nn.Linear(int(d_model) * int(ff_mult), int(d_model)),
        )
        self.ffn_norm = nn.LayerNorm(int(d_model))

        self.lstm = nn.LSTM(input_size=int(d_model), hidden_size=int(lstm_hidden), batch_first=True)
        self.lstm_proj = nn.Linear(int(lstm_hidden), int(d_model))
        self.lstm_norm = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.mha_norm(x + attn_out)

        ff_out = self.ffn(x)
        x = self.ffn_norm(x + ff_out)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        x = self.lstm_norm(x + lstm_out)
        return x


class TCLNStrict(nn.Module):
    def __init__(
        self,
        *,
        tw: int,
        kernel_sizes: Sequence[int] = (1, 3, 5),
        cnn_out_channels: int = 8,
        d_model: int = 64,
        nhead: int = 3,
        num_layers: int = 2,
        ff_mult: int = 4,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.tw = int(tw)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        self.mk = MultiKernelCNN2D(in_channels=1, out_channels=int(cnn_out_channels), kernel_sizes=kernel_sizes)
        mk_feat_dim = int(cnn_out_channels) * len(tuple(kernel_sizes))
        self.token_proj = nn.Linear(mk_feat_dim, int(d_model))

        self.pe = CosinePositionalEncoding(int(d_model))
        self.layers = nn.ModuleList(
            [
                TCLNEncoderLayer(d_model=int(d_model), nhead=int(nhead), ff_mult=int(ff_mult), lstm_hidden=int(lstm_hidden), dropout=float(dropout))
                for _ in range(int(num_layers))
            ]
        )

        self.ar = nn.Linear(int(tw), 1, bias=True)

        fuse_in = int(num_layers) * int(d_model) + 1
        self.relu = nn.ReLU()
        if hadamard:
            self.head = StarHead(in_dim=fuse_in, hidden_dim=int(star_hidden), out_dim=1)
        else:
            self.head = nn.Linear(fuse_in, 1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, (tuple, list)):
            x_img_seq, y_hist = inputs
        else:
            raise ValueError("TCLNStrict expects inputs=(x_img_seq, y_hist)")

        b, tw, c, h, w = x_img_seq.shape
        if tw != self.tw:
            raise ValueError(f"expected tw={self.tw}, got {tw}")
        if c != 1:
            raise ValueError(f"expected channel=1, got {c}")

        x2 = x_img_seq.view(b * tw, c, h, w)
        feats = self.mk(x2)
        feats = feats.view(b, tw, -1)
        tokens = self.token_proj(feats)
        x = self.pe(tokens)

        o_list: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            o_list.append(x[:, -1, :])

        o_ar = self.ar(y_hist)
        z = torch.cat(o_list + [o_ar], dim=1)
        z = self.relu(z)
        y = self.head(z).squeeze(-1)
        return y
