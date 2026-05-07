from __future__ import annotations

import torch
from torch import nn


class StarOp(nn.Module):
    """Star operation from StarNet paper (rewritten form):

    Given x, compute:
        y = (W1 x + b1) ⊙ (W2 x + b2)

    This is essentially an element-wise (Hadamard) multiplication of two linear projections.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.proj2 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj1(x) * self.proj2(x)


class HadamardConcatFusion(nn.Module):
    """A generic Hadamard+concat fusion.

    For two inputs a,b:
      a' = Wa(a), b' = Wb(b)  (project to same dim d)
      h  = a' ⊙ b'
      out = Wo([a', b', h])

    This is a practical way to plug "StarNet/Hadamard fusion" into a forecasting head.
    """

    def __init__(self, a_dim: int, b_dim: int, hidden_dim: int, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.a_proj = nn.Linear(a_dim, hidden_dim)
        self.b_proj = nn.Linear(b_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden_dim * 3, out_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = self.a_proj(a)
        b = self.b_proj(b)
        h = a * b
        x = torch.cat([a, b, h], dim=-1)
        x = self.dropout(x)
        return self.out(x)


class StarHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 1):
        super().__init__()
        self.fc_gelu = nn.Linear(in_dim, hidden_dim)
        self.fc_id = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act(self.fc_gelu(x))
        b = self.fc_id(x)
        h = a * b
        return self.out(h)


###############################################################################
#标准starnet的模型
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWConvLayer, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn(x)
        return x

class StarBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(StarBlock, self).__init__()
        self.dw_conv1 = DWConvLayer(in_channels, in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_channels)
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        identity = x

        # DW-Conv
        x = self.dw_conv1(x)

        # FC layers
        b, c, _, _ = x.size()
        x = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)

        # Element-wise multiplication
        x = x.view(b, c, 1, 1)
        x = x * identity

        return x

class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, hidden_dim):
        super(Stage, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels)
        self.blocks = nn.Sequential(*[StarBlock(out_channels, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        return x

# class StarNet_plain(nn.Module):
#     def __init__(
#     self, 
#     input_dim: int,
#     in_channels :int = 64, 
#     out_channels:int = 64, 
#     num_blocks:int = 1, 
#     hidden_dim=64,
#     output_dim :int = 1,):
#         super(StarNet_plain, self).__init__()
#         self.stage1 = Stage(input_dim, in_channels, num_blocks, hidden_dim)
#         self.stage2 = Stage(in_channels, out_channels, num_blocks, hidden_dim)
#         self.stage3 = Stage(in_channels, out_channels, num_blocks, hidden_dim)
#         self.stage4 = Stage(in_channels, out_channels, num_blocks, hidden_dim)

#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(out_channels, output_dim)

#     def forward(self, x):
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)

#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         y = self.fc(x)

#         return y.squeeze(-1)
