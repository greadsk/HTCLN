from __future__ import annotations

import torch
from torch import nn

from .starnet import StarHead ,Stage
#请在starnet.py中改动StarNet_plain的参数


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.fc = StarHead(in_dim=hidden_dim, hidden_dim=star_hidden, out_dim=1)
        else:
            self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class ImgSeqCNNLSTMRegressor(nn.Module):
    """Per-time-step CNN feature extractor + LSTM.

    Input: (B, T, 1, H, W)
    Output: (B,)
    """

    def __init__(self, cnn_hidden: int = 32, lstm_hidden: int = 64, lstm_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.cnn(x).squeeze(-1).squeeze(-1)
        feat = feat.view(b, t, -1)
        out, _ = self.lstm(feat)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=hidden_dim, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y.squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=d_model, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        x = self.in_proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        vec = x[:, -1, :]
        y = self.head(vec)
        return y.squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=channels, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        feat = self.net(x)
        vec = feat.mean(dim=-1)
        y = self.head(vec)
        return y.squeeze(-1)

#################################################
#开始修改
"""
CNN2DRegressor
"""
class CNN2DRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        hidden_dim1: int = 64,
        hidden_dim2: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim1, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim1, hidden_dim2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim2, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=channels, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        feat = self.net(x)
        vec = feat.mean(dim=-1)
        y = self.head(vec)
        return y.squeeze(-1)


"""
CNN3DRegressor
"""
class CNN3DRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 64,
        hidden_dim1: int = 64,
        hidden_dim2: int = 64,
        hidden_dim3: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim1, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim1, hidden_dim2, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim2, hidden_dim3, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden_dim3, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),            
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=channels, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        feat = self.net(x)
        vec = feat.mean(dim=-1)
        y = self.head(vec)
        return y.squeeze(-1)
#修改结束
###################################
#加入新模型，标准StarNet_plain模型
class StarNet_plain(nn.Module):
    def __init__(
    self, 
    input_dim: int,
    in_channels :int = 64, 
    out_channels:int = 64, 
    num_blocks:int = 1, 
    hidden_dim=64,
    output_dim :int = 1,
    ):
        super(StarNet_plain, self).__init__()
        self.stage1 = Stage(input_dim, in_channels, num_blocks, hidden_dim)
        self.stage2 = Stage(in_channels, out_channels, num_blocks, hidden_dim)
        self.stage3 = Stage(in_channels, out_channels, num_blocks, hidden_dim)
        self.stage4 = Stage(in_channels, out_channels, num_blocks, hidden_dim)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        return y.squeeze(-1)
#######################################################3


class DARNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        enc_hidden: int = 64,
        dec_hidden: int = 64,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.enc_hidden = int(enc_hidden)
        self.dec_hidden = int(dec_hidden)

        self.enc_cell = nn.LSTMCell(self.input_dim, self.enc_hidden)
        self.enc_score_hc = nn.Linear(self.enc_hidden * 2, self.input_dim)
        self.enc_score_x = nn.Linear(self.input_dim, self.input_dim)

        self.dec_cell = nn.LSTMCell(self.enc_hidden, self.dec_hidden)
        self.dec_score_hc = nn.Linear(self.dec_hidden * 2, self.enc_hidden)
        self.dec_score_e = nn.Linear(self.enc_hidden, self.enc_hidden)
        self.dec_score_out = nn.Linear(self.enc_hidden, 1)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        fuse_dim = self.dec_hidden + self.enc_hidden
        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=fuse_dim, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(fuse_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        device = x.device

        h_e = torch.zeros((b, self.enc_hidden), device=device, dtype=x.dtype)
        c_e = torch.zeros((b, self.enc_hidden), device=device, dtype=x.dtype)

        enc_h_list = []
        for i in range(t):
            x_t = x[:, i, :]
            hc = torch.cat([h_e, c_e], dim=1)
            score = torch.tanh(self.enc_score_hc(hc) + self.enc_score_x(x_t))
            alpha = torch.softmax(score, dim=1)
            x_att = alpha * x_t
            h_e, c_e = self.enc_cell(x_att, (h_e, c_e))
            enc_h_list.append(h_e)

        enc_h = torch.stack(enc_h_list, dim=1)

        h_d = torch.zeros((b, self.dec_hidden), device=device, dtype=x.dtype)
        c_d = torch.zeros((b, self.dec_hidden), device=device, dtype=x.dtype)

        context = enc_h[:, -1, :]
        for _ in range(t):
            hc_d = torch.cat([h_d, c_d], dim=1)
            q = self.dec_score_hc(hc_d).unsqueeze(1)
            k = self.dec_score_e(enc_h)
            e = torch.tanh(q + k)
            beta = self.dec_score_out(e).squeeze(-1)
            beta = torch.softmax(beta, dim=1)
            context = torch.sum(enc_h * beta.unsqueeze(-1), dim=1)
            h_d, c_d = self.dec_cell(context, (h_d, c_d))
            h_d = self.dropout(h_d)

        z = torch.cat([h_d, context], dim=1)
        y = self.head(z)
        return y.squeeze(-1)


class TPALSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 64,
        conv_channels: int = 32,
        conv_kernel: int = 3,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, batch_first=True)
        pad = (conv_kernel - 1) // 2
        self.conv = nn.Conv1d(lstm_hidden, conv_channels, kernel_size=conv_kernel, padding=pad)
        self.query_proj = nn.Linear(lstm_hidden, conv_channels)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        fuse_dim = lstm_hidden * 2
        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=fuse_dim, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(fuse_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.lstm(x)
        h_seq = self.dropout(h_seq)
        h_last = h_seq[:, -1, :]

        conv_feat = self.conv(h_seq.transpose(1, 2))
        q = self.query_proj(h_last).unsqueeze(-1)
        scores = torch.sum(conv_feat * q, dim=1)
        attn = torch.softmax(scores, dim=1)
        context = torch.sum(h_seq * attn.unsqueeze(-1), dim=1)

        z = torch.cat([h_last, context], dim=1)
        y = self.head(z)
        return y.squeeze(-1)


class DAConvLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attn_hidden: int = 64,
        conv_channels: int = 64,
        conv_kernel: int = 3,
        lstm_hidden: int = 64,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.attn_hidden = int(attn_hidden)

        self.cell = nn.LSTMCell(self.input_dim, self.attn_hidden)
        self.score_hc = nn.Linear(self.attn_hidden * 2, self.input_dim)
        self.score_x = nn.Linear(self.input_dim, self.input_dim)

        pad = (conv_kernel - 1) // 2
        self.conv = nn.Conv1d(self.input_dim, conv_channels, kernel_size=conv_kernel, padding=pad)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=lstm_hidden, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        device = x.device

        h = torch.zeros((b, self.attn_hidden), device=device, dtype=x.dtype)
        c = torch.zeros((b, self.attn_hidden), device=device, dtype=x.dtype)
        xs = []
        for i in range(t):
            x_t = x[:, i, :]
            hc = torch.cat([h, c], dim=1)
            score = torch.tanh(self.score_hc(hc) + self.score_x(x_t))
            alpha = torch.softmax(score, dim=1)
            x_att = alpha * x_t
            h, c = self.cell(x_att, (h, c))
            xs.append(x_att)

        x_att_seq = torch.stack(xs, dim=1)
        conv_feat = self.conv(x_att_seq.transpose(1, 2)).transpose(1, 2)
        conv_feat = self.dropout(conv_feat)

        out, _ = self.lstm(conv_feat)
        last = out[:, -1, :]
        y = self.head(last)
        return y.squeeze(-1)


class TSConvLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 32,
        conv_kernel_t: int = 3,
        conv_kernel_f: int = 3,
        lstm_hidden: int = 64,
        dropout: float = 0.0,
        hadamard: bool = False,
        star_hidden: int = 64,
    ):
        super().__init__()
        pad_t = (conv_kernel_t - 1) // 2
        pad_f = (conv_kernel_f - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=(conv_kernel_t, conv_kernel_f), padding=(pad_t, pad_f)),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(conv_kernel_t, conv_kernel_f), padding=(pad_t, pad_f)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.hadamard = bool(hadamard)
        if self.hadamard:
            self.head = StarHead(in_dim=lstm_hidden, hidden_dim=star_hidden, out_dim=1)
        else:
            self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F) -> (B,1,T,F)
        x2 = x.unsqueeze(1)
        feat = self.conv(x2)  # (B,C,T,F)
        feat = feat.mean(dim=3)  # (B,C,T)
        feat = feat.transpose(1, 2)  # (B,T,C)
        feat = self.dropout(feat)

        out, _ = self.lstm(feat)
        last = out[:, -1, :]
        y = self.head(last)
        return y.squeeze(-1)
