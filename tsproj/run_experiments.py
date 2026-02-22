from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
import torch
from torch import nn

from .config import Ablation2DConfig, ExperimentConfig
from .data import (
    build_loaders_seq,
    build_loaders_tcln_strict,
    build_loaders_time_2d,
    build_loaders_var_img_seq,
    default_dataset_paths,
)
from .models.baselines import (
    CNN1DRegressor,
    CNN2DRegressor,
    CNN3DRegressor,
    DAConvLSTMRegressor,
    DARNNRegressor,
    GRURegressor,
    ImgSeqCNNLSTMRegressor,
    LSTMRegressor,
    TPALSTMRegressor,
    TSConvLSTMRegressor,
    TransformerRegressor,
    StarNet_plain,
    
)
from .models.starnet import HadamardConcatFusion
from .models.tcln import TCLNLike
from .models.tcln_strict import TCLNStrict
from .profile import profile_model
from .train import evaluate_timed, train_model


def _to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, device) for v in x)
    return x


class Simple2DStarNetRegressor(nn.Module):
    """A minimal research-model baseline that demonstrates StarNet/Hadamard fusion.

    Pipeline:
      2D input (B,C,H,W) -> GAP -> feature vector
      + a simple scalar branch -> fuse with HadamardConcatFusion -> output

    This is not yet the full TCLN; it's a clean hook point for your StarNet/Hadamard requirement.
    """

    def __init__(self, in_channels: int, feat_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.scalar_head = nn.Linear(in_channels, 1)
        self.fusion = HadamardConcatFusion(a_dim=feat_dim, b_dim=1, hidden_dim=feat_dim, out_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B,feat_dim)
        scalar = self.scalar_head(x.mean(dim=(2, 3)))  # (B,1)
        y = self.fusion(feat, scalar).squeeze(-1)
        return y


def run_one(
    *,
    exp: ExperimentConfig,
    dataset_csv: str,
    dataset_name: str,
    model_name: str,
    horizon: int,
    device: torch.device,
    ablation_2d: Optional[Ablation2DConfig] = None,
    forced_p: Optional[int] = None,
    feature_stride: int = 1,
    time_stride: int = 1,
):
    is_star = model_name.endswith("_star")
    base_model_name = model_name[: -len("_star")] if is_star else model_name

    if model_name in {"lstm", "lstm_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = LSTMRegressor(
            input_dim=meta["num_features"],
            hidden_dim=64,
            num_layers=2,
            hadamard=(model_name == "lstm_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    elif model_name in {"gru", "gru_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = GRURegressor(
            input_dim=meta["num_features"],
            hidden_dim=64,
            num_layers=2,
            hadamard=(model_name == "gru_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    elif model_name in {"transformer", "transformer_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = TransformerRegressor(
            input_dim=meta["num_features"],
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            hadamard=(model_name == "transformer_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))
###########################################
#开始修改
    elif model_name in {"cnn1d", "cnn1d_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = CNN1DRegressor(
            input_dim=meta["num_features"],
            channels=96,
            kernel_size=3,
            dropout=0.0,
            hadamard=(model_name == "cnn1d_star"),
            star_hidden=32,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    #新加的模型2d
    elif model_name in {"cnn2d", "cnn2d_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = CNN2DRegressor(
            input_dim=meta["num_features"],
            channels=96,
            hidden_dim1= 96,
            hidden_dim2= 64,
            kernel_size=3,
            dropout=0.0,
            hadamard=(model_name == "cnn2d_star"),
            star_hidden=32,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))    

#新加的模型3d
    elif model_name in {"cnn3d", "cnn3d_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = CNN3DRegressor(
            input_dim=meta["num_features"],
            channels=96,
            hidden_dim1= 96,
            hidden_dim2= 96,
            hidden_dim3= 64,
            kernel_size=3,
            dropout=0.0,
            hadamard=(model_name == "cnn3d_star"),
            star_hidden=32,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))    

###############
#新加入starnet_plain模型
    elif model_name == "StarNet_plain":
        if ablation_2d is None:
            ablation_2d = Ablation2DConfig(window_points=25, time_stride=1, padding=1, upsample_to=None)
        train_dl, val_dl, test_dl, meta = build_loaders_time_2d(dataset_csv, exp=exp, horizon=horizon, cfg=ablation_2d)
        model = StarNet_plain(
            input_dim=meta["num_features"],
            in_channels=64,
            out_channels=64,
            num_blocks=1,
            hidden_dim=64,
            output_dim=1,
        )
        example_input = torch.zeros((1, meta["num_features"], meta["H"], meta["W"]))



##结束修改        

    elif model_name in {"da_rnn", "da_rnn_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = DARNNRegressor(
            input_dim=meta["num_features"],
            enc_hidden=64,
            dec_hidden=64,
            dropout=0.0,
            hadamard=(model_name == "da_rnn_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    elif model_name in {"tpa_lstm", "tpa_lstm_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = TPALSTMRegressor(
            input_dim=meta["num_features"],
            lstm_hidden=64,
            conv_channels=32,
            conv_kernel=3,
            dropout=0.0,
            hadamard=(model_name == "tpa_lstm_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    elif model_name in {"da_conv_lstm", "da_conv_lstm_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = DAConvLSTMRegressor(
            input_dim=meta["num_features"],
            attn_hidden=64,
            conv_channels=64,
            conv_kernel=3,
            lstm_hidden=64,
            dropout=0.0,
            hadamard=(model_name == "da_conv_lstm_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    elif model_name in {"ts_conv_lstm", "ts_conv_lstm_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_seq(dataset_csv, exp=exp, horizon=horizon, time_stride=time_stride)
        model = TSConvLSTMRegressor(
            input_dim=meta["num_features"],
            conv_channels=32,
            conv_kernel_t=3,
            conv_kernel_f=3,
            lstm_hidden=64,
            dropout=0.0,
            hadamard=(model_name == "ts_conv_lstm_star"),
            star_hidden=64,
        )
        example_input = torch.zeros((1, exp.seq_len, meta["num_features"]))

    elif model_name == "img_cnnlstm":
        train_dl, val_dl, test_dl, meta = build_loaders_var_img_seq(
            dataset_csv,
            exp=exp,
            horizon=horizon,
            forced_p=forced_p,
            feature_stride=feature_stride,
            time_stride=time_stride,
        )
        model = ImgSeqCNNLSTMRegressor(cnn_hidden=32, lstm_hidden=64, lstm_layers=2)
        example_input = torch.zeros((1, exp.seq_len, 1, meta["H"], meta["W"]))

    elif model_name == "time2d_starnet":
        if ablation_2d is None:
            ablation_2d = Ablation2DConfig(window_points=25, time_stride=1, padding=1, upsample_to=None)
        train_dl, val_dl, test_dl, meta = build_loaders_time_2d(dataset_csv, exp=exp, horizon=horizon, cfg=ablation_2d)
        model = Simple2DStarNetRegressor(in_channels=meta["num_features"], feat_dim=64)
        example_input = torch.zeros((1, meta["num_features"], meta["H"], meta["W"]))

    elif model_name in {"tcln_plain", "tcln_starnet"}:
        if ablation_2d is None:
            ablation_2d = Ablation2DConfig(window_points=25, time_stride=1, padding=1, upsample_to=None)
        train_dl, val_dl, test_dl, meta = build_loaders_time_2d(dataset_csv, exp=exp, horizon=horizon, cfg=ablation_2d)
        fusion = "plain" if model_name == "tcln_plain" else "starnet"
        model = TCLNLike(
            in_channels=meta["num_features"],
            h=meta["H"],
            w=meta["W"],
            kernel_sizes=(1, 3, 5),
            d_model=64,
            lstm_hidden=64,
            fusion=fusion,
            starnet_hidden=64,
            dropout=0.1,
        )
        example_input = torch.zeros((1, meta["num_features"], meta["H"], meta["W"]))

    elif model_name in {"tcln_strict", "tcln_strict_star"}:
        train_dl, val_dl, test_dl, meta = build_loaders_tcln_strict(
            dataset_csv,
            exp=exp,
            horizon=horizon,
            forced_p=forced_p,
            feature_stride=feature_stride,
            time_stride=time_stride,
        )
        model = TCLNStrict(
            tw=int(exp.seq_len),
            kernel_sizes=(1, 3, 5),
            cnn_out_channels=8,
            d_model=64,
            nhead=4,
            num_layers=2,
            ff_mult=4,
            lstm_hidden=64,
            dropout=0.1,
            hadamard=(model_name == "tcln_strict_star"),
            star_hidden=64,
        )
        example_input = (
            torch.zeros((1, exp.seq_len, 1, meta["H"], meta["W"])),
            torch.zeros((1, exp.seq_len)),
        )

    else:
        raise ValueError(f"unsupported model_name: {model_name}")

    suffix = ""
    if forced_p is not None:
        suffix += f"__p{forced_p}"
    if feature_stride != 1:
        suffix += f"__fstride{feature_stride}"
    if time_stride != 1:
        suffix += f"__tstride{time_stride}"
    if ablation_2d is not None and model_name == "time2d_starnet":
        suffix += f"__w{ablation_2d.window_points}__dt{ablation_2d.time_stride}"
        if ablation_2d.upsample_to is not None:
            suffix += f"__up{ablation_2d.upsample_to}"

    # Naming convention requirement:
    # - normal:  <model>__... (prefix)
    # - star:    <model>__..._star (suffix)
    run_name = f"{base_model_name}__{dataset_name}__h{horizon}{suffix}"
    if is_star:
        run_name = f"{run_name}_star"

    csv_path = train_model(model, train_dl, val_dl, exp=exp, run_name=run_name, device=device)

    loss_fn = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_infer_ms, test_infer_ms_per_sample = evaluate_timed(model.to(device), test_dl, loss_fn, device)

    prof = profile_model(model.to(device), _to_device(example_input, device))

    last_epoch_row = pd.read_csv(csv_path).iloc[-1]

    summary_row = {
        "run_name": run_name,
        "model": base_model_name,
        "star": bool(is_star),
        "dataset": dataset_name,
        "horizon": horizon,
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_infer_time_ms": float(test_infer_ms),
        "test_infer_ms_per_sample": float(test_infer_ms_per_sample),
        "params": int(prof.params),
        "flops": (int(prof.flops) if prof.flops is not None else None),
        "flops_backend": prof.backend,
        "last_epoch_time_sec": float(last_epoch_row["epoch_time_sec"]),
        "last_epoch_time_ms": float(last_epoch_row.get("epoch_time_ms", float("nan"))),
        "last_train_time_ms": float(last_epoch_row.get("train_time_ms", float("nan"))),
        "last_val_infer_time_ms": float(last_epoch_row.get("val_infer_time_ms", float("nan"))),
    }

    # attach meta for traceability
    for k, v in meta.items():
        if k == "cfg":
            continue
        summary_row[k] = v

    return csv_path, summary_row


def run_batch(exp: ExperimentConfig, *, models=("lstm", "img_cnnlstm")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = default_dataset_paths()

    all_rows = []
    for dataset_name, csv_path in paths.items():
        for horizon in exp.horizons:
            for model_name in models:
                csv_path_out, row = run_one(
                    exp=exp,
                    dataset_csv=csv_path,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    horizon=horizon,
                    device=device,
                )
                all_rows.append(row)
                print("done:", csv_path_out)

    os.makedirs(exp.results_dir, exist_ok=True)
    out = os.path.join(exp.results_dir, "summary.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print("summary saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "lstm",
            "lstm_star",
            "gru",
            "gru_star",
            "transformer",
            "transformer_star",
            "cnn1d",
            "cnn1d_star",
            "da_rnn",
            "da_rnn_star",
            "tpa_lstm",
            "tpa_lstm_star",
            "da_conv_lstm",
            "da_conv_lstm_star",
            # "ts_conv_lstm",
            # "ts_conv_lstm_star",
            "img_cnnlstm",
            "time2d_starnet",
            "tcln_plain",
            "tcln_starnet",
            "tcln_strict",
            "tcln_strict_star",
            "cnn2d",
            "cnn2d_star",
            "cnn3d",
            "cnn3d_star",
            "StarNet_plain"
        ],
    )

    # ablation knobs for image-seq models
    parser.add_argument("--forced-p", type=int, default=None)
    parser.add_argument("--feature-stride", type=int, default=1)
    parser.add_argument("--time-stride", type=int, default=1)

    # research-model knobs for time2d
    parser.add_argument("--window-points", type=int, default=25)
    parser.add_argument("--upsample-to", type=int, default=None)

    args = parser.parse_args()

    exp = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = default_dataset_paths()

    # Single run mode (recommended for sanity checks)
    if args.dataset is not None or args.horizon is not None or args.model is not None:
        if args.dataset is None or args.horizon is None or args.model is None:
            raise ValueError("single-run mode requires --dataset, --horizon and --model")
        if args.dataset not in paths:
            raise ValueError(f"unknown dataset: {args.dataset}. valid={list(paths.keys())}")

        ablation_2d = None
        if args.model == "time2d_starnet":
            ablation_2d = Ablation2DConfig(
                window_points=int(args.window_points),
                time_stride=int(args.time_stride),
                padding=1,
                upsample_to=(int(args.upsample_to) if args.upsample_to is not None else None),
            )

        csv_path_out, row = run_one(
            exp=exp,
            dataset_csv=paths[args.dataset],
            dataset_name=args.dataset,
            model_name=args.model,
            horizon=int(args.horizon),
            device=device,
            ablation_2d=ablation_2d,
            forced_p=(int(args.forced_p) if args.forced_p is not None else None),
            feature_stride=int(args.feature_stride),
            time_stride=int(args.time_stride),
        )
        print("done:", csv_path_out)
        print(row)
    else:
        # Full batch mode
        run_batch(exp)
