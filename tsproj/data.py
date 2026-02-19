from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import Ablation2DConfig, ExperimentConfig


def default_dataset_paths(root: Optional[str] = None) -> Dict[str, str]:
    root = root or os.getcwd()
    return {
        "ETTh2": os.path.join(root, "all_six_datasets", "ETT-small", "ETTh2.csv"),
        "electricity": os.path.join(root, "all_six_datasets", "electricity", "electricity.csv"),
        "exchange_rate": os.path.join(root, "all_six_datasets", "exchange_rate", "exchange_rate.csv"),
        "illness": os.path.join(root, "all_six_datasets", "illness", "national_illness.csv"),
        "traffic": os.path.join(root, "all_six_datasets", "traffic", "traffic.csv"),
        "weather": os.path.join(root, "all_six_datasets", "weather", "weather.csv"),
        "gold": os.path.join(root, "all_six_datasets", "gold", "gold.csv"),
        "NASDAQ100": os.path.join(root, "all_six_datasets", "NASDAQ100", "nasdaq100_small.csv"),
        "sml2010": os.path.join(root, "all_six_datasets", "sml2010", "sml2010_1.csv")
    }


def load_features_and_target(csv_path: str, target_column: str = "OT") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    time_col = df.columns[0]
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in {csv_path}")

    feature_cols = [c for c in df.columns if c not in [time_col, target_column]]
    X = df[feature_cols].astype(np.float32).values
    y = df[target_column].astype(np.float32).values
    return X, y, feature_cols


def make_supervised_samples(
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    seq_len: int,
    horizon: int,
    time_stride: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    if time_stride <= 0:
        raise ValueError("time_stride must be >= 1")

    n = len(X_all)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all[:train_end])
    X_scaled = np.concatenate([X_scaled, scaler.transform(X_all[train_end:])], axis=0)

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    min_t = seq_len * time_stride
    for t in range(min_t, n - horizon + 1):
        y_idx = t + horizon - 1
        x_seq = X_scaled[t - seq_len * time_stride : t : time_stride]
        if len(x_seq) != seq_len:
            continue

        y_point = y_all[y_idx]

        if y_idx < train_end:
            X_train.append(x_seq)
            y_train.append(y_point)
        elif y_idx < val_end:
            X_val.append(x_seq)
            y_val.append(y_point)
        else:
            X_test.append(x_seq)
            y_test.append(y_point)

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(y_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        scaler,
    )


def make_supervised_samples_with_yhist(
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    seq_len: int,
    horizon: int,
    time_stride: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    if time_stride <= 0:
        raise ValueError("time_stride must be >= 1")

    n = len(X_all)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all[:train_end])
    X_scaled = np.concatenate([X_scaled, scaler.transform(X_all[train_end:])], axis=0)

    X_train, yhist_train, y_train = [], [], []
    X_val, yhist_val, y_val = [], [], []
    X_test, yhist_test, y_test = [], [], []

    min_t = seq_len * time_stride
    for t in range(min_t, n - horizon + 1):
        y_idx = t + horizon - 1
        x_seq = X_scaled[t - seq_len * time_stride : t : time_stride]
        y_hist = y_all[t - seq_len * time_stride : t : time_stride]
        if len(x_seq) != seq_len or len(y_hist) != seq_len:
            continue

        y_point = y_all[y_idx]

        if y_idx < train_end:
            X_train.append(x_seq)
            yhist_train.append(y_hist)
            y_train.append(y_point)
        elif y_idx < val_end:
            X_val.append(x_seq)
            yhist_val.append(y_hist)
            y_val.append(y_point)
        else:
            X_test.append(x_seq)
            yhist_test.append(y_hist)
            y_test.append(y_point)

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(yhist_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(yhist_val, dtype=np.float32),
        np.asarray(y_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(yhist_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        scaler,
    )


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class VarToImageSeqDataset(Dataset):
    """Main-experiment 2D reconstruction: each time step M vars -> p×p, pad zeros, then padding=1.

    Output: (seq_len, 1, H, W)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seq_len: int,
        forced_p: Optional[int] = None,
        feature_stride: int = 1,
        padding: int = 1,
    ):
        if X.shape[1] != seq_len:
            raise ValueError(f"expected seq_len={seq_len}, got {X.shape[1]}")
        if feature_stride <= 0:
            raise ValueError("feature_stride must be >= 1")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.M = int(self.X.shape[2])
        self.p = int(forced_p) if forced_p is not None else int(math.ceil(math.sqrt(self.M)))
        self.flat_size = self.p * self.p
        self.padding = int(padding)

        idxs = list(range(0, self.M, feature_stride))
        self.select_len = min(len(idxs), self.flat_size)
        self.select_idx = idxs[: self.select_len]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx]  # (T,M)
        y_val = self.y[idx]

        x_sel = x_seq[:, self.select_idx]  # (T,K)
        flat = torch.zeros((x_seq.shape[0], self.flat_size), dtype=torch.float32)
        flat[:, : self.select_len] = x_sel
        img = flat.view(x_seq.shape[0], self.p, self.p)  # (T,p,p)

        if self.padding > 0:
            img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0.0)

        img = img.unsqueeze(1)  # (T,1,H,W)
        return img, y_val


class VarToImageSeqWithYHistDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_hist: np.ndarray,
        y: np.ndarray,
        *,
        seq_len: int,
        forced_p: Optional[int] = None,
        feature_stride: int = 1,
        padding: int = 1,
    ):
        if X.shape[1] != seq_len:
            raise ValueError(f"expected seq_len={seq_len}, got {X.shape[1]}")
        if y_hist.shape[1] != seq_len:
            raise ValueError(f"expected y_hist seq_len={seq_len}, got {y_hist.shape[1]}")
        if feature_stride <= 0:
            raise ValueError("feature_stride must be >= 1")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_hist = torch.tensor(y_hist, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.M = int(self.X.shape[2])
        self.p = int(forced_p) if forced_p is not None else int(math.ceil(math.sqrt(self.M)))
        self.flat_size = self.p * self.p
        self.padding = int(padding)

        idxs = list(range(0, self.M, feature_stride))
        self.select_len = min(len(idxs), self.flat_size)
        self.select_idx = idxs[: self.select_len]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx]  # (T,M)
        y_hist = self.y_hist[idx]  # (T,)
        y_val = self.y[idx]

        x_sel = x_seq[:, self.select_idx]  # (T,K)
        flat = torch.zeros((x_seq.shape[0], self.flat_size), dtype=torch.float32)
        flat[:, : self.select_len] = x_sel
        img = flat.view(x_seq.shape[0], self.p, self.p)  # (T,p,p)

        if self.padding > 0:
            img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0.0)

        img = img.unsqueeze(1)  # (T,1,H,W)
        return (img, y_hist), y_val


class TimeTo2DWindowDataset(Dataset):
    """Research-model preprocessing.

    For each sample, select window_points along time with interval time_stride.
    Then reshape (window_points,) to (side, side) per variable/channel.

    Output: (C=p, H, W) where H=W=side (+2 if padding=1), and optional upsample.

    This matches the spec items you mentioned:
    - window_points: 25/36/49/64/100 -> 5×5~10×10
    - time_stride: 5/6/7/8/10 (or others)
    - padding=1
    - optional upsample
    - input shape (k+1)×(k+1)×p where (k+1)=side
    """

    def __init__(
        self,
        X_all_scaled: np.ndarray,
        y_all: np.ndarray,
        *,
        horizon: int,
        cfg: Ablation2DConfig,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split: str = "train",  # train/val/test
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train/val/test")

        self.X_all = X_all_scaled
        self.y_all = y_all
        self.horizon = int(horizon)
        self.cfg = cfg

        self.side = cfg.side()
        self.window_points = int(cfg.window_points)
        self.time_stride = int(cfg.time_stride)
        self.padding = int(cfg.padding)
        self.upsample_to = cfg.upsample_to

        n = len(X_all_scaled)
        self.train_end = int(n * train_ratio)
        self.val_end = int(n * (train_ratio + val_ratio))

        # precompute valid end indices for the split
        self.indices: List[int] = []
        min_t = self.window_points * self.time_stride
        for t in range(min_t, n - self.horizon + 1):
            y_idx = t + self.horizon - 1
            if split == "train" and y_idx < self.train_end:
                self.indices.append(t)
            elif split == "val" and self.train_end <= y_idx < self.val_end:
                self.indices.append(t)
            elif split == "test" and y_idx >= self.val_end:
                self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        y_idx = t + self.horizon - 1

        window = self.X_all[t - self.window_points * self.time_stride : t : self.time_stride]  # (W,p)
        if len(window) != self.window_points:
            raise RuntimeError("unexpected window length")

        # (W,p) -> (p,W) -> (p,side,side)
        x = torch.tensor(window.T, dtype=torch.float32).contiguous()
        x = x.view(x.shape[0], self.side, self.side)

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0.0)

        if self.upsample_to is not None:
            x = x.unsqueeze(0)  # (1,C,H,W)
            x = F.interpolate(x, size=(self.upsample_to, self.upsample_to), mode="bilinear", align_corners=False)
            x = x.squeeze(0)

        y = torch.tensor(self.y_all[y_idx], dtype=torch.float32)
        return x, y


def build_loaders_seq(
    csv_path: str,
    *,
    exp: ExperimentConfig,
    horizon: int,
    time_stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    X_all, y_all, feat_cols = load_features_and_target(csv_path)
    X_tr, y_tr, X_va, y_va, X_te, y_te, scaler = make_supervised_samples(
        X_all,
        y_all,
        seq_len=exp.seq_len,
        horizon=horizon,
        time_stride=time_stride,
    )

    meta = {"num_features": len(feat_cols), "feature_cols": feat_cols}

    train_dl = DataLoader(SeqDataset(X_tr, y_tr), batch_size=exp.batch_size, shuffle=True, num_workers=exp.num_workers)
    val_dl = DataLoader(SeqDataset(X_va, y_va), batch_size=exp.batch_size, shuffle=False, num_workers=exp.num_workers)
    test_dl = DataLoader(SeqDataset(X_te, y_te), batch_size=exp.batch_size, shuffle=False, num_workers=exp.num_workers)
    return train_dl, val_dl, test_dl, meta


def build_loaders_tcln_strict(
    csv_path: str,
    *,
    exp: ExperimentConfig,
    horizon: int,
    forced_p: Optional[int] = None,
    feature_stride: int = 1,
    time_stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    X_all, y_all, feat_cols = load_features_and_target(csv_path)
    X_tr, yh_tr, y_tr, X_va, yh_va, y_va, X_te, yh_te, y_te, scaler = make_supervised_samples_with_yhist(
        X_all,
        y_all,
        seq_len=exp.seq_len,
        horizon=horizon,
        time_stride=time_stride,
    )

    num_features = len(feat_cols)
    p = int(forced_p) if forced_p is not None else int(math.ceil(math.sqrt(num_features)))

    meta = {
        "num_features": num_features,
        "p": p,
        "H": p + 2,
        "W": p + 2,
        "forced_p": forced_p,
        "feature_stride": int(feature_stride),
        "time_stride": int(time_stride),
        "tw": int(exp.seq_len),
    }

    train_dl = DataLoader(
        VarToImageSeqWithYHistDataset(
            X_tr,
            yh_tr,
            y_tr,
            seq_len=exp.seq_len,
            forced_p=forced_p,
            feature_stride=feature_stride,
            padding=1,
        ),
        batch_size=exp.batch_size,
        shuffle=True,
        num_workers=exp.num_workers,
    )
    val_dl = DataLoader(
        VarToImageSeqWithYHistDataset(
            X_va,
            yh_va,
            y_va,
            seq_len=exp.seq_len,
            forced_p=forced_p,
            feature_stride=feature_stride,
            padding=1,
        ),
        batch_size=exp.batch_size,
        shuffle=False,
        num_workers=exp.num_workers,
    )
    test_dl = DataLoader(
        VarToImageSeqWithYHistDataset(
            X_te,
            yh_te,
            y_te,
            seq_len=exp.seq_len,
            forced_p=forced_p,
            feature_stride=feature_stride,
            padding=1,
        ),
        batch_size=exp.batch_size,
        shuffle=False,
        num_workers=exp.num_workers,
    )

    return train_dl, val_dl, test_dl, meta


def build_loaders_var_img_seq(
    csv_path: str,
    *,
    exp: ExperimentConfig,
    horizon: int,
    forced_p: Optional[int] = None,
    feature_stride: int = 1,
    time_stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    X_all, y_all, feat_cols = load_features_and_target(csv_path)
    X_tr, y_tr, X_va, y_va, X_te, y_te, scaler = make_supervised_samples(
        X_all,
        y_all,
        seq_len=exp.seq_len,
        horizon=horizon,
        time_stride=time_stride,
    )

    num_features = len(feat_cols)
    p = int(forced_p) if forced_p is not None else int(math.ceil(math.sqrt(num_features)))

    meta = {
        "num_features": num_features,
        "p": p,
        "H": p + 2,
        "W": p + 2,
        "forced_p": forced_p,
        "feature_stride": int(feature_stride),
        "time_stride": int(time_stride),
    }

    train_dl = DataLoader(
        VarToImageSeqDataset(X_tr, y_tr, seq_len=exp.seq_len, forced_p=forced_p, feature_stride=feature_stride, padding=1),
        batch_size=exp.batch_size,
        shuffle=True,
        num_workers=exp.num_workers,
    )
    val_dl = DataLoader(
        VarToImageSeqDataset(X_va, y_va, seq_len=exp.seq_len, forced_p=forced_p, feature_stride=feature_stride, padding=1),
        batch_size=exp.batch_size,
        shuffle=False,
        num_workers=exp.num_workers,
    )
    test_dl = DataLoader(
        VarToImageSeqDataset(X_te, y_te, seq_len=exp.seq_len, forced_p=forced_p, feature_stride=feature_stride, padding=1),
        batch_size=exp.batch_size,
        shuffle=False,
        num_workers=exp.num_workers,
    )

    return train_dl, val_dl, test_dl, meta


def build_loaders_time_2d(
    csv_path: str,
    *,
    exp: ExperimentConfig,
    horizon: int,
    cfg: Ablation2DConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    X_all, y_all, feat_cols = load_features_and_target(csv_path)

    # scale using train segment only
    n = len(X_all)
    train_end = int(n * 0.8)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all[:train_end])
    X_scaled = np.concatenate([X_scaled, scaler.transform(X_all[train_end:])], axis=0)

    train_ds = TimeTo2DWindowDataset(X_scaled, y_all, horizon=horizon, cfg=cfg, split="train")
    val_ds = TimeTo2DWindowDataset(X_scaled, y_all, horizon=horizon, cfg=cfg, split="val")
    test_ds = TimeTo2DWindowDataset(X_scaled, y_all, horizon=horizon, cfg=cfg, split="test")

    meta = {
        "num_features": len(feat_cols),
        "side": cfg.side(),
        "window_points": cfg.window_points,
        "time_stride": cfg.time_stride,
        "padding": cfg.padding,
        "upsample_to": cfg.upsample_to,
        "H": (cfg.side() + 2) if cfg.padding else cfg.side(),
        "W": (cfg.side() + 2) if cfg.padding else cfg.side(),
        "cfg": asdict(cfg),
    }

    train_dl = DataLoader(train_ds, batch_size=exp.batch_size, shuffle=True, num_workers=exp.num_workers)
    val_dl = DataLoader(val_ds, batch_size=exp.batch_size, shuffle=False, num_workers=exp.num_workers)
    test_dl = DataLoader(test_ds, batch_size=exp.batch_size, shuffle=False, num_workers=exp.num_workers)
    return train_dl, val_dl, test_dl, meta
