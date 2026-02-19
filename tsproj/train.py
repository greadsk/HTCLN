from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig


def init_weights_xavier(model: nn.Module):
    for _, p in model.named_parameters():
        if p is None:
            continue
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _mae_rmse(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred = pred.view(-1)
    target = target.view(-1)
    mae = torch.mean(torch.abs(pred - target))
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return mae.item(), rmse.item()


def create_scheduler(optimizer, schedule: str, *, epochs: int, warmup_epochs: int, exp_gamma: float):
    if epochs <= 0:
        raise ValueError("epochs must be > 0")

    def lr_lambda(epoch_idx: int):
        if schedule == "exp":
            return exp_gamma ** float(epoch_idx)
        if schedule == "warmup_exp":
            if warmup_epochs <= 0:
                return exp_gamma ** float(epoch_idx)
            if epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)
            return exp_gamma ** float(epoch_idx - warmup_epochs)
        raise ValueError(f"unsupported lr_schedule: {schedule}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_device(x: Any, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def _batch_size(xb: Any, yb: torch.Tensor) -> int:
    if torch.is_tensor(xb):
        return int(xb.size(0))
    if isinstance(xb, (tuple, list)) and len(xb) > 0 and torch.is_tensor(xb[0]):
        return int(xb[0].size(0))
    return int(yb.size(0))


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn, device: torch.device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    for xb, yb in dataloader:
        xb = _to_device(xb, device)
        yb = yb.to(device)

        out = model(xb).view(-1)
        yb = yb.view(-1)

        loss = loss_fn(out, yb)
        total_loss += loss.item() * _batch_size(xb, yb)

        preds.append(out.detach().cpu())
        trues.append(yb.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    mae, rmse = _mae_rmse(preds, trues)

    return total_loss / max(1, len(dataloader.dataset)), mae, rmse


@torch.no_grad()
def evaluate_timed(model: nn.Module, dataloader: DataLoader, loss_fn, device: torch.device):
    """Evaluate with GPU-synchronized wall-time measurement (ms)."""
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    _sync_if_cuda(device)
    t0 = time.perf_counter()

    for xb, yb in dataloader:
        xb = _to_device(xb, device)
        yb = yb.to(device)

        out = model(xb).view(-1)
        yb = yb.view(-1)

        loss = loss_fn(out, yb)
        total_loss += loss.item() * _batch_size(xb, yb)

        preds.append(out.detach().cpu())
        trues.append(yb.detach().cpu())

    _sync_if_cuda(device)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    preds = torch.cat(preds, dim=0) if preds else torch.empty((0,))
    trues = torch.cat(trues, dim=0) if trues else torch.empty((0,))
    mae, rmse = _mae_rmse(preds, trues) if len(preds) else (0.0, 0.0)

    n = max(1, len(dataloader.dataset))
    return (total_loss / n), mae, rmse, float(infer_ms), float(infer_ms / float(n))


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    exp: ExperimentConfig,
    run_name: str,
    device: torch.device,
) -> str:
    os.makedirs(exp.results_dir, exist_ok=True)

    model = model.to(device)
    init_weights_xavier(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=exp.base_lr, weight_decay=exp.weight_decay)

    warmup_epochs = int(exp.epochs * exp.warmup_ratio) if exp.lr_schedule == "warmup_exp" else 0
    scheduler = create_scheduler(optimizer, exp.lr_schedule, epochs=exp.epochs, warmup_epochs=warmup_epochs, exp_gamma=exp.exp_gamma)

    csv_path = os.path.join(exp.results_dir, f"{run_name}.csv")
    params = count_parameters(model)
    records = []

    for epoch in range(1, exp.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_train_loss = 0.0

        current_lr = float(optimizer.param_groups[0]["lr"])

        _sync_if_cuda(device)
        train_start = time.perf_counter()
        for xb, yb in train_dl:
            xb = _to_device(xb, device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb).view(-1)
            yb = yb.view(-1)
            loss = loss_fn(out, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=exp.grad_clip_norm)
            optimizer.step()

            total_train_loss += loss.item() * _batch_size(xb, yb)

        _sync_if_cuda(device)
        train_end = time.perf_counter()
        train_time_ms = (train_end - train_start) * 1000.0

        val_loss, val_mae, val_rmse, val_infer_ms, val_infer_ms_per_sample = evaluate_timed(model, val_dl, loss_fn, device)
        epoch_end = time.perf_counter()
        epoch_time_ms = (epoch_end - epoch_start) * 1000.0
        train_loss = total_train_loss / max(1, len(train_dl.dataset))

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "val_rmse": float(val_rmse),
            "lr": float(current_lr),
            "epoch_time_sec": float(epoch_time_ms / 1000.0),
            "epoch_time_ms": float(epoch_time_ms),
            "train_time_ms": float(train_time_ms),
            "val_infer_time_ms": float(val_infer_ms),
            "val_infer_ms_per_sample": float(val_infer_ms_per_sample),
            "params": int(params),
        }
        records.append(row)
        pd.DataFrame(records).to_csv(csv_path, index=False)

        if epoch == 1 or epoch % 10 == 0:
            print(f"[{run_name}] epoch={epoch:03d} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f} lr={current_lr:.6e}")

        scheduler.step()

    return csv_path
