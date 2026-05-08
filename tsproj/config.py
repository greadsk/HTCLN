import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    # data
    seq_len: int = 36
    horizons: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    batch_size: int = 64
    num_workers: int = 0

    # training
    epochs: int = 50
    base_lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # lr schedule
    lr_schedule: str = "exp"  # exp / warmup_exp
    warmup_ratio: float = 0.1
    exp_gamma: float = 0.98  # per-epoch

    # logging
    results_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "results"))


@dataclass
class Ablation2DConfig:
    """Research-model preprocessing knobs.

    - window_points: {25,36,49,64,100} -> side = sqrt(window_points) (5~10)
    - time_stride: {1,5,6,7,8,10,...}: sampling interval along time
    - padding: fixed 1 (per requirement)
    - upsample_to: optional (H,W) to enforce unified resolution
    """

    window_points: int = 36
    time_stride: int = 1
    padding: int = 1
    upsample_to: Optional[int] = None  # if set, upsample padded square to (upsample_to, upsample_to)

    def side(self) -> int:
        s = int(self.window_points ** 0.5)
        if s * s != self.window_points:
            raise ValueError(f"window_points must be a perfect square, got {self.window_points}")
        return s
