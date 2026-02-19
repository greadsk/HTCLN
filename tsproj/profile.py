from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class ProfileResult:
    params: int
    flops: Optional[int]  # total MACs/ops, depending on backend
    backend: str


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: nn.Module, example_input: Any) -> Tuple[Optional[int], str]:
    """Compute FLOPs/MACs with optional backends.

    Priority:
    - thop (most common)
    - fvcore

    If none available, returns (None, 'none').

    Note: Different libraries count FLOPs/MACs slightly differently.
    """

    # thop
    try:
        from thop import profile as thop_profile  # type: ignore

        model.eval()
        with torch.no_grad():
            macs, params = thop_profile(model, inputs=(example_input,), verbose=False)
        return int(macs), "thop_macs"
    except Exception:
        pass

    # fvcore
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        model.eval()
        flops = FlopCountAnalysis(model, (example_input,)).total()
        return int(flops), "fvcore_flops"
    except Exception:
        pass

    return None, "none"


def profile_model(model: nn.Module, example_input: Any) -> ProfileResult:
    p = count_parameters(model)
    flops, backend = compute_flops(model, example_input)
    return ProfileResult(params=p, flops=flops, backend=backend)
