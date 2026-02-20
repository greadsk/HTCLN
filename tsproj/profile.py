from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import nn


@dataclass
class ProfileResult:
    params: int
    flops: Optional[int]  # total MACs/ops, depending on backend
    backend: str


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _manual_flops(model: nn.Module, example_input: Any) -> Optional[int]:
    """
    手动计算模型的近似 MACs（乘加运算次数）。
    支持以下层类型的精确计算：
      - nn.Linear       每个 token 计算一次，乘以序列长度 T：2 * T * in * out
      - nn.MultiheadAttention
      - nn.Conv1d / nn.Conv2d
    其余层忽略（通常是激活函数、Dropout、LayerNorm 等，计算量很小）。

    对于 nn.MultiheadAttention，需要知道序列长度 T 和 embed_dim E：
      - in_proj (Q/K/V): 3 * 2 * T * E * E
      - attn scores (Q @ K^T): 2 * T * T * E
      - attn weighted sum (attn @ V): 2 * T * T * E
      - out_proj: 2 * T * E * E
      合计: 8 * T * E^2 + 4 * T^2 * E
    
    序列长度 T 从 example_input 推断：
      - 若 example_input 是 Tensor 且 ndim==3，T = shape[1]（batch_first=True 约定）
      - 否则 T=1（保守估计）
    """
    # 推断序列长度 T
    if isinstance(example_input, torch.Tensor) and example_input.ndim == 3:
        T = int(example_input.shape[1])
    elif isinstance(example_input, (tuple, list)) and len(example_input) > 0:
        first = example_input[0]
        T = int(first.shape[1]) if isinstance(first, torch.Tensor) and first.ndim == 3 else 1
    else:
        T = 1

    total_macs = 0
    mha_prefixes = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            mha_prefixes.add(name)

    for name, module in model.named_modules():
        # 判断当前模块是否是某个 MHA 的子孙模块
        is_inside_mha = any(
            name.startswith(prefix + ".") for prefix in mha_prefixes
        )

        if isinstance(module, nn.MultiheadAttention):
            E = module.embed_dim
            in_proj_macs = 3 * 2 * T * E * E
            score_macs = 2 * T * T * E
            wsum_macs = 2 * T * T * E
            out_proj_macs = 2 * T * E * E
            total_macs += in_proj_macs + score_macs + wsum_macs + out_proj_macs

        elif isinstance(module, nn.Linear) and not is_inside_mha:
            total_macs += 2 * module.in_features * module.out_features * T

        elif isinstance(module, nn.Conv1d) and not is_inside_mha:
            total_macs += 2 * module.in_channels * module.out_channels * module.kernel_size[0] * T

        elif isinstance(module, nn.Conv2d) and not is_inside_mha:
            kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            total_macs += 2 * module.in_channels * module.out_channels * kH * kW

    return total_macs if total_macs > 0 else None


def compute_flops(model: nn.Module, example_input: Any) -> Tuple[Optional[int], str]:
    """Compute FLOPs/MACs with optional backends.

    Priority order:
    1. torchinfo  (handles nn.MultiheadAttention natively)
    2. thop       (with custom_ops hook for nn.MultiheadAttention)
    3. fvcore
    4. manual     (pure Python fallback, no external deps)

    Note: Different libraries count FLOPs/MACs slightly differently.
    """

    # ------------------------------------------------------------------ #
    # 1. torchinfo — best support for Transformer / MHA                   #
    # ------------------------------------------------------------------ #
    try:
        from torchinfo import summary as ti_summary  # type: ignore

        model.eval()
        with torch.no_grad():
            if isinstance(example_input, (tuple, list)):
                input_data = example_input
            else:
                input_data = (example_input,)
            stats = ti_summary(
                model,
                input_data=input_data,
                verbose=0,
                device=next(model.parameters()).device,
            )
        total_macs = stats.total_mult_adds
        if total_macs is not None and total_macs > 0:
            return int(total_macs), "torchinfo_macs"
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"[profile] torchinfo failed: {e}")

    # ------------------------------------------------------------------ #
    # 2. thop — with custom_ops hook for nn.MultiheadAttention            #
    # ------------------------------------------------------------------ #
    try:
        from thop import profile as thop_profile  # type: ignore

        def _mha_thop_hook(module: nn.MultiheadAttention, inputs, outputs):
            """
            thop custom_ops hook for nn.MultiheadAttention.
            thop 要求在 forward 结束后将当前层的 MACs 写入 module.__flops__。
            必须用 setattr 而非 +=，因为 thop 在 hook 前已将 __flops__ 初始化为 0。
            """
            query = inputs[0]
            # query shape: (B, T, E) if batch_first=True, else (T, B, E)
            if module.batch_first:
                T = int(query.shape[1])
            else:
                T = int(query.shape[0])
            E = module.embed_dim

            in_proj_macs  = 3 * 2 * T * E * E   # Q, K, V projections
            score_macs    = 2 * T * T * E         # Q @ K^T
            wsum_macs     = 2 * T * T * E         # attn @ V
            out_proj_macs = 2 * T * E * E         # output projection
            total = in_proj_macs + score_macs + wsum_macs + out_proj_macs

            # thop 内部用 __flops__ 累计，初始为 0，此处直接设置
            module.__flops__ = getattr(module, "__flops__", 0) + total

        model.eval()
        with torch.no_grad():
            if isinstance(example_input, (tuple, list)):
                inputs_tuple = tuple(example_input)
            else:
                inputs_tuple = (example_input,)
            macs, _ = thop_profile(
                model,
                inputs=inputs_tuple,
                custom_ops={nn.MultiheadAttention: _mha_thop_hook},
                verbose=False,
            )
        if macs is not None and macs > 0:
            return int(macs), "thop_macs"
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"[profile] thop failed: {e}")

    # ------------------------------------------------------------------ #
    # 3. fvcore                                                           #
    # ------------------------------------------------------------------ #
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        model.eval()
        with torch.no_grad():
            if isinstance(example_input, (tuple, list)):
                fvcore_input = tuple(example_input)
            else:
                fvcore_input = (example_input,)
            fca = FlopCountAnalysis(model, fvcore_input)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            flops = fca.total()
        if flops is not None and flops > 0:
            return int(flops), "fvcore_flops"
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"[profile] fvcore failed: {e}")

    # ------------------------------------------------------------------ #
    # 4. Manual fallback — pure Python, no external deps                  #
    # ------------------------------------------------------------------ #
    try:
        manual = _manual_flops(model, example_input)
        if manual is not None and manual > 0:
            return int(manual), "manual_macs"
    except Exception as e:
        warnings.warn(f"[profile] manual flops calculation failed: {e}")

    return None, "none"


def profile_model(model: nn.Module, example_input: Any) -> ProfileResult:
    p = count_parameters(model)
    flops, backend = compute_flops(model, example_input)
    return ProfileResult(params=p, flops=flops, backend=backend)
