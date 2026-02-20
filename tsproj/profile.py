# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Dict, Optional, Tuple

# import torch
# from torch import nn


# @dataclass
# class ProfileResult:
#     params: int
#     flops: Optional[int]  # total MACs/ops, depending on backend
#     backend: str


# def count_parameters(model: nn.Module) -> int:
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def compute_flops(model: nn.Module, example_input: Any) -> Tuple[Optional[int], str]:
#     """Compute FLOPs/MACs with optional backends.

#     Priority:
#     - thop (most common)
#     - fvcore

#     If none available, returns (None, 'none').

#     Note: Different libraries count FLOPs/MACs slightly differently.
#     """

#     # thop
#     try:
#         from thop import profile as thop_profile  # type: ignore

#         model.eval()
#         with torch.no_grad():
#             macs, params = thop_profile(model, inputs=(example_input,), verbose=False)
#         return int(macs), "thop_macs"
#     except Exception:
#         pass

#     # fvcore
#     try:
#         from fvcore.nn import FlopCountAnalysis  # type: ignore

#         model.eval()
#         flops = FlopCountAnalysis(model, (example_input,)).total()
#         return int(flops), "fvcore_flops"
#     except Exception:
#         pass

#     return None, "none"


# def profile_model(model: nn.Module, example_input: Any) -> ProfileResult:
#     p = count_parameters(model)
#     flops, backend = compute_flops(model, example_input)
#     return ProfileResult(params=p, flops=flops, backend=backend)


###########################################
##为了计算transformer修改后代码
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


def _mha_flops_counter_hook(module: nn.MultiheadAttention, input, output):
    """
    为 nn.MultiheadAttention 手动注册 thop FLOPs 计数钩子。

    对于 MultiheadAttention，主要的 MACs 来自：
    1. Q/K/V 投影：3 * 2 * T * d_model * d_model  (thop 按 2*in*out 计算 linear MACs)
    2. Attention scores：2 * T * T * d_model  (Q @ K^T)
    3. Attention output weighted sum：2 * T * T * d_model  (attn @ V)
    4. 输出投影：2 * T * d_model * d_model

    其中 T = 序列长度, d_model = embed_dim
    """
    # input[0] 是 query，shape 为 (B, T, E) (batch_first=True) 或 (T, B, E)
    query = input[0]
    if module.batch_first:
        # batch_first=True: (B, T, E)
        B, T, E = query.shape
    else:
        # batch_first=False: (T, B, E)
        T, B, E = query.shape

    d_model = module.embed_dim

    # 1. Q, K, V 投影 (in_proj): 3 组 Linear(d_model -> d_model)
    #    每个 Linear MACs = 2 * T * d_model * d_model，共 3 个
    in_proj_macs = 3 * 2 * T * d_model * d_model

    # 2. Attention scores: Q @ K^T -> (B, nhead, T, T)
    #    每个 head: 2 * T * T * (d_model/nhead)，共 nhead 个 head
    #    总计 = 2 * T * T * d_model
    attn_score_macs = 2 * T * T * d_model

    # 3. Attention weighted sum: attn @ V -> (B, nhead, T, d_head)
    #    同上 = 2 * T * T * d_model
    attn_out_macs = 2 * T * T * d_model

    # 4. 输出投影: out_proj Linear(d_model -> d_model)
    #    MACs = 2 * T * d_model * d_model
    out_proj_macs = 2 * T * d_model * d_model

    total_macs = in_proj_macs + attn_score_macs + attn_out_macs + out_proj_macs

    # thop 通过 module.__flops__ 累计 MACs（注意 thop 内部叫 flops，实际是 MACs）
    module.__flops__ += total_macs


def compute_flops(model: nn.Module, example_input: Any) -> Tuple[Optional[int], str]:
    """Compute FLOPs/MACs with optional backends.

    Priority:
    - thop (most common)
    - fvcore

    If none available, returns (None, 'none').

    Note: Different libraries count FLOPs/MACs slightly differently.
    
    Fix: 为 thop 注册 nn.MultiheadAttention 的自定义 FLOPs 计数钩子，
         防止 Transformer 模型的 FLOPs 被严重低估（thop 默认不支持 MHA）。
    """

    # thop
    try:
        from thop import profile as thop_profile  # type: ignore

        # 为 MultiheadAttention 注册自定义计数钩子
        custom_ops: Dict[type, Any] = {
            nn.MultiheadAttention: _mha_flops_counter_hook,
        }

        model.eval()
        with torch.no_grad():
            macs, params = thop_profile(
                model,
                inputs=(example_input,),
                custom_ops=custom_ops,
                verbose=False,
            )
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