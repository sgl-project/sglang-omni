"""Vendor wrapper for sglang.srt.layers.*

Centralize third-party imports and apply monkey patches here.

Patches applied to RMSNorm.forward_cuda:
  - Empty tensor early return (avoids CUDA kernel launch on zero-element tensors)
  - dtype mismatch fallback when residual or post_residual_addition differ from x.dtype
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

# sglang 0.5.16 dropped the dp_attention.get_attention_tp_{rank,size} aliases.
# Re-alias the parallel_state functions they forwarded to, so callers keep the
# zero-arg spelling and the pre-0.5.16 semantics.
from sglang.srt.distributed.parallel_state import (
    get_attn_tensor_model_parallel_rank as get_attention_tp_rank,
)
from sglang.srt.distributed.parallel_state import (
    get_attn_tensor_model_parallel_world_size as get_attention_tp_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

# ---------------------------------------------------------------------------
# RMSNorm.forward_cuda monkey-patch
# ---------------------------------------------------------------------------
_orig_forward_cuda = RMSNorm.forward_cuda


def _patched_forward_cuda(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.numel() == 0:
        # Mirror upstream's zero-token contract exactly: callers unpack a
        # 2-tuple whenever they passed a residual, and post_residual_addition
        # is folded into it.
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            return x, residual
        return x
    if residual is not None and residual.dtype != x.dtype:
        return self.forward_native(
            x,
            residual,
            post_residual_addition=post_residual_addition,
            **kwargs,
        )
    if post_residual_addition is not None and post_residual_addition.dtype != x.dtype:
        return self.forward_native(
            x,
            residual,
            post_residual_addition=post_residual_addition,
            **kwargs,
        )
    return _orig_forward_cuda(
        self,
        x,
        residual,
        post_residual_addition=post_residual_addition,
        **kwargs,
    )


RMSNorm.forward_cuda = _patched_forward_cuda

__all__ = [
    "AttentionType",
    "RadixAttention",
    "VocabParallelEmbedding",
    "MRotaryEmbedding",
    "get_rope",
    "get_layer_id",
    "RMSNorm",
    "SiluAndMul",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "StandardTopKOutput",
    "TopK",
    "get_moe_a2a_backend",
    "should_skip_post_experts_all_reduce",
    "should_use_flashinfer_cutlass_moe_fp4_allgather",
    "get_moe_impl_class",
    "RoutingMethodType",
    "get_attention_tp_rank",
    "get_attention_tp_size",
    "QuantizationConfig",
    "LayerCommunicator",
    "LayerScatterModes",
    "FusedMoE",
]
