"""Vendor wrapper for sglang.srt.layers.*

Centralize third-party imports and apply monkey patches here.

Patches applied to RMSNorm.forward_cuda:
  - Empty tensor early return (avoids CUDA kernel launch on zero-element tensors)
  - cast_x_before_out_mul fallback to forward_native (HF-compatible RMSNorm cast order)
  - dtype mismatch fallback when residual or post_residual_addition differ from x.dtype
These patches can be removed once upstream SGLang merges equivalent changes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from sgl_kernel import top_k_top_p_sampling_from_probs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
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
        return x
    if self.cast_x_before_out_mul:
        return self.forward_native(
            x,
            residual,
            post_residual_addition=post_residual_addition,
            **kwargs,
        )
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

# ---------------------------------------------------------------------------
# RMSNorm.forward_with_allreduce_fusion monkey-patch
# ---------------------------------------------------------------------------
_orig_forward_with_allreduce_fusion = RMSNorm.forward_with_allreduce_fusion


def _patched_forward_with_allreduce_fusion(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        from sglang.srt.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_reduce,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        if get_tensor_model_parallel_world_size() > 1:
            fused_result = flashinfer_allreduce_residual_rmsnorm(
                input_tensor=x,
                residual=residual,
                weight=self.weight,
                eps=self.variance_epsilon,
            )
            if fused_result[0] is not None:
                return fused_result

            x = tensor_model_parallel_all_reduce(x)
            return self.forward(
                x,
                residual,
                post_residual_addition=post_residual_addition,
                **kwargs,
            )

    return self.forward(
        x,
        residual,
        post_residual_addition=post_residual_addition,
        **kwargs,
    )


RMSNorm.forward_with_allreduce_fusion = _patched_forward_with_allreduce_fusion

__all__ = [
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
    "TopK",
    "get_moe_a2a_backend",
    "should_use_flashinfer_cutlass_moe_fp4_allgather",
    "get_moe_impl_class",
    "RoutingMethodType",
    "get_attention_tp_rank",
    "get_attention_tp_size",
    "QuantizationConfig",
    "LayerCommunicator",
    "LayerScatterModes",
    "FusedMoE",
    "top_k_top_p_sampling_from_probs",
]
