"""Vendor wrapper for sglang.srt.layers.*

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

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
