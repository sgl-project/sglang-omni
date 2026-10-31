# SPDX-License-Identifier: Apache-2.0
"""Optional BF16 decode blocks with token-major engine-owned caches."""

from __future__ import annotations

import logging

import torch
from torch import nn

from sglang_omni.models.qwen3_tts.cutedsl.api import (
    PackedPostAttention,
    SlotPreAttention,
)
from sglang_omni.models.qwen3_tts.cutedsl.runtime import (
    HEAD_DIM,
    KV_HEADS,
    QKV_WIDTH,
    QUERY_HEADS,
    QUERY_WIDTH,
)

logger = logging.getLogger(__name__)

# note (Hao Wang): Wide CTAs use over half of Hopper SMEM, so 96 resident CTAs need at least 96 SMs.
MIN_DECODE_SM_COUNT = 96


class CuTeDSLDecoderLayer(nn.Module):
    """Pack loaded weights without changing the engine's attention or cache ownership."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self._derived_weight_cache_error = (  # noqa: leading-underscore - SGLang contract
            "Online weight updates are unsupported with Qwen3-TTS CuTeDSL packed "
            "weights. Restart with SGLANG_OMNI_QTTS_BACKEND=default to enable them."
        )
        attention = layer.self_attn
        self.pre_attention = SlotPreAttention(
            layer.input_layernorm.weight,
            attention.qkv_proj.weight,
            attention.q_norm.weight,
            attention.k_norm.weight,
            layer.input_layernorm.variance_epsilon,
        )
        gate_weight, up_weight = layer.mlp.gate_up_proj.weight.chunk(2, dim=0)
        self.post_attention = PackedPostAttention(
            attention.o_proj.weight,
            gate_weight,
            up_weight,
            layer.mlp.down_proj.weight,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )


def build_cutedsl_layers(layers: nn.ModuleList) -> nn.ModuleList | None:
    """Keep unsupported weights, devices, and tensor-parallel layouts on the default path."""
    for layer in layers:
        attention = layer.self_attn
        weight = attention.qkv_proj.weight
        hidden_size = layer.input_layernorm.weight.numel()
        if (
            not weight.is_cuda
            or weight.dtype != torch.bfloat16
            or torch.cuda.get_device_capability(weight.device)
            not in ((9, 0), (10, 0), (10, 3))
            or torch.cuda.get_device_properties(weight.device).multi_processor_count
            < MIN_DECODE_SM_COUNT
            or hidden_size not in (1024, 2048)
            or (attention.num_heads, attention.num_kv_heads, attention.head_dim)
            != (QUERY_HEADS, KV_HEADS, HEAD_DIM)
            or weight.shape != (QKV_WIDTH, hidden_size)
            or attention.o_proj.weight.shape != (hidden_size, QUERY_WIDTH)
            or layer.mlp.gate_up_proj.weight.shape != (6 * hidden_size, hidden_size)
            or layer.mlp.down_proj.weight.shape != (hidden_size, 3 * hidden_size)
            or attention.qkv_proj.bias is not None
            or attention.o_proj.bias is not None
            or not attention.rotary_emb.is_neox_style
            or attention.rotary_emb.cos_sin_cache.shape[-1] != HEAD_DIM
            or attention.rotary_emb.cos_sin_cache.dtype
            not in (torch.float32, torch.bfloat16)
        ):
            logger.warning(
                "CuTeDSL decode is unsupported for this stack; using default"
            )
            return None
        else:
            pass
    return nn.ModuleList(CuTeDSLDecoderLayer(layer) for layer in layers)
