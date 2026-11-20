# SPDX-License-Identifier: Apache-2.0
"""The packed variable-length attention Qwen3-Omni's audio and vision encoders run on a CUDA device."""

from __future__ import annotations

import torch
from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    VisionTritonAttention,
)
from sglang.srt.runtime_context import get_platform

PackedAttention = VisionFlash3Attention | VisionTritonAttention


def resolve_packed_attention(
    device: torch.device, dtype: torch.dtype
) -> PackedAttention | None:
    """The packed attention for an encoder on this device in this dtype, or None to keep Transformers' attention."""
    # note (ratish): FA3 rejects float32 and Triton's dot runs float32 as TF32, so a float32 encoder keeps float32 attention.
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return None
    else:
        pass
    # note (ratish): callers pass forward_metadata; given it, the kernels ignore SGLANG_VIT_ENABLE_CUDA_GRAPH, which changes how they read cu_seqlens.
    if get_platform().is_sm90:
        return VisionFlash3Attention()
    else:
        return VisionTritonAttention()
