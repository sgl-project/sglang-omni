# SPDX-License-Identifier: Apache-2.0
"""The packed variable-length attention Qwen3-Omni's audio and vision encoders run on a CUDA device."""

from __future__ import annotations

from typing import Literal, get_args

import torch
from sglang.kernels.ops.attention.flash_attention_v4 import (
    is_flash_attention_v4_available,
)
from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    VisionFlash4Attention,
    VisionTritonAttention,
)
from sglang.srt.runtime_context import get_platform

PackedAttention = VisionFlash3Attention | VisionFlash4Attention | VisionTritonAttention
PackedAttentionBackend = Literal["triton_attn", "fa3", "fa4"]
PACKED_ATTENTION_BACKENDS: tuple[PackedAttentionBackend, ...] = get_args(
    PackedAttentionBackend
)


def resolve_packed_attention(
    device: torch.device,
    dtype: torch.dtype,
    *,
    mm_attention_backend: PackedAttentionBackend | None,
    fa4_on_blackwell: bool = False,
) -> PackedAttention | None:
    """The packed attention for an encoder on this device in this dtype, or None to keep Transformers' attention.

    mm_attention_backend names the kernel; None picks FA3 on Hopper and Triton elsewhere, or FA4 on SM100 and SM120
    Blackwell when fa4_on_blackwell is set.
    """
    if mm_attention_backend is not None and (
        mm_attention_backend not in PACKED_ATTENTION_BACKENDS
    ):
        raise ValueError(
            f"mm_attention_backend={mm_attention_backend!r} is not supported by the "
            f"Qwen3-Omni encoders; choose one of {', '.join(PACKED_ATTENTION_BACKENDS)}"
        )
    else:
        pass
    # note (ratish): FA3 rejects float32 and Triton's dot runs float32 as TF32, so a float32 encoder keeps float32 attention.
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return None
    else:
        pass
    platform = get_platform()
    if mm_attention_backend is not None:
        backend = mm_attention_backend
    elif platform.is_sm90:
        backend = "fa3"
    elif (
        fa4_on_blackwell
        and (platform.is_sm100 or platform.is_sm120)
        and is_flash_attention_v4_available()
    ):
        backend = "fa4"
    else:
        backend = "triton_attn"
    # note (ratish): callers pass forward_metadata; given it, the kernels ignore SGLANG_VIT_ENABLE_CUDA_GRAPH, which changes how they read cu_seqlens.
    if backend == "fa3" and platform.is_blackwell:
        raise ValueError("mm_attention_backend='fa3' does not run on Blackwell GPUs")
    elif backend == "fa3":
        return VisionFlash3Attention()
    elif backend == "fa4" and not is_flash_attention_v4_available():
        raise RuntimeError(
            "mm_attention_backend='fa4' needs the FlashAttention 4 kernels, which "
            "failed to import in this environment"
        )
    elif backend == "fa4":
        return VisionFlash4Attention()
    else:
        return VisionTritonAttention()
