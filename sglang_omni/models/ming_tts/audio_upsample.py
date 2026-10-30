# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for shared Ming AudioVAE upsampling."""

from sglang_omni.modules.ming_audio.audio_upsample import (
    HAS_TRITON,
    MINIMUM_BF16_COMPUTE_CAPABILITY,
    UPSAMPLE_BLOCK_ELEMENTS,
    can_upsample_linear_bth,
    upsample_linear_bth,
    upsample_linear_bth_kernel,
)

__all__ = [
    "HAS_TRITON",
    "MINIMUM_BF16_COMPUTE_CAPABILITY",
    "UPSAMPLE_BLOCK_ELEMENTS",
    "can_upsample_linear_bth",
    "upsample_linear_bth",
    "upsample_linear_bth_kernel",
]
