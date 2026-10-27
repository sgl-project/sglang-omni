# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for shared Ming audio components."""

from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import (
    AudioVAE,
    current_platform,
)

__all__ = ["AudioVAE", "current_platform"]
