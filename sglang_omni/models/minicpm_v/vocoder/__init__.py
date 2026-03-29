# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o vocoder components for audio output.

This module provides CosyVoice vocoder integration for MiniCPM-o 2.6
audio output generation.
"""

from sglang_omni.models.minicpm_v.vocoder.cosyvoice import (
    CosyVoiceVocoder,
    load_cosyvoice_model,
)

__all__ = [
    "CosyVoiceVocoder",
    "load_cosyvoice_model",
]
