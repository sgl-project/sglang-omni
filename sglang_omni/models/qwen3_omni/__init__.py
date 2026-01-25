# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni model adapters."""

from sglang_omni.models.qwen3_omni.hf_split import (
    Qwen3OmniAudioEncoder,
    Qwen3OmniImageEncoder,
    Qwen3OmniSpec,
    Qwen3OmniSplitThinker,
)
from sglang_omni.models.qwen3_omni.pipeline import create_text_first_pipeline_config

__all__ = [
    "Qwen3OmniSpec",
    "Qwen3OmniAudioEncoder",
    "Qwen3OmniImageEncoder",
    "Qwen3OmniSplitThinker",
    "create_text_first_pipeline_config",
]
