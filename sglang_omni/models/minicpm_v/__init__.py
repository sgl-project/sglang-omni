# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-V 2.6 model components and pipeline helpers.

This module provides Phase 1 and Phase 2 support for MiniCPM-V 2.6:
- Phase 1: Vision-only input (images), text output with HF backend
- Phase 2: SGLang-backed LLM for production performance

Key components:
- LLaVA-style slice-based image processing
- SigLIP ViT + Perceiver Resampler architecture
- MiniCPMVSGLangLLM for SGLang paged attention
"""

from sglang_omni.models.minicpm_v.components.common import (
    MiniCPMVSpec,
    get_image_token_id,
    load_llm_config,
    load_minicpm_config,
)
from sglang_omni.models.minicpm_v.components.image_encoder import MiniCPMVImageEncoder
from sglang_omni.models.minicpm_v.components.preprocessor import MiniCPMVPreprocessor
from sglang_omni.models.minicpm_v.sglang_llm import MiniCPMVSGLangLLM

from . import config

__all__ = [
    "MiniCPMVPreprocessor",
    "MiniCPMVSpec",
    "MiniCPMVImageEncoder",
    "MiniCPMVSGLangLLM",
    "get_image_token_id",
    "load_llm_config",
    "load_minicpm_config",
    "config",
]
