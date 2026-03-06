# SPDX-License-Identifier: Apache-2.0
"""FishAudio-S1 (DualAR) model support for sglang-omni."""

from .factory import create_dual_ar_engine
from .runtime.dual_ar import DualARRequestData, DualARStepOutput
from .tokenizer import FishTokenizerAdapter, Reference

__all__ = [
    "create_dual_ar_engine",
    "create_tts_pipeline_config",
    "DualARRequestData",
    "DualARStepOutput",
    "FishTokenizerAdapter",
    "Reference",
]


def __getattr__(name: str):
    # Lazy import to avoid circular dependency with sglang_omni.config
    if name == "create_tts_pipeline_config":
        from .pipeline.config import create_tts_pipeline_config

        return create_tts_pipeline_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
