# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni model components and pipeline helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_omni import config
    from sglang_omni.models.qwen3_omni.components.audio_encoder import (
        Qwen3OmniAudioEncoder,
    )
    from sglang_omni.models.qwen3_omni.components.common import Qwen3OmniSpec
    from sglang_omni.models.qwen3_omni.components.image_encoder import (
        Qwen3OmniImageEncoder,
    )
    from sglang_omni.models.qwen3_omni.components.preprocessor import (
        Qwen3OmniPreprocessor,
    )
    from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker

__all__ = [
    "Qwen3OmniPreprocessor",
    "Qwen3OmniSpec",
    "Qwen3OmniAudioEncoder",
    "Qwen3OmniImageEncoder",
    "Qwen3OmniSplitThinker",
    "config",
]


def __getattr__(name: str):
    if name == "config":
        return import_module(f"{__name__}.config")
    if name == "Qwen3OmniPreprocessor":
        from sglang_omni.models.qwen3_omni.components.preprocessor import (
            Qwen3OmniPreprocessor,
        )

        return Qwen3OmniPreprocessor
    if name == "Qwen3OmniSpec":
        from sglang_omni.models.qwen3_omni.components.common import Qwen3OmniSpec

        return Qwen3OmniSpec
    if name == "Qwen3OmniAudioEncoder":
        from sglang_omni.models.qwen3_omni.components.audio_encoder import (
            Qwen3OmniAudioEncoder,
        )

        return Qwen3OmniAudioEncoder
    if name == "Qwen3OmniImageEncoder":
        from sglang_omni.models.qwen3_omni.components.image_encoder import (
            Qwen3OmniImageEncoder,
        )

        return Qwen3OmniImageEncoder
    if name == "Qwen3OmniSplitThinker":
        from sglang_omni.models.qwen3_omni.components.thinker import (
            Qwen3OmniSplitThinker,
        )

        return Qwen3OmniSplitThinker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
