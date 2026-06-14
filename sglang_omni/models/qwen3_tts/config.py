# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-TTS Base."""

from __future__ import annotations

import re
from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.qwen3_tts"
_QWEN3_TTS_CUSTOM_VARIANT_MARKERS = (
    "custom_voice",
    "customvoice",
    "voice_design",
    "voicedesign",
)


class Qwen3TTSPipelineConfig(PipelineConfig):
    """3-stage Qwen3-TTS Base pipeline: preprocessing -> engine -> vocoder."""

    architecture: ClassVar[str] = "Qwen3TTSForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            terminal=True,
        ),
    ]

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return _is_qwen3_tts_base_model(self.model_path)

    def supports_uploaded_voice_references(self) -> bool:
        return _is_qwen3_tts_base_model(self.model_path)


def _is_qwen3_tts_base_model(model_path: str) -> bool:
    qwen3_tts_parts = [
        part.replace("-", "_").casefold()
        for part in re.split(r"[/\\]+", model_path.strip())
        if "qwen3_tts" in part.replace("-", "_").casefold()
    ]
    if any(
        marker in part
        for part in qwen3_tts_parts
        for marker in _QWEN3_TTS_CUSTOM_VARIANT_MARKERS
    ):
        return False
    return any(part.endswith("_base") or "_base_" in part for part in qwen3_tts_parts)


EntryClass = Qwen3TTSPipelineConfig
