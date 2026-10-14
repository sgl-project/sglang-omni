# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-TTS-Nano."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import StageConfig
from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig
from sglang_omni.models.moss_tts_local.config import stages as local_stages

_PKG = "sglang_omni.models.moss_tts_nano"


def stages() -> list[StageConfig]:
    stages = local_stages(codec_gpu=0, colocated=True)
    for stage in stages:
        stage.factory_path = f"{_PKG}.stages.create_{stage.name}_executor"
        if stage.name in {"preprocessing", "vocoder"}:
            stage.factory.compute_dtype = "float32"
        else:
            pass
    return stages


class MossTTSNanoPipelineConfig(MossTTSLocalPipelineConfig):
    """Single-GPU MOSS-TTS-Nano pipeline."""

    architecture: ClassVar[str] = "MossTTSNanoForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = ()
    vocoder_cuda_graph_model_name: ClassVar[str] = "MOSS-TTS-Nano"
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        {
            "Arabic",
            "Czech",
            "Danish",
            "Greek",
            "Hebrew",
            "Hungarian",
            "Persian (Farsi)",
            "Polish",
            "Swedish",
            "Turkish",
        }
    )

    stages: list[StageConfig] = Field(default_factory=stages)


EntryClass = MossTTSNanoPipelineConfig
