# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for AURA Omni."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.aura_omni"
_ASR_PKG = "sglang_omni.models.qwen3_asr"
_TTS_PKG = "sglang_omni.models.qwen3_tts"
_ROUTING = f"{_PKG}.request_builders"

TTS_STAGES = ("tts_preprocessing", "vocoder", "tts_engine")


def aura_omni_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next=["asr", "aura"],
            route_fn=f"{_ROUTING}.resolve_preprocessing_next_stages",
            project_payload={
                "asr": f"{_ROUTING}.project_preprocessing_to_asr",
                "aura": f"{_ROUTING}.project_preprocessing_to_aura",
            },
        ),
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_ASR_PKG}.stages.create_sglang_qwen3_asr_executor",
            engine=EngineArgs(max_running_requests=8, enable_torch_compile=False),
            gpu=0,
            next="aura",
        ),
        EngineStageConfig(
            name="aura",
            process="aura",
            factory_path=f"{_PKG}.stages.create_aura_executor",
            factory=FactoryArgs(max_new_tokens=256, request_build_max_workers=2),
            gpu=1,
            wait_for=["preprocessing", "asr"],
            wait_for_fn=f"{_ROUTING}.resolve_aura_wait_sources",
            merge_fn=f"{_ROUTING}.merge_for_aura",
            next=["decode", "tts_preprocessing", "vocoder"],
            route_fn=f"{_ROUTING}.resolve_aura_next_stages",
            stream_to=["decode", "vocoder"],
            stream_done_to_fn=f"{_ROUTING}.resolve_aura_stream_done_targets",
            project_payload={
                "tts_preprocessing": f"{_ROUTING}.project_aura_to_tts",
                "vocoder": f"{_ROUTING}.project_aura_to_vocoder",
            },
        ),
        StageConfig(
            name="decode",
            process="aura",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        StageConfig(
            name="tts_preprocessing",
            process="tts",
            factory_path=f"{_TTS_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        # note (siju): the vocoder is built before the engine sharing its process,
        # so its weights and graphs are resident when the KV pool is sized.
        StageConfig(
            name="vocoder",
            process="tts",
            factory_path=f"{_TTS_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=2,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        EngineStageConfig(
            name="tts_engine",
            process="tts",
            factory_path=f"{_TTS_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=2,
            next="vocoder",
            stream_to=["vocoder"],
        ),
    ]


class AuraOmniPipelineConfig(PipelineConfig):
    """Qwen3-ASR -> AURA (Qwen3-VL) -> Qwen3-TTS, answering in text and speech."""

    architecture: ClassVar[str] = "Qwen3VLForConditionalGeneration"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
        "aura": EngineStageConfig,
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    asr_model_path: str = "Qwen/Qwen3-ASR-1.7B"
    tts_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    entry_stage: str = "preprocessing"
    terminal_stages_fn: str | None = f"{_ROUTING}.resolve_terminal_stages"
    stages: list[StageConfig] = Field(default_factory=aura_omni_stages)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str | bool]:
        if stage_name == "asr":
            return {"model_path": self.asr_model_path}
        elif stage_name == "vocoder" and (
            not current_platform.enable_tts_vocoder_fast_path()
        ):
            # The vocoder captures its decode graphs during the asynchronous
            # decode warmup, which a platform that has not measured it skips.
            return {"model_path": self.tts_model_path, "async_decode": False}
        elif stage_name in TTS_STAGES:
            return {"model_path": self.tts_model_path}
        else:
            return {}


EntryClass = AuraOmniPipelineConfig
