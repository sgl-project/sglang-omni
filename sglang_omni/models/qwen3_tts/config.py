# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-TTS Base."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

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
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, Any]:
        from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

        defaults = Qwen3TtsEngineBuilder().generation_defaults(dtype="bfloat16")
        return {k: defaults[k] for k in ("max_running_requests", "max_queued_requests")}

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (Akazaakane): preprocessing stores prepared requests in the module-level
        # _PREPROCESSING_CONTEXT/_PREPARED_REQUESTS registries that the AR engine
        # builder reads in-process.
        return frozenset({("preprocessing", "tts_engine")})

    model_path: str
    # note (0xtoward): Keep deterministic inference opt-in because it serializes
    # preprocessing and vocoder decoding and disables Talker compilation and the
    # initial vocoder CUDA Graph, reducing throughput.
    enable_deterministic_inference: bool = False
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if not self.enable_deterministic_inference:
            return {}
        # note (0xtoward): deterministic inference serializes preprocessing
        # and vocoder decoding and disables the initial vocoder CUDA graph;
        # applied at launch so an explicit factory.* value still wins.
        if stage_name == "preprocessing":
            return {"max_concurrency": 1}
        if stage_name == "tts_engine":
            return {"server_args_overrides": {"enable_deterministic_inference": True}}
        if stage_name == "vocoder":
            return {
                "enable_deterministic_inference": True,
                "initial_cuda_graph": False,
            }
        return {}

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
