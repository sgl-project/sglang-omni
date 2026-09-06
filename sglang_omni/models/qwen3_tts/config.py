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

    model_path: str
    # note (0xtoward): Keep deterministic inference opt-in because it serializes
    # preprocessing and vocoder decoding and disables the vocoder CUDA graphs,
    # reducing throughput.
    enable_deterministic_inference: bool = False
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            # Note (Jiaxin Deng): no gpu declaration here. Sharing the engine's
            # process the stage holds no GPU budget of its own, and declaring one
            # makes every layout that shares the card demand a fraction for it. A
            # split frontend passes --preprocessing.gpu with its own fraction.
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

    def preprocessing_in_own_process(self) -> bool:
        stages = {stage.name: stage for stage in self.stages}
        return stages["preprocessing"].process != stages["tts_engine"].process

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        # Note (Jiaxin Deng): outside the engine process the preprocessing stage
        # loads its own prompt frontend and ships prepared tensors in the payload.
        if stage_name == "preprocessing" and self.preprocessing_in_own_process():
            kwargs["load_frontend"] = True
        if not self.enable_deterministic_inference:
            return kwargs
        # note (0xtoward): deterministic inference serializes preprocessing
        # and vocoder decoding and disables the vocoder CUDA graphs.
        # Applied at launch so an explicit factory.* value still wins.
        if stage_name == "preprocessing":
            return {**kwargs, "max_concurrency": 1}
        if stage_name == "tts_engine":
            return {"server_args_overrides": {"enable_deterministic_inference": True}}
        if stage_name == "vocoder":
            return {
                "enable_deterministic_inference": True,
                "initial_cuda_graph": False,
                "followup_cuda_graph": False,
            }
        return kwargs

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return is_qwen3_tts_base_model(self.model_path)

    def supports_uploaded_voice_references(self) -> bool:
        return is_qwen3_tts_base_model(self.model_path)


def qwen3_tts_checkpoint_model_type(checkpoint_dir: str) -> str:
    """Read ``tts_model_type`` from a resolved checkpoint.

    The directory name is not a reliable signal: a Base checkpoint served from
    a path like ``/srv/checkpoints/current`` carries no marker at all. The
    config does, and it is the same value the request path validates against.
    Returns ``"base"`` when the field is absent, matching that path's default.
    """
    import json
    import os

    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        return "base"
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle).get("tts_model_type")
    normalized = str(raw or "base").replace("-", "_").strip().lower()
    if normalized == "customvoice":
        return "custom_voice"
    if normalized == "voicedesign":
        return "voice_design"
    return normalized


def is_qwen3_tts_base_model(model_path: str) -> bool:
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
