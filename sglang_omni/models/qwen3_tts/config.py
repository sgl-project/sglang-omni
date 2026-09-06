# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-TTS."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

from sglang_omni.config import (
    CustomVoiceConfig,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)

_PKG = "sglang_omni.models.qwen3_tts"
_QWEN3_TTS_CUSTOM_VARIANT_MARKERS = (
    "custom_voice",
    "customvoice",
    "voice_design",
    "voicedesign",
)


class Qwen3TTSPipelineConfig(PipelineConfig):
    """3-stage Qwen3-TTS pipeline: preprocessing -> engine -> vocoder."""

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

    def resolve_custom_voice_config(self) -> CustomVoiceConfig | None:
        engine_stage = self.stage_named("tts_engine")
        # Note(yzxiao): Inspect the checkpoint the engine factory will actually
        # receive, including pipeline-authored and user-set factory overrides.
        engine_factory_kwargs = {
            "model_path": self.model_path,
            **resolve_stage_factory_kwargs(engine_stage, self),
            **resolve_stage_typed_kwargs(engine_stage),
        }
        checkpoint_config = _load_qwen3_tts_checkpoint_config(
            engine_factory_kwargs["model_path"]
        )
        model_type = _normalize_qwen3_tts_model_type(
            checkpoint_config.get("tts_model_type")
        )
        if model_type in {"base", "voice_design"}:
            return None
        if model_type == "custom_voice":
            spk_id = checkpoint_config.get("talker_config", {}).get("spk_id")
            if (
                not isinstance(spk_id, dict)
                or not spk_id
                or any(not isinstance(name, str) or not name.strip() for name in spk_id)
            ):
                raise ValueError(
                    "CustomVoice requires a non-empty talker_config.spk_id speaker mapping"
                )
            return CustomVoiceConfig(
                speakers=tuple(spk_id),
                task_type="CustomVoice",
            )
        return None


def _load_qwen3_tts_checkpoint_config(model_path: str) -> dict[str, Any]:
    checkpoint_dir = Path(model_path).expanduser()
    if checkpoint_dir.is_dir():
        config_path = checkpoint_dir / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_qwen3_tts_model_type(raw: Any) -> str:
    normalized = str(raw or "base").replace("-", "_").strip().lower()
    if normalized == "customvoice":
        return "custom_voice"
    if normalized == "voicedesign":
        return "voice_design"
    return normalized


def qwen3_tts_checkpoint_model_type(checkpoint_dir: str) -> str:
    """Read ``tts_model_type`` from a resolved checkpoint.

    The directory name is not a reliable signal: a Base checkpoint served from
    a path like ``/srv/checkpoints/current`` carries no marker at all. The
    config does, and it is the same value the request path validates against.
    Returns ``"base"`` when the field is absent, matching that path's default.
    """
    if not (Path(checkpoint_dir) / "config.json").is_file():
        return "base"
    config = _load_qwen3_tts_checkpoint_config(checkpoint_dir)
    return _normalize_qwen3_tts_model_type(config.get("tts_model_type"))


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
