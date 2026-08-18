# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-TTS Base."""

from __future__ import annotations

import re
from typing import Any, ClassVar

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
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, Any]:
        from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

        defaults = Qwen3TtsEngineBuilder().generation_defaults(dtype="bfloat16")
        return {k: defaults[k] for k in ("max_running_requests", "max_queued_requests")}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        # Note (Akazaakane): preprocessing -> tts_engine is excluded because
        # preprocessing stores prepared requests in the module-level
        # _PREPROCESSING_CONTEXT/_PREPARED_REQUESTS registries that the AR engine
        # builder reads in-process. The vocoder loads its own speech tokenizer and
        # reads audio_codes from the payload.
        return frozenset({("tts_engine", "vocoder")})

    @classmethod
    def process_edge_resources(
        cls,
    ) -> dict[tuple[str, str], dict[str, float]]:
        return {
            ("tts_engine", "vocoder"): {
                "tts_engine": 0.85,
                "vocoder": 0.10,
            }
        }

    model_path: str
    # note (0xtoward): Keep deterministic inference opt-in because it serializes
    # preprocessing and vocoder decoding and disables Talker compilation and the
    # initial vocoder CUDA Graph, reducing throughput.
    enable_deterministic_inference: bool = False
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
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if not self.enable_deterministic_inference:
            return

        self.runtime_overrides.setdefault("preprocessing", {})["max_concurrency"] = 1
        tts_engine = self.runtime_overrides.setdefault("tts_engine", {})
        server_args = tts_engine.setdefault("server_args_overrides", {})
        server_args["enable_deterministic_inference"] = True
        vocoder = self.runtime_overrides.setdefault("vocoder", {})
        vocoder["enable_deterministic_inference"] = True
        vocoder["initial_cuda_graph"] = False

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
