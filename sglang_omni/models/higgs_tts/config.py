# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.higgs_tts"
_PREPROCESS_MAX_WORKERS = 2


class HiggsTtsPipelineConfig(PipelineConfig):
    """4-stage TTS pipeline: preprocessing → audio_encoder → tts_engine → vocoder.

    Preprocessing normalizes text/reference inputs; audio_encoder codec-encodes
    raw reference audio and builds the prompt; tts_engine composes the delayed
    reference-code embeddings at ``-100`` placeholder positions and drives the
    SGLang AR loop; vocoder reverses the delay pattern and decodes the waveform.
    """

    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="tts_frontend",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=_PREPROCESS_MAX_WORKERS),
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="tts_frontend",
            factory_path=f"{_PKG}.stages.create_audio_encoder_executor",
            factory=FactoryArgs(device="cuda"),
            gpu=0,
            gpu_memory_fraction=0.03,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(
                device="cuda", max_new_tokens=2048, enable_async_decode=True
            ),
            gpu=0,
            gpu_memory_fraction=0.85,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            # Keep the LM and vocoder in one CUDA context by default.  Splitting
            # them into same-GPU processes time-slices the H100 at ordinary
            # serving concurrency and prevents decode/vocoder overlap.
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(device="cuda"),
            gpu=0,
            gpu_memory_fraction=0.10,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    # Stream cadence is owned by the vocoder stage; the tts_engine emits on
    # the same cadence, so a tts_engine value either mirrors the vocoder's or
    # is refused.
    _STREAM_CADENCE_KEYS: ClassVar[tuple[str, ...]] = (
        "stream_stride",
        "stream_followup_stride",
        "initial_chunk_frames",
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "tts_engine":
            vocoder_extra = self.stage_named("vocoder").factory.model_extra or {}
            return {
                key: vocoder_extra[key]
                for key in self._STREAM_CADENCE_KEYS
                if key in vocoder_extra
            }
        if stage_name == "vocoder":
            return {
                "compile_decode": False,
                # Before the steady cursor is established, a decode window is
                # bounded by the default 75-row stride plus its 75-row
                # follow-up. Capture that complete finite domain so terminal
                # flushes cannot silently fall back to eager execution.
                "decode_cuda_graph_frame_counts": tuple(range(1, 151)),
            }
        return {}

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        stages = {stage.name: stage for stage in self.stages}
        preprocessing = stages["preprocessing"]
        if "OMP_NUM_THREADS" not in self.env_defaults:
            preprocessing.env.setdefault(
                "OMP_NUM_THREADS",
                str(
                    bounded_intraop_threads(
                        worker_count=_PREPROCESS_MAX_WORKERS,
                        max_threads=8,
                    )
                ),
            )
        vocoder_extra = stages["vocoder"].factory.model_extra or {}
        tts_engine_extra = stages["tts_engine"].factory.model_extra or {}
        for key in self._STREAM_CADENCE_KEYS:
            if key not in vocoder_extra:
                if key in tts_engine_extra:
                    raise ValueError(
                        f"Higgs TTS {key!r} must be configured on the vocoder stage"
                    )
                continue
            if key in tts_engine_extra and tts_engine_extra[key] != vocoder_extra[key]:
                raise ValueError(
                    f"Higgs TTS {key!r} must match between the tts_engine and "
                    "vocoder stages; omit the tts_engine value to derive it "
                    "from the vocoder"
                )

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
