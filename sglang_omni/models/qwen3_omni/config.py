# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.config.schema import MemFractionOverrideStages, StreamTargetConfig
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    CODE2WAV_STAGE,
    CODE_PREDICTOR_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_AR_STAGE,
    THINKER_STAGE,
)


def _validate_qwen3_speech_gpu_placement(
    gpu_placement: dict[str, int],
    *,
    tp_size: int = 1,
) -> None:
    thinker_gpu = gpu_placement.get("thinker", 0)
    thinker_range = range(thinker_gpu, thinker_gpu + tp_size)
    for speech_stage_name in ("talker_ar", "code_predictor", "code2wav"):
        stage_gpu = gpu_placement.get(speech_stage_name, 1)
        if stage_gpu in thinker_range:
            raise ValueError(
                f"Speech stage '{speech_stage_name}' GPU {stage_gpu} collides with "
                f"thinker TP range [{thinker_gpu}, {thinker_gpu + tp_size}). "
                f"Place speech stages on GPU >= {thinker_gpu + tp_size}."
            )


class Qwen3OmniPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    mem_fraction_override_stages: MemFractionOverrideStages = Field(
        default_factory=lambda: MemFractionOverrideStages(thinker=THINKER_STAGE)
    )
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                merge_fn="sglang_omni.models.qwen3_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={
                    "thinker_max_seq_len": 8192,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.thinker_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]

    def apply_server_args_overrides(
        self, *, stage_name: str, overrides: dict[str, Any]
    ) -> None:
        tp_size = overrides.get("tp_size", 1)
        if stage_name == THINKER_STAGE and tp_size > 1:
            raise NotImplementedError(
                "The TP runtime under sglang_omni/engines/tp/ is model-agnostic; "
                "Qwen3-Omni support will land as a follow-up after Ming-flash-omni "
                "TP lands."
            )
        super().apply_server_args_overrides(
            stage_name=stage_name,
            overrides=overrides,
        )

    def apply_thinker_server_args_overrides(self, overrides: dict[str, Any]) -> None:
        self.apply_server_args_overrides(
            stage_name=THINKER_STAGE,
            overrides=overrides,
        )


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """9-stage pipeline config for Qwen3 Omni with text + speech output."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    terminal_stages: list[str] = [DECODE_STAGE, CODE2WAV_STAGE]
    mem_fraction_override_stages: MemFractionOverrideStages = Field(
        default_factory=lambda: MemFractionOverrideStages(
            thinker=THINKER_STAGE,
            talker=TALKER_AR_STAGE,
        )
    )
    gpu_placement: dict[str, int] = {
        "thinker": 0,
        "talker_ar": 1,
        "code_predictor": 1,
        "code2wav": 1,
    }

    stages: list[StageConfig] = [
        # Stages 1-4: same as text-only
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                merge_fn="sglang_omni.models.qwen3_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 5: Thinker (speech_enabled, fan-out)
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={"thinker_max_seq_len": 8192, "speech_enabled": True},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.thinker_next_speech",
            relay=RelayConfig(device="cuda"),
            stream_to=[StreamTargetConfig(to_stage=TALKER_AR_STAGE)],
        ),
        # Stage 6: Decode (terminal)
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 7: Talker AR
        StageConfig(
            name=TALKER_AR_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_talker_ar_executor_from_config",
                args={
                    "talker_max_seq_len": 8192,
                    "speech_enabled": True,
                    "feedback_enabled": True,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.talker_ar_next",
            relay=RelayConfig(device="cuda"),
            stream_to=[StreamTargetConfig(to_stage=CODE_PREDICTOR_STAGE)],
        ),
        # Stage 8: Code Predictor (streaming: consumes chunks from Talker, sends chunks to Code2Wav)
        StageConfig(
            name=CODE_PREDICTOR_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.components.code_predictor_executor.create_code_predictor_executor_from_config",
                args={"code_predictor_max_seq_len": 256},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.code_predictor_next",
            relay=RelayConfig(device="cuda"),
            stream_to=[
                StreamTargetConfig(to_stage=CODE2WAV_STAGE),
                StreamTargetConfig(to_stage=TALKER_AR_STAGE, bootstrap=False),
            ],
        ),
        # Stage 9: Code2Wav (terminal)
        StageConfig(
            name=CODE2WAV_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.components.code2wav_executor.create_code2wav_executor",
                args={"device": "cuda"},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.code2wav_next",
            relay=RelayConfig(device="cuda"),
        ),
    ]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        _validate_qwen3_speech_gpu_placement(self.gpu_placement)

    def apply_server_args_overrides(
        self, *, stage_name: str, overrides: dict[str, Any]
    ) -> None:
        if stage_name == THINKER_STAGE:
            tp_size = overrides.get("tp_size", 1)
            _validate_qwen3_speech_gpu_placement(
                self.gpu_placement,
                tp_size=tp_size,
            )
        super().apply_server_args_overrides(
            stage_name=stage_name,
            overrides=overrides,
        )

    def apply_thinker_server_args_overrides(self, overrides: dict[str, Any]) -> None:
        self.apply_server_args_overrides(
            stage_name=THINKER_STAGE,
            overrides=overrides,
        )


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
}
