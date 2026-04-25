# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.config.schema import StreamTargetConfig
from sglang_omni.models.qwen3_omni.pipeline.engine_io import DEFAULT_THINKER_MAX_SEQ_LEN
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


class Qwen3OmniPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_preprocessing_executor",
                args={"max_seq_len": DEFAULT_THINKER_MAX_SEQ_LEN},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            payload_filter="sglang_omni.models.qwen3_omni.pipeline.payload_filter.preprocessing_payload_filter",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                    "max_batch_size": 32,
                },
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            payload_filter="sglang_omni.models.qwen3_omni.pipeline.payload_filter.encoder_payload_filter",
            relay=RelayConfig(device="cpu"),
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
            payload_filter="sglang_omni.models.qwen3_omni.pipeline.payload_filter.encoder_payload_filter",
            relay=RelayConfig(device="cpu"),
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
                    "thinker_max_seq_len": DEFAULT_THINKER_MAX_SEQ_LEN,
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

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE}

    def apply_server_args_overrides(
        self, *, stage_name: str, overrides: dict[str, Any]
    ) -> None:
        if (
            stage_name == THINKER_STAGE
            and "tp_size" in overrides
            and overrides["tp_size"] > 1
        ):
            raise NotImplementedError("Qwen3-Omni TP is not supported yet.")
        remaining = _route_thinker_executor_args(self.stages, stage_name, overrides)
        if remaining:
            super().apply_server_args_overrides(
                stage_name=stage_name,
                overrides=remaining,
            )


def _route_thinker_executor_args(
    stages: list[StageConfig],
    stage_name: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Pop thinker-factory kwargs onto the thinker stage; return the rest."""
    remaining = dict(overrides)
    if stage_name != THINKER_STAGE:
        return remaining

    casted: dict[str, Any] = {}

    seq_len = remaining.pop("thinker_max_seq_len", None)
    if seq_len is not None:
        seq_len = int(seq_len)
        casted["thinker_max_seq_len"] = seq_len

    reserve = remaining.pop("encoder_mem_reserve", None)
    if reserve is not None:
        reserve = float(reserve)
        if not 0.0 <= reserve < 1.0:
            raise ValueError(f"encoder_mem_reserve must be in [0, 1), got {reserve}")
        casted["encoder_mem_reserve"] = reserve

    if casted:
        for stage in stages:
            if stage.name == THINKER_STAGE:
                stage.executor.args.update(casted)
            elif seq_len is not None and stage.name == PREPROCESSING_STAGE:
                stage.executor.args["max_seq_len"] = seq_len
            elif seq_len is not None and stage.name == TALKER_AR_STAGE:
                stage.executor.args["talker_max_seq_len"] = seq_len
    return remaining


def _validate_qwen3_speech_gpu_placement(
    gpu_placement: dict[str, int],
    *,
    tp_size: int,
) -> None:
    thinker_gpu = gpu_placement.get(THINKER_STAGE, 0)
    thinker_range = range(thinker_gpu, thinker_gpu + tp_size)
    for stage_name in (TALKER_AR_STAGE, CODE_PREDICTOR_STAGE, CODE2WAV_STAGE):
        stage_gpu = gpu_placement.get(stage_name, 1)
        if stage_gpu in thinker_range:
            raise ValueError(
                f"Speech stage {stage_name!r} GPU {stage_gpu} collides with "
                f"thinker TP range [{thinker_gpu}, {thinker_gpu + tp_size}). "
                f"Place speech stages on GPU >= {thinker_gpu + tp_size}."
            )


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """9-stage pipeline config for Qwen3 Omni with text + speech output."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    terminal_stages: list[str] = [DECODE_STAGE, CODE2WAV_STAGE]
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
                args={"max_seq_len": DEFAULT_THINKER_MAX_SEQ_LEN},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.preprocessing_next",
            payload_filter="sglang_omni.models.qwen3_omni.pipeline.payload_filter.preprocessing_payload_filter",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_image_encoder_executor",
                args={"device": "cuda", "dtype": None, "max_batch_size": 32},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            payload_filter="sglang_omni.models.qwen3_omni.pipeline.payload_filter.encoder_payload_filter",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_audio_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.encoder_next",
            payload_filter="sglang_omni.models.qwen3_omni.pipeline.payload_filter.encoder_payload_filter",
            relay=RelayConfig(device="cpu"),
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
                args={
                    "thinker_max_seq_len": DEFAULT_THINKER_MAX_SEQ_LEN,
                    "speech_enabled": True,
                },
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
                args={"server_args_overrides": {}},
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
                factory="sglang_omni.models.qwen3_omni.components.code2wav_executor.create_code2wav_executor_from_config",
                args={"server_args_overrides": {}},
            ),
            get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.code2wav_next",
            relay=RelayConfig(device="cuda"),
        ),
    ]

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE, "talker": TALKER_AR_STAGE}

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        _validate_qwen3_speech_gpu_placement(self.gpu_placement, tp_size=1)

    def apply_server_args_overrides(
        self, *, stage_name: str, overrides: dict[str, Any]
    ) -> None:
        # TODO (Ratish, Chenyang):
        # Order matters: validate placement first so users get an actionable
        # collision error currently, and the same check gates TP after support lands.
        if stage_name in (THINKER_STAGE, TALKER_AR_STAGE) and "tp_size" in overrides:
            tp_size = overrides["tp_size"]

            # TODO (Ratish, Chenyang):
            # Validate placement for whichever AR stage is being scaled.
            # Currently, talker_ar's range-check reuses the same thinker-range
            # helper; once per-stage TP lands, extend this with a
            # talker_ar-specific collision check rather than letting the
            # outer guard silently pass.
            _validate_qwen3_speech_gpu_placement(
                self.gpu_placement,
                tp_size=tp_size,
            )
            if tp_size > 1:
                raise NotImplementedError("Qwen3-Omni TP is not supported yet.")
        remaining = _route_thinker_executor_args(self.stages, stage_name, overrides)
        if remaining:
            super().apply_server_args_overrides(
                stage_name=stage_name,
                overrides=remaining,
            )


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
}
