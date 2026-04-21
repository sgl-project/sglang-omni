# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
)


class MingOmniPipelineConfig(PipelineConfig):
    """6-stage text/vision pipeline for Ming-Omni.

    preprocessing → audio_encoder + image_encoder → mm_aggregate → thinker → decode
    """

    architecture: ClassVar[str] = "BailingMM2NativeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_audio_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, AUDIO_STAGE, IMAGE_STAGE],
                merge_fn="sglang_omni.models.ming_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={
                    "thinker_max_seq_len": 8192,
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.thinker_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE}


def _validate_ming_speech_gpu_placement(
    gpu_placement: dict[str, int],
    *,
    tp_size: int,
) -> None:
    thinker_gpu = gpu_placement.get("thinker", 0)
    talker_gpu = gpu_placement.get("talker", 1)
    thinker_range = range(thinker_gpu, thinker_gpu + tp_size)
    if talker_gpu in thinker_range:
        raise ValueError(
            f"Talker GPU {talker_gpu} collides with thinker TP range "
            f"[{thinker_gpu}, {thinker_gpu + tp_size}). "
            f"Set --gpu-talker >= {thinker_gpu + tp_size}."
        )


class MingOmniSpeechPipelineConfig(PipelineConfig):
    """7-stage pipeline for Ming-Omni with text + speech output.

    Adds a talker stage that generates audio from thinker's decoded text.
    The talker is a self-contained MingOmniTalker (own LLM + CFM + AudioVAE).
    """

    architecture: ClassVar[str] = "BailingMM2NativeForConditionalGeneration"

    model_path: str
    entry_stage: str = "preprocessing"
    terminal_stages: list[str] = [DECODE_STAGE, TALKER_STAGE]
    gpu_placement: dict[str, int] = {
        "thinker": 0,
        "talker": 1,
    }

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_preprocessing_executor",
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_audio_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_image_encoder_executor",
                args={"device": "cuda", "dtype": None},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_aggregate_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, AUDIO_STAGE, IMAGE_STAGE],
                merge_fn="sglang_omni.models.ming_omni.pipeline.merge.merge_for_thinker",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=THINKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_sglang_thinker_executor_from_config",
                args={"thinker_max_seq_len": 8192},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.thinker_next_speech",
            relay=RelayConfig(device="cuda"),
        ),
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_decode_executor",
                args={},
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=TALKER_STAGE,
            executor=ExecutorConfig(
                factory="sglang_omni.models.ming_omni.pipeline.stages.create_talker_executor",
                args={
                    "device": "cuda",
                    "voice": "DB30",
                },
            ),
            get_next="sglang_omni.models.ming_omni.pipeline.next_stage.talker_next",
            relay=RelayConfig(device="cuda"),
        ),
    ]

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE}

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        _validate_ming_speech_gpu_placement(self.gpu_placement, tp_size=1)

    def apply_server_args_overrides(
        self, *, stage_name: str, overrides: dict[str, Any]
    ) -> None:
        if stage_name == THINKER_STAGE and "tp_size" in overrides:
            _validate_ming_speech_gpu_placement(
                self.gpu_placement,
                tp_size=overrides["tp_size"],
            )
        super().apply_server_args_overrides(
            stage_name=stage_name,
            overrides=overrides,
        )


EntryClass = MingOmniPipelineConfig
