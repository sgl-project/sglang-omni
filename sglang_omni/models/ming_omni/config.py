# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config.schema import PipelineConfig, PlacementConfig, StageConfig
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_GEN_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    SEGMENTER_STAGE,
    TALKER_STAGE,
    TALKER_STREAM_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.ming_omni.tp_utils import validate_stage_tp_support

_PKG = "sglang_omni.models.ming_omni"


def _stage_by_name(stages: list[StageConfig], name: str) -> StageConfig | None:
    return next((stage for stage in stages if stage.name == name), None)


def _stage_gpu_set(gpu: int | list[int] | None, tp_size: int) -> set[int]:
    """Return GPUs occupied by a stage.

    Explicit list placement is authoritative; scalar placement preserves the
    legacy contiguous TP range interpretation.
    """
    if isinstance(gpu, list):
        return {int(gpu_id) for gpu_id in gpu}
    if gpu is None:
        return set()
    return set(range(int(gpu), int(gpu) + tp_size))


def _validate_ming_stage_tp_support(stages: list[StageConfig]) -> None:
    for stage in stages:
        validate_stage_tp_support(stage_name=stage.name, tp_size=stage.tp_size)


def _preprocessing_stage(
    *,
    process: str,
    enable_image_gen: bool = False,
) -> StageConfig:
    factory_args = {"enable_image_gen": True} if enable_image_gen else {}
    return StageConfig(
        name=PREPROCESSING_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_preprocessing_executor",
        factory_args=factory_args,
        next=[AUDIO_STAGE, IMAGE_STAGE, AGGREGATE_STAGE],
        project_payload={
            AUDIO_STAGE: f"{_PKG}.stages.project_preprocessing_to_audio_encoder",
            IMAGE_STAGE: f"{_PKG}.stages.project_preprocessing_to_image_encoder",
            AGGREGATE_STAGE: (f"{_PKG}.stages.project_preprocessing_to_mm_aggregate"),
        },
    )


def _audio_encoder_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name=AUDIO_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_audio_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        next=AGGREGATE_STAGE,
        project_payload={
            AGGREGATE_STAGE: f"{_PKG}.stages.project_encoder_to_mm_aggregate"
        },
    )


def _image_encoder_stage(
    *, gpu: int | list[int], tp_size: int = 1, process: str
) -> StageConfig:
    return StageConfig(
        name=IMAGE_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_image_encoder_executor",
        factory_args={"device": "cuda", "dtype": None},
        gpu=gpu,
        tp_size=tp_size,
        next=AGGREGATE_STAGE,
        project_payload={
            AGGREGATE_STAGE: f"{_PKG}.stages.project_encoder_to_mm_aggregate"
        },
    )


def _aggregate_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name=AGGREGATE_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_aggregate_executor",
        wait_for=[PREPROCESSING_STAGE, AUDIO_STAGE, IMAGE_STAGE],
        merge_fn=f"{_PKG}.pipeline.merge.merge_for_thinker",
        next=THINKER_STAGE,
        disable_direct_cuda_ipc_payload=True,
    )


def _thinker_stage(
    *,
    gpu: int,
    speech_enabled: bool,
    process: str,
    image_gen_enabled: bool = False,
) -> StageConfig:
    factory_args: dict[str, Any] = {"thinker_max_seq_len": 8192}
    if image_gen_enabled:
        factory_args["capture_hidden"] = True

    next_stages = [DECODE_STAGE]
    if speech_enabled:
        next_stages.append(TALKER_STAGE)
    if image_gen_enabled:
        next_stages.append(IMAGE_GEN_STAGE)

    return StageConfig(
        name=THINKER_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory_args=factory_args,
        gpu=gpu,
        next=next_stages if len(next_stages) > 1 else DECODE_STAGE,
        stream_to=[DECODE_STAGE],
    )


def _streaming_thinker_stage(*, gpu: int, process: str) -> StageConfig:
    """Thinker stage variant for streaming TTS.

    Fans out to decode + segmenter and streams completion to both. The
    segmenter consumes TTS text chunks; decode needs stream_done to finalize
    stream=true requests.
    """
    return StageConfig(
        name=THINKER_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory_args={"thinker_max_seq_len": 8192, "enable_streaming_tts": True},
        gpu=gpu,
        next=[DECODE_STAGE, SEGMENTER_STAGE],
        stream_to=[DECODE_STAGE, SEGMENTER_STAGE],
    )


def _segmenter_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name=SEGMENTER_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_streaming_segmenter_executor",
        next=TALKER_STREAM_STAGE,
        stream_to=[TALKER_STREAM_STAGE],
        can_accept_stream_before_payload=True,
    )


def _talker_stream_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name=TALKER_STREAM_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_streaming_talker_executor",
        factory_args={"device": "cuda", "voice": "DB30"},
        gpu=gpu,
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name=DECODE_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _talker_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name=TALKER_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_talker_executor",
        factory_args={"device": "cuda", "voice": "DB30"},
        gpu=gpu,
        terminal=True,
    )


def _image_gen_stage(
    gpu: int,
    process: str,
    dit_type: str = "zimage",
    dit_model_path: str | None = None,
    enable_standalone_semantic_encoder: bool = False,
    enable_byt5_text_rendering: bool = False,
) -> StageConfig:
    factory_args = {"device": "cuda", "dit_type": dit_type}
    if dit_model_path is not None:
        factory_args["dit_model_path"] = dit_model_path
    if enable_standalone_semantic_encoder:
        factory_args["enable_standalone_semantic_encoder"] = True
    if enable_byt5_text_rendering:
        factory_args["enable_byt5_text_rendering"] = True
    return StageConfig(
        name=IMAGE_GEN_STAGE,
        process=process,
        factory=f"{_PKG}.stages.create_image_gen_executor",
        factory_args=factory_args,
        gpu=gpu,
        terminal=True,
    )


def _ming_text_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="preprocessing"),
        _audio_encoder_stage(gpu=0, process="audio_encoder"),
        _image_encoder_stage(gpu=0, process="image_encoder"),
        _aggregate_stage(process="mm_aggregate"),
        _thinker_stage(gpu=0, speech_enabled=False, process="thinker"),
        _decode_stage(process="decode"),
    ]


def _ming_speech_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="preprocessing"),
        _audio_encoder_stage(gpu=0, process="audio_encoder"),
        _image_encoder_stage(gpu=0, process="image_encoder"),
        _aggregate_stage(process="mm_aggregate"),
        _thinker_stage(gpu=0, speech_enabled=True, process="thinker"),
        _decode_stage(process="decode"),
        _talker_stage(gpu=1, process="talker"),
    ]


def _ming_streaming_speech_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="preprocessing"),
        _audio_encoder_stage(gpu=0, process="audio_encoder"),
        _image_encoder_stage(gpu=0, process="image_encoder"),
        _aggregate_stage(process="mm_aggregate"),
        _streaming_thinker_stage(gpu=0, process="thinker"),
        _decode_stage(process="decode"),
        _segmenter_stage(process="segmenter"),
        _talker_stream_stage(gpu=1, process="talker_stream"),
    ]


def _ming_image_stages(
    *,
    dit_type: str = "zimage",
    dit_model_path: str | None = None,
    enable_standalone_semantic_encoder: bool = False,
    enable_byt5_text_rendering: bool = False,
) -> list[StageConfig]:
    return [
        _preprocessing_stage(process="preprocessing", enable_image_gen=True),
        _audio_encoder_stage(gpu=0, process="audio_encoder"),
        _image_encoder_stage(gpu=0, process="image_encoder"),
        _aggregate_stage(process="mm_aggregate"),
        _thinker_stage(
            gpu=0,
            speech_enabled=False,
            image_gen_enabled=True,
            process="thinker",
        ),
        _decode_stage(process="decode"),
        _image_gen_stage(
            gpu=1,
            process="image_gen",
            dit_type=dit_type,
            dit_model_path=dit_model_path,
            enable_standalone_semantic_encoder=enable_standalone_semantic_encoder,
            enable_byt5_text_rendering=enable_byt5_text_rendering,
        ),
    ]


def _apply_image_gen_factory_args(
    stages: list[StageConfig],
    *,
    dit_type: str,
    dit_model_path: str | None,
    enable_standalone_semantic_encoder: bool = False,
    enable_byt5_text_rendering: bool = False,
) -> None:
    image_gen = _stage_by_name(stages, IMAGE_GEN_STAGE)
    if image_gen is None:
        return

    factory_args = dict(image_gen.factory_args)
    factory_args["device"] = factory_args.get("device", "cuda")
    factory_args["dit_type"] = dit_type
    if dit_model_path is not None:
        factory_args["dit_model_path"] = dit_model_path
    else:
        factory_args.pop("dit_model_path", None)
    if enable_standalone_semantic_encoder:
        factory_args["enable_standalone_semantic_encoder"] = True
    else:
        factory_args.pop("enable_standalone_semantic_encoder", None)
    if enable_byt5_text_rendering:
        factory_args["enable_byt5_text_rendering"] = True
    else:
        factory_args.pop("enable_byt5_text_rendering", None)
    image_gen.factory_args = factory_args


def _validate_stage_gpus_do_not_overlap(
    stages: list[StageConfig],
    *,
    subject_name: str,
    subject_label: str,
    owner_name: str,
    owner_label: str,
    error_prefix: str,
) -> None:
    subject = _stage_by_name(stages, subject_name)
    owner = _stage_by_name(stages, owner_name)
    if subject is None or owner is None:
        return

    subject_gpus = _stage_gpu_set(subject.gpu, subject.tp_size)
    owner_gpus = _stage_gpu_set(owner.gpu, owner.tp_size)
    collisions = subject_gpus & owner_gpus
    if not collisions:
        return

    raise ValueError(
        f"{error_prefix}: "
        f"{subject_label} gpus={sorted(subject_gpus)}, "
        f"{owner_label} gpus={sorted(owner_gpus)}, "
        f"collisions={sorted(collisions)}"
    )


class _MingOmniBasePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BailingMM2NativeForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("BailingMoeV2ForCausalLM",)
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = (
        THINKER_STAGE,
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": THINKER_STAGE}

    @classmethod
    def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
        return {THINKER_STAGE}


class MingOmniPipelineConfig(_MingOmniBasePipelineConfig):
    """6-stage text pipeline."""

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_ming_text_stages)

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        _validate_ming_stage_tp_support(self.stages)


class MingOmniSpeechPipelineConfig(_MingOmniBasePipelineConfig):
    """7-stage speech pipeline."""

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TALKER_STAGE}

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_ming_speech_stages)

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        _validate_ming_stage_tp_support(self.stages)
        self._validate_talker_gpu_not_in_thinker_tp_range()

    def _validate_talker_gpu_not_in_thinker_tp_range(self) -> None:
        _validate_stage_gpus_do_not_overlap(
            self.stages,
            subject_name=TALKER_STAGE,
            subject_label="talker",
            owner_name=THINKER_STAGE,
            owner_label="thinker",
            error_prefix=("Ming-Omni speech talker GPU collides with thinker TP range"),
        )


class MingOmniImagePipelineConfig(_MingOmniBasePipelineConfig):
    """7-stage image generation pipeline."""

    architecture: ClassVar[str] = "BailingMoeV2ForCausalLM"

    model_path: str
    dit_type: str = "zimage"
    dit_model_path: str | None = None
    enable_standalone_semantic_encoder: bool = False
    enable_byt5_text_rendering: bool = False
    entry_stage: str = PREPROCESSING_STAGE
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_ming_image_stages)

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        _validate_ming_stage_tp_support(self.stages)
        _apply_image_gen_factory_args(
            self.stages,
            dit_type=self.dit_type,
            dit_model_path=self.dit_model_path,
            enable_standalone_semantic_encoder=(
                self.enable_standalone_semantic_encoder
            ),
            enable_byt5_text_rendering=self.enable_byt5_text_rendering,
        )
        self._validate_image_gen_gpu_not_in_thinker_tp_range()

    def _validate_image_gen_gpu_not_in_thinker_tp_range(self) -> None:
        _validate_stage_gpus_do_not_overlap(
            self.stages,
            subject_name=IMAGE_GEN_STAGE,
            subject_label="image_gen",
            owner_name=THINKER_STAGE,
            owner_label="thinker",
            error_prefix=("Ming-Omni image_gen GPU collides with thinker TP range"),
        )


class MingOmniStreamingSpeechPipelineConfig(_MingOmniBasePipelineConfig):
    """8-stage streaming-TTS speech pipeline.

    Adds a ``segmenter`` stage between ``thinker`` and ``talker_stream``
    that converts incremental thinker text deltas into speakable segments.
    The thinker fans out final payloads to ``decode`` and ``segmenter``,
    and streams per-token deltas to ``segmenter`` via stream_to. The
    streaming talker emits audio chunks to the coordinator (terminal).
    """

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TALKER_STREAM_STAGE}

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_ming_streaming_speech_stages)

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        _validate_ming_stage_tp_support(self.stages)
        self._validate_talker_stream_gpu_not_in_thinker_tp_range()

    def _validate_talker_stream_gpu_not_in_thinker_tp_range(self) -> None:
        thinker = _stage_by_name(self.stages, THINKER_STAGE)
        talker = _stage_by_name(self.stages, TALKER_STREAM_STAGE)
        if thinker is None or talker is None:
            return

        thinker_gpus = _stage_gpu_set(thinker.gpu, thinker.tp_size)
        talker_gpus = _stage_gpu_set(talker.gpu, talker.tp_size)
        collisions = thinker_gpus & talker_gpus
        if not collisions:
            return

        raise ValueError(
            "Ming-Omni streaming-speech talker GPU collides with thinker TP range: "
            f"talker gpus={sorted(talker_gpus)}, "
            f"thinker gpus={sorted(thinker_gpus)}, "
            f"collisions={sorted(collisions)}"
        )


EntryClass = MingOmniSpeechPipelineConfig

Variants = {
    "text": MingOmniPipelineConfig,
    "speech": MingOmniSpeechPipelineConfig,
    "streaming_speech": MingOmniStreamingSpeechPipelineConfig,
    "image": MingOmniImagePipelineConfig,
}
