# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.qwen3_omni"
_PLACEMENT_POLICY = f"{_PKG}.placement.Qwen3OmniPlacementPolicy"
THINKER_STAGE = "thinker"
THINKER_PREFILL_STAGE = "thinker_prefill"
THINKER_DECODE_STAGE = "thinker_decode"
MIN_PARTIAL_START_CHUNKS = 3

# SGLang reads this when DeepGEMM compile utilities are imported. Qwen AR
# stages can first hit some dense FP8 shapes after readiness; disable all-M
# precompile so that miss does not become a long post-ready compile session.
# FIXME (Ratish): Replace this with a bounded/pre-ready SGLang DeepGEMM compile
# policy once that exists outside import-time environment globals.
_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS = {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}

# A colocated worker launches seven stage processes. Letting every PyTorch
# process size its OpenMP pool to the full host oversubscribes launch-side CPU
# work when multiple workers share a node. Preprocessing handles one prompt per
# scheduler call, so a host-wide tokenizer Rayon pool only adds contention.
_COLOCATED_STAGE_ENV_DEFAULTS = {
    **_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS,
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
}


def _preprocessing_stage(
    *,
    process: str,
    speech_enabled: bool = False,
    thinker_stage: str = THINKER_STAGE,
) -> StageConfig:
    if speech_enabled:
        next_stages = ["image_encoder", "audio_encoder", thinker_stage, "talker_ar"]
        route_name = (
            "resolve_preprocessing_next_stages_speech_pd"
            if thinker_stage == THINKER_PREFILL_STAGE
            else "resolve_preprocessing_next_stages_speech"
        )
        route_fn = f"{_PKG}.request_builders.{route_name}"
        join_targets = (thinker_stage, "talker_ar")
    else:
        next_stages = ["image_encoder", "audio_encoder", "mm_aggregate"]
        route_fn = f"{_PKG}.request_builders.resolve_preprocessing_next_stages"
        join_targets = ("mm_aggregate",)
    return StageConfig(
        name="preprocessing",
        process=process,
        factory_path=f"{_PKG}.stages.create_preprocessing_executor",
        factory=FactoryArgs(max_seq_len=8192),
        next=next_stages,
        route_fn=route_fn,
        project_payload={
            "image_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_image_encoder"
            ),
            "audio_encoder": (
                f"{_PKG}.request_builders.project_preprocessing_to_audio_encoder"
            ),
            **{
                target: (
                    f"{_PKG}.request_builders.project_preprocessing_to_mm_aggregate"
                )
                for target in join_targets
            },
        },
    )


def _encoder_join_edges(
    *, speech_enabled: bool, thinker_stage: str = THINKER_STAGE
) -> dict[str, object]:
    if speech_enabled:
        route_name = (
            "resolve_encoder_next_stages_pd"
            if thinker_stage == THINKER_PREFILL_STAGE
            else "resolve_encoder_next_stages"
        )
        return {
            "next": [thinker_stage, "talker_ar"],
            "route_fn": f"{_PKG}.request_builders.{route_name}",
            "project_payload": {
                thinker_stage: (
                    f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
                ),
                "talker_ar": (f"{_PKG}.request_builders.project_encoder_to_talker_ar"),
            },
        }
    return {
        "next": "mm_aggregate",
        "project_payload": {
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    }


def _image_encoder_stage(
    *,
    gpu: int,
    process: str,
    speech_enabled: bool = False,
    thinker_stage: str = THINKER_STAGE,
) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory_path=f"{_PKG}.stages.create_image_encoder_executor",
        gpu=gpu,
        **_encoder_join_edges(
            speech_enabled=speech_enabled,
            thinker_stage=thinker_stage,
        ),
    )


def _audio_encoder_stage(
    *,
    gpu: int,
    process: str,
    speech_enabled: bool = False,
    thinker_stage: str = THINKER_STAGE,
) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        process=process,
        factory_path=f"{_PKG}.stages.create_audio_encoder_executor",
        factory=FactoryArgs(enable_layer_cuda_graph=True),
        gpu=gpu,
        disable_direct_cuda_ipc_payload=True,
        **_encoder_join_edges(
            speech_enabled=speech_enabled,
            thinker_stage=thinker_stage,
        ),
    )


def _aggregate_stage(
    *, process: str, gpu: int, thinker_stage: str = THINKER_STAGE
) -> StageConfig:
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory_path=f"{_PKG}.stages.create_aggregate_executor",
        gpu=gpu,
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next=thinker_stage,
        disable_direct_cuda_ipc_payload=True,
    )


def _thinker_stage(*, gpu: int, speech_enabled: bool, process: str) -> StageConfig:
    # note (jiaxin deng): async decode defaults on;
    # --thinker.factory.enable_async_decode false overrides it.
    factory_group = FactoryArgs(max_seq_len=8192, enable_async_decode=True)
    if speech_enabled:
        factory_group = FactoryArgs(
            max_seq_len=8192, enable_async_decode=True, speech_enabled=True
        )
    join_kwargs: dict = {}
    if speech_enabled:
        join_kwargs = {
            "wait_for": ["preprocessing", "image_encoder", "audio_encoder"],
            "wait_for_fn": (
                f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources"
            ),
            "merge_fn": f"{_PKG}.merge.merge_for_thinker",
        }
    return EngineStageConfig(
        name="thinker",
        process=process,
        factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
        factory=factory_group,
        gpu=gpu,
        next="decode",
        **join_kwargs,
        stream_to=["talker_ar", "decode"] if speech_enabled else ["decode"],
        route_fn=(
            f"{_PKG}.request_builders.resolve_thinker_next_stages"
            if speech_enabled
            else None
        ),
        stream_done_to_fn=(
            f"{_PKG}.request_builders.resolve_thinker_stream_done_targets"
            if speech_enabled
            else None
        ),
        project_payload={
            "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
        },
    )


def _thinker_pd_stages(
    *,
    prefill_gpu: int,
    decode_gpu: int,
    speech_enabled: bool,
    prefill_process: str,
    decode_process: str,
) -> list[StageConfig]:
    """Declare the two PD halves as ordinary engine stages."""
    common_factory = {
        "max_seq_len": 8192,
        **({"speech_enabled": True} if speech_enabled else {}),
    }
    stream_targets = ["talker_ar", "decode"] if speech_enabled else ["decode"]
    join_kwargs: dict[str, object] = {}
    if speech_enabled:
        join_kwargs = {
            "wait_for": ["preprocessing", "image_encoder", "audio_encoder"],
            "wait_for_fn": (
                f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources"
            ),
            "merge_fn": f"{_PKG}.merge.merge_for_thinker",
        }

    return [
        EngineStageConfig(
            name=THINKER_PREFILL_STAGE,
            process=prefill_process,
            factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory=FactoryArgs(
                **common_factory,
                scheduler_role="prefill",
                enable_async_decode=False,
            ),
            engine=EngineArgs(disable_radix_cache=True, page_size=1),
            gpu=prefill_gpu,
            next=THINKER_DECODE_STAGE,
            stream_to=stream_targets,
            **join_kwargs,
        ),
        EngineStageConfig(
            name=THINKER_DECODE_STAGE,
            process=decode_process,
            factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory=FactoryArgs(
                **common_factory,
                scheduler_role="decode",
                enable_async_decode=True,
            ),
            engine=EngineArgs(disable_radix_cache=True, page_size=1),
            gpu=decode_gpu,
            next="decode",
            stream_to=stream_targets,
            route_fn=(
                f"{_PKG}.request_builders.resolve_thinker_next_stages"
                if speech_enabled
                else None
            ),
            stream_done_to_fn=(
                f"{_PKG}.request_builders.resolve_thinker_stream_done_targets"
                if speech_enabled
                else None
            ),
            project_payload={
                "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
            },
        ),
    ]


def _decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory_path=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _talker_stage(
    *,
    gpu: int,
    process: str,
    enable_partial_start: bool,
) -> StageConfig:
    return EngineStageConfig(
        name="talker_ar",
        process=process,
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.request_builders.merge_for_talker",
        factory_path=f"{_PKG}.stages.create_talker_ar_executor_from_config",
        # Note (Xuesong): max_seq_len must exceed talker_max_new_tokens (4096)
        # + prefill, else req_to_token_pool OOBs and crashes talker_ar.
        # Note (Chenyang): bumped 8192 → 32768 because the V1 talker
        # prefill replays the full thinker prompt as projected
        # embeddings, and a 30-frame video prompt is ~22K positions,
        # which overflows 8192 and triggers a FusedAddRMSNorm illegal
        # memory access in the talker forward.
        factory=FactoryArgs(
            max_seq_len=32768,
            enable_partial_start=enable_partial_start,
            partial_start_min_chunks=5,
        ),
        gpu=gpu,
        next="code2wav",
        stream_to=["code2wav"],
        project_payload={
            "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav",
        },
        can_accept_stream_before_payload=True,
    )


def _code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory_path=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        gpu=gpu,
        gpu_memory_fraction=0.02,
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def _text_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="pipeline"),
        _image_encoder_stage(gpu=0, process="pipeline"),
        _audio_encoder_stage(gpu=0, process="pipeline"),
        _aggregate_stage(process="pipeline", gpu=0),
        _thinker_stage(gpu=0, speech_enabled=False, process="pipeline"),
        _decode_stage(process="pipeline"),
    ]


def _text_pd_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(process="preprocessing"),
        _image_encoder_stage(gpu=0, process="image_encoder"),
        _audio_encoder_stage(gpu=0, process="audio_encoder"),
        _aggregate_stage(
            process="mm_aggregate",
            gpu=0,
            thinker_stage=THINKER_PREFILL_STAGE,
        ),
        *_thinker_pd_stages(
            prefill_gpu=0,
            decode_gpu=1,
            speech_enabled=False,
            prefill_process=THINKER_PREFILL_STAGE,
            decode_process=THINKER_DECODE_STAGE,
        ),
        _decode_stage(process="decode"),
    ]


def _speech_stages(
    *,
    thinker_gpu: int,
    talker_gpu: int,
    process_by_stage: dict[str, str],
    enable_partial_start: bool,
) -> list[StageConfig]:
    return [
        _preprocessing_stage(
            process=process_by_stage["preprocessing"],
            speech_enabled=True,
        ),
        _image_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["image_encoder"],
            speech_enabled=True,
        ),
        _audio_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["audio_encoder"],
            speech_enabled=True,
        ),
        _thinker_stage(
            gpu=thinker_gpu,
            speech_enabled=True,
            process=process_by_stage["thinker"],
        ),
        _decode_stage(process=process_by_stage["decode"]),
        _talker_stage(
            gpu=talker_gpu,
            process=process_by_stage["talker_ar"],
            enable_partial_start=enable_partial_start,
        ),
        _code2wav_stage(gpu=thinker_gpu, process=process_by_stage["code2wav"]),
    ]


def _speech_pd_stages() -> list[StageConfig]:
    return [
        _preprocessing_stage(
            process="preprocessing",
            speech_enabled=True,
            thinker_stage=THINKER_PREFILL_STAGE,
        ),
        _image_encoder_stage(
            gpu=0,
            process="image_encoder",
            speech_enabled=True,
            thinker_stage=THINKER_PREFILL_STAGE,
        ),
        _audio_encoder_stage(
            gpu=0,
            process="audio_encoder",
            speech_enabled=True,
            thinker_stage=THINKER_PREFILL_STAGE,
        ),
        *_thinker_pd_stages(
            prefill_gpu=0,
            decode_gpu=1,
            speech_enabled=True,
            prefill_process=THINKER_PREFILL_STAGE,
            decode_process=THINKER_DECODE_STAGE,
        ),
        _decode_stage(process="decode"),
        _talker_stage(
            gpu=2,
            process="talker_ar",
            enable_partial_start=True,
        ),
        _code2wav_stage(gpu=0, process="code2wav"),
    ]


_SPEECH_DEFAULT_PROCESSES = {
    "preprocessing": "preprocessing",
    "image_encoder": "image_encoder",
    "audio_encoder": "audio_encoder",
    "thinker": "thinker",
    "decode": "decode",
    "talker_ar": "talker_ar",
    "code2wav": "code2wav",
}


class _Qwen3OmniBasePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = (
        THINKER_STAGE,
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    @classmethod
    def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
        return {THINKER_STAGE}

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        speech_enabled = any(stage.name == "talker_ar" for stage in self.stages)
        if stage_name in ("image_encoder", "audio_encoder"):
            # Device selection is deferred to the worker; the encoders read
            # the platform default at construction.
            return {}
        if (
            stage_name
            in {
                THINKER_STAGE,
                THINKER_PREFILL_STAGE,
                THINKER_DECODE_STAGE,
            }
            and speech_enabled
        ):
            return {"speech_enabled": True}
        return {}


class Qwen3OmniPipelineConfig(_Qwen3OmniBasePipelineConfig):
    """6-stage text-only pipeline."""

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_text_stages)


class Qwen3OmniPDPipelineConfig(_Qwen3OmniBasePipelineConfig):
    """Text-only pipeline with explicit Prefill and Decode engine stages."""

    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = (
        THINKER_PREFILL_STAGE,
        THINKER_DECODE_STAGE,
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_PREFILL_STAGE: EngineStageConfig,
        THINKER_DECODE_STAGE: EngineStageConfig,
    }

    @classmethod
    def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
        return {THINKER_PREFILL_STAGE, THINKER_DECODE_STAGE}

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_text_pd_stages)

    def model_post_init(self, context: Any = None) -> None:
        super().model_post_init(context)
        stages = {stage.name: stage for stage in self.stages}
        try:
            prefill = stages[THINKER_PREFILL_STAGE]
            decode = stages[THINKER_DECODE_STAGE]
        except KeyError as exc:
            raise ValueError(
                "Qwen PD pipelines require thinker_prefill and thinker_decode"
            ) from exc

        expected = (
            (prefill, "prefill", THINKER_DECODE_STAGE),
            (decode, "decode", "decode"),
        )
        effective_gpus: list[int] = []
        for stage, role, next_stage in expected:
            if stage.tp_size != 1:
                raise ValueError(f"Qwen PD stage {stage.name!r} requires tp_size=1")
            if stage.next != next_stage:
                raise ValueError(
                    f"Qwen PD stage {stage.name!r} must route to {next_stage!r}"
                )
            if (stage.factory.model_extra or {}).get("scheduler_role") != role:
                raise ValueError(
                    f"Qwen PD stage {stage.name!r} requires scheduler_role={role!r}"
                )
            if stage.engine is None:
                raise ValueError(f"Qwen PD stage {stage.name!r} requires engine config")
            engine = stage.engine.overrides()
            if engine.get("page_size") != 1:
                raise ValueError(f"Qwen PD stage {stage.name!r} requires page_size=1")
            if engine.get("disable_radix_cache") is not True:
                raise ValueError(
                    f"Qwen PD stage {stage.name!r} requires disable_radix_cache=true"
                )
            gpu = stage.gpu
            if gpu is None or isinstance(gpu, list):
                raise ValueError(
                    f"Qwen PD stage {stage.name!r} requires one explicit GPU"
                )

            process_name = stage.process
            assert process_name is not None
            process = self.processes.get(process_name)
            if process is not None and process.num_replicas != 1:
                raise ValueError(
                    f"Qwen PD stage {stage.name!r} does not support replicas"
                )
            if process is not None and process.replica_devices is not None:
                if len(process.replica_devices) != 1:
                    raise ValueError(
                        f"Qwen PD stage {stage.name!r} requires one process GPU"
                    )
                effective_gpus.append(process.replica_devices[0])
            else:
                effective_gpus.append(gpu)

        if prefill.process == decode.process:
            raise ValueError("Qwen PD stages must run in separate processes")
        if effective_gpus[0] == effective_gpus[1]:
            raise ValueError("Qwen PD stages must run on separate GPUs")
        if prefill.factory.enable_async_decode is not False:
            raise ValueError("Qwen PD Prefill requires enable_async_decode=false")


class Qwen3OmniSpeechPipelineConfig(_Qwen3OmniBasePipelineConfig):
    """7-stage speech pipeline (text + audio output)."""

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
        "talker_ar": EngineStageConfig,
    }

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "code2wav"

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(
            thinker_gpu=0,
            talker_gpu=1,
            process_by_stage=_SPEECH_DEFAULT_PROCESSES,
            enable_partial_start=True,
        )
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "talker_ar":
            return {
                "speech_enabled": True,
                "feedback_enabled": True,
            }
        if stage_name == "code2wav":
            return {
                "enable_cuda_graph": current_platform.enable_code2wav_graph(),
            }
        return super().stage_factory_kwargs(stage_name)


class Qwen3OmniSpeechPDPipelineConfig(Qwen3OmniPDPipelineConfig):
    """Speech pipeline with explicit Prefill and Decode engine stages."""

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_PREFILL_STAGE: EngineStageConfig,
        THINKER_DECODE_STAGE: EngineStageConfig,
        "talker_ar": EngineStageConfig,
    }

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "code2wav"

    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"
    stages: list[StageConfig] = Field(default_factory=_speech_pd_stages)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "talker_ar":
            return {
                "speech_enabled": True,
                "feedback_enabled": True,
            }
        if stage_name == "code2wav":
            return {
                "enable_cuda_graph": current_platform.enable_code2wav_graph(),
            }
        return super().stage_factory_kwargs(stage_name)


class Qwen3OmniSpeechColocatedPipelineConfig(Qwen3OmniSpeechPipelineConfig):
    """7-stage speech pipeline for single-GPU stage colocation.

    The topology places image_encoder, audio_encoder, thinker, talker_ar, and
    code2wav on the same GPU while keeping preprocessing and decode as CPU
    stages. Per-stage memory budgets are supplied by the selected config
    file so deployments can use hardware-appropriate stage fractions and
    SGLang AR cache fractions.
    """

    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_COLOCATED_STAGE_ENV_DEFAULTS)
    )

    stages: list[StageConfig] = Field(
        default_factory=lambda: _speech_stages(
            thinker_gpu=0,
            talker_gpu=0,
            process_by_stage=_SPEECH_DEFAULT_PROCESSES,
            enable_partial_start=False,
        )
    )


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
    "speech-colocated": Qwen3OmniSpeechColocatedPipelineConfig,
}
