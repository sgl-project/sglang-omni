# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-Omni."""

from __future__ import annotations

import base64
from typing import ClassVar

from pydantic import Field, model_validator

from sglang_omni.admission import REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    ProcessConfig,
    StageConfig,
)
from sglang_omni.config.session_audio import AudioSessionPipelineConfig, StageGroup
from sglang_omni.models.qwen3_omni.session_node import SESSION_PORTS
from sglang_omni.platforms import current_platform
from sglang_omni.proto.session import GRAPH_INPUT
from sglang_omni.proto.streams import IMAGE, SessionPorts
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage
from sglang_omni.serve.server_warmup import WARMUP_MAX_TOKENS, warmup_tone_wav_bytes
from sglang_omni.utils.cpu import effective_cpu_count

_PKG = "sglang_omni.models.qwen3_omni"
_PLACEMENT_POLICY = f"{_PKG}.placement.Qwen3OmniPlacementPolicy"
THINKER_STAGE = "thinker"
# Note (wenyao): Config and pre-boot gates preserve the legacy three-chunk floor;
# TALKER_START_MIN_CHUNKS reflects the prompt topology's one-chunk minimum.
MIN_PARTIAL_START_CHUNKS = 3

# Note (wenyao): the talker assistant prompt needs one chunk
# for a 9-row tail (3 template + 4 pad + BOS + text); later chunks feed decode.
TALKER_START_MIN_CHUNKS = 1

ENABLE_TALKER_START_TOPOLOGY = False

TALKER_MAX_RUNNING_REQUESTS = 32

# SGLang reads this when DeepGEMM compile utilities are imported. Qwen AR
# stages can first hit some dense FP8 shapes after readiness; disable all-M
# precompile so that miss does not become a long post-ready compile session.
# FIXME (Ratish): Replace this with a bounded/pre-ready SGLang DeepGEMM compile
# policy once that exists outside import-time environment globals.
_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS = {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}

# A colocated worker launches six stage processes. Letting every PyTorch
# process size its OpenMP pool to the full host oversubscribes launch-side CPU
# work when multiple workers share a node. Preprocessing handles one prompt per
# scheduler call, so a host-wide tokenizer Rayon pool only adds contention.
_COLOCATED_STAGE_ENV_DEFAULTS = {
    **_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS,
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
}


def preprocessing_stage(*, process: str, speech_enabled: bool = False) -> StageConfig:
    if speech_enabled:
        next_stages = ["image_encoder", "audio_encoder", "thinker", "talker_ar"]
        route_fn = f"{_PKG}.request_builders.resolve_preprocessing_next_stages_speech"
        join_targets = ("thinker", "talker_ar")
    else:
        next_stages = ["image_encoder", "audio_encoder", "mm_aggregate"]
        route_fn = f"{_PKG}.request_builders.resolve_preprocessing_next_stages"
        join_targets = ("mm_aggregate",)
    return StageConfig(
        name="preprocessing",
        process=process,
        factory_path=f"{_PKG}.stages.create_preprocessing_executor",
        factory=FactoryArgs(),
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


def encoder_join_edges(*, speech_enabled: bool) -> dict[str, object]:
    if speech_enabled:
        return {
            "next": ["thinker", "talker_ar"],
            "route_fn": f"{_PKG}.request_builders.resolve_encoder_next_stages",
            "project_payload": {
                "thinker": (f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"),
                "talker_ar": (f"{_PKG}.request_builders.project_encoder_to_talker_ar"),
            },
        }
    else:
        pass
    return {
        "next": "mm_aggregate",
        "project_payload": {
            "mm_aggregate": f"{_PKG}.request_builders.project_encoder_to_mm_aggregate"
        },
    }


def image_encoder_stage(
    *, gpu: int, process: str, speech_enabled: bool = False
) -> StageConfig:
    return StageConfig(
        name="image_encoder",
        process=process,
        factory_path=f"{_PKG}.stages.create_image_encoder_executor",
        gpu=gpu,
        **encoder_join_edges(speech_enabled=speech_enabled),
    )


def audio_encoder_stage(
    *, gpu: int, process: str, speech_enabled: bool = False
) -> StageConfig:
    return StageConfig(
        name="audio_encoder",
        process=process,
        factory_path=f"{_PKG}.stages.create_audio_encoder_executor",
        factory=FactoryArgs(enable_layer_cuda_graph=True),
        gpu=gpu,
        disable_direct_cuda_ipc_payload=True,
        **encoder_join_edges(speech_enabled=speech_enabled),
    )


def aggregate_stage(*, process: str, gpu: int) -> StageConfig:
    return StageConfig(
        name="mm_aggregate",
        process=process,
        factory_path=f"{_PKG}.stages.create_aggregate_executor",
        gpu=gpu,
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.merge.merge_for_thinker",
        next="thinker",
        disable_direct_cuda_ipc_payload=True,
    )


def thinker_stage(*, gpu: int, speech_enabled: bool, process: str) -> StageConfig:
    # note (jiaxin deng): async decode defaults on;
    # --thinker.factory.enable_async_decode false overrides it.
    factory_group = FactoryArgs(enable_async_decode=True)
    if speech_enabled:
        factory_group = FactoryArgs(enable_async_decode=True, speech_enabled=True)
    else:
        pass
    join_kwargs: dict = {}
    if speech_enabled:
        join_kwargs = {
            "wait_for": ["preprocessing", "image_encoder", "audio_encoder"],
            "wait_for_fn": (
                f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources"
            ),
            "merge_fn": f"{_PKG}.merge.merge_for_thinker",
        }
    else:
        pass
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


def decode_stage(*, process: str) -> StageConfig:
    return StageConfig(
        name="decode",
        process=process,
        factory_path=f"{_PKG}.stages.create_decode_executor",
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def talker_stage_env() -> dict[str, str]:
    if current_platform.is_rocm():
        # Note (zijiecode): aiter.greedy_sample returns wrong ids for vocab sizes below
        # 16384 (gfx950, aiter c16d44b9) and the Talker codec head has 3072, so a
        # greedy Talker request would corrupt its first codec token.
        return {"SGLANG_DISABLE_AITER_GREEDY_SAMPLE": "1"}
    else:
        return {}


def talker_stage(
    *,
    gpu: int,
    process: str,
    enable_partial_start: bool,
) -> StageConfig:
    return EngineStageConfig(
        name="talker_ar",
        process=process,
        env=talker_stage_env(),
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",
        merge_fn=f"{_PKG}.request_builders.merge_for_talker",
        factory_path=f"{_PKG}.stages.create_talker_ar_executor_from_config",
        factory=FactoryArgs(
            enable_partial_start=enable_partial_start,
            partial_start_min_chunks=5,
            # Note (wenyao): Match the default serial Code2Wav window so later
            # 10-row messages keep the captured 10/20/30/35-frame graph shapes.
            codec_coalesce_frames=10,
            codec_coalesce_early_frames=10,
            codec_coalesce_first_frames=0,
        ),
        gpu=gpu,
        next="code2wav",
        stream_to=["code2wav"],
        project_payload={
            "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav",
        },
        can_accept_stream_before_payload=True,
    )


def code2wav_stage(*, gpu: int, process: str) -> StageConfig:
    return StageConfig(
        name="code2wav",
        process=process,
        factory_path=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        gpu=gpu,
        gpu_memory_fraction=0.02,
        terminal=True,
        can_accept_stream_before_payload=True,
    )


def text_stages() -> list[StageConfig]:
    return [
        preprocessing_stage(process="pipeline"),
        image_encoder_stage(gpu=0, process="pipeline"),
        audio_encoder_stage(gpu=0, process="pipeline"),
        aggregate_stage(process="pipeline", gpu=0),
        thinker_stage(gpu=0, speech_enabled=False, process="pipeline"),
        decode_stage(process="pipeline"),
    ]


def speech_stages(
    *,
    thinker_gpu: int,
    talker_gpu: int,
    process_by_stage: dict[str, str],
    enable_partial_start: bool,
) -> list[StageConfig]:
    return [
        preprocessing_stage(
            process=process_by_stage["preprocessing"],
            speech_enabled=True,
        ),
        image_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["image_encoder"],
            speech_enabled=True,
        ),
        audio_encoder_stage(
            gpu=thinker_gpu,
            process=process_by_stage["audio_encoder"],
            speech_enabled=True,
        ),
        thinker_stage(
            gpu=thinker_gpu,
            speech_enabled=True,
            process=process_by_stage["thinker"],
        ),
        decode_stage(process=process_by_stage["decode"]),
        talker_stage(
            gpu=talker_gpu,
            process=process_by_stage["talker_ar"],
            enable_partial_start=enable_partial_start,
        ),
        code2wav_stage(gpu=thinker_gpu, process=process_by_stage["code2wav"]),
    ]


SPEECH_DEFAULT_PROCESSES = {
    "preprocessing": "preprocessing",
    "image_encoder": "image_encoder",
    "audio_encoder": "audio_encoder",
    "thinker": "thinker",
    "decode": "decode",
    "talker_ar": "talker_ar",
    "code2wav": "code2wav",
}

# note (ratish): on one card the GPU time-slices between the stage processes,
# so code2wav decodes inside the talker's process on a priority stream instead
# of waiting for its own turn.
COLOCATED_SPEECH_PROCESSES = {**SPEECH_DEFAULT_PROCESSES, "code2wav": "talker_ar"}

WARMUP_IMAGE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAA"
    "bUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y"
    "6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw7"
    "3kKCSgAAAABJRU5ErkJggg=="
)


def build_server_warmup_request(
    pipeline_config: PipelineConfig,
) -> ChatCompletionRequest:
    """An image, a second of audio and a short instruction through every stage."""
    if pipeline_config.code2wav_stage() is None:
        modalities = ["text"]
        talker_max_new_tokens = None
    else:
        modalities = ["text", "audio"]
        talker_max_new_tokens = WARMUP_MAX_TOKENS
    return ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{WARMUP_IMAGE_PNG_BASE64}"
                        },
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(warmup_tone_wav_bytes()).decode(
                                "ascii"
                            ),
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "Describe the image and the audio."},
                ],
            )
        ],
        modalities=modalities,
        max_tokens=WARMUP_MAX_TOKENS,
        talker_max_new_tokens=talker_max_new_tokens,
        temperature=0.0,
    )


class Qwen3OmniBasePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"
    server_warmup_request_factory: ClassVar[str] = (
        f"{_PKG}.config.build_server_warmup_request"
    )
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = (
        THINKER_STAGE,
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: EngineStageConfig,
    }
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_DEEPGEMM_PRECOMPILE_ENV_DEFAULTS)
    )

    def resolved_stage_env_defaults(self, stage_name: str) -> dict[str, str]:
        """Keep CPU-sensitive preprocessing parallel unless OMP is configured."""
        env_defaults = super().resolved_stage_env_defaults(stage_name)
        if stage_name == "preprocessing" and "OMP_NUM_THREADS" not in env_defaults:
            env_defaults["OMP_NUM_THREADS"] = str(effective_cpu_count())
        else:
            pass
        return env_defaults

    @classmethod
    def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
        return {THINKER_STAGE}

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, bool]:
        speech_enabled = any(stage.name == "talker_ar" for stage in self.stages)
        if stage_name in ("image_encoder", "audio_encoder"):
            # Device selection is deferred to the worker; the encoders read
            # the platform default at construction.
            return {}
        else:
            pass
        if stage_name == "thinker" and speech_enabled:
            return {"speech_enabled": True}
        else:
            pass
        return {}


class Qwen3OmniPipelineConfig(Qwen3OmniBasePipelineConfig):
    """6-stage text-only pipeline."""

    model_path: str
    placement_policy: str | None = _PLACEMENT_POLICY
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=text_stages)


class Qwen3OmniSpeechPipelineConfig(Qwen3OmniBasePipelineConfig):
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
        default_factory=lambda: speech_stages(
            thinker_gpu=0,
            talker_gpu=1,
            process_by_stage=SPEECH_DEFAULT_PROCESSES,
            enable_partial_start=True,
        )
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, bool | int]:
        process_by_stage = {stage.name: stage.process for stage in self.stages}
        code2wav_shares_talker_process = (
            process_by_stage["code2wav"] == process_by_stage["talker_ar"]
        )
        if stage_name == "talker_ar":
            return {
                "speech_enabled": True,
                "feedback_enabled": True,
                "code2wav_in_process": code2wav_shares_talker_process,
                "max_running_requests": TALKER_MAX_RUNNING_REQUESTS,
            }
        else:
            pass
        if stage_name == "code2wav":
            talker = self.stage_named("talker_ar")
            talker_process = self.processes.get(talker.process, ProcessConfig())
            return {
                "enable_cuda_graph": current_platform.enable_code2wav_graph(),
                "talker_in_process": code2wav_shares_talker_process,
                "codec_state_slots": self.talker_max_running_requests()
                * talker_process.num_replicas,
            }
        else:
            pass
        return super().stage_factory_kwargs(stage_name)

    def talker_max_running_requests(self) -> int:
        """The talker's running-request cap as its factory resolves it: an engine override, then
        a factory override, then this pipeline's default."""
        talker = self.stage_named("talker_ar")
        engine_overrides = (
            talker.engine.overrides() if talker.engine is not None else {}
        )
        factory_overrides = talker.factory.model_extra or {}
        return int(
            engine_overrides.get(
                "max_running_requests",
                factory_overrides.get(
                    "max_running_requests", TALKER_MAX_RUNNING_REQUESTS
                ),
            )
        )


class Qwen3OmniSpeechColocatedPipelineConfig(Qwen3OmniSpeechPipelineConfig):
    """7-stage speech pipeline for single-GPU stage colocation.

    The topology places image_encoder, audio_encoder, thinker, talker_ar, and
    code2wav on the same GPU, with code2wav inside the talker's process, while
    keeping preprocessing and decode as CPU stages. Per-stage memory budgets
    are supplied by the selected config file so deployments can use
    hardware-appropriate stage fractions and SGLang AR cache fractions.
    """

    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_COLOCATED_STAGE_ENV_DEFAULTS)
    )

    stages: list[StageConfig] = Field(
        default_factory=lambda: speech_stages(
            thinker_gpu=0,
            talker_gpu=0,
            process_by_stage=COLOCATED_SPEECH_PROCESSES,
            enable_partial_start=False,
        )
    )


def qwen_session_stages() -> list[StageConfig]:
    native = f"{_PKG}.native_stages"
    return [
        StageConfig(
            name="dialog",
            process="dialog",
            terminal=True,
            factory_path=f"{native}.create_input_scheduler",
        ),
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            next="image_encoder",
            factory_path=f"{native}.create_preprocessing_scheduler",
            factory=FactoryArgs(max_seq_len=32768),
        ),
        StageConfig(
            name="image_encoder",
            process="image_encoder",
            gpu=0,
            next="audio_encoder",
            factory_path=f"{native}.create_encoder_scheduler",
            factory=FactoryArgs(stage_name="image_encoder"),
        ),
        StageConfig(
            name="audio_encoder",
            process="audio_encoder",
            gpu=0,
            next="thinker",
            factory_path=f"{native}.create_encoder_scheduler",
            factory=FactoryArgs(
                stage_name="audio_encoder", enable_layer_cuda_graph=True
            ),
        ),
        EngineStageConfig(
            name="thinker",
            process="thinker",
            gpu=0,
            next="talker_ar",
            stream_to=["talker_ar"],
            factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory=FactoryArgs(
                execution_mode="session",
                max_seq_len=32768,
                speech_enabled=True,
                enable_async_decode=False,
                encoder_mem_reserve=0.0,
            ),
        ),
        EngineStageConfig(
            name="talker_ar",
            process="talker",
            env=talker_stage_env(),
            gpu=0,
            next="code2wav",
            stream_to=["code2wav"],
            project_payload={
                "code2wav": f"{_PKG}.request_builders.project_talker_to_code2wav",
            },
            can_accept_stream_before_payload=True,
            factory_path=f"{_PKG}.stages.create_talker_ar_executor_from_config",
            # The same prompt, partial-start and codec window settings as the speech profile's talker stage.
            factory=FactoryArgs(
                execution_mode="session",
                max_seq_len=32768,
                enable_partial_start=True,
                partial_start_min_chunks=5,
                codec_coalesce_frames=10,
                codec_coalesce_early_frames=10,
                codec_coalesce_first_frames=0,
            ),
        ),
        StageConfig(
            name="code2wav",
            process="speech",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
            factory_path=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
        ),
    ]


class Qwen3OmniSessionPipelineConfig(AudioSessionPipelineConfig):
    """Incremental Qwen stage routes over the shared Session schedulers."""

    stage_groups: ClassVar[tuple[StageGroup, ...]] = (
        *AudioSessionPipelineConfig.stage_groups,
        StageGroup(qwen_session_stages, None, SESSION_PORTS),
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        **AudioSessionPipelineConfig.stage_config_types,
        "thinker": EngineStageConfig,
        "talker_ar": EngineStageConfig,
    }
    max_sessions: int = Field(ge=1)
    # The audio the dialog holds for one user turn before speech.stopped or a commit ends it; a client that never ends a turn is cut here.
    max_utterance_seconds: float = Field(default=60, gt=0)
    # Finished utterances waiting behind the turn being answered; a client that keeps committing without listening is cut here.
    max_queued_turns: int = Field(default=8, ge=1)
    # The newest camera frames of an utterance that go to the model with it, as many as the executor route sent.
    max_utterance_frames: int = Field(default=4, ge=0)
    placement_policy: str | None = None
    env_defaults: dict[str, str] = Field(
        default_factory=lambda: dict(_COLOCATED_STAGE_ENV_DEFAULTS)
    )

    @model_validator(mode="before")
    @classmethod
    def select_entry_stage(cls, configuration: object) -> object:
        if isinstance(configuration, dict):
            return {"entry_stage": "dialog", **configuration}
        else:
            return configuration

    def stage_factory_kwargs(
        self, stage_name: str
    ) -> dict[str, float | int | dict[str, int]]:
        if stage_name in {"thinker", "talker_ar"}:
            kwargs = {
                "server_args_overrides": {
                    "max_running_requests": self.max_sessions
                    + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV,
                }
            }
            if stage_name == "talker_ar":
                kwargs["max_running_requests"] = (
                    self.max_sessions + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
                )
            else:
                pass
            return kwargs
        elif stage_name == "code2wav":
            return {}
        elif stage_name == "dialog":
            return {
                "max_open_sessions": self.max_sessions,
                "max_utterance_seconds": self.max_utterance_seconds,
                "max_queued_turns": self.max_queued_turns,
                "max_utterance_frames": self.max_utterance_frames,
            }
        else:
            return {"max_open_sessions": self.max_sessions}

    def session_ports(self) -> SessionPorts:
        ports = super().session_ports()
        blind_readers = sorted(
            name
            for name, node in self.session_graph().nodes.items()
            if node.edges().get(GRAPH_INPUT, frozenset()) is None
            and (
                self.node_ports(name, self.session_nodes[name]) is None
                or IMAGE
                not in self.node_ports(name, self.session_nodes[name]).input_modalities
            )
        )
        if IMAGE in ports.input_modalities and blind_readers:
            raise ValueError(
                f"nodes {blind_readers} read every format of {GRAPH_INPUT!r}, including camera frames they cannot consume"
            )
        else:
            return ports


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "session": Qwen3OmniSessionPipelineConfig,
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
    "speech-colocated": Qwen3OmniSpeechColocatedPipelineConfig,
}
