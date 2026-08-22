# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for pipeline wiring."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

REPLICA_SEPARATOR = "@r"


def replica_instance_name(logical_name: str, replica_id: int) -> str:
    return f"{logical_name}{REPLICA_SEPARATOR}{replica_id}"


def parse_replica_instance_name(name: str) -> tuple[str, int | None]:
    """Split ``stage@rN`` into ``(stage, N)``; plain names get ``None``."""
    logical, sep, suffix = name.rpartition(REPLICA_SEPARATOR)
    if not sep or not suffix.isdigit():
        return name, None
    return logical, int(suffix)


def stage_process_name(stage: "StageConfig") -> str:
    """Process Name that owns *stage*.

    Non-TP stages declare it explicitly. A TP stage owns its process outright,
    so it falls back to the stage name when no process is declared.
    """
    if stage.tp_size > 1:
        return stage.process or stage.name
    if not stage.process:
        raise ValueError(f"Stage {stage.name!r} must declare process")
    return stage.process


class CommConfig(BaseModel):
    """Per-stage communication buffer and Mooncake options.

    Transport selection is owned by ``CommRouter`` from stage locality and
    placement. This config only tunes buffer pools and backend-specific
    connection options for transports the router selects.
    """

    model_config = ConfigDict(extra="forbid")

    slot_size_mb: int = 512
    credits: int = 2
    cuda_ipc_slot_size_kb: int = 64
    cuda_ipc_pool_size_mb: int | None = None
    mooncake_protocol: str = "rdma"
    mooncake_hostname: str | None = None
    mooncake_device_name: str = ""


class EndpointsConfig(BaseModel):
    """Endpoint allocation settings."""

    model_config = ConfigDict(extra="forbid")

    base_path: str = "/tmp/sglang_omni"


class ParallelismConfig(BaseModel):
    """Supported parallelism for one logical stage."""

    model_config = ConfigDict(extra="forbid")

    tp: int = 1

    def model_post_init(self, __context: Any = None) -> None:
        if self.tp < 1:
            raise ValueError("parallelism.tp must be >= 1")


class StageResourceConfig(BaseModel):
    """Placement-resource intent for one logical stage rank."""

    model_config = ConfigDict(extra="forbid")

    total_gpu_memory_fraction: float | None = Field(
        default=None,
        description=(
            "Per-stage-rank budget as a fraction of total physical GPU memory. "
            "After TP expansion, each rank contributes this budget to its "
            "assigned GPU; stages sharing an OS process contribute jointly to "
            "that process's budget."
        ),
    )

    def model_post_init(self, __context: Any = None) -> None:
        value = self.total_gpu_memory_fraction
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(
                "runtime.resources.total_gpu_memory_fraction must be in (0, 1]"
            )


class SGLangServerArgsConfig(BaseModel):
    """Typed subset of SGLang ServerArgs exposed through pipeline config."""

    model_config = ConfigDict(extra="forbid")

    mem_fraction_static: float | None = None

    def model_post_init(self, __context: Any = None) -> None:
        value = self.mem_fraction_static
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError(
                "runtime.sglang_server_args.mem_fraction_static must be in (0, 1)"
            )


class StageRuntimeConfig(BaseModel):
    """Typed runtime intent for one stage.

    Backend-specific values stay namespaced. For example,
    sglang_server_args is translated into SGLang ServerArgs by the
    runtime adapter, not by placement planning.
    """

    model_config = ConfigDict(extra="forbid")

    resources: StageResourceConfig = Field(default_factory=StageResourceConfig)
    max_seq_len: int | None = None
    video_fps: float | None = None
    sglang_server_args: SGLangServerArgsConfig = Field(
        default_factory=SGLangServerArgsConfig
    )

    def model_post_init(self, __context: Any = None) -> None:
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError("runtime.max_seq_len must be positive")
        if self.video_fps is not None and self.video_fps <= 0:
            raise ValueError("runtime.video_fps must be positive")


class PlacementConfig(BaseModel):
    """Pipeline-level placement planning limits."""

    model_config = ConfigDict(extra="forbid")

    max_total_gpu_memory_fraction_per_gpu: float = 1.0
    require_memory_fraction_for_colocation: bool = True

    def model_post_init(self, __context: Any = None) -> None:
        value = self.max_total_gpu_memory_fraction_per_gpu
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "placement.max_total_gpu_memory_fraction_per_gpu must be in (0, 1]"
            )


# Note (kaige): validation follows the context each layer owns. StageConfig and
# ProcessConfig check object-local fields, PipelineConfig checks declarations
# and references, and logical-process compilation checks derived topology once
# before workers start. Runtime only consumes the compiled plans.
class ProcessConfig(BaseModel):
    """Replica policy for one logical process.

    Keyed by Process Name in ``PipelineConfig.processes``. Member stages come
    from ``StageConfig.process``, so this never repeats them.
    """

    model_config = ConfigDict(extra="forbid")

    num_replicas: int = 1
    replica_devices: list[int] | None = None

    @field_validator("replica_devices", mode="before")
    @classmethod
    def _parse_replica_devices(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            if any(not part for part in parts):
                raise ValueError("processes.replica_devices must contain GPU ids")
            try:
                return [int(part) for part in parts]
            except ValueError as exc:
                raise ValueError(
                    "processes.replica_devices must contain only integer GPU ids"
                ) from exc
        return value

    @field_validator("replica_devices")
    @classmethod
    def _validate_replica_devices(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("processes.replica_devices must not be empty")
        if any(device_id < 0 for device_id in value):
            raise ValueError("processes.replica_devices GPU ids must be >= 0")
        return value

    def model_post_init(self, __context: Any = None) -> None:
        if self.num_replicas < 1:
            raise ValueError("processes.num_replicas must be >= 1")


class StageConfig(BaseModel):
    """Single pipeline stage configuration.

    Minimal example::

        StageConfig(name="decode", factory="...create_decode", terminal=True)

    Fan-in example::

        StageConfig(
            name="aggregate",
            factory="...create_aggregate",
            wait_for=["preprocessor", "image_enc", "audio_enc"],
            merge_fn="...merge_for_thinker",
            next="thinker",
        )
    """

    model_config = ConfigDict(extra="forbid")

    # --- Identity ---
    name: str

    # --- Factory ---
    factory: str
    factory_args: dict[str, Any] = Field(default_factory=dict)

    # --- Routing (set `next` for static routing or `terminal`) ---
    next: str | list[str] | None = None
    terminal: bool = False
    route_fn: str | None = None

    # --- GPU / parallelism ---
    gpu: int | list[int] | None = None
    tp_size: int = 1
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    process: str | None = None

    # --- Runtime intent ---
    runtime: StageRuntimeConfig = Field(default_factory=StageRuntimeConfig)
    runtime_arg_map: dict[str, str] = Field(default_factory=dict)
    # Note (Yueying Li): per-stage env defaults applied in this stage's worker process at spawn
    # (merged over the pipeline-level env_defaults; never overrides os.environ).
    env: dict[str, str] = Field(default_factory=dict)

    # --- Fan-in ---
    wait_for: list[str] | None = None
    wait_for_fn: str | None = None
    merge_fn: str | None = None

    # --- Streaming ---
    stream_to: list[str] = Field(default_factory=list)
    stream_done_to_fn: str | None = None
    can_accept_stream_before_payload: bool = False

    # --- Payload transport ---
    disable_direct_cuda_ipc_payload: bool = False

    # --- Route-specific payload projection ---
    project_payload: dict[str, str] = Field(default_factory=dict)

    # --- Communication pool tuning ---
    comm: CommConfig | None = None

    def model_post_init(self, __context: Any = None) -> None:
        fields_set = self.__pydantic_fields_set__
        tp_size_set = "tp_size" in fields_set
        parallelism_set = "parallelism" in fields_set
        if self.tp_size < 1:
            raise ValueError(f"Stage {self.name!r} must have tp_size >= 1")
        if self.process is not None:
            self.process = self.process.strip()
            if not self.process:
                raise ValueError(f"Stage {self.name!r} process must not be empty")
        if parallelism_set and tp_size_set and self.parallelism.tp != self.tp_size:
            raise ValueError(
                f"Stage {self.name!r}: tp_size={self.tp_size} conflicts with "
                f"parallelism.tp={self.parallelism.tp}"
            )
        if not parallelism_set and self.tp_size != self.parallelism.tp:
            self.parallelism.tp = self.tp_size
        elif (
            parallelism_set and not tp_size_set and self.tp_size != self.parallelism.tp
        ):
            self.tp_size = self.parallelism.tp

        gpu = self.gpu
        if gpu is None:
            if self.tp_size > 1:
                raise ValueError(
                    f"Stage {self.name!r}: gpu is required when tp_size={self.tp_size}"
                )
            return

        gpu_ids = [gpu] if isinstance(gpu, int) else gpu
        if len(gpu_ids) != self.tp_size:
            raise ValueError(
                f"Stage {self.name!r}: gpu has {len(gpu_ids)} entries "
                f"but tp_size={self.tp_size}"
            )
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            raise ValueError(f"Stage {self.name!r}: GPU ids must be >= 0")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(f"Stage {self.name!r}: GPU ids must be unique")


class AudioChunkingConfig(BaseModel):
    """Per-model long-audio policy for the transcription endpoint.

    Each ASR model declares the longest clip it can take in one request; anything
    longer gets split into non-overlapping chunks that are transcribed
    independently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Some models can't correctly transcribe an isolated chunk (e.g. diarization needs to track speakers across the whole
    # recording), so we leave the default value of `allow_audio_chunking` = False.
    allow_audio_chunking: bool = False
    # Note (Jeffro): Longest clip (chunk length) sent to the engine in one request.
    # Must stay within what the model's context can hold (Qwen3-ASR sizes its
    # context for the official 1,200s native limit); below that ceiling it
    # is a scheduling trade-off: shorter chunks batch better and keep a
    # long upload from monopolizing the engine, at the cost of more seams.
    max_audio_clip_s: float = Field(default=60.0, gt=0)

    max_native_clip_s: float | None = Field(default=None, gt=0)

    max_total_audio_s: float | None = Field(default=3600.0, gt=0)

    # Shortest final chunk worth transcribing.
    min_tail_s: float = Field(default=0.5, ge=0)

    # Note (Jeffro): How many chunks of one HTTP request may run in the engine at once.
    # This is a fairness cap: to avoid a single long
    # upload grabs every batch slot and queues out everyone else's requests.
    # This is a pre-request cap.
    max_concurrent_chunks: int = Field(default=8, ge=1)

    def model_post_init(self, __context: Any = None) -> None:
        if (
            self.max_total_audio_s is not None
            and self.max_total_audio_s < self.max_audio_clip_s
        ):
            raise ValueError(
                f"max_total_audio_s={self.max_total_audio_s} must be at least "
                f"max_audio_clip_s={self.max_audio_clip_s}"
            )
        if (
            self.max_native_clip_s is not None
            and self.max_native_clip_s < self.max_audio_clip_s
        ):
            raise ValueError(
                f"max_native_clip_s={self.max_native_clip_s} must be at least "
                f"max_audio_clip_s={self.max_audio_clip_s}"
            )

    @property
    def stream_clip_limit_s(self) -> float:
        """Longest clip the un-chunkable streaming path accepts."""
        return (
            self.max_native_clip_s
            if self.max_native_clip_s is not None
            else self.max_audio_clip_s
        )

    def chunk_samples(self, sample_rate: int) -> int:
        """Chunk length in samples, at least one sample."""
        return max(int(self.max_audio_clip_s * sample_rate), 1)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    Subclasses set ``requires_model_capabilities`` when their model package
    must export static architecture-level capability metadata.
    """

    model_config = ConfigDict(extra="forbid")

    architecture: ClassVar[str | None] = None
    architecture_aliases: ClassVar[tuple[str, ...]] = ()
    requires_model_capabilities: ClassVar[bool] = False
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = ()
    required_speech_reference_count: ClassVar[int | None] = None
    speech_reference_text_required: ClassVar[bool] = False
    speech_reference_text_excludes_instructions: ClassVar[bool] = False
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset()
    audio_chunking: ClassVar[AudioChunkingConfig] = AudioChunkingConfig()

    model_path: str
    stages: list[StageConfig]
    name: str | None = None
    entry_stage: str | None = None
    processes: dict[str, ProcessConfig] = Field(default_factory=dict)
    runtime_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_defaults: dict[str, str] = Field(default_factory=dict)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    placement_policy: str | None = None
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    terminal_stages_fn: str | None = None
    config_cls: str | None = None

    def model_post_init(self, __context: Any = None) -> None:
        self._validate_general()
        self._validate_processes()
        self.config_cls = self.__class__.__name__
        if self.name is None:
            self.name = self.model_path

    @property
    def resolved_entry_stage(self) -> str:
        if self.entry_stage is not None:
            return self.entry_stage
        return self.stages[0].name

    @property
    def terminal_stages(self) -> list[str]:
        return [s.name for s in self.stages if s.terminal]

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        """Pipeline edges whose stages must stay in the same process.

        Keyed by edge rather than by stage because correctness depends on which
        handoff crosses a process boundary, not on which stage moved. Grouping
        ``preprocessing`` with ``audio_encoder`` leaves their shared handoff
        local and permits ``audio_encoder -> tts_engine`` to cross processes.

        Declare an edge when the downstream stage depends on process-local
        state that the payload does not carry. A model may also retain an edge
        temporarily to preserve an established support boundary; document that
        compatibility guard at the declaration.
        """
        return frozenset()

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for SGLang mem_fraction_static overrides."""
        return {}

    @classmethod
    def encoder_mem_reserve_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for encoder memory reserve overrides."""
        return {}

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for talker placement overrides."""
        return {}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for talker SGLang ServerArgs overrides."""
        return {}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        """Class-level public role map for generation SGLang ServerArgs overrides."""
        return {}

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, Any]:
        """Coordinator in-flight cap defaults (running + queued). Overlay with CLI."""
        return {}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        """Return the code2wav stage name when the pipeline supports it."""
        return None

    @classmethod
    def tensor_parallel_server_args_overrides(
        cls,
        *,
        stage_name: str,
        tp_size: int,
    ) -> dict[str, object]:
        """Return SGLang ServerArgs overrides implied by stage TP settings."""
        if (
            tp_size > 1
            and stage_name in cls.tensor_parallel_disable_custom_all_reduce_stages
        ):
            return {"disable_custom_all_reduce": True}
        return {}

    @classmethod
    def topology_gated_custom_all_reduce_stages(cls) -> set[str]:
        """Stages whose TP custom all-reduce disable is topology-relaxable."""
        return set()

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        """Return whether non-default TTS voice names must be uploaded voices."""
        return False

    def supports_uploaded_voice_references(self) -> bool:
        """Return whether uploaded voices can be lowered as reference audio."""
        return False

    def supports_audio_translation(self) -> bool:
        """Return whether this pipeline can serve /v1/audio/translations."""
        return False

    @property
    def gpu_placement(self) -> dict[str, int | list[int]]:
        out: dict[str, int | list[int]] = {}
        for s in self.stages:
            if s.gpu is not None:
                out[s.name] = s.gpu
        return out

    def _validate_general(self) -> None:
        if not self.model_path:
            raise ValueError("Model path is required")

        names = [s.name for s in self.stages]
        if not names:
            raise ValueError("Pipeline must define at least one stage")
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")
        entry = self.resolved_entry_stage
        if entry not in names:
            raise ValueError(f"entry_stage {entry!r} is not defined")

        for s in self.stages:
            if not s.factory:
                raise ValueError(f"Stage {s.name!r} missing factory")
            has_next = s.next is not None
            if has_next == bool(s.terminal):
                raise ValueError(
                    f"Stage {s.name!r} must set exactly one of 'next' or 'terminal'"
                )
            if s.terminal and s.route_fn is not None:
                raise ValueError(
                    f"Stage {s.name!r} cannot set route_fn on a terminal stage"
                )
            if s.stream_done_to_fn is not None and not s.stream_to:
                raise ValueError(
                    f"Stage {s.name!r} cannot set stream_done_to_fn without stream_to"
                )
            if s.wait_for:
                if not s.merge_fn:
                    raise ValueError(f"Stage {s.name!r} has wait_for but no merge_fn")
                unknown = set(s.wait_for) - set(names)
                if unknown:
                    raise ValueError(
                        f"Stage {s.name!r} wait_for has unknown stages: {sorted(unknown)}"
                    )
            elif s.wait_for_fn is not None:
                raise ValueError(f"Stage {s.name!r} has wait_for_fn but no wait_for")
            if s.next is not None:
                targets = [s.next] if isinstance(s.next, str) else s.next
                unknown = set(targets) - set(names)
                if unknown:
                    raise ValueError(
                        f"Stage {s.name!r} next has unknown stages: {sorted(unknown)}"
                    )
            for t in s.stream_to:
                if t not in names:
                    raise ValueError(
                        f"Stage {s.name!r} stream_to references unknown stage {t!r}"
                    )
            for t in s.project_payload:
                if t not in names:
                    raise ValueError(
                        f"Stage {s.name!r} project_payload references unknown stage {t!r}"
                    )

        for s in self.stages:
            if parse_replica_instance_name(s.name)[1] is not None:
                raise ValueError(
                    f"Stage name {s.name!r} uses the '@r<N>' suffix reserved "
                    "for replica instances"
                )

        for stage_name in self.runtime_overrides:
            if stage_name not in names:
                raise ValueError(
                    f"runtime_overrides references unknown stage {stage_name!r}"
                )

        missing_process = [
            s.name for s in self.stages if s.tp_size == 1 and not s.process
        ]
        if missing_process:
            raise ValueError(
                "Non-TP stages must declare process; "
                f"missing process for {missing_process}"
            )

    def _validate_processes(self) -> None:
        """Check Process Names and the sparse ``processes`` replica policy.

        Membership grouping, cross-process edges, and device counts belong to
        the logical-process compile step; only declaration-level facts that
        need nothing but this config are checked here.
        """
        members: dict[str, list[StageConfig]] = {}
        for stage in self.stages:
            members.setdefault(stage_process_name(stage), []).append(stage)

        for process_name, stages in members.items():
            if parse_replica_instance_name(process_name)[1] is not None:
                raise ValueError(
                    f"Process name {process_name!r} uses the '@r<N>' suffix "
                    "reserved for replica instances"
                )
            tp_stages = [stage.name for stage in stages if stage.tp_size > 1]
            if len(tp_stages) > 1:
                raise ValueError(
                    f"Process name {process_name!r} is claimed by multiple TP "
                    f"stages: {tp_stages}"
                )
            if tp_stages and len(stages) > 1:
                others = [s.name for s in stages if s.name not in tp_stages]
                raise ValueError(
                    f"Process {process_name!r} holds TP stage {tp_stages[0]!r} "
                    f"and cannot be shared with {others}"
                )

        unknown = sorted(set(self.processes) - set(members))
        if unknown:
            raise ValueError(
                f"processes references unknown process name(s): {unknown}. "
                f"Declared process names: {sorted(members)}"
            )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        return PipelineConfig(**data)
