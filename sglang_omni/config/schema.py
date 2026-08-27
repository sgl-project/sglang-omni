# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for pipeline wiring."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


logger = logging.getLogger(__name__)


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


class EngineArgs(BaseModel):
    """SGLang ServerArgs overrides for one engine stage.

    The commonly tuned keys are declared so they validate and show up in path
    enumeration; every other ServerArgs key passes through as a free-form
    entry. The block only exists on engine stages -- stage types that drive an
    SGLang engine declare ``engine_stage = True`` on their ``StageConfig``
    subclass, and writing ``engine.*`` on any other stage is a path error.
    """

    model_config = ConfigDict(extra="allow")

    mem_fraction_static: float | None = Field(default=None, gt=0, lt=1)
    max_running_requests: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    cuda_graph_max_bs: int | None = Field(default=None, ge=1)
    disable_cuda_graph: bool | None = None
    enable_torch_compile: bool | None = None
    torch_compile_max_bs: int | None = Field(default=None, ge=1)
    cpu_offload_gb: int | None = Field(default=None, ge=0)
    quantization: str | None = Field(default=None, min_length=1)
    disable_custom_all_reduce: bool | None = None

    def model_post_init(self, __context: Any = None) -> None:
        if self.quantization is not None and not self.quantization.strip():
            raise ValueError("engine.quantization must not be empty")

    def overrides(self) -> dict[str, Any]:
        """Return the keys set on this block, declared and free-form alike.

        A declared key left at ``None`` means "not set" and is omitted, so
        SGLang's own defaults stay in charge; free-form keys are always
        included because writing one is itself the intent.
        """
        extra = self.model_extra or {}
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None or key in extra
        }


class FactoryArgs(BaseModel):
    """Constructor kwargs for one stage's factory.

    Every declared field defaults to ``None``, meaning "use the factory's
    default"; a set field is passed to the stage factory under its own name.
    The declared fields are the commonly tuned ones and validate eagerly; any
    other key passes through untouched -- whether the factory accepts it is
    the factory's own call, made where the kwargs are applied.
    """

    model_config = ConfigDict(extra="allow")

    # --- model construction ---
    device: str | None = None
    dtype: str | None = None
    max_seq_len: int | None = Field(default=None, gt=0)
    video_fps: float | None = Field(default=None, gt=0)
    max_new_tokens: int | None = Field(default=None, gt=0)
    context_length: int | None = Field(default=None, gt=0)

    # --- executor / batching ---
    max_concurrency: int | None = Field(default=None, ge=1)
    max_batch_size: int | None = Field(default=None, ge=1)
    max_batch_wait_ms: float | None = Field(default=None, ge=0)
    enable_async_decode: bool | None = None
    async_decode_min_batch_size: int | None = Field(default=None, ge=1)
    prefill_coalesce_requests: int | None = Field(default=None, ge=0)
    prefill_coalesce_wait_ms: float | None = Field(default=None, gt=0)
    prefill_coalesce_when_idle: bool | None = None
    prefill_coalesce_requires_pending_builds: bool | None = None
    prefill_coalesce_after_builds_during_decode: bool | None = None
    request_build_max_workers: int | None = Field(default=None, ge=1)
    request_build_max_pending: int | None = Field(default=None, ge=1)
    encoder_mem_reserve: float | None = Field(default=None, ge=0, lt=1)
    enable_partial_start: bool | None = None
    partial_start_min_chunks: int | None = Field(default=None, ge=1)

    def model_post_init(self, __context: Any = None) -> None:
        if self.prefill_coalesce_requests == 1:
            logger.warning(
                "prefill_coalesce_requests=1 disables coalescing: the "
                "admission gate only engages at >= 2 (a batch of one has "
                "nothing to coalesce with). Use 0 to disable explicitly, "
                "or >= 2 to enable."
            )


class PlacementConfig(BaseModel):
    """Pipeline-level placement planning limits."""

    model_config = ConfigDict(extra="forbid")

    max_total_gpu_memory_fraction_per_gpu: float = Field(default=1.0, gt=0, le=1)
    require_memory_fraction_for_colocation: bool = True


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

    Stage settings are grouped by consumer: fields at the top level are read
    by the parent process (placement, process planning, wiring), ``engine.*``
    by SGLang ServerArgs, and ``factory.*`` by the stage factory's signature.

    Minimal example::

        StageConfig(name="decode", factory_path="...create_decode", terminal=True)

    Fan-in example::

        StageConfig(
            name="aggregate",
            factory_path="...create_aggregate",
            wait_for=["preprocessor", "image_enc", "audio_enc"],
            merge_fn="...merge_for_thinker",
            next="thinker",
        )
    """

    model_config = ConfigDict(extra="forbid")

    engine_stage: ClassVar[bool] = False
    """Whether this stage type drives an SGLang engine.

    Declared per stage *type* so path compilation can refuse ``engine.*``
    writes on stages that would silently ignore them.
    """

    # --- Identity ---
    name: str

    # --- Factory ---
    factory_path: str
    """Dotted import path of the stage factory."""

    # --- Routing (set `next` for static routing or `terminal`) ---
    next: str | list[str] | None = None
    terminal: bool = False
    route_fn: str | None = None

    # --- GPU / parallelism / process placement (parent-process consumers) ---
    gpu: int | list[int] | None = None
    tp_size: int = Field(default=1, ge=1)
    process: str | None = None
    gpu_memory_fraction: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Per-stage-rank budget as a fraction of total physical GPU memory. "
            "After TP expansion, each rank contributes this budget to its "
            "assigned GPU; stages sharing an OS process contribute jointly to "
            "that process's budget."
        ),
    )

    # --- Consumer groups ---
    engine: EngineArgs | None = None
    factory: FactoryArgs = Field(default_factory=FactoryArgs)

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
        if isinstance(self.gpu, int) and self.tp_size > 1:
            raise ValueError(
                f"Stage {self.name!r}: TP placement requires a list of "
                f"{self.tp_size} unique GPU ids, got scalar gpu={self.gpu}"
            )
        if isinstance(self.gpu, list):
            if len(self.gpu) != self.tp_size:
                raise ValueError(
                    f"Stage {self.name!r}: gpu has {len(self.gpu)} entries "
                    f"but tp_size={self.tp_size}"
                )
            if len(set(self.gpu)) != len(self.gpu):
                raise ValueError(
                    f"Stage {self.name!r}: TP placement requires unique GPU "
                    f"ids, got {list(self.gpu)}"
                )
        if self.process is not None:
            self.process = self.process.strip()
            if not self.process:
                raise ValueError(f"Stage {self.name!r} process must not be empty")
        if self.engine is not None and not type(self).engine_stage:
            raise ValueError(
                f"Stage {self.name!r} is not an engine stage; the engine block "
                "only exists on stages whose factory drives an SGLang engine"
            )

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


class EngineStageConfig(StageConfig):
    """Stage whose factory drives an SGLang engine, so ``engine.*`` exists."""

    engine_stage: ClassVar[bool] = True

    engine: EngineArgs | None = Field(default_factory=EngineArgs)


class AudioChunkingConfig(BaseModel):
    """Per-model long-audio policy for the transcription endpoint.

    Each ASR model declares the longest clip it can take in one request; anything
    longer gets split into non-overlapping chunks. Models may opt into ordered
    previous-text conditioning; otherwise chunks are transcribed independently.
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

    condition_on_previous_text: bool = False

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

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {}
    """Stage name -> ``StageConfig`` subclass for this pipeline's stage types.

    The mapping is what makes a stage's type survive a dump/rebuild round
    trip: the resolver mutates ``model_dump()`` output and reconstructs the
    config, and this is how each stage document gets validated against its
    own subclass (engine marker, model-specific ``model.*`` fields) instead
    of the base ``StageConfig``. Stage names absent from the mapping --
    including stages a user file adds -- validate as plain ``StageConfig``.
    """

    model_path: str
    stages: list[StageConfig]
    name: str | None = None
    entry_stage: str | None = None
    processes: dict[str, ProcessConfig] = Field(default_factory=dict)
    env_defaults: dict[str, str] = Field(default_factory=dict)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    placement_policy: str | None = None
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    terminal_stages_fn: str | None = None
    config_cls: str | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump with each stage serialized by its runtime class.

        Pydantic serializes a ``list[StageConfig]`` field by the declared
        item type, which would strip a per-stage subclass's fields (the
        engine block, model-specific ``model.*`` fields). The resolver's
        dump/mutate/rebuild round trip depends on those fields surviving,
        and subclasses redeclare ``stages`` for their defaults, so the fix
        lives here rather than in a per-field annotation.
        """
        data = super().model_dump(**kwargs)
        data["stages"] = [stage.model_dump(**kwargs) for stage in self.stages]
        return data

    @model_validator(mode="before")
    @classmethod
    def _materialize_stage_types(cls, data: Any) -> Any:
        """Validate stage documents against their declared per-stage types."""
        if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
            return data
        stages: list[Any] = []
        for stage in data["stages"]:
            stage_cls = (
                cls.stage_config_types.get(stage.get("name"))
                if isinstance(stage, dict)
                else None
            )
            stages.append(
                stage_cls.model_validate(stage) if stage_cls is not None else stage
            )
        return {**data, "stages": stages}

    def model_post_init(self, __context: Any = None) -> None:
        self._validate_general()
        self._validate_processes()
        self.config_cls = self.__class__.__name__
        if self.name is None:
            self.name = self.model_path

    @classmethod
    def stage_config_cls(cls, stage_name: str) -> type[StageConfig]:
        """The ``StageConfig`` subclass a named stage compiles paths against."""
        return cls.stage_config_types.get(stage_name, StageConfig)

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

    def stage_named(self, stage_name: str) -> StageConfig:
        """The stage with this name; raises ``KeyError`` when absent."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        raise KeyError(stage_name)

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        """Constructor kwargs the pipeline author passes to this stage's factory.

        This is a code-level hook, not a configuration surface: values
        returned here are wiring owned by the pipeline class, evaluated at
        launch time after every configuration source has been resolved.
        User-tunable knobs belong in the stage's typed ``factory``/``engine``
        groups, which override kwargs returned here whenever both name the
        same factory parameter.
        """
        return {}
        return {}

    def resolved_env_defaults(self) -> dict[str, str]:
        """Process-environment defaults, evaluated at launch.

        A code-level hook like :meth:`stage_factory_kwargs`: models override
        it to derive environment values from the resolved configuration
        (thread pools sized from worker settings, for instance). Deriving
        here rather than in validation keeps the derivation out of the
        config's dump, so a rebuild cannot mistake yesterday's derivation
        for a written value. An entry written in ``env_defaults`` always
        wins over a derivation.
        """
        return dict(self.env_defaults)

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
            if not s.factory_path:
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
