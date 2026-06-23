# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for pipeline wiring."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class RelayConfig(BaseModel):
    """Relay configuration for stage data transfer."""

    model_config = ConfigDict(extra="forbid")

    slot_size_mb: int = 512
    credits: int = 2
    rank: int | None = None
    world_size: int | None = None
    device: str = "cpu"


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
    """Placement-resource intent for one stage rank/process."""

    model_config = ConfigDict(extra="forbid")

    total_gpu_memory_fraction: float | None = Field(
        default=None,
        description=(
            "Per-rank/process budget as a fraction of total physical GPU "
            "memory. After TP expansion, each rank contributes this budget to "
            "its assigned GPU."
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

    # --- Fan-in ---
    wait_for: list[str] | None = None
    wait_for_fn: str | None = None
    merge_fn: str | None = None

    # --- Streaming ---
    stream_to: list[str] = Field(default_factory=list)
    stream_done_to_fn: str | None = None
    can_accept_stream_before_payload: bool = False

    # --- Route-specific payload projection ---
    project_payload: dict[str, str] = Field(default_factory=dict)

    # --- Relay (auto-inferred from gpu when None) ---
    relay: RelayConfig | None = None

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


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    architecture_aliases: ClassVar[tuple[str, ...]] = ()
    tensor_parallel_disable_custom_all_reduce_stages: ClassVar[tuple[str, ...]] = ()

    model_path: str
    stages: list[StageConfig]
    name: str | None = None
    entry_stage: str | None = None
    relay_backend: Literal["shm", "nccl", "nixl", "mooncake"] = "shm"
    fused_stages: list[list[str]] = Field(default_factory=list)
    runtime_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_defaults: dict[str, str] = Field(default_factory=dict)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    placement_policy: str | None = None
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    terminal_stages_fn: str | None = None
    config_cls: str | None = None

    def model_post_init(self, __context: Any = None) -> None:
        self._validate_general()
        self._validate_fusion()
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

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        """Return whether non-default TTS voice names must be uploaded voices."""
        return False

    def supports_uploaded_voice_references(self) -> bool:
        """Return whether uploaded voices can be lowered as reference audio."""
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
            if s.tp_size < 1:
                raise ValueError(f"Stage {s.name!r} must have tp_size >= 1")
            if s.parallelism.tp != s.tp_size:
                raise ValueError(
                    f"Stage {s.name!r}: tp_size={s.tp_size} conflicts with "
                    f"parallelism.tp={s.parallelism.tp}"
                )
            if isinstance(s.gpu, list) and len(s.gpu) != s.tp_size:
                raise ValueError(
                    f"Stage {s.name!r}: gpu has {len(s.gpu)} entries "
                    f"but tp_size={s.tp_size}"
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

    def _validate_fusion(self) -> None:
        names = [s.name for s in self.stages]
        fused = self.fused_stages or []
        if not fused:
            return
        index_map = {n: i for i, n in enumerate(names)}
        stage_by_name = {s.name: s for s in self.stages}
        seen: set[str] = set()
        for group in fused:
            if not group or len(group) < 2:
                raise ValueError("fused_stages groups must have at least 2 stage names")
            for n in group:
                if n not in index_map:
                    raise ValueError(f"fused stage {n!r} is not defined")
                if n in seen:
                    raise ValueError(f"stage {n!r} appears in multiple fused groups")
                seen.add(n)
            indices = [index_map[n] for n in group]
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError(f"fused group not adjacent/ordered: {group}")
            self._validate_fused_group_contract(group, stage_by_name)

    def _validate_fused_group_contract(
        self,
        group: list[str],
        stage_by_name: dict[str, StageConfig],
    ) -> None:
        stages = [stage_by_name[name] for name in group]

        for stage in stages:
            if stage.tp_size != 1:
                raise ValueError(
                    f"fused group {group} cannot include TP stage {stage.name!r}"
                )

        gpu_ids = {
            gpu_id for stage in stages for gpu_id in _stage_gpu_ids_for_fusion(stage)
        }
        if len(gpu_ids) > 1:
            raise ValueError(
                f"fused group {group} must fit on one GPU; got {sorted(gpu_ids)}"
            )

        for index, stage in enumerate(stages):
            if index > 0 and stage.wait_for:
                raise ValueError(
                    f"fused group {group} cannot include internal fan-in stage "
                    f"{stage.name!r}"
                )
            if index < len(stages) - 1:
                expected_next = group[index + 1]
                if stage.terminal or stage.route_fn is not None:
                    raise ValueError(
                        f"fused group {group} must be linear; internal stage "
                        f"{stage.name!r} cannot be terminal or dynamic-routed"
                    )
                if _target_list(stage.next) != [expected_next]:
                    raise ValueError(
                        f"fused group {group} must be linear; stage "
                        f"{stage.name!r} must route only to {expected_next!r}"
                    )

    def apply_fusion(self) -> tuple[list[StageConfig], dict[str, str], str]:
        name_map = {s.name: s.name for s in self.stages}
        return list(self.stages), name_map, self.resolved_entry_stage

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        return PipelineConfig(**data)


def _target_list(targets: str | list[str] | None) -> list[str]:
    if targets is None:
        return []
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _stage_gpu_ids_for_fusion(stage: StageConfig) -> tuple[int, ...]:
    gpu = stage.gpu
    if gpu is None:
        return ()
    if isinstance(gpu, int):
        return (gpu,)
    return tuple(int(gpu_id) for gpu_id in gpu)
