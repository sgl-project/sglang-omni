# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for pipeline wiring."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutorConfig(BaseModel):
    """Executor factory configuration."""

    model_config = ConfigDict(extra="forbid")

    factory: str
    args: dict[str, Any] = Field(default_factory=dict)


class InputHandlerConfig(BaseModel):
    """Stage input handler configuration."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["direct", "aggregated"] = "direct"
    sources: list[str] | None = None
    merge_fn: str | None = None


class RelayConfig(BaseModel):
    """Relay configuration for stage data transfer."""

    model_config = ConfigDict(extra="forbid")

    slot_size_mb: int = 512
    credits: int = 2
    rank: int | None = None
    world_size: int | None = None
    device: str = "cpu"


class StreamTargetConfig(BaseModel):
    """Streaming target for inter-stage streaming data transfer."""

    model_config = ConfigDict(extra="forbid")
    to_stage: str
    bootstrap: bool = True


class StageConfig(BaseModel):
    """Single pipeline stage configuration."""

    model_config = ConfigDict(extra="forbid")
    name: str
    executor: ExecutorConfig
    get_next: str
    input_handler: InputHandlerConfig = Field(default_factory=InputHandlerConfig)
    relay: RelayConfig = Field(default_factory=RelayConfig)
    num_workers: int = 1
    stream_to: list[StreamTargetConfig] = Field(default_factory=list)


class EndpointsConfig(BaseModel):
    """Endpoint allocation settings."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["ipc", "tcp"] = "ipc"
    base_path: str = "/tmp/sglang_omni"
    base_port: int = 16000


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    entry_stage: str
    stages: list[StageConfig]
    name: str = "model"  # default for all
    terminal_stages: list[str] = Field(default_factory=list)
    relay_backend: Literal["shm", "nccl", "nixl", "mooncake"] = "shm"
    fused_stages: list[list[str]] = Field(default_factory=list)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    gpu_placement: dict[str, int] = Field(default_factory=dict)
    completion_endpoint: str | None = None
    abort_endpoint: str | None = None
    config_cls: str | None = None

    def model_post_init(self, __context: Any = None) -> None:
        self._validate_general()
        self._validate_fusion()

        # we set this attribute to enable saving to and loading from the same pipeline class
        self.config_cls = self.__class__.__name__

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        """
        Create a PipelineConfig from a dictionary.
        """
        return PipelineConfig(**data)

    def _validate_general(self) -> None:
        if not self.model_path:
            raise ValueError("Model path is required")

        stage_names = [stage_cfg.name for stage_cfg in self.stages]
        if not stage_names:
            raise ValueError("Pipeline must define at least one stage")

        if len(stage_names) != len(set(stage_names)):
            raise ValueError("Stage names must be unique")

        if self.entry_stage not in stage_names:
            raise ValueError(f"entry_stage {self.entry_stage!r} is not defined")

        for stage_cfg in self.stages:
            if stage_cfg.num_workers < 1:
                raise ValueError(
                    f"Stage {stage_cfg.name!r} must have at least one worker"
                )
            if not stage_cfg.executor.factory:
                raise ValueError(f"Stage {stage_cfg.name!r} missing executor factory")
            if not stage_cfg.get_next:
                raise ValueError(f"Stage {stage_cfg.name!r} missing get_next")
            if stage_cfg.input_handler.type == "aggregated":
                sources = stage_cfg.input_handler.sources or []
                unknown = set(sources) - set(stage_names)
                if unknown:
                    raise ValueError(
                        f"Stage {stage_cfg.name!r} has unknown sources: {sorted(unknown)}"
                    )

        # Validate stream_to targets
        for stage_cfg in self.stages:
            for st in stage_cfg.stream_to:
                if st.to_stage not in stage_names:
                    raise ValueError(
                        f"Stage {stage_cfg.name!r} stream_to references "
                        f"unknown stage {st.to_stage!r}"
                    )

    def _validate_fusion(self) -> None:
        """
        Conduct sanity checks for stage fusion configuration. We want to ensure that:
        - Each fused group must have at least 2 stage names
        - A stage can only appear at most in one fused group
        - The stages in a fused group must be adjacent and in order
        """
        stage_names = [stage_cfg.name for stage_cfg in self.stages]

        fused = self.fused_stages or []
        if not fused:
            return
        index_map = {name: idx for idx, name in enumerate(stage_names)}
        seen: set[str] = set()
        for group in fused:
            if not group or len(group) < 2:
                raise ValueError("fused_stages groups must have at least 2 stage names")
            for name in group:
                if name not in index_map:
                    raise ValueError(f"fused stage {name!r} is not defined")
                if name in seen:
                    raise ValueError(f"stage {name!r} appears in multiple fused groups")
                seen.add(name)

            indices = [index_map[name] for name in group]
            if indices != sorted(indices):
                raise ValueError(f"fused group is out of order: {group}")
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError(
                    f"stages in the fused group are not adjacent: {group}, please ensure that the stages are adjacent and in order"
                )

    def apply_fusion(self) -> tuple[list[StageConfig], dict[str, str], str]:
        stage_by_name = {stage.name: stage for stage in self.stages}
        fused_groups = self.fused_stages or []

        name_map = {name: name for name in stage_by_name}
        group_by_last: dict[str, list[str]] = {}

        for group in fused_groups:
            last = group[-1]
            group_by_last[last] = group
            for name in group:
                name_map[name] = last

        stages_out: list[StageConfig] = []
        for stage in self.stages:
            mapped = name_map.get(stage.name, stage.name)
            if mapped != stage.name:
                continue  # fused into another stage
            if stage.name in group_by_last:
                group = group_by_last[stage.name]
                first = stage_by_name[group[0]]
                executors = [
                    {
                        "factory": stage_by_name[name].executor.factory,
                        "args": stage_by_name[name].executor.args,
                    }
                    for name in group
                ]
                fused_stage = StageConfig(
                    name=stage.name,
                    executor=ExecutorConfig(
                        factory="sglang_omni.executors.fused_executor.create_fused_executor",
                        args={"executors": executors},
                    ),
                    get_next=stage.get_next,
                    input_handler=first.input_handler,
                    relay=first.relay,
                    num_workers=first.num_workers,
                    stream_to=first.stream_to,
                )
                stages_out.append(fused_stage)
            else:
                stages_out.append(stage)

        entry_stage = name_map[self.entry_stage]
        return stages_out, name_map, entry_stage
