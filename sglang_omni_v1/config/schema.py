# SPDX-License-Identifier: Apache-2.0
"""Configuration schema for pipeline wiring."""

from __future__ import annotations

from typing import Any, Literal

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

    scheme: Literal["ipc", "tcp"] = "ipc"
    base_path: str = "/tmp/sglang_omni_v1"
    base_port: int = 16000


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

    # --- GPU / parallelism ---
    gpu: int | list[int] | None = None
    tp_size: int = 1

    # --- Fan-in ---
    wait_for: list[str] | None = None
    merge_fn: str | None = None

    # --- Streaming ---
    stream_to: list[str] = Field(default_factory=list)

    # --- Route-specific payload projection ---
    project_payload: dict[str, str] = Field(default_factory=dict)

    # --- Relay (auto-inferred from gpu when None) ---
    relay: RelayConfig | None = None


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    stages: list[StageConfig]
    name: str = "model"
    entry_stage: str | None = None
    relay_backend: Literal["shm", "nccl", "nixl", "mooncake"] = "shm"
    fused_stages: list[list[str]] = Field(default_factory=list)
    runtime_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    completion_endpoint: str | None = None
    abort_endpoint: str | None = None
    config_cls: str | None = None

    def model_post_init(self, __context: Any = None) -> None:
        self._validate_general()
        self._validate_fusion()
        self.config_cls = self.__class__.__name__

    @property
    def resolved_entry_stage(self) -> str:
        if self.entry_stage is not None:
            return self.entry_stage
        return self.stages[0].name

    @property
    def terminal_stages(self) -> list[str]:
        return [s.name for s in self.stages if s.terminal]

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
            if s.next is None and not s.terminal:
                raise ValueError(f"Stage {s.name!r} must set 'next' or 'terminal'")
            if s.tp_size < 1:
                raise ValueError(f"Stage {s.name!r} must have tp_size >= 1")
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

    def _validate_fusion(self) -> None:
        names = [s.name for s in self.stages]
        fused = self.fused_stages or []
        if not fused:
            return
        index_map = {n: i for i, n in enumerate(names)}
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

    def apply_fusion(self) -> tuple[list[StageConfig], dict[str, str], str]:
        if self.fused_stages:
            raise NotImplementedError("fused_stages not yet supported")
        name_map = {s.name: s.name for s in self.stages}
        return list(self.stages), name_map, self.resolved_entry_stage

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PipelineConfig:
        return PipelineConfig(**data)
