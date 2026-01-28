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

    type: Literal["shm", "nccl", "nixl"] = "nixl"
    slot_size_mb: int = 64
    credits: int = 2
    rank: int | None = None
    world_size: int | None = None
    device: str = "cpu"


class StageConfig(BaseModel):
    """Single pipeline stage configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    executor: ExecutorConfig
    get_next: str
    input_handler: InputHandlerConfig = Field(default_factory=InputHandlerConfig)
    relay: RelayConfig = Field(default_factory=RelayConfig)
    num_workers: int = 1


class EndpointsConfig(BaseModel):
    """Endpoint allocation settings."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["ipc", "tcp"] = "ipc"
    base_path: str = "/tmp/sglang_omni"
    base_port: int = 16000


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entry_stage: str
    stages: list[StageConfig]
    fused_stages: list[list[str]] = Field(default_factory=list)
    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    completion_endpoint: str | None = None
    abort_endpoint: str | None = None
