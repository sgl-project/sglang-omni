# SPDX-License-Identifier: Apache-2.0
"""Typed YAML configuration for managed local Omni router launches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from sglang_omni_router.config import Capability


class LocalLauncherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["local"] = "local"
    model_path: str
    model_name: str | None = None
    num_workers: int = 1
    num_gpus_per_worker: int = 1
    worker_host: str = "127.0.0.1"
    worker_base_port: int = 8011
    worker_gpu_ids: list[str] | None = None
    worker_capabilities: set[Capability] | None = None
    worker_extra_args: str = ""
    wait_timeout: int = 600

    @field_validator("model_path", "model_name", "worker_host", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("worker_extra_args", mode="before")
    @classmethod
    def _normalize_extra_args(cls, value: object) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("worker_extra_args must be a string")
        return value.strip()

    @field_validator(
        "num_workers",
        "num_gpus_per_worker",
        "wait_timeout",
    )
    @classmethod
    def _validate_positive_int(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("value must be > 0")
        return value

    @field_validator("worker_base_port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if value <= 0 or value > 65535:
            raise ValueError("worker_base_port must be in [1, 65535]")
        return value

    @field_validator("worker_gpu_ids")
    @classmethod
    def _validate_worker_gpu_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("worker_gpu_ids entries must be strings")
            item = item.strip()
            if not item:
                raise ValueError("worker_gpu_ids entries must not be empty")
            normalized.append(item)
        return normalized

    @field_validator("worker_capabilities")
    @classmethod
    def _validate_worker_capabilities(
        cls, value: set[Capability] | None
    ) -> set[Capability] | None:
        if value is not None and not value:
            raise ValueError("worker_capabilities must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_launch_shape(self) -> "LocalLauncherConfig":
        if self.worker_base_port + self.num_workers - 1 > 65535:
            raise ValueError("worker port range exceeds 65535")
        if (
            self.worker_gpu_ids is not None
            and len(self.worker_gpu_ids) != self.num_workers
        ):
            raise ValueError("worker_gpu_ids must contain exactly num_workers entries")
        return self


def load_launcher_config(path: str | Path) -> LocalLauncherConfig:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"failed to read launcher config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid launcher config YAML: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("launcher"), dict):
        raise ValueError("launcher config must contain a top-level launcher object")

    try:
        return LocalLauncherConfig(**payload["launcher"])
    except ValidationError as exc:
        raise ValueError(f"invalid launcher config: {exc}") from exc
