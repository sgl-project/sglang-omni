# SPDX-License-Identifier: Apache-2.0
"""Types for same-GPU CUDA IPC weight sharing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SCHEMA_VERSION = 1

WeightIpcRole = Literal["off", "leader", "follower"]


@dataclass(frozen=True)
class IpcTensorMeta:
    """Metadata for one shared CUDA tensor (from ``UntypedStorage._share_cuda_``)."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    nbytes: int
    device_index: int
    handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    allocation_offset_bytes: int
    tensor_offset: int
    requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool


@dataclass
class WeightIpcBundle:
    schema_version: int
    model_path: str
    model_revision: str | None
    cuda_driver: str | None
    created_unix_s: float
    leader_pid: int
    tensors: list[IpcTensorMeta] = field(default_factory=list)
    name_digest: str = ""


@dataclass(frozen=True)
class WeightIpcConfig:
    role: WeightIpcRole = "off"
    store_dir: str | None = None
    model_path: str = ""
    model_revision: str | None = None
    timeout_s: float = 120.0
