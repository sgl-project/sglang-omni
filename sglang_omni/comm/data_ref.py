# SPDX-License-Identifier: Apache-2.0
"""Typed references to data-plane buffers carried by control messages."""
from __future__ import annotations

from enum import Enum
from typing import Any

import msgspec


class TransportKind(str, Enum):
    LOCAL_OBJECT = "local_object"
    CUDA_IPC = "cuda_ipc"
    SHM = "shm"
    MOONCAKE = "mooncake"


class DataKind(str, Enum):
    STAGE_PAYLOAD = "stage_payload"
    STREAM_CHUNK = "stream_chunk"
    STREAM_METADATA_TENSOR = "stream_metadata_tensor"
    KV_PAGES = "kv_pages"
    WEIGHT_BUCKET = "weight_bucket"
    MOE_EXPERT_PAYLOAD = "moe_expert_payload"


class DataLayout(str, Enum):
    PACKED_TENSORS = "packed_tensors"
    RAW_TENSOR = "raw_tensor"
    PAGED = "paged"
    BUCKETED = "bucketed"
    SCATTER = "scatter"


class TensorMeta(msgspec.Struct, frozen=True):
    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    offset: int
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "offset": self.offset,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TensorMeta":
        return cls(
            path=_required(value, "path", str),
            shape=_int_tuple(value, "shape"),
            dtype=_required(value, "dtype", str),
            device=_required(value, "device", str),
            offset=_required(value, "offset", int),
            size=_required(value, "size", int),
        )


class BackendRef(msgspec.Struct, frozen=True):
    transport: TransportKind
    info: dict[str, Any]
    length: int

    @classmethod
    def from_relay_info(
        cls, *, transport: TransportKind, relay_info: dict[str, Any]
    ) -> "BackendRef":
        transfer_info = _required(relay_info, "transfer_info", dict)
        return cls(
            transport=transport,
            info=relay_info,
            length=_required(transfer_info, "size", int),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport.value,
            "info": self.info,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BackendRef":
        return cls(
            transport=TransportKind(_required(value, "transport", str)),
            info=_required(value, "info", dict),
            length=_required(value, "length", int),
        )


class MetadataTensorRef(msgspec.Struct, frozen=True):
    path: str
    ref: "DataRef"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "ref": self.ref.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MetadataTensorRef":
        return cls(
            path=_required(value, "path", str),
            ref=DataRef.from_dict(_required(value, "ref", dict)),
        )


class DataRef(msgspec.Struct, frozen=True):
    """Control-plane pointer to one data-plane object."""

    version: int
    kind: DataKind
    object_id: str
    transport: TransportKind
    layout: DataLayout
    buffer: BackendRef
    header: str | None = None
    tensors: tuple[TensorMeta, ...] = ()
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    offset: int | None = None
    metadata: dict[str, Any] | None = None
    metadata_tensors: tuple[MetadataTensorRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "_type": "DataRef",
            "version": self.version,
            "kind": self.kind.value,
            "object_id": self.object_id,
            "transport": self.transport.value,
            "layout": self.layout.value,
            "buffer": self.buffer.to_dict(),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "metadata_tensors": [
                tensor_ref.to_dict() for tensor_ref in self.metadata_tensors
            ],
        }
        if self.header is not None:
            value["header"] = self.header
        if self.shape is not None:
            value["shape"] = list(self.shape)
        if self.dtype is not None:
            value["dtype"] = self.dtype
        if self.offset is not None:
            value["offset"] = self.offset
        if self.metadata is not None:
            value["metadata"] = self.metadata
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataRef":
        if _required(value, "_type", str) != "DataRef":
            raise ValueError("data_ref must have _type='DataRef'")
        version = _required(value, "version", int)
        if version != 1:
            raise ValueError(f"unsupported DataRef version {version}")
        return cls(
            version=version,
            kind=DataKind(_required(value, "kind", str)),
            object_id=_required(value, "object_id", str),
            transport=TransportKind(_required(value, "transport", str)),
            layout=DataLayout(_required(value, "layout", str)),
            buffer=BackendRef.from_dict(_required(value, "buffer", dict)),
            header=_optional(value, "header", str),
            tensors=tuple(
                TensorMeta.from_dict(item) for item in _required(value, "tensors", list)
            ),
            shape=_int_tuple(value, "shape") if "shape" in value else None,
            dtype=_optional(value, "dtype", str),
            offset=_required(value, "offset", int) if "offset" in value else None,
            metadata=(
                _required(value, "metadata", dict) if "metadata" in value else None
            ),
            metadata_tensors=tuple(
                MetadataTensorRef.from_dict(item)
                for item in _required(value, "metadata_tensors", list)
            ),
        )


def _required(value: dict[str, Any], key: str, expected: type) -> Any:
    item = value[key]
    if type(item) is not expected:
        raise TypeError(f"{key} must be {expected.__name__}, got {type(item).__name__}")
    return item


def _optional(value: dict[str, Any], key: str, expected: type) -> Any | None:
    item = value.get(key)
    if item is None:
        return None
    if type(item) is not expected:
        raise TypeError(
            f"{key} must be {expected.__name__} or None, " f"got {type(item).__name__}"
        )
    return item


def _int_tuple(value: dict[str, Any], key: str) -> tuple[int, ...]:
    items = _required(value, key, list)
    if not all(type(item) is int for item in items):
        raise TypeError(f"{key} must be list[int]")
    return tuple(items)
