# SPDX-License-Identifier: Apache-2.0
"""Lazy references for large tensors forwarded through pipeline stages."""
from __future__ import annotations

from typing import Any

import msgspec

from sglang_omni.comm.data_ref import DataKind, DataRef

_TYPE = "TensorRef"
_VERSION = 1


class TensorRef(msgspec.Struct, frozen=True):
    request_id: str
    producer_stage: str
    consumer_stage: str
    path: str
    nbytes: int
    data_ref: DataRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "_type": _TYPE,
            "version": _VERSION,
            "request_id": self.request_id,
            "producer_stage": self.producer_stage,
            "consumer_stage": self.consumer_stage,
            "path": self.path,
            "nbytes": self.nbytes,
            "data_ref": self.data_ref.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TensorRef":
        if _required(value, "_type", str) != _TYPE:
            raise ValueError("value is not a tensor_ref")
        version = _required(value, "version", int)
        if version != _VERSION:
            raise ValueError(f"unsupported TensorRef version {version}")
        data_ref = DataRef.from_dict(_required(value, "data_ref", dict))
        if data_ref.kind is not DataKind.TENSOR_REF:
            raise ValueError(
                f"tensor_ref data_ref kind must be tensor_ref, got "
                f"{data_ref.kind.value}"
            )
        return cls(
            request_id=_required(value, "request_id", str),
            producer_stage=_required(value, "producer_stage", str),
            consumer_stage=_required(value, "consumer_stage", str),
            path=_required(value, "path", str),
            nbytes=_required(value, "nbytes", int),
            data_ref=data_ref,
        )


def is_tensor_ref(value: Any) -> bool:
    return isinstance(value, dict) and value.get("_type") == _TYPE


def collect_tensor_refs(value: Any, seen: set[int] | None = None) -> list[TensorRef]:
    if value is None:
        return []
    seen = set() if seen is None else seen
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)
    if is_tensor_ref(value):
        return [TensorRef.from_dict(value)]
    if isinstance(value, dict):
        return [
            ref for item in value.values() for ref in collect_tensor_refs(item, seen)
        ]
    if isinstance(value, (list, tuple)):
        return [ref for item in value for ref in collect_tensor_refs(item, seen)]
    return []


def _required(value: dict[str, Any], key: str, expected: type) -> Any:
    item = value[key]
    if type(item) is not expected:
        raise TypeError(f"tensor_ref {key} must be {expected.__name__}")
    return item
