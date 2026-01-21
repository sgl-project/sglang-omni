# SPDX-License-Identifier: Apache-2.0
"""Shared pipeline work types for metadata-only scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sglang_omni.proto import StagePayload

RelayMetadata = dict[str, Any]
MergeFn = Callable[[dict[str, StagePayload]], StagePayload]


@dataclass(frozen=True)
class InputRef:
    """Reference to an input payload, either inline or via relay metadata."""

    source: str
    payload: StagePayload | None = None
    metadata: RelayMetadata | None = None

    def __post_init__(self) -> None:
        if (self.payload is None) == (self.metadata is None):
            raise ValueError("InputRef must set exactly one of payload or metadata")

    @classmethod
    def from_payload(cls, source: str, payload: StagePayload) -> "InputRef":
        return cls(source=source, payload=payload)

    @classmethod
    def from_metadata(cls, source: str, metadata: RelayMetadata) -> "InputRef":
        return cls(source=source, metadata=metadata)


@dataclass
class WorkDescriptor:
    """Unit of work for a worker: input refs and optional merge."""

    request_id: str
    inputs: list[InputRef]
    merge: MergeFn | None = None
