# SPDX-License-Identifier: Apache-2.0
"""Control-plane types for page-oriented KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import msgspec


@dataclass(frozen=True)
class KVBufferSpec:
    """Wire-visible shape of one independently registered KV buffer."""

    name: str
    bytes_per_page: int


@dataclass(frozen=True)
class KVPoolLayout:
    """Layout contract that source and destination pools must share."""

    layout_id: str
    page_size: int
    buffers: tuple[KVBufferSpec, ...]

    def compatible_with(self, other: "KVPoolLayout") -> bool:
        """Return whether pages can be copied without a layout transform."""

        if self.layout_id != other.layout_id or self.page_size != other.page_size:
            return False
        if len(self.buffers) != len(other.buffers):
            return False
        return all(
            source.name == destination.name
            and source.bytes_per_page == destination.bytes_per_page
            for source, destination in zip(self.buffers, other.buffers, strict=True)
        )


@dataclass(frozen=True)
class KVTransferPrepareMessage:
    """Ask a receiver to reserve destination pages for one transfer."""

    request_id: str
    transfer_id: str
    from_stage: str
    to_stage: str
    source_pool_id: str
    target_pool_id: str
    source_page_indices: tuple[int, ...]
    source_layout: KVPoolLayout
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = msgspec.to_builtins(self)
        value["metadata"] = self.metadata
        return {"type": "kv_transfer_prepare", **value}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVTransferPrepareMessage":
        return msgspec.convert(value, type=cls, strict=True)


@dataclass(frozen=True)
class KVTransferReadyMessage:
    """Return a reserved destination or a prepare failure to the sender."""

    request_id: str
    transfer_id: str
    from_stage: str
    to_stage: str
    success: bool
    destination_pool_id: str | None = None
    destination_page_indices: tuple[int, ...] = ()
    destination_ref: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "kv_transfer_ready", **msgspec.to_builtins(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KVTransferReadyMessage":
        return msgspec.convert(value, type=cls, strict=True)
