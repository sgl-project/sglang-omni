# SPDX-License-Identifier: Apache-2.0
"""Control-plane types for page-oriented KV cache transfers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import msgspec


@dataclass(frozen=True)
class KVBufferSpec:
    """Wire-visible shape of one independently registered KV buffer."""

    name: str
    bytes_per_page: int
    dtype: str | None = None
    token_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class KVPoolLayout:
    """Layout contract that source and destination pools must share."""

    layout_id: str
    page_size: int
    buffers: tuple[KVBufferSpec, ...]
    cache_dtype: str | None = None

    def compatible_with(self, other: "KVPoolLayout") -> bool:
        """Return whether pages can be copied without a layout transform."""

        return (
            self.layout_id == other.layout_id
            and self.page_size == other.page_size
            and self.cache_dtype == other.cache_dtype
            and self.buffers == other.buffers
        )


@dataclass(frozen=True)
class PDStageKVLayout:
    role: Literal["prefill", "decode"]
    pool_id: str
    layout: KVPoolLayout
    partner_stage: str | None = None


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
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        value = msgspec.to_builtins(self)
        # Metadata is intentionally opaque. In particular, continuation bytes
        # must not be converted to a JSON-style list of integers.
        value["metadata"] = self.metadata
        return {"type": "kv_transfer_prepare", **value}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "KVTransferPrepareMessage":
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
    destination_ref: dict[str, object] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"type": "kv_transfer_ready", **msgspec.to_builtins(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "KVTransferReadyMessage":
        return msgspec.convert(value, type=cls, strict=True)
