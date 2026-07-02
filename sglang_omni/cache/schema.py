# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import socket
from dataclasses import dataclass, field, replace
from typing import Any, Literal

CacheStatus = Literal["building", "ready", "failed"]


@dataclass(frozen=True)
class CacheKey:
    """Stable identity for a cacheable artifact.

    The key describes content identity and routing dimensions only. The actual
    tensor/object bytes stay owned by the stage cache, relay, or upstream
    SGLang cache that produced them.
    """

    namespace: str
    kind: str
    digest: str
    model_id: str | None = None
    weight_version: str | None = None
    processor_version: str | None = None
    stage_name: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("namespace", "kind", "digest"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CacheKey.{field_name} must be a non-empty string")


@dataclass(frozen=True)
class CacheOwner:
    """Where a cache entry is materialized."""

    stage_name: str | None = None
    process_id: int | None = None
    worker_id: str | None = None
    gpu_id: int | None = None
    device: str | None = None
    host: str = field(default_factory=socket.gethostname)


@dataclass(frozen=True)
class ArtifactHandle:
    """Pointer to the owner-specific artifact storage."""

    backend: str
    ref: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("ArtifactHandle.backend must be a non-empty string")


@dataclass(frozen=True)
class CacheMeta:
    owner: CacheOwner
    size_bytes: int = 0
    device: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    created_at: float = 0.0
    last_access_at: float = 0.0
    ttl_s: float | None = None
    ref_count: int = 0
    status: CacheStatus = "ready"
    error: str | None = None

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("CacheMeta.size_bytes must be non-negative")
        if self.ref_count < 0:
            raise ValueError("CacheMeta.ref_count must be non-negative")
        if self.ttl_s is not None and self.ttl_s <= 0:
            raise ValueError("CacheMeta.ttl_s must be positive when set")

    def with_access(self, timestamp: float) -> "CacheMeta":
        return replace(self, last_access_at=timestamp)

    def with_ref_count(self, ref_count: int) -> "CacheMeta":
        return replace(self, ref_count=ref_count)


@dataclass(frozen=True)
class CacheEntry:
    key: CacheKey
    handle: ArtifactHandle
    meta: CacheMeta


@dataclass(frozen=True)
class CacheLease:
    key: CacheKey
    request_id: str


@dataclass(frozen=True)
class CacheAffinity:
    """Cache locality score for a materialized owner."""

    owner: CacheOwner
    entry_count: int = 0
    total_bytes: int = 0
    session_entry_count: int = 0
    bound_session: bool = False

    def __post_init__(self) -> None:
        if self.entry_count < 0:
            raise ValueError("CacheAffinity.entry_count must be non-negative")
        if self.total_bytes < 0:
            raise ValueError("CacheAffinity.total_bytes must be non-negative")
        if self.session_entry_count < 0:
            raise ValueError("CacheAffinity.session_entry_count must be non-negative")


@dataclass(frozen=True)
class CacheSelector:
    namespace: str | None = None
    kind: str | None = None
    model_id: str | None = None
    stage_name: str | None = None
    session_id: str | None = None
    keys: frozenset[CacheKey] | None = None

    def matches(self, key: CacheKey) -> bool:
        if self.keys is not None and key not in self.keys:
            return False
        if self.namespace is not None and key.namespace != self.namespace:
            return False
        if self.kind is not None and key.kind != self.kind:
            return False
        if self.model_id is not None and key.model_id != self.model_id:
            return False
        if self.stage_name is not None and key.stage_name != self.stage_name:
            return False
        if self.session_id is not None and key.session_id != self.session_id:
            return False
        return True


@dataclass(frozen=True)
class CacheStats:
    entries: int
    ready_entries: int
    building_entries: int
    pinned_entries: int
    total_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    invalidation_count: int
    session_bindings: int = 0
