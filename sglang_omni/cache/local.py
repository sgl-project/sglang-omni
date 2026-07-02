# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from sglang_omni.cache.schema import (
    ArtifactHandle,
    CacheAffinity,
    CacheEntry,
    CacheKey,
    CacheLease,
    CacheMeta,
    CacheOwner,
    CacheSelector,
    CacheStats,
)


class LocalCachePlane:
    """In-process metadata registry for cacheable artifacts.

    Owners keep the actual payload bytes. The plane records where entries live,
    their approximate size, pin state, and readiness so higher layers can make
    routing, invalidation, and observability decisions without taking over
    tensor or SGLang KV ownership.
    """

    def __init__(
        self,
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        clock=time.monotonic,
    ) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive when set")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive when set")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._clock = clock
        self._entries: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._conditions: dict[CacheKey, threading.Condition] = {}
        self._session_owners: dict[str, CacheOwner] = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._invalidation_count = 0

    def publish(
        self,
        key: CacheKey,
        handle: ArtifactHandle,
        *,
        owner: CacheOwner | None = None,
        size_bytes: int = 0,
        device: str | None = None,
        dtype: str | None = None,
        shape: tuple[int, ...] | None = None,
        ttl_s: float | None = None,
    ) -> CacheEntry:
        now = self._clock()
        owner = owner or CacheOwner()
        meta = CacheMeta(
            owner=owner,
            size_bytes=int(size_bytes),
            device=device,
            dtype=dtype,
            shape=shape,
            created_at=now,
            last_access_at=now,
            ttl_s=ttl_s,
            status="ready",
        )
        entry = CacheEntry(key=key, handle=handle, meta=meta)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            if key.session_id is not None:
                self._session_owners[key.session_id] = owner
            self._notify_locked(key)
            self._evict_over_budget_locked()
            return self._entries.get(key, entry)

    def start_build(
        self,
        key: CacheKey,
        *,
        owner: CacheOwner | None = None,
        ttl_s: float | None = None,
    ) -> bool:
        """Register a single-flight build.

        Returns True for the leader. Followers get False and can call
        wait_ready(). Failed builds are removed, so a later caller can retry.
        """

        now = self._clock()
        owner = owner or CacheOwner()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and not self._is_expired(entry, now):
                self._entries.move_to_end(key)
                return False
            if entry is not None:
                self._remove_locked(key)
            meta = CacheMeta(
                owner=owner,
                created_at=now,
                last_access_at=now,
                ttl_s=ttl_s,
                status="building",
            )
            self._entries[key] = CacheEntry(
                key=key,
                handle=ArtifactHandle(backend="building"),
                meta=meta,
            )
            self._entries.move_to_end(key)
            if key.session_id is not None:
                self._session_owners[key.session_id] = owner
            self._conditions.setdefault(key, threading.Condition(self._lock))
            return True

    def fail_build(self, key: CacheKey, error: BaseException | str) -> None:
        del error
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.meta.status == "building":
                self._remove_locked(key)
            self._notify_locked(key)

    def wait_ready(
        self, key: CacheKey, timeout_s: float | None = None
    ) -> CacheEntry | None:
        deadline = None if timeout_s is None else self._clock() + timeout_s
        with self._lock:
            condition = self._conditions.setdefault(
                key, threading.Condition(self._lock)
            )
            while True:
                entry = self._entries.get(key)
                if entry is None:
                    return None
                if self._is_expired(entry, self._clock()):
                    self._remove_locked(key)
                    self._notify_locked(key)
                    return None
                if entry.meta.status == "ready":
                    self._entries.move_to_end(key)
                    return entry
                if entry.meta.status != "building":
                    return None
                if deadline is None:
                    condition.wait()
                    continue
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                condition.wait(remaining)

    def lookup(self, key: CacheKey) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            now = self._clock()
            if entry is None or self._is_expired(entry, now):
                if entry is not None:
                    self._remove_locked(key)
                self._miss_count += 1
                return None
            if entry.meta.status != "ready":
                self._miss_count += 1
                return None
            entry = CacheEntry(
                key=entry.key, handle=entry.handle, meta=entry.meta.with_access(now)
            )
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._hit_count += 1
            return entry

    def peek(self, key: CacheKey) -> CacheEntry | None:
        """Return a ready entry without updating hit/miss counters."""

        with self._lock:
            entry = self._entries.get(key)
            now = self._clock()
            if entry is None:
                return None
            if self._is_expired(entry, now):
                self._remove_locked(key)
                return None
            if entry.meta.status != "ready":
                return None
            return entry

    def record_miss(self, key: CacheKey) -> None:
        del key
        with self._lock:
            self._miss_count += 1

    def touch(self, key: CacheKey) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            now = self._clock()
            self._entries[key] = CacheEntry(
                key=entry.key,
                handle=entry.handle,
                meta=entry.meta.with_access(now),
            )
            self._entries.move_to_end(key)

    def pin(self, key: CacheKey, request_id: str) -> CacheLease | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.meta.status != "ready":
                return None
            meta = entry.meta.with_ref_count(entry.meta.ref_count + 1)
            self._entries[key] = CacheEntry(
                key=entry.key, handle=entry.handle, meta=meta
            )
            self._entries.move_to_end(key)
            return CacheLease(key=key, request_id=str(request_id))

    def release(self, lease: CacheLease) -> None:
        with self._lock:
            entry = self._entries.get(lease.key)
            if entry is None:
                return
            meta = entry.meta.with_ref_count(max(0, entry.meta.ref_count - 1))
            self._entries[lease.key] = CacheEntry(
                key=entry.key,
                handle=entry.handle,
                meta=meta,
            )

    def bind_session(self, session_id: str, owner: CacheOwner) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        with self._lock:
            self._session_owners[session_id] = owner

    def unbind_session(self, session_id: str) -> bool:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        with self._lock:
            return self._session_owners.pop(session_id, None) is not None

    def session_owner(self, session_id: str) -> CacheOwner | None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        with self._lock:
            return self._session_owners.get(session_id)

    def rank_owners(
        self,
        selector: CacheSelector | None = None,
        *,
        session_id: str | None = None,
    ) -> list[CacheAffinity]:
        selector = selector or CacheSelector()
        affinity_session_id = session_id or selector.session_id
        with self._lock:
            scores: dict[CacheOwner, CacheAffinity] = {}
            now = self._clock()
            for key, entry in self._entries.items():
                if entry.meta.status != "ready":
                    continue
                if self._is_expired(entry, now):
                    continue
                if not selector.matches(key):
                    continue
                affinity = scores.get(entry.meta.owner)
                session_entry = (
                    affinity_session_id is not None
                    and key.session_id == affinity_session_id
                )
                scores[entry.meta.owner] = CacheAffinity(
                    owner=entry.meta.owner,
                    entry_count=(affinity.entry_count if affinity else 0) + 1,
                    total_bytes=(affinity.total_bytes if affinity else 0)
                    + entry.meta.size_bytes,
                    session_entry_count=(
                        affinity.session_entry_count if affinity else 0
                    )
                    + int(session_entry),
                    bound_session=affinity.bound_session if affinity else False,
                )

            bound_owner = (
                self._session_owners.get(affinity_session_id)
                if affinity_session_id is not None
                else None
            )
            if bound_owner is not None:
                affinity = scores.get(bound_owner)
                scores[bound_owner] = CacheAffinity(
                    owner=bound_owner,
                    entry_count=affinity.entry_count if affinity else 0,
                    total_bytes=affinity.total_bytes if affinity else 0,
                    session_entry_count=(
                        affinity.session_entry_count if affinity else 0
                    ),
                    bound_session=True,
                )

            return sorted(
                scores.values(),
                key=lambda affinity: (
                    not affinity.bound_session,
                    -affinity.session_entry_count,
                    -affinity.total_bytes,
                    -affinity.entry_count,
                    affinity.owner.host,
                    affinity.owner.stage_name or "",
                    affinity.owner.worker_id or "",
                    affinity.owner.device or "",
                ),
            )

    def remove(self, key: CacheKey) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._remove_locked(key)
            self._notify_locked(key)
            return True

    def invalidate(
        self,
        selector: CacheSelector | None = None,
        *,
        include_pinned: bool = False,
    ) -> int:
        selector = selector or CacheSelector()
        removed = 0
        with self._lock:
            for key, entry in list(self._entries.items()):
                if not selector.matches(key):
                    continue
                if entry.meta.ref_count > 0 and not include_pinned:
                    continue
                self._remove_locked(key)
                self._notify_locked(key)
                removed += 1
            if selector.session_id is not None and removed:
                self._session_owners.pop(selector.session_id, None)
            self._invalidation_count += removed
        return removed

    def evict(
        self,
        *,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        selector: CacheSelector | None = None,
    ) -> list[CacheKey]:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive when set")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive when set")
        selector = selector or CacheSelector()
        evicted: list[CacheKey] = []
        with self._lock:
            while self._over_budget_locked(max_entries, max_bytes):
                victim = self._oldest_evictable_locked(selector)
                if victim is None:
                    break
                self._remove_locked(victim)
                self._notify_locked(victim)
                evicted.append(victim)
            self._eviction_count += len(evicted)
        return evicted

    def stats(self, selector: CacheSelector | None = None) -> CacheStats:
        selector = selector or CacheSelector()
        with self._lock:
            entries = [
                entry for key, entry in self._entries.items() if selector.matches(key)
            ]
            return CacheStats(
                entries=len(entries),
                ready_entries=sum(entry.meta.status == "ready" for entry in entries),
                building_entries=sum(
                    entry.meta.status == "building" for entry in entries
                ),
                pinned_entries=sum(entry.meta.ref_count > 0 for entry in entries),
                total_bytes=sum(entry.meta.size_bytes for entry in entries),
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                eviction_count=self._eviction_count,
                invalidation_count=self._invalidation_count,
                session_bindings=self._session_binding_count_locked(selector),
            )

    def clear(self) -> None:
        with self._lock:
            keys = list(self._entries)
            for key in keys:
                self._notify_locked(key)
            self._entries.clear()
            self._conditions.clear()
            self._session_owners.clear()

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        ttl_s = entry.meta.ttl_s
        return ttl_s is not None and now - entry.meta.created_at >= ttl_s

    def _over_budget_locked(
        self,
        max_entries: int | None = None,
        max_bytes: int | None = None,
    ) -> bool:
        entry_budget = max_entries if max_entries is not None else self.max_entries
        byte_budget = max_bytes if max_bytes is not None else self.max_bytes
        if entry_budget is not None and len(self._entries) > entry_budget:
            return True
        if byte_budget is not None and self._total_bytes_locked() > byte_budget:
            return True
        return False

    def _evict_over_budget_locked(self) -> None:
        evicted = 0
        while self._over_budget_locked():
            victim = self._oldest_evictable_locked(CacheSelector())
            if victim is None:
                break
            self._remove_locked(victim)
            self._notify_locked(victim)
            evicted += 1
        self._eviction_count += evicted

    def _oldest_evictable_locked(self, selector: CacheSelector) -> CacheKey | None:
        for key, entry in self._entries.items():
            if not selector.matches(key):
                continue
            if entry.meta.status != "ready":
                continue
            if entry.meta.ref_count > 0:
                continue
            return key
        return None

    def _total_bytes_locked(self) -> int:
        return sum(entry.meta.size_bytes for entry in self._entries.values())

    def _remove_locked(self, key: CacheKey) -> None:
        self._entries.pop(key, None)

    def _session_binding_count_locked(self, selector: CacheSelector) -> int:
        if selector.session_id is not None:
            return int(selector.session_id in self._session_owners)
        return len(self._session_owners)

    def _notify_locked(self, key: CacheKey) -> None:
        condition = self._conditions.get(key)
        if condition is not None:
            condition.notify_all()
