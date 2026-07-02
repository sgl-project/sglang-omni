# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.cache import (
    ArtifactHandle,
    CacheKey,
    CacheOwner,
    CacheSelector,
    LocalCachePlane,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache


def _key(digest: str, *, kind: str = "encoder_output") -> CacheKey:
    return CacheKey(namespace="qwen3_omni", kind=kind, digest=digest)


def _session_key(digest: str, session_id: str) -> CacheKey:
    return CacheKey(
        namespace="qwen3_omni",
        kind="encoder_output",
        digest=digest,
        session_id=session_id,
    )


def test_local_cache_plane_tracks_owner_pin_and_eviction() -> None:
    plane = LocalCachePlane(max_entries=2)
    owner = CacheOwner(stage_name="image_encoder", gpu_id=0, device="cuda:0")

    plane.publish(
        _key("a"),
        ArtifactHandle(backend="stage_output", ref={"key": "a"}),
        owner=owner,
        size_bytes=16,
    )
    plane.publish(
        _key("b"),
        ArtifactHandle(backend="stage_output", ref={"key": "b"}),
        owner=owner,
        size_bytes=16,
    )

    assert plane.lookup(_key("a")) is not None
    lease = plane.pin(_key("a"), request_id="req-1")
    assert lease is not None

    plane.publish(
        _key("c"),
        ArtifactHandle(backend="stage_output", ref={"key": "c"}),
        owner=owner,
        size_bytes=16,
    )

    assert plane.lookup(_key("a")) is not None
    assert plane.lookup(_key("b")) is None
    assert plane.lookup(_key("c")) is not None
    assert plane.stats().eviction_count == 1
    assert plane.stats().pinned_entries == 1

    plane.release(lease)
    evicted = plane.evict(max_entries=1)

    assert len(evicted) == 1
    assert plane.stats().entries == 1
    assert plane.stats().pinned_entries == 0


def test_local_cache_plane_ranks_owners_for_session_locality() -> None:
    plane = LocalCachePlane()
    owner_a = CacheOwner(
        stage_name="image_encoder",
        worker_id="worker-a",
        gpu_id=0,
        device="cuda:0",
    )
    owner_b = CacheOwner(
        stage_name="audio_encoder",
        worker_id="worker-b",
        gpu_id=1,
        device="cuda:1",
    )

    plane.publish(
        _session_key("image-a", "session-a"),
        ArtifactHandle(backend="stage_output", ref={"key": "image-a"}),
        owner=owner_a,
        size_bytes=32,
    )
    plane.publish(
        _session_key("audio-a", "session-a"),
        ArtifactHandle(backend="stage_output", ref={"key": "audio-a"}),
        owner=owner_a,
        size_bytes=16,
    )
    plane.publish(
        _session_key("image-b", "session-b"),
        ArtifactHandle(backend="stage_output", ref={"key": "image-b"}),
        owner=owner_b,
        size_bytes=128,
    )
    plane.bind_session("session-a", owner_a)

    ranked = plane.rank_owners(
        CacheSelector(namespace="qwen3_omni"),
        session_id="session-a",
    )

    assert ranked[0].owner == owner_a
    assert ranked[0].bound_session
    assert ranked[0].entry_count == 2
    assert ranked[0].session_entry_count == 2
    assert ranked[0].total_bytes == 48
    assert plane.session_owner("session-a") == owner_a
    assert plane.stats().session_bindings == 2

    assert plane.unbind_session("session-a")
    assert plane.session_owner("session-a") is None
    assert plane.rank_owners(session_id="session-a")[0].owner == owner_a


def test_local_cache_plane_singleflight_failure_does_not_poison_key() -> None:
    plane = LocalCachePlane()
    key = _key("build-me")

    assert plane.start_build(key, owner=CacheOwner(stage_name="audio_encoder"))
    assert not plane.start_build(key, owner=CacheOwner(stage_name="audio_encoder"))

    plane.fail_build(key, RuntimeError("boom"))

    assert plane.wait_ready(key, timeout_s=0.01) is None
    assert plane.start_build(key, owner=CacheOwner(stage_name="audio_encoder"))

    plane.publish(
        key,
        ArtifactHandle(backend="stage_output", ref={"key": "build-me"}),
        owner=CacheOwner(stage_name="audio_encoder"),
        size_bytes=8,
    )

    assert plane.wait_ready(key, timeout_s=0.01) is not None


def test_local_cache_plane_peek_does_not_count_hit_or_miss() -> None:
    plane = LocalCachePlane()
    key = _key("peek-me")
    plane.publish(key, ArtifactHandle(backend="stage_output", ref={"key": "peek-me"}))

    assert plane.peek(key) is not None
    assert plane.peek(_key("missing")) is None

    stats = plane.stats()
    assert stats.hit_count == 0
    assert stats.miss_count == 0


def test_stage_output_cache_registers_and_removes_plane_entries() -> None:
    plane = LocalCachePlane()
    owner = CacheOwner(stage_name="image_encoder", device="cuda")
    cache = StageOutputCache(
        max_size=1,
        cache_plane=plane,
        cache_namespace="qwen3_omni",
        cache_kind="image_encoder_output",
        cache_owner=owner,
        cache_device="cpu",
    )

    cache.put("image-a", {"x": torch.ones(2, dtype=torch.float32)})
    key_a = CacheKey(
        namespace="qwen3_omni",
        kind="image_encoder_output",
        digest="image-a",
        stage_name="image_encoder",
    )
    entry_a = plane.lookup(key_a)

    assert entry_a is not None
    assert entry_a.meta.owner.stage_name == "image_encoder"
    assert entry_a.meta.device == "cpu"
    assert entry_a.meta.size_bytes == 8
    assert cache.get("image-a") is not None
    assert plane.stats().hit_count == 2

    cache.put("image-b", torch.ones(1, dtype=torch.float32))
    key_b = CacheKey(
        namespace="qwen3_omni",
        kind="image_encoder_output",
        digest="image-b",
        stage_name="image_encoder",
    )

    assert cache.get("image-a") is None
    assert plane.lookup(key_a) is None
    assert plane.lookup(key_b) is not None

    cache.clear()

    assert plane.lookup(key_b) is None


def test_stage_output_cache_singleflight_waits_for_ready_entry() -> None:
    plane = LocalCachePlane()
    cache = StageOutputCache(
        cache_plane=plane,
        cache_namespace="qwen3_omni",
        cache_kind="image_encoder_output",
        cache_owner=CacheOwner(stage_name="image_encoder"),
    )
    results: list[object | None] = []

    assert cache.start_build("image-a")

    import threading

    waiter = threading.Thread(
        target=lambda: results.append(cache.wait_ready("image-a", timeout_s=1.0))
    )
    waiter.start()
    cache.put("image-a", {"x": torch.ones(1)})
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert len(results) == 1
    assert results[0] is not None
    assert plane.stats().ready_entries == 1


def test_stage_output_cache_singleflight_failure_allows_retry() -> None:
    plane = LocalCachePlane()
    cache = StageOutputCache(
        cache_plane=plane,
        cache_namespace="qwen3_omni",
        cache_kind="audio_encoder_output",
        cache_owner=CacheOwner(stage_name="audio_encoder"),
    )

    assert cache.start_build("audio-a")
    cache.fail_build("audio-a", RuntimeError("boom"))

    assert cache.wait_ready("audio-a", timeout_s=0.01) is None
    assert cache.start_build("audio-a")


def test_stage_output_cache_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="max_size"):
        StageOutputCache(max_size=0)
    with pytest.raises(ValueError, match="max_bytes"):
        StageOutputCache(max_bytes=0)


def test_cache_selector_invalidates_matching_namespace() -> None:
    plane = LocalCachePlane()
    plane.publish(
        _key("a", kind="image"),
        ArtifactHandle(backend="stage_output", ref={"key": "a"}),
    )
    plane.publish(
        _key("b", kind="audio"),
        ArtifactHandle(backend="stage_output", ref={"key": "b"}),
    )

    removed = plane.invalidate(CacheSelector(namespace="qwen3_omni", kind="image"))

    assert removed == 1
    assert plane.lookup(_key("a", kind="image")) is None
    assert plane.lookup(_key("b", kind="audio")) is not None
