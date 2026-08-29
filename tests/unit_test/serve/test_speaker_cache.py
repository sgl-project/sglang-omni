# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import threading
import time

import numpy as np
import pytest

from sglang_omni.scheduling.speaker_cache import SpeakerArtifactCache, SpeakerCacheKey


def _wait_for_misses(cache: SpeakerArtifactCache, expected: int) -> None:
    deadline = time.monotonic() + 5.0
    while cache.stats()["miss_count"] < expected and time.monotonic() < deadline:
        time.sleep(0.001)
    assert cache.stats()["miss_count"] >= expected


def test_speaker_cache_tracks_hits_misses_and_voice_invalidation() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey(
        model_type="higgs",
        voice_name="speaker-a",
        voice_version=1,
        artifact_kind="ref_codes",
    )

    assert cache.get(key) is None
    cache.put(key, np.arange(16, dtype=np.float32))

    assert cache.get(key).shape == (16,)
    cache.clear_voice("SPEAKER-A")
    assert cache.get(key) is None

    stats = cache.stats()
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 2
    assert stats["delete_invalidation_counter"] == 1
    assert stats["entries"] == 0

    cache.clear_voice("SPEAKER-A")
    assert cache.stats()["delete_invalidation_counter"] == 1


def test_speaker_cache_evicts_oldest_entry_under_memory_pressure() -> None:
    cache = SpeakerArtifactCache(max_bytes=64)
    first = SpeakerCacheKey("higgs", "a", 1, "embedding")
    second = SpeakerCacheKey("higgs", "b", 1, "embedding")

    cache.put(first, np.arange(12, dtype=np.float32))
    cache.put(second, np.arange(12, dtype=np.float32))

    assert cache.get(first) is None
    assert cache.get(second) is not None
    assert cache.stats()["eviction_count"] == 1


def test_speaker_cache_singleflights_same_key_cold_miss() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    leader_started = threading.Event()
    release_leader = threading.Event()
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        leader_started.set()
        assert release_leader.wait(timeout=5.0)
        return "encoded"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(cache.get_or_compute, key, compute)
        assert leader_started.wait(timeout=5.0)
        follower = executor.submit(cache.get_or_compute, key, compute)
        _wait_for_misses(cache, 2)
        release_leader.set()

        assert leader.result(timeout=5.0) == "encoded"
        assert follower.result(timeout=5.0) == "encoded"

    assert calls == 1
    assert cache.get(key) == "encoded"
    assert cache.stats()["singleflight_merged_count"] == 1


def test_speaker_cache_singleflight_propagates_failure_and_allows_retry() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    leader_started = threading.Event()
    release_leader = threading.Event()
    calls = 0

    def fail() -> str:
        nonlocal calls
        calls += 1
        leader_started.set()
        assert release_leader.wait(timeout=5.0)
        raise ValueError("encode failed")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(cache.get_or_compute, key, fail)
        assert leader_started.wait(timeout=5.0)
        follower = executor.submit(cache.get_or_compute, key, fail)
        _wait_for_misses(cache, 2)
        release_leader.set()

        for future in (leader, follower):
            with pytest.raises(ValueError, match="encode failed"):
                future.result(timeout=5.0)

    assert calls == 1
    assert cache.get_or_compute(key, lambda: "retry") == "retry"


def test_speaker_cache_singleflight_propagates_leader_cancellation() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    leader_started = threading.Event()
    release_leader = threading.Event()
    calls = 0

    def cancel() -> str:
        nonlocal calls
        calls += 1
        leader_started.set()
        assert release_leader.wait(timeout=5.0)
        raise concurrent.futures.CancelledError("encode cancelled")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(cache.get_or_compute, key, cancel)
        assert leader_started.wait(timeout=5.0)
        follower = executor.submit(cache.get_or_compute, key, cancel)
        _wait_for_misses(cache, 2)
        release_leader.set()

        for future in (leader, follower):
            with pytest.raises(concurrent.futures.CancelledError):
                future.result(timeout=5.0)

    assert calls == 1
    assert cache.get_or_compute(key, lambda: "retry") == "retry"


def test_speaker_cache_singleflight_store_failure_does_not_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    original_put = cache._cache.put

    def fail_store(encoded_key, value) -> None:
        del encoded_key, value
        raise RuntimeError("cache store failed")

    monkeypatch.setattr(cache._cache, "put", fail_store)
    with pytest.raises(RuntimeError, match="cache store failed"):
        cache.get_or_compute(key, lambda: "encoded")

    assert not cache._inflight

    monkeypatch.setattr(cache._cache, "put", original_put)
    assert cache.get_or_compute(key, lambda: "retry") == "retry"


def test_speaker_cache_put_wins_over_late_singleflight_leader() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    leader_started = threading.Event()
    release_leader = threading.Event()

    def compute() -> str:
        leader_started.set()
        assert release_leader.wait(timeout=5.0)
        return "stale"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        leader = executor.submit(cache.get_or_compute, key, compute)
        assert leader_started.wait(timeout=5.0)
        cache.put(key, "replacement")
        release_leader.set()
        assert leader.result(timeout=5.0) == "stale"

    assert cache.get(key) == "replacement"


def test_speaker_cache_delete_does_not_resurrect_inflight_voice() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    leader_started = threading.Event()
    release_leader = threading.Event()

    def compute() -> str:
        leader_started.set()
        assert release_leader.wait(timeout=5.0)
        return "stale"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        leader = executor.submit(cache.get_or_compute, key, compute)
        assert leader_started.wait(timeout=5.0)
        cache.clear_voice("GUIDE")
        release_leader.set()
        assert leader.result(timeout=5.0) == "stale"

    assert cache.get(key) is None


def test_speaker_cache_delete_detaches_invalidated_flight() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    old_started = threading.Event()
    release_old = threading.Event()
    fresh_started = threading.Event()

    def compute_old() -> str:
        old_started.set()
        assert release_old.wait(timeout=5.0)
        return "stale"

    def compute_fresh() -> str:
        fresh_started.set()
        return "fresh"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        old = executor.submit(cache.get_or_compute, key, compute_old)
        assert old_started.wait(timeout=5.0)
        cache.clear_voice("guide")
        fresh = executor.submit(cache.get_or_compute, key, compute_fresh)
        assert fresh_started.wait(timeout=5.0)
        assert fresh.result(timeout=5.0) == "fresh"
        release_old.set()
        assert old.result(timeout=5.0) == "stale"

    assert cache.get(key) == "fresh"


def test_speaker_cache_clear_does_not_resurrect_inflight_artifact() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    leader_started = threading.Event()
    release_leader = threading.Event()

    def compute() -> str:
        leader_started.set()
        assert release_leader.wait(timeout=5.0)
        return "stale"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        leader = executor.submit(cache.get_or_compute, key, compute)
        assert leader_started.wait(timeout=5.0)
        cache.clear()
        release_leader.set()
        assert leader.result(timeout=5.0) == "stale"

    assert cache.get(key) is None


def test_speaker_cache_distinct_keys_compute_concurrently() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    keys = (
        SpeakerCacheKey("qwen3_tts_icl", "first", 1, "voice_clone_prompt"),
        SpeakerCacheKey("qwen3_tts_icl", "second", 1, "voice_clone_prompt"),
    )
    both_started = threading.Barrier(2)

    def compute(value: str) -> str:
        both_started.wait(timeout=5.0)
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                cache.get_or_compute,
                key,
                lambda value=value: compute(value),
            )
            for key, value in zip(keys, ("first", "second"), strict=True)
        ]
        assert [future.result(timeout=5.0) for future in futures] == [
            "first",
            "second",
        ]


def test_speaker_cache_replaces_rejected_cached_artifact() -> None:
    cache = SpeakerArtifactCache(max_bytes=1024)
    key = SpeakerCacheKey("qwen3_tts_icl", "guide", 1, "voice_clone_prompt")
    cache.put(key, {"artifact_type": "wrong"})

    result = cache.get_or_compute(
        key,
        lambda: {"artifact_type": "qwen3_tts_voice_clone_prompt"},
        accept_cached=lambda value: value.get("artifact_type")
        == "qwen3_tts_voice_clone_prompt",
    )

    assert result == {"artifact_type": "qwen3_tts_voice_clone_prompt"}
    assert cache.get(key) == result
