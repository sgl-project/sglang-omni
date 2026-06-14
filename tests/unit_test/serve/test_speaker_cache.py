# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from sglang_omni.scheduling.speaker_cache import SpeakerArtifactCache, SpeakerCacheKey


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
