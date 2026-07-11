# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared reference-audio path cache-key helpers."""

from sglang_omni.preprocessing import cache_key


def test_reference_path_cache_key_tracks_file_content(tmp_path) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"a")
    first_key = cache_key.reference_path_cache_key(ref_audio)

    # Same content -> stable key (so repeat requests hit the cache).
    assert first_key == cache_key.reference_path_cache_key(ref_audio)

    ref_audio.write_bytes(b"longer")
    second_key = cache_key.reference_path_cache_key(ref_audio)

    # Different content -> different key (so a replaced file is not stale-served).
    assert first_key is not None and first_key.startswith("file:")
    assert second_key is not None and second_key.startswith("file:")
    assert first_key != second_key


def test_reference_path_cache_key_same_size_edit_and_non_files(tmp_path) -> None:
    # Same path, same size, same head/tail, different middle must not stale-hit.
    head, tail = b"H" * 8192, b"T" * 8192
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(head + b"a" * 4096 + tail)
    key_a = cache_key.reference_path_cache_key(ref_audio)
    ref_audio.write_bytes(head + b"b" * 4096 + tail)  # same size, middle differs
    assert key_a is not None
    assert key_a != cache_key.reference_path_cache_key(ref_audio)

    # URLs and missing files resolve to no key (callers bypass the cache).
    assert cache_key.reference_path_cache_key("https://example.com/ref.wav") is None
    assert cache_key.reference_path_cache_key(str(tmp_path / "missing.wav")) is None


def test_reference_path_cache_key_memoizes_stable_file_hash(
    monkeypatch, tmp_path
) -> None:
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake wav bytes")
    cache_key._REF_PATH_HASH_MEMO.clear()
    read_calls = 0
    original_read_bytes = cache_key.Path.read_bytes

    def counting_read_bytes(path):
        nonlocal read_calls
        if path == ref_audio:
            read_calls += 1
        return original_read_bytes(path)

    monkeypatch.setattr(cache_key.Path, "read_bytes", counting_read_bytes)

    first_key = cache_key.reference_path_cache_key(ref_audio)
    second_key = cache_key.reference_path_cache_key(ref_audio)

    assert first_key == second_key
    assert read_calls == 1


def test_reference_path_cache_key_trust_stat_skips_sentinel_on_hit(
    monkeypatch, tmp_path
) -> None:
    # The opt-in trust_stat fast path (absorbed from #740) trusts the
    # (size, mtime_ns, ctime_ns) stat tuple as content identity and skips the
    # sentinel byte-read on memo hits. Co-authored idea: GaokaiZhang (#740).
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"fake wav bytes")
    cache_key._REF_PATH_HASH_MEMO.clear()

    sentinel_calls = 0
    original_sentinel = cache_key._reference_path_sentinel

    def counting_sentinel(path, file_size):
        nonlocal sentinel_calls
        sentinel_calls += 1
        return original_sentinel(path, file_size)

    monkeypatch.setattr(cache_key, "_reference_path_sentinel", counting_sentinel)

    # First call (memo miss) must still compute the sentinel once so the memo
    # entry stays valid for default (trust_stat=False) callers like Higgs.
    first = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)
    assert sentinel_calls == 1
    # Second call (memo hit) takes the fast path: no further sentinel read.
    second = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)
    assert sentinel_calls == 1
    assert first == second


def test_reference_path_cache_key_trust_stat_keyspace_matches_default(
    tmp_path,
) -> None:
    # trust_stat must produce a byte-identical key to the default path for the
    # same content, so both callers share one keyspace.
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"shared content bytes")

    cache_key._REF_PATH_HASH_MEMO.clear()
    key_default = cache_key.reference_path_cache_key(ref_audio)
    cache_key._REF_PATH_HASH_MEMO.clear()
    key_trust = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)

    assert key_default is not None and key_default.startswith("file:")
    assert key_default == key_trust


def test_reference_path_cache_key_trust_stat_invalidates_on_stat_change(
    tmp_path,
) -> None:
    # A real content replacement that changes the stat tuple still invalidates.
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"a" * 64)
    cache_key._REF_PATH_HASH_MEMO.clear()
    key_a = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)

    ref_audio.write_bytes(b"b" * 128)  # different size -> stat tuple changes
    key_b = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)

    assert key_a is not None and key_b is not None
    assert key_a != key_b
