# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared reference-audio path cache-key helpers."""

import pytest
import torch
import xxhash

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
    cache_key._REF_PATH_HASH_MEMO.clear()  # noqa: leading-underscore  # production name
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
    cache_key._REF_PATH_HASH_MEMO.clear()  # noqa: leading-underscore  # production name

    sentinel_calls = 0
    original_sentinel = cache_key.reference_path_sentinel

    def counting_sentinel(path, file_size):
        nonlocal sentinel_calls
        sentinel_calls += 1
        return original_sentinel(path, file_size)

    monkeypatch.setattr(cache_key, "reference_path_sentinel", counting_sentinel)

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

    cache_key._REF_PATH_HASH_MEMO.clear()  # noqa: leading-underscore  # production name
    key_default = cache_key.reference_path_cache_key(ref_audio)
    cache_key._REF_PATH_HASH_MEMO.clear()  # noqa: leading-underscore  # production name
    key_trust = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)

    assert key_default is not None and key_default.startswith("file:")
    assert key_default == key_trust


def test_reference_path_cache_key_trust_stat_invalidates_on_stat_change(
    tmp_path,
) -> None:
    # A real content replacement that changes the stat tuple still invalidates.
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"a" * 64)
    cache_key._REF_PATH_HASH_MEMO.clear()  # noqa: leading-underscore  # production name
    key_a = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)

    ref_audio.write_bytes(b"b" * 128)  # different size -> stat tuple changes
    key_b = cache_key.reference_path_cache_key(ref_audio, trust_stat=True)

    assert key_a is not None and key_b is not None
    assert key_a != key_b


def test_hash_media_item_keys_content_not_mutable_addresses() -> None:
    # An address can serve new bytes, so only loaded content may key the cache.
    for url in (
        "http://example.com/a.png",
        "https://example.com/a.png",
        "file:///tmp/a.png",
    ):
        assert cache_key.hash_media_item(url) is None

    # Inline data URLs carry their bytes and keep a content key.
    first = cache_key.hash_media_item("data:image/png,content-a")
    assert first is not None
    assert first == cache_key.hash_media_item("data:image/png,content-a")
    assert first != cache_key.hash_media_item("data:image/png,content-b")


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(360, dtype=torch.float32).reshape(4, 3, 5, 6),
        torch.arange(360, dtype=torch.float32).reshape(4, 5, 6, 3).permute(0, 3, 1, 2),
        torch.arange(128, dtype=torch.int16).reshape(2, 64).T,
    ],
    ids=["contiguous-clip", "channels-last-clip", "transposed-rows"],
)
def test_hash_media_item_tensor_digest_matches_whole_buffer(tensor) -> None:
    # Strided frames are hashed one at a time, and the key must stay the hash
    # of the whole C-order buffer so existing cache keys do not change.
    whole = xxhash.xxh3_64(tensor.numpy().tobytes()).hexdigest()
    expected = f"pt:{tensor.dtype}|{tuple(tensor.shape)}:{whole}"
    assert cache_key.hash_media_item(tensor) == expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64])
def test_prompt_row_cache_keys_follow_row_content(dtype) -> None:
    rows = (torch.arange(32).reshape(4, 8) * 7 - 50).to(dtype)
    rows[3] = rows[0]
    keys = cache_key.PromptRowCacheKeys.for_rows(
        row_bytes=8 * rows.element_size(), device=torch.device("cpu")
    )
    changed = rows.clone()
    changed[1, 5] += 1

    ids = keys.key_ids(rows)
    changed_ids = keys.key_ids(changed)

    assert ids == keys.key_ids(rows.t().contiguous().t())
    assert ids[0] == ids[3]
    assert len(set(ids)) == 3
    assert changed_ids[1] != ids[1]
    assert changed_ids[:1] + changed_ids[2:] == ids[:1] + ids[2:]
    assert all(0 <= key < 2**63 for key in ids + changed_ids)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prompt_row_cache_keys_match_on_cpu_and_cuda() -> None:
    rows = torch.randn(64, 2048).to(torch.bfloat16)
    row_bytes = 2048 * rows.element_size()

    cpu_ids = cache_key.PromptRowCacheKeys.for_rows(
        row_bytes=row_bytes, device=torch.device("cpu")
    ).key_ids(rows)
    cuda_ids = cache_key.PromptRowCacheKeys.for_rows(
        row_bytes=row_bytes, device=torch.device("cuda")
    ).key_ids(rows.cuda())

    assert cpu_ids == cuda_ids
    assert len(set(cpu_ids)) == len(cpu_ids)
