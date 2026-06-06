# SPDX-License-Identifier: Apache-2.0
"""Reference-audio cache keys and reference-code fingerprints for Higgs TTS."""

from __future__ import annotations

import base64
import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

_REF_PATH_HASH_MEMO_MAX_ITEMS = 1024
_REF_PATH_HASH_SENTINEL_BYTES = 8192
_REF_PATH_HASH_MEMO: OrderedDict[str, tuple[str, str]] = OrderedDict()
_REF_PATH_HASH_MEMO_LOCK = threading.Lock()


def _reference_path_hash_memo_key(path: Path) -> tuple[str, int] | None:
    try:
        if not path.is_file():
            return None
        stat_result = path.stat()
        memo_key = (
            f"{path.resolve()}:"
            f"{stat_result.st_size}:"
            f"{stat_result.st_mtime_ns}:"
            f"{stat_result.st_ctime_ns}"
        )
        return memo_key, int(stat_result.st_size)
    except OSError:
        return None


def _reference_path_sentinel(path: Path, file_size: int) -> str | None:
    from sglang_omni.preprocessing.cache_key import hash_bytes

    try:
        chunk_size = min(_REF_PATH_HASH_SENTINEL_BYTES, file_size)
        with path.open("rb") as f:
            chunks = [f.read(chunk_size)]
            if file_size > _REF_PATH_HASH_SENTINEL_BYTES:
                middle_offset = max((file_size - chunk_size) // 2, 0)
                f.seek(middle_offset)
                chunks.append(f.read(chunk_size))
            if file_size > 2 * _REF_PATH_HASH_SENTINEL_BYTES:
                f.seek(max(file_size - chunk_size, 0))
                chunks.append(f.read(chunk_size))
        return hash_bytes(b"".join(chunks) + f"|size:{file_size}".encode())
    except OSError:
        return None


def _get_reference_path_hash(memo_key: str, sentinel: str) -> str | None:
    with _REF_PATH_HASH_MEMO_LOCK:
        cached = _REF_PATH_HASH_MEMO.get(memo_key)
        if cached is None:
            return None
        cached_sentinel, digest = cached
        if cached_sentinel != sentinel:
            _REF_PATH_HASH_MEMO.pop(memo_key, None)
            return None
        _REF_PATH_HASH_MEMO.move_to_end(memo_key)
        return digest


def _put_reference_path_hash(memo_key: str, sentinel: str, digest: str) -> None:
    with _REF_PATH_HASH_MEMO_LOCK:
        _REF_PATH_HASH_MEMO[memo_key] = (sentinel, digest)
        _REF_PATH_HASH_MEMO.move_to_end(memo_key)
        while len(_REF_PATH_HASH_MEMO) > _REF_PATH_HASH_MEMO_MAX_ITEMS:
            _REF_PATH_HASH_MEMO.popitem(last=False)


def _reference_path_cache_key(path_like: str | Path) -> str | None:
    # Memoized full-content hash. The stat tuple avoids rereading stable local
    # refs while still invalidating normal and rapid same-size replacements.
    path = Path(str(path_like)).expanduser()
    memo = _reference_path_hash_memo_key(path)
    if memo is None:
        return None
    memo_key, file_size = memo
    sentinel = _reference_path_sentinel(path, file_size)
    if sentinel is None:
        return None
    digest = _get_reference_path_hash(memo_key, sentinel)
    if digest is not None:
        return f"file:{digest}"
    from sglang_omni.preprocessing.cache_key import hash_bytes

    try:
        digest = hash_bytes(path.read_bytes())
    except OSError:
        return None
    if _reference_path_hash_memo_key(path) == memo:
        _put_reference_path_hash(memo_key, sentinel, digest)
    return f"file:{digest}"


def reference_audio_cache_key(reference_audio: Any) -> str | None:
    """Stable cache key for a reference-audio input."""
    if isinstance(reference_audio, (str, Path)):
        return _reference_path_cache_key(reference_audio)
    if not isinstance(reference_audio, dict):
        return None
    path = reference_audio.get("audio_path") or reference_audio.get("path")
    if path:
        return _reference_path_cache_key(path)
    if "bytes" in reference_audio:
        from sglang_omni.preprocessing.cache_key import hash_media_item

        data = reference_audio["bytes"]
        if isinstance(data, str):
            data = data.encode()
        return hash_media_item(data)
    encoded = reference_audio.get("base64") or reference_audio.get("data")
    if encoded is None:
        return None
    from sglang_omni.preprocessing.cache_key import hash_media_item

    raw = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
    return hash_media_item(raw)


def reference_codes_fingerprint(codes: list[list[int]] | None) -> str | None:
    """Stable hash of the full N-codebook ref-audio sequence.

    Returned as a short hex string used as ``Req.extra_key``. ``None`` for
    zero-shot (no ref audio) so all zero-shot requests share the radix subtree.
    Each codec value packs into 2 bytes (range 0..1025) so the hash is
    sensitive to every codebook, not just cb0.
    """
    if not codes:
        return None
    buf = bytearray(2 * sum(len(row) for row in codes))
    i = 0
    for row in codes:
        for c in row:
            buf[i] = c & 0xFF
            buf[i + 1] = (c >> 8) & 0xFF
            i += 2
    return hashlib.blake2b(bytes(buf), digest_size=16).hexdigest()
