# SPDX-License-Identifier: Apache-2.0
"""Stable reference-audio identity helpers shared by TTS models."""

from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

_REF_PATH_HASH_MEMO_MAX_ITEMS = 1024
_REF_PATH_HASH_SENTINEL_BYTES = 8192


class ReferenceAudioHashMemo:
    def __init__(
        self,
        *,
        max_items: int = _REF_PATH_HASH_MEMO_MAX_ITEMS,
        sentinel_bytes: int = _REF_PATH_HASH_SENTINEL_BYTES,
    ) -> None:
        self.max_items = int(max_items)
        self.sentinel_bytes = int(sentinel_bytes)
        self.memo: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self.lock = threading.Lock()

    def path_cache_key(self, path_like: str | Path) -> str | None:
        # Memoized full-content hash. The stat tuple avoids rereading stable
        # local refs while still invalidating rapid same-size replacements.
        path = Path(str(path_like)).expanduser()
        memo = self._path_hash_memo_key(path)
        if memo is None:
            return None
        memo_key, file_size = memo
        sentinel = self._path_sentinel(path, file_size)
        if sentinel is None:
            return None
        digest = self._get_path_hash(memo_key, sentinel)
        if digest is not None:
            return f"file:{digest}"
        from sglang_omni.preprocessing.cache_key import hash_bytes

        try:
            digest = hash_bytes(path.read_bytes())
        except OSError:
            return None
        if self._path_hash_memo_key(path) == memo:
            self._put_path_hash(memo_key, sentinel, digest)
        return f"file:{digest}"

    def _path_hash_memo_key(self, path: Path) -> tuple[str, int] | None:
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

    def _path_sentinel(self, path: Path, file_size: int) -> str | None:
        from sglang_omni.preprocessing.cache_key import hash_bytes

        try:
            chunk_size = min(self.sentinel_bytes, file_size)
            with path.open("rb") as f:
                chunks = [f.read(chunk_size)]
                if file_size > self.sentinel_bytes:
                    middle_offset = max((file_size - chunk_size) // 2, 0)
                    f.seek(middle_offset)
                    chunks.append(f.read(chunk_size))
                if file_size > 2 * self.sentinel_bytes:
                    f.seek(max(file_size - chunk_size, 0))
                    chunks.append(f.read(chunk_size))
            return hash_bytes(b"".join(chunks) + f"|size:{file_size}".encode())
        except OSError:
            return None

    def _get_path_hash(self, memo_key: str, sentinel: str) -> str | None:
        with self.lock:
            cached = self.memo.get(memo_key)
            if cached is None:
                return None
            cached_sentinel, digest = cached
            if cached_sentinel != sentinel:
                self.memo.pop(memo_key, None)
                return None
            self.memo.move_to_end(memo_key)
            return digest

    def _put_path_hash(self, memo_key: str, sentinel: str, digest: str) -> None:
        with self.lock:
            self.memo[memo_key] = (sentinel, digest)
            self.memo.move_to_end(memo_key)
            while len(self.memo) > self.max_items:
                self.memo.popitem(last=False)


_DEFAULT_REFERENCE_AUDIO_HASH_MEMO = ReferenceAudioHashMemo()


def _reference_path_cache_key(path_like: str | Path) -> str | None:
    return _DEFAULT_REFERENCE_AUDIO_HASH_MEMO.path_cache_key(path_like)


def reference_audio_cache_key(
    reference_audio: Any,
    *,
    memo: ReferenceAudioHashMemo = _DEFAULT_REFERENCE_AUDIO_HASH_MEMO,
) -> str | None:
    """Stable cache key for a reference-audio input."""
    if isinstance(reference_audio, (str, Path)):
        return memo.path_cache_key(reference_audio)
    if not isinstance(reference_audio, dict):
        return None
    path = reference_audio.get("audio_path") or reference_audio.get("path")
    if path:
        return memo.path_cache_key(path)
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
