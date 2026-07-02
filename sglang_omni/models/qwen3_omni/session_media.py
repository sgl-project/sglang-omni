# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class SessionMediaEntry:
    """Media references and cache keys retained for one logical session."""

    raw_images: Any = None
    raw_videos: Any = None
    raw_audios: Any = None
    image_cache_key: str | None = None
    audio_cache_key: str | None = None
    num_images: int = 0
    num_videos: int = 0
    num_audios: int = 0
    image_grid_thw: Any = None
    video_grid_thw: Any = None
    video_second_per_grid: Any = None
    audio_placeholder_lengths: Any = None
    audio_target_sr: int = 16000
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None
    use_audio_in_video: bool | None = None
    video_seconds_per_chunk: float | None = None
    video_position_id_per_seconds: float | None = None

    @property
    def has_media(self) -> bool:
        return bool(
            self.raw_images
            or self.raw_videos
            or self.raw_audios
            or self.image_cache_key
            or self.audio_cache_key
        )


class SessionMediaRegistry:
    """Small process-local LRU for session-scoped media reuse.

    The registry keeps references and cache keys only. Heavy encoder artifacts
    remain owned by StageOutputCache/cache plane entries.
    """

    def __init__(self, *, max_sessions: int = 1024) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = int(max_sessions)
        self._entries: OrderedDict[str, SessionMediaEntry] = OrderedDict()
        self._lock = RLock()

    def get(self, session_id: str) -> SessionMediaEntry | None:
        _validate_session_id(session_id)
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            self._entries.move_to_end(session_id)
            return entry

    def put(self, session_id: str, entry: SessionMediaEntry) -> None:
        _validate_session_id(session_id)
        if not entry.has_media:
            return
        with self._lock:
            self._entries[session_id] = entry
            self._entries.move_to_end(session_id)
            while len(self._entries) > self.max_sessions:
                self._entries.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        _validate_session_id(session_id)
        with self._lock:
            return self._entries.pop(session_id, None) is not None

    def clear_all(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
