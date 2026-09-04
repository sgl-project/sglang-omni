# SPDX-License-Identifier: Apache-2.0
"""Tests for video audio extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from sglang_omni.models.qwen3_omni.components.preprocessor import (
    _merge_extracted_video_audio,
)
from sglang_omni.preprocessing import video


def _write_video_with_audio(path: Path) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=10:duration=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000:duration=0.2",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ],
        check=True,
    )


def test_extract_audio_from_path_decodes_resamples_and_downmixes(
    tmp_path: Path,
) -> None:
    media = tmp_path / "audio.mp4"
    _write_video_with_audio(media)

    audio = video._extract_audio_from_path(media, 16_000)

    assert audio is not None
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert 3_000 <= audio.size <= 4_500
    assert np.max(np.abs(audio)) > 0.01


def test_extract_audio_from_path_returns_none_without_audio(monkeypatch) -> None:
    class Container:
        def __init__(self) -> None:
            self.streams = [SimpleNamespace(type="video")]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(video.av, "open", lambda _path: Container())

    assert video._extract_audio_from_path(Path("silent.mp4"), 16_000) is None


def test_extract_audio_from_path_surfaces_decode_failure(monkeypatch) -> None:
    class Container:
        def __init__(self) -> None:
            self.streams = [SimpleNamespace(type="audio", index=0)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, *, audio):
            assert audio == 0
            raise RuntimeError("broken stream")

    monkeypatch.setattr(video.av, "open", lambda _path: Container())

    with pytest.raises(video.VideoDecodeError, match="broken stream"):
        video._extract_audio_from_path(Path("broken.mp4"), 16_000)


def test_merge_extracted_video_audio_rejects_mixed_audio_presence() -> None:
    explicit = [np.ones(2, dtype=np.float32)]
    video_audio = [np.ones(3, dtype=np.float32), None]

    with pytest.raises(ValueError, match="every video"):
        _merge_extracted_video_audio(explicit, video_audio)


def test_merge_extracted_video_audio_allows_video_without_audio() -> None:
    explicit = [np.ones(2, dtype=np.float32)]
    merged, enabled = _merge_extracted_video_audio(explicit, [None])
    assert merged is explicit
    assert enabled is False


def test_merge_extracted_video_audio_preserves_alignment() -> None:
    explicit = [np.ones(2, dtype=np.float32)]
    video_audio = [np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32)]

    merged, enabled = _merge_extracted_video_audio(explicit, video_audio)

    assert enabled is True
    expected = [*explicit, *video_audio]
    assert len(merged) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(merged, expected))
