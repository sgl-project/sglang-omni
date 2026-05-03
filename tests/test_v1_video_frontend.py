# SPDX-License-Identifier: Apache-2.0
"""Unit tests for v1 video preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from sglang_omni_v1.preprocessing import (
    compute_video_cache_key,
    ensure_video_list_async,
)
from sglang_omni_v1.preprocessing import video as video_module
from sglang_omni_v1.preprocessing.video import (
    VideoDecodeError,
    _check_if_video_has_audio,
    load_video_path,
)

VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
CACHE_DIR = Path.home().joinpath(".cache/omni-ci")
CACHE_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def video_path():
    video_dir = CACHE_DIR.joinpath("test_video.mp4")
    if video_dir.exists():
        return video_dir
    with open(video_dir, "wb") as f:
        response = requests.get(VIDEO_URL, timeout=30)
        response.raise_for_status()
        f.write(response.content)
    return video_dir


class TestVideoPreprocessing:
    def test_video_decode_error_includes_path_and_backend_context(
        self, monkeypatch, tmp_path
    ):
        video = tmp_path / "broken.mp4"
        video.write_bytes(b"not a real video")
        reader_calls: list[str] = []

        def fail_torchcodec(_):
            reader_calls.append("torchcodec")
            raise ValueError("total_frames must be a positive integer")

        def fail_torchvision(_):
            reader_calls.append("torchvision")
            raise ValueError("nframes should in interval [2, 0], but got 0")

        monkeypatch.setattr(
            video_module.qwen_vision,
            "get_video_reader_backend",
            lambda: "torchcodec",
        )
        monkeypatch.setitem(
            video_module.qwen_vision.VIDEO_READER_BACKENDS,
            "torchcodec",
            fail_torchcodec,
        )
        monkeypatch.setitem(
            video_module.qwen_vision.VIDEO_READER_BACKENDS,
            "torchvision",
            fail_torchvision,
        )

        with pytest.raises(VideoDecodeError) as exc_info:
            load_video_path(video, fps=2.0, max_frames=128)

        message = str(exc_info.value)
        assert reader_calls == ["torchcodec", "torchvision"]
        assert str(video) in message
        assert "torchcodec failed with ValueError" in message
        assert "torchvision failed with ValueError" in message

    @pytest.mark.asyncio
    async def test_video_loading_and_normalization(self, video_path):
        videos, fps_list, _ = await ensure_video_list_async(video_path, fps=2.0)

        assert len(videos) == 1
        assert videos[0].dim() == 4
        assert videos[0].shape[1] == 3
        assert fps_list is not None
        assert len(fps_list) == 1
        assert fps_list[0] > 0

    @pytest.mark.asyncio
    async def test_audio_extraction_from_video(self, video_path):
        has_audio = _check_if_video_has_audio(video_path)
        videos, fps_list, audios = await ensure_video_list_async(
            video_path, extract_audio=True, audio_target_sr=16000
        )

        assert len(videos) == 1
        if has_audio:
            assert audios is not None
            assert len(audios) == 1
            assert audios[0] is not None
            assert audios[0].ndim == 1
        else:
            assert audios is None or (len(audios) == 1 and audios[0] is None)

    def test_video_cache_key(self, video_path):
        key = compute_video_cache_key(video_path)

        assert key is not None
        assert isinstance(key, str)
        assert key.startswith("video:")
        assert key == compute_video_cache_key(video_path)

    @pytest.mark.asyncio
    async def test_complete_video_pipeline(self, video_path):
        cache_key = compute_video_cache_key(video_path)
        assert cache_key is not None

        has_audio = _check_if_video_has_audio(video_path)
        videos, fps_list, audios = await ensure_video_list_async(
            video_path, fps=2.0, extract_audio=True
        )
        assert len(videos) == 1
        assert videos[0].dim() == 4

        if has_audio:
            assert audios is not None
            assert len(audios) == 1
            assert audios[0] is not None
        else:
            assert audios is None or (len(audios) == 1 and audios[0] is None)

    @pytest.mark.asyncio
    async def test_video_loading_from_url(self):
        videos, fps_list, audios = await ensure_video_list_async(
            VIDEO_URL, fps=2.0, extract_audio=True, audio_target_sr=16000
        )

        assert len(videos) == 1
        assert videos[0].dim() == 4
        assert videos[0].shape[1] == 3
        assert fps_list is not None
        assert len(fps_list) == 1
        assert fps_list[0] > 0
        assert audios is not None
        assert len(audios) == 1
        assert audios[0] is not None
        assert audios[0].ndim == 1
