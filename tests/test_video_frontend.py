# SPDX-License-Identifier: Apache-2.0
"""Unit tests for video preprocessing."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from sglang_omni.preprocessing import (
    compute_video_cache_key,
    ensure_video_list,
    extract_audio_from_video_inputs,
)
from sglang_omni.preprocessing.video import _check_if_video_has_audio

# Remote test resources
VIDEO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
CACHE_DIR = Path.home().joinpath(".cache/omni-ci")
CACHE_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def video_path():
    """Download test video once for all tests."""
    video_dir = CACHE_DIR.joinpath("test_video.mp4")
    if video_dir.exists():
        return video_dir
    else:
        with open(video_dir, "wb") as f:
            response = requests.get(VIDEO_URL, timeout=30)
            response.raise_for_status()
            f.write(response.content)
        return video_dir


@pytest.fixture(scope="module")
def image_path():
    """Download test image once for all tests."""
    img_dir = CACHE_DIR.joinpath("test_image.jpg")
    if img_dir.exists():
        return img_dir
    else:
        with open(img_dir, "wb") as f:
            response = requests.get(IMAGE_URL, timeout=30)
            response.raise_for_status()
            f.write(response.content)
        return img_dir


class TestVideoPreprocessing:
    """Test core video preprocessing functionality."""

    def test_video_loading_and_normalization(self, video_path):
        """Test loading video from path and normalizing to tensor."""
        videos, fps = ensure_video_list(video_path, fps=2.0)

        assert len(videos) == 1
        assert videos[0].dim() == 4  # (T, C, H, W)
        assert videos[0].shape[1] == 3  # RGB channels
        assert fps is not None
        assert len(fps) == 1
        assert fps[0] > 0

    def test_audio_extraction_from_video(self, video_path):
        """Test extracting audio from video when use_audio_in_video=True."""
        has_audio = _check_if_video_has_audio(video_path)
        audios, flag = extract_audio_from_video_inputs(
            video_path, use_audio_in_video=True, target_sr=16000
        )

        if has_audio:
            if audios is not None:
                assert len(audios) == 1
                assert audios[0].ndim == 1
                assert flag is True
        else:
            assert audios is None
            assert flag is False

    def test_video_cache_key(self, video_path):
        """Test cache key generation for videos."""
        key = compute_video_cache_key(video_path)

        assert key is not None
        assert isinstance(key, str)
        assert key.startswith("video:")

        # Cache key should be consistent
        key2 = compute_video_cache_key(video_path)
        assert key == key2

    def test_complete_video_pipeline(self, video_path):
        """Test complete video processing pipeline without model inference."""
        # 1. Compute cache key
        cache_key = compute_video_cache_key(video_path)
        assert cache_key is not None

        # 2. Load and normalize video
        videos, fps = ensure_video_list(video_path, fps=2.0)
        assert len(videos) == 1
        assert videos[0].dim() == 4

        # 3. Extract audio if available
        has_audio = _check_if_video_has_audio(video_path)
        audios, use_audio = extract_audio_from_video_inputs(
            video_path, use_audio_in_video=True
        )

        if has_audio:
            assert (audios is not None and use_audio is True) or (
                audios is None and use_audio is False
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
