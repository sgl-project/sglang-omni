# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic video frontend utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import av
import librosa
import torch
from qwen_omni_utils.v2_5 import vision_process as qwen_vision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_f

from .cache_key import compute_media_cache_key

logger = logging.getLogger(__name__)


def ensure_video_list(
    videos: Any,
    *,
    fps: float | None = None,
) -> tuple[list[Any], list[float] | None]:
    """Normalize video inputs into a list, sampling like qwen_omni_utils."""
    if videos is None:
        return [], None
    if isinstance(videos, list):
        items = videos
    else:
        items = [videos]
    normalized: list[Any] = []
    sample_fps_list: list[float] = []
    all_paths = True
    for item in items:
        if isinstance(item, (str, Path)) and Path(item).exists():
            video, sample_fps = load_video_path(item, fps=fps)
            normalized.append(video)
            sample_fps_list.append(sample_fps)
        else:
            normalized.append(item)
            all_paths = False
    if all_paths:
        return normalized, sample_fps_list
    return normalized, None


def load_video_path(
    path: str | Path,
    *,
    fps: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Load a local video into a torch tensor (T, C, H, W) on CPU."""
    ele: dict[str, Any] = {"video": str(path)}
    if fps is not None:
        ele["fps"] = float(fps)
    video, sample_fps = qwen_vision._read_video_torchvision(ele)
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", qwen_vision.VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", qwen_vision.VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(
            qwen_vision.VIDEO_MAX_PIXELS,
            total_pixels / nframes * qwen_vision.FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = qwen_vision.smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=qwen_vision.IMAGE_FACTOR,
        )
    else:
        resized_height, resized_width = qwen_vision.smart_resize(
            height,
            width,
            factor=qwen_vision.IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = tv_f.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video, sample_fps


def build_video_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard video tensors from HF processor outputs.

    This is a placeholder schema; refine it after more models are integrated.
    """
    return {
        "pixel_values_videos": hf_inputs.get("pixel_values_videos"),
        "video_grid_thw": hf_inputs.get("video_grid_thw"),
        "video_second_per_grid": hf_inputs.get("video_second_per_grid"),
    }


def compute_video_cache_key(videos: Any) -> str | None:
    """Compute cache key from raw video inputs (paths, URLs, tensors).

    This should be called BEFORE ensure_video_list() to capture original
    paths/URLs which are much cheaper to hash than decoded frames.
    """
    return compute_media_cache_key(videos, prefix="video")


def _check_if_video_has_audio(video_path: str | Path) -> bool:
    """Check if a video file has an audio track."""
    try:
        container = av.open(str(video_path))
        audio_streams = [
            stream for stream in container.streams if stream.type == "audio"
        ]
        container.close()
        return len(audio_streams) > 0
    except Exception as e:
        logger.debug(f"Failed to check audio in video {video_path}: {e}")
        return False


def extract_audio_from_video_inputs(
    videos: Any, *, use_audio_in_video: bool, target_sr: int = 16000
) -> tuple[list[Any] | None, bool]:
    """Extract audio from local video paths when use_audio_in_video=True.

    Returns (audio_list, use_audio_in_video) where:
    - audio_list: List of numpy arrays (one per video) if extraction succeeds, None otherwise
    - use_audio_in_video: Updated flag (False if extraction fails or videos are not local paths)

    This follows the logic from qwen_omni_utils.v2_5.audio_process.process_audio_info.
    """
    if not use_audio_in_video or not videos:
        return None, use_audio_in_video

    # Normalize to list
    if isinstance(videos, list):
        video_items = videos
    else:
        video_items = [videos]

    extracted_audios: list[Any] = []
    for item in video_items:
        # Only extract from local file paths
        if not isinstance(item, (str, Path)):
            logger.warning(
                f"Video item {item} is not a local path, cannot extract audio. "
                "Falling back to use_audio_in_video=False"
            )
            return None, False

        video_path = Path(item)
        if not video_path.exists():
            logger.warning(
                f"Video path {video_path} does not exist, cannot extract audio. "
                "Falling back to use_audio_in_video=False"
            )
            return None, False

        # Check if video has audio track
        if not _check_if_video_has_audio(video_path):
            logger.warning(
                f"Video {video_path} has no audio track. "
                "Falling back to use_audio_in_video=False"
            )
            return None, False

        # Extract audio using librosa (same as qwen_omni_utils)
        try:
            audio, _ = librosa.load(str(video_path), sr=target_sr)
            extracted_audios.append(audio)
        except Exception as e:
            logger.warning(
                f"Failed to extract audio from video {video_path}: {e}. "
                "Falling back to use_audio_in_video=False"
            )
            return None, False

    if not extracted_audios:
        return None, False

    return extracted_audios, True
