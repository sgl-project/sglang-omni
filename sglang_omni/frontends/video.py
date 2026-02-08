# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic video frontend utilities."""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

import av
import librosa
import torch
from qwen_vl_utils import vision_process as qwen_vision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_f

from .base import MediaIO, _is_url
from .cache_key import compute_media_cache_key

logger = logging.getLogger(__name__)


class VideoMediaIO(MediaIO[tuple[torch.Tensor, float]]):
    """MediaIO implementation for video files."""

    def __init__(
        self,
        *,
        fps: float | None = None,
        image_mode: str = "RGB",
        **kwargs,
    ) -> None:
        """Initialize VideoMediaIO.

        Args:
            fps: Target FPS for video loading.
            image_mode: Target image mode (default: "RGB").
            **kwargs: Additional arguments (for compatibility with MediaConnector).
        """
        super().__init__()
        self.fps = fps
        self.image_mode = image_mode
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[torch.Tensor, float]:
        """Load video from raw bytes."""
        # qwen_vision._read_video_torchvision requires a file path,
        # so we need to write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(data)

        try:
            return load_video_path(tmp_path, fps=self.fps)
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[torch.Tensor, float]:
        """Load video from base64-encoded data."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[torch.Tensor, float]:
        """Load video from a local file path."""
        return load_video_path(filepath, fps=self.fps)


def ensure_video_list(
    videos: Any,
    *,
    fps: float | None = None,
    image_mode: str = "RGB",
    media_connector: Any | None = None,
) -> tuple[list[Any], list[float] | None]:
    """Normalize video inputs into a list.

    Supports local file paths, URLs (HTTP/HTTPS, data URLs, file URLs),
    and torch Tensors.

    Args:
        videos: Video input(s) - can be a path, URL, torch Tensor, or list.
        fps: Target FPS for video loading.
        image_mode: Target image mode (default: "RGB").
        media_connector: Optional MediaConnector instance. If None, uses
                        the global connector.

    Returns:
        Tuple of (normalized video list, sample_fps_list or None).
    """
    if videos is None:
        return [], None
    if isinstance(videos, list):
        items = videos
    else:
        items = [videos]
    normalized: list[Any] = []
    sample_fps_list: list[float] = []
    all_paths = True

    # Import here to avoid circular dependency
    if media_connector is None:
        from .media_connector import get_global_media_connector

        media_connector = get_global_media_connector()

    for video_item in items:
        if isinstance(video_item, (str, Path)):
            if _is_url(video_item):
                # Load from URL
                video, sample_fps = media_connector.fetch_video(
                    str(video_item), fps=fps, image_mode=image_mode
                )
                normalized.append(video)
                sample_fps_list.append(sample_fps)
            elif Path(video_item).exists():
                # Load from local path
                video, sample_fps = load_video_path(video_item, fps=fps)
                normalized.append(video)
                sample_fps_list.append(sample_fps)
            else:
                # Path doesn't exist, treat as already processed
                normalized.append(video_item)
                all_paths = False
        else:
            # Already processed (torch Tensor, etc.)
            normalized.append(video_item)
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
    return {
        "pixel_values_videos": hf_inputs.get("pixel_values_videos"),
        "video_grid_thw": hf_inputs.get("video_grid_thw"),
        "video_second_per_grid": hf_inputs.get("video_second_per_grid"),
    }


def compute_video_cache_key(videos: Any) -> str | None:
    return compute_media_cache_key(videos, prefix="video")


def _check_if_video_has_audio(video_path: str | Path) -> bool:
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
    if not use_audio_in_video or not videos:
        return None, use_audio_in_video

    def _disable(reason: str) -> tuple[list[Any] | None, bool]:
        logger.warning(f"{reason} Falling back to use_audio_in_video=False")
        return None, False

    # Normalize to list
    video_items = videos if isinstance(videos, list) else [videos]

    extracted_audios: list[Any] = []
    for item in video_items:
        if not isinstance(item, (str, Path)):
            return _disable(
                f"Video item {item} is not a local path, cannot extract audio."
            )

        video_path = Path(item)
        if not video_path.exists():
            return _disable(
                f"Video path {video_path} does not exist, cannot extract audio."
            )

        if not _check_if_video_has_audio(video_path):
            return _disable(f"Video {video_path} has no audio track.")

        try:
            audio, _ = librosa.load(str(video_path), sr=target_sr)
            extracted_audios.append(audio)
        except Exception as e:
            return _disable(
                f"Failed to extract audio from {video_path}: {e}. "
                "This may require ffmpeg to be installed."
            )

    return (extracted_audios, True) if extracted_audios else (None, False)
