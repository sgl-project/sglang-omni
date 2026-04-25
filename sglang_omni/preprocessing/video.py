# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic video preprocessing utilities."""

from __future__ import annotations

import asyncio
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
from .resource_connector import global_thread_pool

logger = logging.getLogger(__name__)


class VideoMediaIO(MediaIO[tuple[torch.Tensor, float, Any | None]]):
    """MediaIO implementation for video files with optional audio extraction."""

    def __init__(
        self,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        image_mode: str = "RGB",
        extract_audio: bool = False,
        audio_target_sr: int = 16000,
        **kwargs,
    ) -> None:
        """Initialize VideoMediaIO.

        Args:
            fps: Target FPS for video loading.
            max_frames: Optional frame cap passed to the video reader backend.
            min_pixels: Optional lower resize budget per frame.
            max_pixels: Optional upper resize budget per frame.
            total_pixels: Optional total video pixel budget.
            image_mode: Target image mode (default: "RGB").
            extract_audio: If True, extract audio from video and return as third element.
            audio_target_sr: Target sample rate for audio extraction (default: 16000).
            **kwargs: Additional arguments (for compatibility with MultiModalResourceConnector).
        """
        super().__init__()
        self.fps = fps
        self.max_frames = max_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.image_mode = image_mode
        self.extract_audio = extract_audio
        self.audio_target_sr = audio_target_sr
        self.kwargs = kwargs

    def _load_path(self, filepath: Path) -> tuple[torch.Tensor, float]:
        return load_video_path(
            filepath,
            fps=self.fps,
            max_frames=self.max_frames,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            total_pixels=self.total_pixels,
        )

    def load_bytes(self, data: bytes) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from raw bytes, optionally extracting audio.

        Returns:
            Tuple of (video_tensor, sample_fps, audio_or_None).
            If extract_audio is False, the third element is None.
        """
        # qwen_vision._read_video_torchvision requires a file path,
        # so we need to write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(data)

        try:
            if self.extract_audio:
                # Load video and extract audio from the same file
                video, sample_fps = self._load_path(tmp_path)
                audio = _extract_audio_from_path(tmp_path, self.audio_target_sr)
                return video, sample_fps, audio
            else:
                video, sample_fps = self._load_path(tmp_path)
                return video, sample_fps, None
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from base64-encoded data, optionally extracting audio."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[torch.Tensor, float, Any | None]:
        """Load video from a local file path, optionally extracting audio."""
        if self.extract_audio:
            # Load video and extract audio from the same file
            video, sample_fps = self._load_path(filepath)
            audio = _extract_audio_from_path(filepath, self.audio_target_sr)
            return video, sample_fps, audio
        else:
            video, sample_fps = self._load_path(filepath)
            return video, sample_fps, None


async def ensure_video_list_async(
    videos: Any,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
    image_mode: str = "RGB",
    resource_connector: Any | None = None,
    extract_audio: bool = False,
    audio_target_sr: int = 16000,
) -> tuple[list[Any], list[float] | None, list[Any] | None]:
    """Asynchronously normalize video inputs into a list.

    Args:
        videos: Video input(s) - can be a path, URL, torch Tensor, or list.
        fps: Target FPS for video loading.
        max_frames: Optional frame cap passed to the video reader backend.
        min_pixels: Optional lower resize budget per frame.
        max_pixels: Optional upper resize budget per frame.
        total_pixels: Optional total video pixel budget.
        image_mode: Target image mode (default: "RGB").
        resource_connector: Optional MultiModalResourceConnector instance. If None, uses
                        the global connector.
        extract_audio: If True, extract audio from videos and return as third element.
        audio_target_sr: Target sample rate for audio extraction (default: 16000).

    Returns:
        Tuple of (normalized video list, sample_fps_list or None, extracted_audio_list or None).
        If extract_audio is False, the third element is None.
    """
    if videos is None:
        return [], None, None
    if isinstance(videos, list):
        items = videos
    else:
        items = [videos]
    normalized: list[Any] = []
    sample_fps_list: list[float] = []
    extracted_audios: list[Any] = [] if extract_audio else []
    all_paths = True

    # Import here to avoid circular dependency
    if resource_connector is None:
        from .resource_connector import get_global_resource_connector

        resource_connector = get_global_resource_connector()

    async def _load_video_with_audio(
        video_item: str | Path, is_url: bool
    ) -> tuple[Any, float, Any | None]:
        """Load video and optionally extract audio."""
        loop = asyncio.get_running_loop()

        if is_url:
            # Use fetch_video_async for URL videos, similar to fetch_image_async
            return await resource_connector.fetch_video_async(
                str(video_item),
                fps=fps,
                max_frames=max_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                total_pixels=total_pixels,
                image_mode=image_mode,
                extract_audio=extract_audio,
                audio_target_sr=audio_target_sr,
            )
        else:
            # Local file path
            video_path = Path(video_item)
            if extract_audio:
                video_task = loop.run_in_executor(
                    global_thread_pool,
                    load_video_path,
                    video_path,
                    fps,
                    max_frames,
                    min_pixels,
                    max_pixels,
                    total_pixels,
                )
                audio_task = loop.run_in_executor(
                    global_thread_pool,
                    _extract_audio_from_path,
                    video_path,
                    audio_target_sr,
                )
                (video, sample_fps), audio = await asyncio.gather(
                    video_task, audio_task
                )
                return video, sample_fps, audio
            else:
                video, sample_fps = await loop.run_in_executor(
                    global_thread_pool,
                    load_video_path,
                    video_path,
                    fps,
                    max_frames,
                    min_pixels,
                    max_pixels,
                    total_pixels,
                )
                return video, sample_fps, None

    # Collect coroutines for URL and local file items
    coroutines: list[asyncio.Task[tuple[Any, float, Any | None]] | None] = []
    url_indices: list[int] = []

    # First pass: identify items that need loading
    for idx, video_item in enumerate(items):
        if isinstance(video_item, (str, Path)):
            if _is_url(video_item):
                # Create coroutine for async URL fetching with optional audio extraction
                coro = _load_video_with_audio(video_item, is_url=True)
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder for video
                sample_fps_list.append(0.0)  # Placeholder for fps
                if extract_audio:
                    extracted_audios.append(None)  # Placeholder for audio
            elif Path(video_item).exists():
                # Load from local path with optional audio extraction
                coro = _load_video_with_audio(video_item, is_url=False)
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder for video
                sample_fps_list.append(0.0)  # Placeholder for fps
                if extract_audio:
                    extracted_audios.append(None)  # Placeholder for audio
            else:
                # Path doesn't exist, treat as already processed
                normalized.append(video_item)
                all_paths = False
                if extract_audio:
                    extracted_audios.append(None)
        else:
            # Already processed (torch Tensor, etc.)
            normalized.append(video_item)
            all_paths = False
            if extract_audio:
                extracted_audios.append(None)

    # Wait for all loads to complete
    if coroutines:
        results = await asyncio.gather(*coroutines)
        # Fill in the results at the correct indices
        for url_idx, (video, sample_fps, audio) in zip(url_indices, results):
            normalized[url_idx] = video
            sample_fps_list[url_idx] = sample_fps
            if extract_audio:
                extracted_audios[url_idx] = audio

    if all_paths:
        return (
            normalized,
            sample_fps_list,
            extracted_audios if extract_audio else None,
        )
    return normalized, None, extracted_audios if extract_audio else None


def _extract_audio_from_path(video_path: Path, target_sr: int) -> Any | None:
    """Extract audio from a video file path."""
    if not _check_if_video_has_audio(video_path):
        return None
    try:
        audio, _ = librosa.load(str(video_path), sr=target_sr)
        return audio
    except Exception as e:
        logger.debug(f"Failed to extract audio from {video_path}: {e}")
        return None


def load_video_path(
    path: str | Path,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Load a local video into a torch tensor (T, C, H, W) on CPU."""
    ele: dict[str, Any] = {"video": str(path)}
    if fps is not None:
        ele["fps"] = float(fps)
    if max_frames is not None:
        ele["max_frames"] = int(max_frames)
    if min_pixels is not None:
        ele["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        ele["max_pixels"] = int(max_pixels)
    if total_pixels is not None:
        ele["total_pixels"] = int(total_pixels)
    backend = qwen_vision.get_video_reader_backend()
    try:
        video, sample_fps = qwen_vision.VIDEO_READER_BACKENDS[backend](ele)
    except Exception:
        logger.warning("Video reader %s failed, falling back to torchvision", backend)
        video, sample_fps = qwen_vision.VIDEO_READER_BACKENDS["torchvision"](ele)
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
