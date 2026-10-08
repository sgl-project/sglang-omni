# SPDX-License-Identifier: Apache-2.0
"""Decode MiniCPM video frames and matching audio intervals on one timeline."""

from __future__ import annotations

import asyncio
import base64
import math
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import audioread
import librosa
import numpy as np
import numpy.typing as npt
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

from sglang_omni.preprocessing.base import MediaIO, is_url
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.preprocessing.video import extract_audio_from_path

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

MAX_VIDEO_FRAMES = 64
AUDIO_SAMPLE_RATE = 16000
MIN_TAIL_AUDIO_SAMPLES = 1600


@dataclass(kw_only=True)
class TimedVideo:
    frames: list[Image.Image] = field(default_factory=list)
    audio_segments: list[npt.NDArray[np.float32]] = field(default_factory=list)
    timestamps_seconds: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0


class MiniCPMVideoIO(MediaIO[TimedVideo]):
    def __init__(
        self,
        *,
        use_audio: bool,
        fps: float | None = None,
        max_frames: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
    ) -> None:
        self.use_audio = use_audio
        self.fps = fps
        self.max_frames = max_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("video_fps must be positive and finite")
        elif any(
            budget is not None and budget <= 0
            for budget in (min_pixels, max_pixels, total_pixels)
        ):
            raise ValueError("Video pixel budgets must be positive")
        else:
            pass

    def load_bytes(self, media_bytes: bytes) -> TimedVideo:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temporary_video:
            temporary_video.write(media_bytes)
            temporary_video.flush()
            return self.load_file(Path(temporary_video.name))

    def load_base64(self, media_type: str, encoded_video: str) -> TimedVideo:
        return self.load_bytes(base64.b64decode(encoded_video, validate=True))

    def load_file(self, video_path: Path) -> TimedVideo:
        if VideoReader is None or cpu is None:
            raise RuntimeError("MiniCPM-o video input requires decord==0.6.0")
        else:
            reader = VideoReader(str(video_path), ctx=cpu(0))
        try:
            source_fps = reader.get_avg_fps()
            duration_seconds = len(reader) / source_fps if source_fps > 0 else 0.0
            if duration_seconds <= 0 or source_fps <= 0:
                raise ValueError("Video input has invalid duration or frame rate")
            else:
                pass
            if self.fps is not None:
                timestamps = np.arange(0, duration_seconds, 1.0 / self.fps).tolist()
            elif duration_seconds > MAX_VIDEO_FRAMES:
                timestamps = [
                    round(index * 0.1, 1)
                    for index in range(int(duration_seconds / 0.1))
                ]
            else:
                timestamps = list(range(math.ceil(duration_seconds)))
            frame_limit = (
                self.max_frames if self.max_frames is not None else MAX_VIDEO_FRAMES
            )
            if frame_limit < 1:
                raise ValueError("video_max_frames must be positive")
            elif len(timestamps) > frame_limit:
                indices = np.linspace(
                    0, len(timestamps) - 1, frame_limit, dtype=int
                ).tolist()
                timestamps = [timestamps[index] for index in indices]
            else:
                pass
            frame_indices = [
                min(int(timestamp * source_fps), len(reader) - 1)
                for timestamp in timestamps
            ]
            pixels = reader.get_batch(frame_indices).asnumpy()
            frames = [Image.fromarray(frame).convert("RGB") for frame in pixels]
        finally:
            del reader
        if self.use_audio:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="PySoundFile failed")
                    waveform, sample_rate = librosa.load(
                        str(video_path), sr=AUDIO_SAMPLE_RATE, mono=True
                    )
            except (audioread.NoBackendError, RuntimeError):
                waveform = extract_audio_from_path(video_path, AUDIO_SAMPLE_RATE)
        else:
            waveform = None
        segments: list[npt.NDArray[np.float32]] = []
        if waveform is not None:
            for index, timestamp in enumerate(timestamps):
                end_seconds = (
                    timestamps[index + 1]
                    if index + 1 < len(timestamps)
                    else duration_seconds
                )
                segment = waveform[
                    int(timestamp * AUDIO_SAMPLE_RATE) : int(
                        end_seconds * AUDIO_SAMPLE_RATE
                    )
                ]
                if (
                    index == len(timestamps) - 1
                    and len(segment) < MIN_TAIL_AUDIO_SAMPLES
                ):
                    segment = np.pad(
                        segment, (0, MIN_TAIL_AUDIO_SAMPLES - len(segment))
                    )
                else:
                    pass
                segments.append(segment.astype(np.float32, copy=False))
        else:
            pass
        return TimedVideo(
            frames=self.resize_frames(frames),
            audio_segments=segments,
            timestamps_seconds=timestamps,
            duration_seconds=duration_seconds,
        )

    def resize_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        if all(
            budget is None
            for budget in (self.min_pixels, self.max_pixels, self.total_pixels)
        ):
            return frames
        else:
            resized: list[Image.Image] = []
            for frame in frames:
                maximum_pixels = self.max_pixels or frame.width * frame.height
                if self.total_pixels is not None:
                    maximum_pixels = min(
                        maximum_pixels, self.total_pixels // len(frames)
                    )
                else:
                    pass
                minimum_pixels = self.min_pixels or min(
                    maximum_pixels, frame.width * frame.height
                )
                if minimum_pixels > maximum_pixels:
                    raise ValueError(
                        "Video minimum pixel budget exceeds maximum budget"
                    )
                else:
                    height, width = smart_resize(
                        frame.height,
                        frame.width,
                        min_pixels=minimum_pixels,
                        max_pixels=maximum_pixels,
                    )
                    resized.append(
                        frame.resize((width, height), Image.Resampling.BICUBIC)
                    )
            return resized


async def load_timed_video(
    source: str,
    *,
    use_audio: bool,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
) -> TimedVideo:
    if fps is not None and fps <= 0:
        raise ValueError("video_fps must be positive")
    else:
        pass
    decoder = MiniCPMVideoIO(
        use_audio=use_audio,
        fps=fps,
        max_frames=max_frames,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
    )
    connector = get_global_resource_connector()
    if is_url(source):
        return await connector.load_resource_async(source, decoder)
    else:
        video_path = Path(connector.local_media_path(source))
        return await asyncio.to_thread(decoder.load_file, video_path)
