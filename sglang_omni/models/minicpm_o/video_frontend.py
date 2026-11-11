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
from pydantic import BaseModel
from qwen_vl_utils.vision_process import smart_resize
from sglang.utils import LazyImport

from sglang_omni.preprocessing.base import MediaIO, is_url
from sglang_omni.preprocessing.resource_connector import (
    await_media_cleanup,
    get_global_resource_connector,
    global_thread_pool,
)
from sglang_omni.preprocessing.video import (
    DECODED_FRAME_BYTES_PER_PIXEL,
    DEVICE_DECODE_LOCK,
    VideoDecodeError,
    VideoDecoder,
    device_decode_frames_per_batch,
    extract_audio_from_path,
    is_invalid_video,
    warn_device_decode_fallback,
)

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

current_platform: LazyImport = LazyImport("sglang_omni.platforms", "current_platform")

MAX_VIDEO_FRAMES = 64
AUDIO_SAMPLE_RATE = 16000
MIN_TAIL_AUDIO_SAMPLES = 1600
LONG_VIDEO_SAMPLE_INTERVAL_SECONDS = 0.1


class VideoProcessingOptions(BaseModel):
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None


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
        self.use_audio: bool = use_audio
        self.frames_per_second: float | None = fps
        self.maximum_frame_count: int | None = max_frames
        self.minimum_pixels_per_frame: int | None = min_pixels
        self.maximum_pixels_per_frame: int | None = max_pixels
        self.total_pixel_budget: int | None = total_pixels
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

    def sample_timeline(
        self, source_frame_count: int, source_frames_per_second: float
    ) -> tuple[list[float], list[int], float]:
        """Sampled timestamps, the source frame index of each, and the clip duration in seconds."""
        duration_seconds = (
            source_frame_count / source_frames_per_second
            if source_frames_per_second > 0
            else 0.0
        )
        if duration_seconds <= 0 or source_frames_per_second <= 0:
            raise ValueError("Video input has invalid duration or frame rate")
        else:
            pass
        if self.frames_per_second is not None:
            sample_interval_seconds = 1.0 / self.frames_per_second
            candidate_frame_count = math.ceil(
                duration_seconds / sample_interval_seconds
            )
        elif duration_seconds > MAX_VIDEO_FRAMES:
            sample_interval_seconds = LONG_VIDEO_SAMPLE_INTERVAL_SECONDS
            candidate_frame_count = int(duration_seconds / sample_interval_seconds)
        else:
            sample_interval_seconds = 1.0
            candidate_frame_count = math.ceil(duration_seconds)
        maximum_frame_count = (
            self.maximum_frame_count
            if self.maximum_frame_count is not None
            else MAX_VIDEO_FRAMES
        )
        if maximum_frame_count < 1:
            raise ValueError("video_max_frames must be positive")
        elif candidate_frame_count > maximum_frame_count:
            sample_indices = [
                index * (candidate_frame_count - 1) // max(maximum_frame_count - 1, 1)
                for index in range(maximum_frame_count)
            ]
        else:
            sample_indices = range(candidate_frame_count)
        timestamps_seconds = [
            index * sample_interval_seconds for index in sample_indices
        ]
        if self.frames_per_second is None and duration_seconds > MAX_VIDEO_FRAMES:
            timestamps_seconds = [round(seconds, 1) for seconds in timestamps_seconds]
        else:
            pass
        frame_indices = [
            min(
                int(timestamp_seconds * source_frames_per_second),
                source_frame_count - 1,
            )
            for timestamp_seconds in timestamps_seconds
        ]
        return timestamps_seconds, frame_indices, duration_seconds

    def read_frames_with_decord(
        self, video_path: Path
    ) -> tuple[list[Image.Image], list[float], float]:
        if VideoReader is None or cpu is None:
            raise RuntimeError(
                "MiniCPM-o timed video input requires decord==0.6.0; "
                "its prebuilt Linux wheel supports x86_64 only"
            )
        else:
            video_reader = VideoReader(str(video_path), ctx=cpu(0))
        try:
            timestamps_seconds, frame_indices, duration_seconds = self.sample_timeline(
                len(video_reader), video_reader.get_avg_fps()
            )
            frame_pixels = video_reader.get_batch(frame_indices).asnumpy()
        finally:
            del video_reader
        frames = [Image.fromarray(frame).convert("RGB") for frame in frame_pixels]
        return frames, timestamps_seconds, duration_seconds

    def read_frames_on_device(
        self, video_path: Path, decode_device: str
    ) -> tuple[list[Image.Image], list[float], float]:
        with DEVICE_DECODE_LOCK:
            decoder = VideoDecoder(
                str(video_path), device=decode_device, dimension_order="NHWC"
            )
            timestamps_seconds, frame_indices, duration_seconds = self.sample_timeline(
                decoder.metadata.num_frames, decoder.metadata.average_fps
            )
            frames_per_batch = device_decode_frames_per_batch(
                decoder, DECODED_FRAME_BYTES_PER_PIXEL
            )
            frame_pixels = [
                decoder.get_frames_at(
                    indices=frame_indices[batch_start : batch_start + frames_per_batch]
                )
                .data.cpu()
                .numpy()
                for batch_start in range(0, len(frame_indices), frames_per_batch)
            ]
        frames = [
            Image.fromarray(frame).convert("RGB")
            for batch in frame_pixels
            for frame in batch
        ]
        return frames, timestamps_seconds, duration_seconds

    def load_file(self, video_path: Path) -> TimedVideo:
        decode_device = current_platform.get_video_decode_device()
        if decode_device is None or VideoDecoder is None:
            frames, timestamps_seconds, duration_seconds = self.read_frames_with_decord(
                video_path
            )
        else:
            try:
                frames, timestamps_seconds, duration_seconds = (
                    self.read_frames_on_device(video_path, decode_device)
                )
            except RuntimeError as device_exc:
                if VideoReader is None and is_invalid_video(video_path, device_exc):
                    raise VideoDecodeError(
                        f"Invalid media data while decoding video path={video_path}: {device_exc}"
                    ) from device_exc
                elif VideoReader is None:
                    raise
                else:
                    # note (Rahul): decord samples the same timeline and reports invalid media itself.
                    warn_device_decode_fallback(decode_device, device_exc)
                    frames, timestamps_seconds, duration_seconds = (
                        self.read_frames_with_decord(video_path)
                    )
            finally:
                # note (Rahul): the caching allocator would otherwise keep the decode's peak reserved on the card.
                current_platform.empty_cache()
        if self.use_audio:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="PySoundFile failed")
                    audio_waveform, sample_rate = librosa.load(
                        str(video_path), sr=AUDIO_SAMPLE_RATE, mono=True
                    )
            except (audioread.NoBackendError, RuntimeError):
                audio_waveform = extract_audio_from_path(video_path, AUDIO_SAMPLE_RATE)
        else:
            audio_waveform = None
        audio_segments: list[npt.NDArray[np.float32]] = []
        if audio_waveform is not None:
            for index, timestamp_seconds in enumerate(timestamps_seconds):
                end_seconds = (
                    timestamps_seconds[index + 1]
                    if index + 1 < len(timestamps_seconds)
                    else duration_seconds
                )
                audio_segment = audio_waveform[
                    int(timestamp_seconds * AUDIO_SAMPLE_RATE) : int(
                        end_seconds * AUDIO_SAMPLE_RATE
                    )
                ]
                if (
                    index == len(timestamps_seconds) - 1
                    and len(audio_segment) < MIN_TAIL_AUDIO_SAMPLES
                ):
                    audio_segment = np.pad(
                        audio_segment, (0, MIN_TAIL_AUDIO_SAMPLES - len(audio_segment))
                    )
                else:
                    pass
                audio_segments.append(audio_segment.astype(np.float32, copy=False))
        else:
            pass
        return TimedVideo(
            frames=self.resize_frames(frames),
            audio_segments=audio_segments,
            timestamps_seconds=timestamps_seconds,
            duration_seconds=duration_seconds,
        )

    def resize_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        if all(
            budget is None
            for budget in (
                self.minimum_pixels_per_frame,
                self.maximum_pixels_per_frame,
                self.total_pixel_budget,
            )
        ):
            return frames
        else:
            resized_frames: list[Image.Image] = []
            for frame in frames:
                maximum_pixels = (
                    self.maximum_pixels_per_frame or frame.width * frame.height
                )
                if self.total_pixel_budget is not None:
                    maximum_pixels = min(
                        maximum_pixels, self.total_pixel_budget // len(frames)
                    )
                else:
                    pass
                minimum_pixels = self.minimum_pixels_per_frame or min(
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
                    resized_frames.append(
                        frame.resize((width, height), Image.Resampling.BICUBIC)
                    )
            return resized_frames


async def load_timed_video(
    media_url: str,
    *,
    use_audio: bool,
    fps: float | None = None,
    max_frames: int | None = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    total_pixels: int | None = None,
) -> TimedVideo:
    video_decoder = MiniCPMVideoIO(
        use_audio=use_audio,
        fps=fps,
        max_frames=max_frames,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
    )
    resource_connector = get_global_resource_connector()
    if is_url(media_url):
        return await resource_connector.load_resource_async(media_url, video_decoder)
    else:
        video_path = Path(resource_connector.local_media_path(media_url))
        decode_future = asyncio.get_running_loop().run_in_executor(
            global_thread_pool, video_decoder.load_file, video_path
        )

        async def cleanup_video_decoder() -> None:
            await asyncio.gather(decode_future, return_exceptions=True)

        try:
            return await asyncio.shield(decode_future)
        finally:
            await await_media_cleanup(cleanup_video_decoder())
