# SPDX-License-Identifier: Apache-2.0
"""Audio, image and video inputs of an LFM3 chat request, loaded under the server's media policy."""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from PIL import Image
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata

from sglang_omni.preprocessing.base import MediaIO, is_url
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector

# note (ruilin): the API layer answers 400 to errors that carry this prefix.
REQUEST_ERROR_PREFIX = "LFM3 request:"


def part_source(part: Mapping[str, object], kind: str) -> object:
    """The media of a content part: input_audio data, else the kind_url field (an object with a url, or a string), else the kind field."""
    if kind == "audio" and part.get("type") == "input_audio":
        audio = part["input_audio"]
        if not isinstance(audio, Mapping) or "data" not in audio:
            raise ValueError(f"{REQUEST_ERROR_PREFIX} input_audio needs a data field")
        else:
            pass
        return f"data:audio/{audio.get('format', 'wav')};base64,{audio['data']}"
    else:
        value = part.get(f"{kind}_url", part.get(kind, part.get(f"input_{kind}")))
        source = value.get("url") if isinstance(value, Mapping) else value
        if source is None:
            raise ValueError(
                f"{REQUEST_ERROR_PREFIX} a {part.get('type')} content part needs its {kind}"
            )
        else:
            return source


class VideoBytesIO(MediaIO[bytes]):
    """A video's encoded bytes, read through the resource connector's media policy."""

    def load_bytes(self, data: bytes) -> bytes:
        return data

    def load_base64(self, media_type: str, data: str) -> bytes:
        del media_type
        return base64.b64decode(data, validate=True)

    def load_file(self, filepath: Path) -> bytes:
        return filepath.read_bytes()


async def load_video_bytes(source: object) -> bytes:
    """A video URL (http, data or file) or local path, under the server's media policy."""
    connector = get_global_resource_connector()
    if isinstance(source, str) and is_url(source):
        return await connector.load_resource_async(source, VideoBytesIO())
    elif isinstance(source, (str, Path)):
        return connector.load_local_path(source, VideoBytesIO())
    else:
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} unsupported video input {type(source).__name__}"
        )


def load_video_frames(
    video_bytes: bytes, video_processor: BaseVideoProcessor
) -> tuple[np.ndarray, VideoMetadata]:
    """The frames the checkpoint's video processor samples, decoded with PyAV, and their metadata.

    Only the sampled frames are kept; the metadata carries their indices and the clip's fps, from which the processor writes the frame timestamps.
    """
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0) or None
        total_frames = stream.frames or sum(
            1 for packet in container.demux(stream) if packet.size
        )
    if not total_frames:
        raise ValueError(f"{REQUEST_ERROR_PREFIX} no frames in the video")
    else:
        pass
    frame_indices = [
        int(index)
        for index in video_processor.sample_frames(
            VideoMetadata(total_num_frames=total_frames, fps=fps)
        )
    ]
    # note (ruilin): the sampled indices are distinct and increasing, so one pass keeps only them, in order.
    wanted_indices = set(frame_indices)
    decoded_frames: list[np.ndarray] = []
    with av.open(io.BytesIO(video_bytes)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted_indices:
                decoded_frames.append(frame.to_ndarray(format="rgb24"))
            else:
                pass
            if index >= frame_indices[-1]:
                break
            else:
                pass
    if len(decoded_frames) != len(frame_indices):
        raise ValueError(
            f"{REQUEST_ERROR_PREFIX} decoded {len(decoded_frames)} of {len(frame_indices)} sampled video frames"
        )
    else:
        pass
    metadata = VideoMetadata(
        total_num_frames=total_frames, fps=fps, frames_indices=frame_indices
    )
    return np.stack(decoded_frames), metadata


@dataclass(kw_only=True)
class Media:
    """A request's input clips (16 kHz), images and videos, in placeholder order."""

    audio: list[np.ndarray] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    videos: list[np.ndarray] = field(default_factory=list)
    video_metadata: list[VideoMetadata] = field(default_factory=list)

    def processor_inputs(self) -> dict[str, object]:
        inputs: dict[str, object] = {}
        if self.audio:
            inputs["audio"] = self.audio
        else:
            pass
        if self.images:
            inputs["images"] = self.images
        else:
            pass
        if self.videos:
            # note (ruilin): already sampled by load_video_frames, with their indices in the metadata.
            inputs.update(
                videos=self.videos,
                video_metadata=self.video_metadata,
                do_sample_frames=False,
            )
        else:
            pass
        return inputs
