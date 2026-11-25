# SPDX-License-Identifier: Apache-2.0
"""CPU tests of LFM3's media loading: content-part sources, the video media policy and sampled video frames."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import av
import numpy as np
import pytest
from transformers.video_utils import VideoMetadata

from sglang_omni.models.lfm3_audio import media
from sglang_omni.preprocessing.resource_connector import (
    MediaPolicyError,
    MultiModalResourceConnector,
    resource_connector_scope,
)


@pytest.mark.parametrize(
    ("part", "kind", "source"),
    [
        (
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}},
            "audio",
            "data:audio/mp3;base64,AAAA",
        ),
        ({"type": "image_url", "image_url": {"url": "a.png"}}, "image", "a.png"),
        ({"type": "video_url", "video_url": "b.mp4"}, "video", "b.mp4"),
        ({"type": "image", "image": "c.png"}, "image", "c.png"),
    ],
)
def test_content_part_sources(part: dict[str, object], kind: str, source: str) -> None:
    assert media.part_source(part, kind) == source


def test_content_part_without_media_is_a_request_error() -> None:
    with pytest.raises(ValueError, match="needs its video"):
        media.part_source({"type": "video_url"}, "video")


def test_local_video_paths_follow_the_media_policy(tmp_path: Path) -> None:
    """A bare video path outside allowed_local_media_path is refused before it is read."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"not a video")
    connector = MultiModalResourceConnector(allowed_local_media_path=str(allowed))
    with resource_connector_scope(connector), pytest.raises(MediaPolicyError):
        asyncio.run(media.load_video_bytes(str(outside)))


def test_video_frames_are_the_sampled_ones_in_order() -> None:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(10):
            image = np.full((32, 32, 3), index * 25, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class EveryThirdFrame:
        def sample_frames(self, metadata: VideoMetadata) -> list[int]:
            return list(range(0, metadata.total_num_frames, 3))

    frames, metadata = media.load_video_frames(buffer.getvalue(), EveryThirdFrame())
    assert metadata.total_num_frames == 10 and metadata.fps == 8
    assert metadata.frames_indices == [0, 3, 6, 9]
    brightness = [int(frame.mean().round()) for frame in frames]
    assert brightness == pytest.approx([0, 75, 150, 225], abs=4)
