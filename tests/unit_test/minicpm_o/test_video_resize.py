# SPDX-License-Identifier: Apache-2.0
"""Project-owned contracts for MiniCPM-o video resizing."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor

import pytest
import torch

from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector
from sglang_omni.preprocessing.video import (
    VideoMediaIO,
    resize_video_tensor,
    resize_video_tensor_parallel,
)


def make_video_tensor(
    frame_count: int,
    dtype: torch.dtype,
    non_contiguous: bool,
) -> torch.Tensor:
    source_tensor = torch.arange(frame_count * 3 * 9 * 11, dtype=torch.int64)
    source_tensor = source_tensor.remainder(251).reshape(frame_count, 3, 9, 11)
    source_tensor = source_tensor.to(dtype)
    if dtype.is_floating_point:
        source_tensor = source_tensor / 17.0
    else:
        pass
    if non_contiguous:
        return source_tensor.transpose(2, 3)
    else:
        pass
    return source_tensor


@pytest.mark.parametrize(
    ("dtype", "non_contiguous", "frame_count", "workers"),
    [
        (torch.uint8, False, 8, 8),
        (torch.uint8, True, 5, 8),
        (torch.float32, False, 5, 3),
        (torch.float32, True, 1, 8),
    ],
)
def test_parallel_resize_matches_serial_resize(
    dtype: torch.dtype,
    non_contiguous: bool,
    frame_count: int,
    workers: int,
) -> None:
    video = make_video_tensor(frame_count, dtype, non_contiguous)
    serial_video = resize_video_tensor(video, 7, 5)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parallel_video = resize_video_tensor_parallel(
            video,
            7,
            5,
            executor=executor,
            workers=workers,
        )
    assert torch.equal(serial_video, parallel_video)


def test_url_video_loading_forwards_resize_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = MultiModalResourceConnector()
    captured_configuration: tuple[Executor | None, int] | None = None

    async def fake_load_resource_async(
        resource_url: str,
        media_io: VideoMediaIO,
        *,
        timeout: float,
    ) -> tuple[torch.Tensor, float, None]:
        nonlocal captured_configuration
        captured_configuration = (media_io.resize_executor, media_io.resize_workers)
        return torch.zeros((2, 3, 2, 2)), 1.0, None

    monkeypatch.setattr(connector, "load_resource_async", fake_load_resource_async)
    with ThreadPoolExecutor(max_workers=2) as executor:
        asyncio.run(
            connector.fetch_video_async(
                "https://example.com/clip.mp4",
                resize_executor=executor,
                resize_workers=8,
            )
        )

    assert captured_configuration == (executor, 8)
