# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from sglang_omni.preprocessing.video import (
    _qwen_video_resize_defaults,
    _unpack_qwen_video_reader_result,
)


def test_unpack_qwen_video_reader_two_value_result() -> None:
    video = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)

    unpacked, sample_fps = _unpack_qwen_video_reader_result((video, 2))

    assert unpacked is video
    assert sample_fps == 2.0


def test_unpack_qwen_video_reader_three_value_result() -> None:
    video = torch.zeros((1, 3, 4, 4), dtype=torch.uint8)

    unpacked, sample_fps = _unpack_qwen_video_reader_result(
        (video, {"video_backend": "torchvision"}, 1.5)
    )

    assert unpacked is video
    assert sample_fps == 1.5


def test_unpack_qwen_video_reader_rejects_unexpected_arity() -> None:
    with pytest.raises(ValueError, match="returned 1 values"):
        _unpack_qwen_video_reader_result((torch.empty(0),))


def test_qwen_video_resize_defaults_are_positive() -> None:
    image_factor, min_pixels, total_pixels, max_pixels = _qwen_video_resize_defaults()

    assert image_factor > 0
    assert min_pixels > 0
    assert total_pixels > 0
    assert max_pixels >= min_pixels
