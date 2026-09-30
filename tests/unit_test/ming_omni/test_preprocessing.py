# SPDX-License-Identifier: Apache-2.0
"""Video preprocessing contracts for Ming-Omni against the pinned Transformers."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.ming_omni.components import preprocessor

TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
FRAME_HEIGHT = 224
FRAME_WIDTH = 224


class FakeVisionConfig:
    """Vision geometry the video processor is constructed from."""

    patch_size = PATCH_SIZE
    temporal_patch_size = TEMPORAL_PATCH_SIZE
    spatial_merge_size = SPATIAL_MERGE_SIZE


class FakeMingConfig:
    """Config surface the constructor reads before any download happens."""

    audio_config = None
    vision_config = FakeVisionConfig()


class FakeMingTokenizer:
    """Token ids the constructor resolves for its modality placeholders."""

    def convert_tokens_to_ids(self, token: str) -> int:
        return len(token)


def fake_load_ming_config(model_path: str) -> FakeMingConfig:
    return FakeMingConfig()


def fake_load_ming_tokenizer(model_path: str) -> FakeMingTokenizer:
    return FakeMingTokenizer()


def build_preprocessor(monkeypatch: pytest.MonkeyPatch) -> preprocessor.MingPreprocessor:
    """Build a real preprocessor with only the checkpoint loaders stubbed out."""
    # note (WinnieSmasher): checkpoint loading is stubbed but the processor lookup
    # runs for real, so a renamed Transformers symbol fails here instead of at
    # server startup.
    monkeypatch.setattr(preprocessor, "load_ming_config", fake_load_ming_config)
    monkeypatch.setattr(preprocessor, "load_ming_tokenizer", fake_load_ming_tokenizer)
    return preprocessor.MingPreprocessor("unused-model-path")


def make_frame_stack(frame_count: int) -> torch.Tensor:
    """A (frame_count, channel, height, width) float stack in 0..255."""
    return torch.randint(0, 256, (frame_count, 3, FRAME_HEIGHT, FRAME_WIDTH)).float()


def test_video_grid_accounts_for_every_input_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every decoded frame must reach the grid without being dropped or re-sampled."""
    frames = make_frame_stack(frame_count=8)

    _, video_grid_thw, _ = build_preprocessor(monkeypatch).process_videos([frames])

    assert video_grid_thw.shape == (1, 3)
    assert int(video_grid_thw[0][0]) == frames.shape[0] // TEMPORAL_PATCH_SIZE


def test_video_token_count_matches_the_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The placeholder count must match the patch count or videoPatch desyncs."""
    frames = make_frame_stack(frame_count=8)

    pixel_values_videos, video_grid_thw, token_counts = build_preprocessor(
        monkeypatch
    ).process_videos([frames])

    grid_time, grid_height, grid_width = (int(value) for value in video_grid_thw[0])
    patch_count = grid_time * grid_height * grid_width
    assert token_counts == [patch_count // (SPATIAL_MERGE_SIZE**2)]
    assert pixel_values_videos.shape[0] == patch_count


def test_out_of_range_frames_leave_the_caller_tensor_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames are clamped before the uint8 cast; the caller's tensor must survive."""
    frames = make_frame_stack(frame_count=4)
    # note (WinnieSmasher): bicubic resampling overshoots past 0..255, so
    # out-of-range values reach this path on real inputs rather than only in theory.
    frames[0, 0, 0, 0] = 300.0
    frames[1, 1, 1, 1] = -20.0
    frames_before = frames.clone()

    build_preprocessor(monkeypatch).process_videos([frames])

    assert torch.equal(frames, frames_before)
