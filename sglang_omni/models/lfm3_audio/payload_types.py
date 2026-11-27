# SPDX-License-Identifier: Apache-2.0
"""Pipeline state of LFM3-Audio and LFM3-Omni requests."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

DEFAULT_MAX_NEW_TOKENS = 2048


@dataclass
class Lfm3AudioState(DeclarativeStateBase):
    """Prompt, sampling settings and outputs of one request."""

    input_ids: list[int] | None = None
    # Log-mel features of the input clips, (clips, 128, frames), and their valid-frame mask.
    input_features: torch.Tensor | None = wire(None, codec="typed_tensor")
    feature_attention_mask: torch.Tensor | None = wire(None, codec="typed_tensor")
    # LFM3-Omni images and videos: flattened patches and their (t, h, w) patch grids.
    pixel_values: torch.Tensor | None = wire(None, codec="typed_tensor")
    image_grid_thw: torch.Tensor | None = wire(None, codec="typed_tensor")
    pixel_values_videos: torch.Tensor | None = wire(None, codec="typed_tensor")
    video_grid_thw: torch.Tensor | None = wire(None, codec="typed_tensor")
    text_only: bool = False
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    # Text sampling, greedy unless the request sets it.
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    # Speech-frame sampling, named as in the checkpoint's generation_config; None takes its value.
    audio_temperature: float | None = None
    audio_top_k: int | None = None
    audio_repetition_penalty: float | None = None

    # AR output: text ids with one audio_out per emitted frame, and the frames (frames, 12).
    generated_ids: list[int] = field(default_factory=list)
    # The engine's reason the turn ended: stop, or length at max_new_tokens.
    finish_reason: str | None = None
    audio_codes: torch.Tensor | None = wire(None, codec="typed_tensor")
