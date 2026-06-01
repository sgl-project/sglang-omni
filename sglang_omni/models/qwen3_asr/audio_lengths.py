# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen3-ASR audio length helpers."""

from __future__ import annotations

from typing import Any

import torch


def qwen3_asr_audio_token_lengths(input_lengths: Any) -> torch.Tensor:
    """Return Qwen3-ASR encoder output lengths for mel-frame lengths.

    This mirrors the upstream audio encoder shape math: 100-frame windows emit
    13 tokens each, and the remainder passes through three stride-2 downsamples.
    """
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths)
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def qwen3_asr_num_audio_tokens(num_mel_frames: int) -> int:
    """Scalar wrapper for scheduler request construction."""
    return int(qwen3_asr_audio_token_lengths(num_mel_frames).item())


__all__ = [
    "qwen3_asr_audio_token_lengths",
    "qwen3_asr_num_audio_tokens",
]
