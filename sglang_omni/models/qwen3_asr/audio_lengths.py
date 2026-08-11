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
    input_lengths_leave = num_mel_frames % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (num_mel_frames // 100) * 13


# Longest clip the model natively accepts, per the official wrapper's
# MAX_ASR_INPUT_SECONDS (QwenLM/Qwen3-ASR).
QWEN3_ASR_MAX_INPUT_SECONDS = 1200

# Mel frames per second of audio (16 kHz, hop 160).
_MEL_FRAMES_PER_SECOND = 100


def qwen3_asr_max_audio_tokens() -> int:
    """Audio tokens of the longest natively supported clip (15,600 for 1,200s)."""
    return qwen3_asr_num_audio_tokens(
        QWEN3_ASR_MAX_INPUT_SECONDS * _MEL_FRAMES_PER_SECOND
    )


# Note(Jeffro): Speech produces 3-5 output tokens per audio second (I measured ~3.3 on SeedTTS EN).
# We budget 10 here, roughly 3x that, to cover faster/denser speech.
QWEN3_ASR_OUTPUT_TOKENS_PER_SECOND = 10


def qwen3_asr_max_output_tokens() -> int:
    """Output-token budget of the longest natively supported clip (12,000)."""
    return QWEN3_ASR_MAX_INPUT_SECONDS * QWEN3_ASR_OUTPUT_TOKENS_PER_SECOND


__all__ = [
    "QWEN3_ASR_MAX_INPUT_SECONDS",
    "QWEN3_ASR_OUTPUT_TOKENS_PER_SECOND",
    "qwen3_asr_audio_token_lengths",
    "qwen3_asr_max_audio_tokens",
    "qwen3_asr_max_output_tokens",
    "qwen3_asr_num_audio_tokens",
]
