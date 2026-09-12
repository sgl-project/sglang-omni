# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

_LOW_FRAME_RATE_STAGES = 3


def fun_asr_low_frame_rate_length(lfr_frames: int) -> int:
    """Historical split-layout audio-embedding length."""
    out = lfr_frames
    for _ in range(_LOW_FRAME_RATE_STAGES):
        out = (out + 1) // 2
    return out


def fun_asr_audio_token_length(lfr_frames: int, *, checkpoint_layout: str) -> int:
    """Number of audio placeholders and embeddings for a checkpoint layout.

    The native HF processor uses one placeholder for each valid LFR frame in
    the flat layout. See transformers/models/fun_asr_nano/processing_fun_asr_nano.py.
    """
    if checkpoint_layout == "flat":
        return lfr_frames
    if checkpoint_layout == "split":
        return fun_asr_low_frame_rate_length(lfr_frames)
    raise ValueError(f"Unknown Fun-ASR checkpoint layout: {checkpoint_layout}")


__all__ = [
    "fun_asr_low_frame_rate_length",
    "fun_asr_audio_token_length",
]
