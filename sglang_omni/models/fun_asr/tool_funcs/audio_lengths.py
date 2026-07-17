# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

_LFR_N = 6
_LOW_FRAME_RATE_STAGES = 3


def fun_asr_lfr_length(mel_frames: int) -> int:
    return (mel_frames + _LFR_N - 1) // _LFR_N


def fun_asr_low_frame_rate_length(lfr_frames: int) -> int:
    out = lfr_frames
    for _ in range(_LOW_FRAME_RATE_STAGES):
        out = (out + 1) // 2
    return out


def fun_asr_audio_token_lengths(input_lengths: Any) -> torch.Tensor:
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths)
    lfr_lengths = (input_lengths + _LFR_N - 1) // _LFR_N
    tokens = lfr_lengths
    for _ in range(_LOW_FRAME_RATE_STAGES):
        tokens = (tokens + 1) // 2
    return tokens


def fun_asr_num_audio_tokens(num_mel_frames: int) -> int:
    return int(fun_asr_audio_token_lengths(num_mel_frames).item())


__all__ = [
    "fun_asr_lfr_length",
    "fun_asr_low_frame_rate_length",
    "fun_asr_audio_token_lengths",
    "fun_asr_num_audio_tokens",
]
