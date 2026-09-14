# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

_LOW_FRAME_RATE_STAGES = 3


def fun_asr_low_frame_rate_length(lfr_frames: int) -> int:
    out = lfr_frames
    for _ in range(_LOW_FRAME_RATE_STAGES):
        out = (out + 1) // 2
    return out


def fun_asr_num_audio_tokens(
    num_samples: int,
    *,
    frame_length_samples: int,
    frame_shift_samples: int,
    lfr_n: int,
) -> int:
    """Derive Fun-ASR adaptor tokens from a resampled waveform length."""
    values = {
        "num_samples": num_samples,
        "frame_length_samples": frame_length_samples,
        "frame_shift_samples": frame_shift_samples,
        "lfr_n": lfr_n,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    mel_frames = (
        1
        if num_samples < frame_length_samples
        else 1 + (num_samples - frame_length_samples) // frame_shift_samples
    )
    lfr_frames = (mel_frames + lfr_n - 1) // lfr_n
    return fun_asr_low_frame_rate_length(lfr_frames)


__all__ = [
    "fun_asr_low_frame_rate_length",
    "fun_asr_num_audio_tokens",
]
