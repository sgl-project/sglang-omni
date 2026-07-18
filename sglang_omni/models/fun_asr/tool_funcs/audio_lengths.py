# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

_LOW_FRAME_RATE_STAGES = 3


def fun_asr_low_frame_rate_length(lfr_frames: int) -> int:
    out = lfr_frames
    for _ in range(_LOW_FRAME_RATE_STAGES):
        out = (out + 1) // 2
    return out


__all__ = [
    "fun_asr_low_frame_rate_length",
]
