# SPDX-License-Identifier: Apache-2.0
"""Delay-pattern helpers for MOSS-TTS audio codes."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.utils.codec_delay import reverse_delay_pattern


def split_moss_audio_segments(
    delayed_audio_codes: Any,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> list[torch.Tensor]:
    """Extract contiguous decoded audio-code segments from delayed rows."""

    if delayed_audio_codes is None:
        return []
    if not isinstance(delayed_audio_codes, torch.Tensor):
        delayed_audio_codes = torch.as_tensor(delayed_audio_codes, dtype=torch.long)
    delayed_audio_codes = delayed_audio_codes.to(dtype=torch.long)
    if delayed_audio_codes.numel() == 0:
        return []

    audio_codes = reverse_delay_pattern(delayed_audio_codes, allow_short=True)
    if audio_codes.numel() == 0:
        return []

    pad_code = int(audio_pad_code)
    is_pad = (audio_codes == pad_code).all(dim=1)
    is_complete_code = ((audio_codes >= 0) & (audio_codes < pad_code)).all(dim=1)
    non_pad = (~is_pad) & is_complete_code
    if not bool(non_pad.any()):
        return []

    idx = torch.nonzero(non_pad, as_tuple=False).squeeze(1)
    break_points = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    if break_points.numel() == 0:
        segments = [idx]
    else:
        segments = list(torch.tensor_split(idx, break_points.cpu().tolist()))

    code_segments = [audio_codes[segment].contiguous() for segment in segments]
    if assistant_start_length > 0 and code_segments:
        trim = min(int(assistant_start_length), int(code_segments[0].shape[0]))
        code_segments[0] = code_segments[0][trim:]
        code_segments = [segment for segment in code_segments if segment.numel() > 0]
    return code_segments
