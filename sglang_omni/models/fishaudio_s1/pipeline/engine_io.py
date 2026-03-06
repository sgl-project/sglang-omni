# SPDX-License-Identifier: Apache-2.0
"""Engine request/result helpers for the FishAudio-S1 TTS stage."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.fishaudio_s1.io import FishAudioState
from sglang_omni.models.fishaudio_s1.runtime.dual_ar import DualARRequestData


def build_tts_request(state: FishAudioState) -> DualARRequestData:
    """Convert pipeline state into a DualARRequestData for the engine."""
    input_values = state.input_values
    if not isinstance(input_values, torch.Tensor):
        input_values = torch.tensor(input_values)

    audio_masks = state.audio_masks
    if audio_masks is not None and not isinstance(audio_masks, torch.Tensor):
        audio_masks = torch.tensor(audio_masks)

    audio_parts = state.audio_parts
    if audio_parts is not None and not isinstance(audio_parts, torch.Tensor):
        audio_parts = torch.tensor(audio_parts)

    return DualARRequestData(
        input_values=input_values,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        num_codebooks=state.num_codebooks,
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        repetition_penalty=state.repetition_penalty,
    )


def apply_tts_result(state: FishAudioState, result: Any) -> None:
    """Extract output codes from engine result and store in pipeline state."""
    if isinstance(result, DualARRequestData):
        if result.output_codes:
            all_codes = torch.cat(result.output_codes, dim=1)
            state.output_codes = all_codes
        else:
            state.output_codes = None
    else:
        state.output_codes = result
