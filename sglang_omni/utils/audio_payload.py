# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for final audio waveform payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def audio_waveform_payload(
    audio: Any,
    *,
    sample_rate: int | None = None,
    modality: str | None = None,
    source_hint: str = "audio",
) -> dict[str, Any]:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().reshape(-1).numpy()
    try:
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported {source_hint} audio output type: {type(audio)}"
        ) from exc
    payload: dict[str, Any] = {
        "audio_waveform": array.tobytes(),
        "audio_waveform_shape": list(array.shape),
        "audio_waveform_dtype": "float32",
    }
    if sample_rate is not None:
        payload["sample_rate"] = int(sample_rate)
    if modality is not None:
        payload["modality"] = modality
    return payload
