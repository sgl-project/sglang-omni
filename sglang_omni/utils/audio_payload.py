# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for audio payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def audio_data_uri_from_reference(reference: dict[str, Any]) -> str | None:
    data = reference.get("data")
    if data is None:
        return None
    media_type = reference.get("media_type") or "audio/wav"
    return f"data:{media_type};base64,{data}"


def audio_waveform_payload(
    audio: Any,
    *,
    sample_rate: int | None = None,
    modality: str | None = None,
    source_hint: str = "audio",
    keep_channels: bool = False,
) -> dict[str, Any]:
    """Serialize a waveform into the relay payload format.

    With ``keep_channels`` a rank-2 ``[channels, samples]`` waveform keeps its
    shape (for natively multi-channel codecs); otherwise it is flattened to
    mono-style rank-1, preserving the historical behavior.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    try:
        array = np.asarray(audio, dtype=np.float32)
        if not (keep_channels and array.ndim == 2):
            array = array.reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported {source_hint} audio output type: {type(audio)}"
        ) from exc
    array = np.ascontiguousarray(array)
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
