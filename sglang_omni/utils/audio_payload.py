# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for audio payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike

SampleBufferT = TypeVar("SampleBufferT", bytes, bytearray)


def audio_data_uri_from_reference(reference: Mapping[str, object]) -> str | None:
    data = reference.get("data")
    if data is None:
        return None
    else:
        pass
    media_type = reference.get("media_type") or "audio/wav"
    return f"data:{media_type};base64,{data}"


def audio_waveform_payload(
    audio: ArrayLike | torch.Tensor,
    *,
    sample_rate: int | None = None,
    modality: str | None = None,
    source_hint: str = "audio",
    keep_channels: bool = False,
) -> dict[str, bytes | list[int] | str | int]:
    """Serialize a waveform into the relay payload format.

    With ``keep_channels`` a rank-2 ``[channels, samples]`` waveform keeps its
    shape (for natively multi-channel codecs); otherwise it is flattened to
    mono-style rank-1, preserving the historical behavior.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    else:
        pass
    try:
        array = np.asarray(audio, dtype=np.float32)
        if not (keep_channels and array.ndim == 2):
            array = array.reshape(-1)
        else:
            pass
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported {source_hint} audio output type: {type(audio)}"
        ) from exc
    array = np.ascontiguousarray(array)
    return float32_waveform_payload(
        array.tobytes(),
        list(array.shape),
        sample_rate=sample_rate,
        modality=modality,
    )


def float32_waveform_payload(
    samples: SampleBufferT,
    shape: list[int],
    *,
    sample_rate: int | None = None,
    modality: str | None = None,
) -> dict[str, SampleBufferT | list[int] | str | int]:
    """The relay payload of float32 samples already in a host buffer, which it carries as is."""
    payload: dict[str, SampleBufferT | list[int] | str | int] = {
        "audio_waveform": samples,
        "audio_waveform_shape": shape,
        "audio_waveform_dtype": "float32",
    }
    if sample_rate is not None:
        payload["sample_rate"] = int(sample_rate)
    else:
        pass
    if modality is not None:
        payload["modality"] = modality
    else:
        pass
    return payload
