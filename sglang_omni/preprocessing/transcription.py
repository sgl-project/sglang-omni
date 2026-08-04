# SPDX-License-Identifier: Apache-2.0
"""Shared audio preparation for ASR request builders.

Resolves the audio source from the ``StagePayload``, decodes/resamples it to
the model's sample rate, then derives the clip duration and cache fingerprint.

Low-level mechanics (decode, load, resample, fingerprint) stay in
``sglang_omni.utils.audio``.

Model-specific: ``source_name`` used in error messages, duration limit where
the model has one, and the custom ``source_resolver`` when the model accepts
sources beyond the default payload keys (e.g. MOSS-Transcribe-Diarize).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from sglang_omni.utils.audio import audio_fingerprint, audio_fingerprint_int, load_audio

if TYPE_CHECKING:
    from sglang_omni.proto import StagePayload

DEFAULT_TARGET_SAMPLE_RATE = 16000

# Byte-like sources take precedence over path-like sources; within each group,
# order is the lookup precedence.
_BYTES_SOURCE_KEYS = ("audio_bytes", "bytes", "file")
_PATH_SOURCE_KEYS = ("audio_path", "path", "url")


def resolve_audio_source(payload: StagePayload) -> Any:
    """Default source resolver shared by the ASR request builders."""
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in _BYTES_SOURCE_KEYS:
            value = inputs.get(key)
            if value is not None:
                return value
        for key in _PATH_SOURCE_KEYS:
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


@dataclass(frozen=True)
class PreparedAudio:
    """Decoded waveform plus the derived per-request audio metadata."""

    waveform: np.ndarray
    sample_rate: int
    duration_s: float
    fingerprint: str

    @property
    def fingerprint_int(self) -> int:
        return audio_fingerprint_int(self.fingerprint)


def prepare_audio(
    payload: StagePayload,
    *,
    source_name: str,
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    source_resolver: Callable[[StagePayload], Any] = resolve_audio_source,
    max_duration_s: float | None = None,
    max_duration_message: str | None = None,
) -> PreparedAudio:
    """Resolve, load, and fingerprint the payload's audio for one request."""

    source = source_resolver(payload)
    waveform = load_audio(
        source,
        source_name=source_name,
        target_sample_rate=target_sample_rate,
    )
    duration_s = float(len(waveform) / target_sample_rate)
    if max_duration_s is not None and duration_s > max_duration_s:
        raise ValueError(
            max_duration_message
            or (
                f"{source_name} accepts audio up to {max_duration_s} seconds, "
                f"got {duration_s:.3f} seconds"
            )
        )
    return PreparedAudio(
        waveform=waveform,
        sample_rate=target_sample_rate,
        duration_s=duration_s,
        fingerprint=audio_fingerprint(waveform),
    )


__all__ = [
    "DEFAULT_TARGET_SAMPLE_RATE",
    "PreparedAudio",
    "prepare_audio",
    "resolve_audio_source",
]
