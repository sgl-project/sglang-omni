# SPDX-License-Identifier: Apache-2.0
"""Requests a server sends itself before it reports ready."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile

from sglang_omni.config import PipelineConfig

WARMUP_AUDIO_SAMPLE_RATE = 16000
WARMUP_TONE_HZ = 440.0
WARMUP_TONE_AMPLITUDE = 0.1
WARMUP_MAX_TOKENS = 8


@dataclass(frozen=True, kw_only=True)
class TranscriptionWarmupRequest:
    """A WAV file the server transcribes before it reports ready."""

    wav_bytes: bytes
    max_new_tokens: int


def warmup_tone_wav_bytes() -> bytes:
    """One second of a 16 kHz tone as a PCM 16 WAV file."""
    sample_times_s = np.arange(WARMUP_AUDIO_SAMPLE_RATE) / WARMUP_AUDIO_SAMPLE_RATE
    tone = WARMUP_TONE_AMPLITUDE * np.sin(2 * np.pi * WARMUP_TONE_HZ * sample_times_s)
    wav_buffer = io.BytesIO()
    soundfile.write(
        wav_buffer, tone, WARMUP_AUDIO_SAMPLE_RATE, format="WAV", subtype="PCM_16"
    )
    return wav_buffer.getvalue()


def build_transcription_warmup_request(
    pipeline_config: PipelineConfig,
) -> TranscriptionWarmupRequest:
    """A second of audio through the transcription route, the same for every model."""
    return TranscriptionWarmupRequest(
        wav_bytes=warmup_tone_wav_bytes(), max_new_tokens=WARMUP_MAX_TOKENS
    )
