# SPDX-License-Identifier: Apache-2.0
"""Audio decoding, downmix, and the Whisper sample rate."""

import io
from typing import Literal

import numpy as np
import pytest
import soundfile
import torch
from whisper.audio import SAMPLE_RATE as WHISPER_SAMPLE_RATE

from sglang_omni.models.ming3_audio.preprocessing import separate_asr_language_prefix
from sglang_omni.modules.ming_audio.audio_media import AudioMediaIO

STEREO_FRAME = np.array([[0.5, -0.25]], dtype=np.float32)
STEREO_FRAME_COUNT = 1600
MEAN_AMPLITUDE = 0.125
SOURCE_SAMPLE_RATE = 48000


def stereo_channel_wav_bytes(subtype: Literal["PCM_16", "FLOAT"]) -> bytes:
    waveform = np.tile(STEREO_FRAME, (STEREO_FRAME_COUNT, 1))
    encoded = io.BytesIO()
    soundfile.write(
        encoded, waveform, WHISPER_SAMPLE_RATE, format="WAV", subtype=subtype
    )
    return encoded.getvalue()


@pytest.mark.parametrize("subtype", ["PCM_16", "FLOAT"])
def test_wav_bytes_downmix_stereo_channels_to_mean(
    subtype: Literal["PCM_16", "FLOAT"],
) -> None:
    actual = AudioMediaIO().load_bytes(stereo_channel_wav_bytes(subtype))
    torch.testing.assert_close(
        actual, torch.full((STEREO_FRAME_COUNT,), MEAN_AMPLITUDE), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("transcript", "recognized_text"),
    [
        ("English\thello there", "hello there"),
        ("hello there", "hello there"),
        ("English\t", ""),
    ],
)
def test_asr_transcript_drops_language_prefix(
    transcript: str, recognized_text: str
) -> None:
    assert separate_asr_language_prefix(transcript) == recognized_text


def test_loader_resamples_to_the_whisper_rate() -> None:
    encoded = io.BytesIO()
    soundfile.write(
        encoded,
        np.zeros(SOURCE_SAMPLE_RATE, dtype=np.float32),
        SOURCE_SAMPLE_RATE,
        format="WAV",
        subtype="FLOAT",
    )
    actual = AudioMediaIO().load_bytes(encoded.getvalue())
    assert actual.shape == (WHISPER_SAMPLE_RATE,)
