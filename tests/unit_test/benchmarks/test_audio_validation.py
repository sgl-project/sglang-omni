# SPDX-License-Identifier: Apache-2.0
"""Tests for WAV duration checks."""

import io
import struct
import wave

import pytest

from benchmarks.tts_serving.audio_validation import validate_audio_response


@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("frame_offset", [-1, 0, 1])
@pytest.mark.parametrize("require_min_audio", [True, False])
def test_wav_duration_threshold(
    sample_rate: int,
    channels: int,
    frame_offset: int,
    require_min_audio: bool,
) -> None:
    num_frames = round(sample_rate * 0.05) + frame_offset
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<h", 1000) * num_frames * channels)

    result = validate_audio_response(
        buffer.getvalue(),
        response_format="wav",
        content_type="audio/wav",
        require_min_audio=require_min_audio,
    )

    expected_ok = not require_min_audio or frame_offset >= 0
    assert result.ok is expected_ok
    if expected_ok:
        assert result.error is None
        assert result.duration_s == pytest.approx(num_frames / sample_rate)
    else:
        assert result.error is not None
        assert "shorter than the minimum generated-audio duration" in result.error
