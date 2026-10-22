# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from sglang_omni_mlx.voxtral_tts.audio import encode_wav


def test_encode_wav_writes_clipped_pcm16_mono() -> None:
    encoded = encode_wav(np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32), 24000)
    with wave.open(io.BytesIO(encoded), "rb") as audio_file:
        assert audio_file.getnchannels() == 1
        assert audio_file.getsampwidth() == 2
        assert audio_file.getframerate() == 24000
        assert (
            audio_file.readframes(4)
            == np.array([-32768, 0, 16384, 32767], dtype="<i2").tobytes()
        )


def test_encode_wav_rejects_stereo_samples() -> None:
    with pytest.raises(ValueError, match="WAV output must be mono"):
        encode_wav(np.zeros((2, 2), dtype=np.float32), sample_rate_hz=24000)


@pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
def test_encode_wav_rejects_non_finite_samples(sample: float) -> None:
    with pytest.raises(ValueError, match="WAV output samples must be finite"):
        encode_wav(np.array([sample], dtype=np.float32), sample_rate_hz=24000)
