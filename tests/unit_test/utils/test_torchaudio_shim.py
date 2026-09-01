from __future__ import annotations

import io
import math
import os
import struct
import tempfile
import wave

import torchaudio


def _write_sine_wav(path: str) -> None:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        frames = b"".join(
            struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * t / 16000)))
            for t in range(160)
        )
        wav_file.writeframes(frames)


def test_torchaudio_shim_load_accepts_path_and_bytesio() -> None:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _write_sine_wav(path)
        audio, sample_rate = torchaudio.load(path)
        assert sample_rate == 16000
        assert audio.shape == (1, 160)

        with open(path, "rb") as f:
            audio_bytes, sample_rate_bytes = torchaudio.load(io.BytesIO(f.read()))

        assert sample_rate_bytes == 16000
        assert audio_bytes.shape == (1, 160)
    finally:
        os.remove(path)
