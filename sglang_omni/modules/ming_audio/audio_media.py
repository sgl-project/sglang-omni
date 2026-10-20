# SPDX-License-Identifier: Apache-2.0
"""Decode request audio into mono Whisper-rate waveforms for Ming 3 audio."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisper

from sglang_omni.preprocessing.audio import decode_audio_bytes_av, parse_wav_bytes
from sglang_omni.preprocessing.base import MediaIO


class AudioMediaIO(MediaIO[torch.Tensor]):
    def load_bytes(self, data: bytes) -> torch.Tensor:
        try:
            waveform, sample_rate = parse_wav_bytes(data)
        except ValueError:
            waveform, sample_rate = decode_audio_bytes_av(data)
        writable_waveform = np.array(waveform, dtype=np.float32, copy=True)
        return torchaudio.functional.resample(
            torch.from_numpy(writable_waveform),
            sample_rate,
            whisper.audio.SAMPLE_RATE,
        )

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        """Decode audio; media_type belongs to the MediaIO interface."""
        return self.load_bytes(base64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        return self.load_bytes(filepath.read_bytes())
