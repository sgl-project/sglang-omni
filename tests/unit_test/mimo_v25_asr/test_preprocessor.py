# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.mimo_v25_asr.preprocessor import (
    MiMoAudioPreprocessor,
    MiMoAudioTokenizerSettings,
)


class _FakeEncoder:
    def encode(self, *, input_features, input_lens, return_codes_only):
        assert return_codes_only is True
        frames = int(input_lens.sum().item()) // 4
        frames = max(frames, 1)
        codes = torch.zeros((20, frames), dtype=torch.long)
        return codes, input_lens


def test_encode_waveform_pads_to_group_size() -> None:
    preprocessor = MiMoAudioPreprocessor(
        MiMoAudioTokenizerSettings(chunk_seconds=1.0),
        device="cpu",
    )
    # Shorter than n_fft so the pad-to-n_fft branch is exercised.
    waveform = torch.zeros(100, dtype=torch.float32)
    codes = preprocessor.encode_waveform(waveform, _FakeEncoder())
    assert codes.ndim == 2
    assert codes.shape[1] == 8
    assert codes.shape[0] % 4 == 0
