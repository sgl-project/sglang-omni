# SPDX-License-Identifier: Apache-2.0
"""The cached-table fbank must stay bit-identical to torchaudio's kaldi.fbank."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
kaldi = pytest.importorskip("torchaudio.compliance.kaldi")

from sglang_omni.utils.audio_features import _mel_banks, cached_fbank

N_MELS = 80
FRAME_SHIFT = 10.0
WINDOW = "hamming"


def _reference(wav, frame_length, sample_rate, window=WINDOW):
    return kaldi.fbank(
        wav,
        num_mel_bins=N_MELS,
        frame_length=frame_length,
        frame_shift=FRAME_SHIFT,
        dither=0.0,
        energy_floor=0.0,
        window_type=window,
        sample_frequency=sample_rate,
        snip_edges=True,
    )


def _wav(seconds, sample_rate, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(seconds * sample_rate), dtype=np.float32)
    return torch.from_numpy(samples).unsqueeze(0) * (1 << 15)


@pytest.mark.parametrize("seconds", [0.3, 1.0, 5.0])
@pytest.mark.parametrize("sample_rate", [16000, 8000])
def test_matches_kaldi_fbank_bit_exactly(seconds, sample_rate):
    wav = _wav(seconds, sample_rate)
    frame_length = min(25.0, seconds * 1000)
    expected = _reference(wav, frame_length, sample_rate)
    got = cached_fbank(
        wav,
        num_mel_bins=N_MELS,
        frame_length=frame_length,
        frame_shift=FRAME_SHIFT,
        window_type=WINDOW,
        sample_frequency=sample_rate,
    )
    assert got.shape == expected.shape
    assert torch.equal(got, expected)


def test_short_audio_caps_frame_length_like_the_caller():
    # The caller shortens frame_length for clips below one window.
    sample_rate = 16000
    seconds = 0.012
    wav = _wav(seconds, sample_rate, seed=3)
    frame_length = min(25.0, seconds * 1000)
    expected = _reference(wav, frame_length, sample_rate)
    got = cached_fbank(
        wav,
        num_mel_bins=N_MELS,
        frame_length=frame_length,
        frame_shift=FRAME_SHIFT,
        window_type=WINDOW,
        sample_frequency=sample_rate,
    )
    assert torch.equal(got, expected)


def test_mel_table_is_reused_across_calls():
    _mel_banks.cache_clear()
    for i in range(4):
        cached_fbank(
            _wav(1.0, 16000, seed=i),
            num_mel_bins=N_MELS,
            frame_length=25.0,
            frame_shift=FRAME_SHIFT,
            window_type=WINDOW,
            sample_frequency=16000,
        )
    info = _mel_banks.cache_info()
    assert info.misses == 1
    assert info.hits == 3


@pytest.mark.parametrize("window", ["hamming", "povey"])
def test_matches_for_each_window_type(window):
    # Fun-ASR uses hamming, the Ming speaker encoders use the kaldi default.
    wav = _wav(1.0, 16000, seed=7)
    expected = _reference(wav, 25.0, 16000, window=window)
    got = cached_fbank(
        wav,
        num_mel_bins=N_MELS,
        frame_length=25.0,
        frame_shift=FRAME_SHIFT,
        window_type=window,
        sample_frequency=16000,
    )
    assert torch.equal(got, expected)
