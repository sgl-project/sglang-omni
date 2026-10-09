# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from sglang_omni_mlx.parakeet.audio import (
    CUT_SEARCH_SECONDS,
    MAX_CHUNK_SECONDS,
    chunk_spans,
    is_silent,
    log_mel_features,
    mel_filter_bank,
)
from sglang_omni_mlx.wav import SAMPLE_RATE

# Normalized features are about unit scale; float32 FFT and reduction order differ.
FEATURE_TOLERANCE = 2e-3


def noise(sample_count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(sample_count).astype(np.float32) * 0.1


@pytest.mark.parametrize("mel_bin_count", [80, 128])
def test_mel_filter_bank_matches_librosa(mel_bin_count: int) -> None:
    librosa = pytest.importorskip("librosa")
    reference = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=512,
        n_mels=mel_bin_count,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2,
        norm="slaney",
    )
    np.testing.assert_allclose(
        mel_filter_bank(mel_bin_count), reference, rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("sample_count", [100, 512, 4000, SAMPLE_RATE * 3 + 37])
def test_features_match_the_reference_extractor(sample_count: int) -> None:
    pytest.importorskip("torch")
    extractor_module = pytest.importorskip(
        "transformers.models.parakeet.feature_extraction_parakeet"
    )
    samples = noise(sample_count)
    padded = np.pad(samples, (0, max(512 - sample_count, 0)))
    reference = extractor_module.ParakeetFeatureExtractor(feature_size=128)(
        padded, sampling_rate=SAMPLE_RATE, return_tensors="np"
    )
    valid = int(reference["attention_mask"][0].sum())
    ours = log_mel_features(samples, mel_filter_bank(128))
    assert ours.shape == (valid, 128)
    np.testing.assert_allclose(
        ours, reference["input_features"][0, :valid], rtol=0, atol=FEATURE_TOLERANCE
    )


def test_short_audio_keeps_every_span() -> None:
    assert chunk_spans(noise(SAMPLE_RATE * 5)) == [(0, SAMPLE_RATE * 5)]


def test_long_audio_is_cut_in_the_quiet_window_before_each_boundary() -> None:
    samples = noise(int(SAMPLE_RATE * MAX_CHUNK_SECONDS * 2.5))
    chunk = int(SAMPLE_RATE * MAX_CHUNK_SECONDS)
    quiet_start = chunk - SAMPLE_RATE
    samples[quiet_start : quiet_start + 1600] = 0.0
    spans = chunk_spans(samples)
    assert spans[0] == (0, quiet_start + 800)
    assert [end for _, end in spans][-1] == len(samples)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start
    for start, end in spans:
        assert 0 < end - start <= chunk
        assert (
            end == len(samples)
            or end >= start + chunk - CUT_SEARCH_SECONDS * SAMPLE_RATE
        )


def test_silence_is_detected_by_peak() -> None:
    assert is_silent(np.zeros(1600, dtype=np.float32))
    assert is_silent(np.full(1600, 5e-4, dtype=np.float32))
    assert not is_silent(noise(1600))
