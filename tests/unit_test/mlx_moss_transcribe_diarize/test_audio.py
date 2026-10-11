# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from transformers import WhisperFeatureExtractor

from sglang_omni_mlx.moss_transcribe_diarize.audio import (
    WINDOW_SAMPLE_COUNT,
    audio_features,
)


def noise(sample_count: int, seed: int = 0) -> np.ndarray:
    return (
        np.random.default_rng(seed).standard_normal(sample_count).astype(np.float32)
        * 0.1
    )


def test_features_match_whisper_frontend() -> None:
    samples = noise(16000 * 2 + 37)
    actual, token_lengths = audio_features(samples)
    extractor = WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    padded = np.pad(samples, (0, WINDOW_SAMPLE_COUNT - len(samples)))
    expected = extractor(
        [padded],
        sampling_rate=16000,
        padding="max_length",
        return_tensors="np",
    )["input_features"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(token_lengths, [26])


def test_long_audio_is_split_into_padded_windows() -> None:
    features, token_lengths = audio_features(noise(WINDOW_SAMPLE_COUNT + 1600))
    assert features.shape == (2, 80, 3000)
    np.testing.assert_array_equal(token_lengths, [375, 2])


def test_empty_audio_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        audio_features(np.array([], dtype=np.float32))
