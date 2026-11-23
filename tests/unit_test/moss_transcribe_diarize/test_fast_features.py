# SPDX-License-Identifier: Apache-2.0
"""Single-window log-mel boundaries and reference fallback."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from sglang_omni.models.moss_transcribe_diarize.fast_features import TrimmedWhisperFbank
from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    extract_audio_features,
    extract_audio_features_fast,
)


@pytest.fixture(scope="module", params=[80, 128])
def fast_fbank(request: pytest.FixtureRequest) -> TrimmedWhisperFbank:
    extractor = WhisperFeatureExtractor(feature_size=request.param)
    fast_fbank = TrimmedWhisperFbank(extractor)
    assert fast_fbank.self_check()
    return fast_fbank


@pytest.mark.parametrize(
    "sample_count",
    [
        1,
        159,
        160,
        199,
        200,
        399,
        400,
        401,
        10039,
        10040,
        10041,
        20000,
        95999,
        96000,
        479599,
        479600,
        479601,
        479999,
        480000,
    ],
)
@pytest.mark.parametrize("amplitude", [0.0, 1e-6, 0.5, 2.0])
def test_trimmed_fbank_preserves_boundary_frames(
    fast_fbank: TrimmedWhisperFbank, sample_count: int, amplitude: float
) -> None:
    """Trimming must retain reflected boundary frames and the silent-tail clamp."""
    random = np.random.default_rng(415 + sample_count)
    audio = (random.standard_normal(sample_count) * amplitude).astype(np.float32)
    np.testing.assert_array_equal(
        fast_fbank.compute(audio), fast_fbank.reference(audio)
    )


def test_trimmed_fbank_preserves_multiwindow_fallback(
    fast_fbank: TrimmedWhisperFbank,
) -> None:
    """The single-window optimization must not accept empty or multiwindow audio."""
    assert not fast_fbank.supports(0)
    assert fast_fbank.supports(fast_fbank.n_samples)
    assert not fast_fbank.supports(fast_fbank.n_samples + 1)


def test_trimmed_fbank_rejects_dithered_extractor() -> None:
    """Silence-tail reconstruction cannot replace reference dither noise."""
    extractor = WhisperFeatureExtractor(feature_size=80, dither=0.001)
    assert TrimmedWhisperFbank.verified(extractor) is None


@pytest.mark.parametrize("sample_count", [1, 400, 10041, 479601, 480000])
def test_fast_features_match_production_extraction(
    fast_fbank: TrimmedWhisperFbank, sample_count: int
) -> None:
    """Fast extraction must preserve production features, lengths, mapping and token count."""
    processor = SimpleNamespace(
        feature_extractor=fast_fbank.feature_extractor, audio_merge_size=2
    )
    audio = np.random.default_rng(415).standard_normal(sample_count).astype(np.float32)
    expected = extract_audio_features(processor, audio)
    actual = extract_audio_features_fast(processor, audio, fast_fbank)
    for actual_tensor, expected_tensor in zip(actual[:3], expected[:3], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert actual[3] == expected[3]
