# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pytest

from sglang_omni_mlx.fun_asr.audio import (
    MEL_BIN_COUNT,
    STACK_STRIDE,
    STACKED_FRAME_COUNT,
    audio_features,
    log_fbank,
    stack_low_frame_rate,
    token_count,
)

# Log-mel values reach about 20; float32 FFT and matmul order differ from torch.
FBANK_TOLERANCE = 2e-3


def noise(sample_count: int, seed: int = 0) -> np.ndarray:
    return (
        np.random.default_rng(seed).standard_normal(sample_count).astype(np.float32)
        * 0.1
    )


def reference_lfr(fbank: np.ndarray) -> np.ndarray:
    """funasr apply_lfr, written out frame by frame."""
    frame_count = fbank.shape[0]
    stacked_frame_count = math.ceil(frame_count / STACK_STRIDE)
    left_pad = (STACKED_FRAME_COUNT - 1) // 2
    padded = np.vstack([np.tile(fbank[:1], (left_pad, 1)), fbank])
    rows = []
    for index in range(stacked_frame_count):
        start = index * STACK_STRIDE
        if STACKED_FRAME_COUNT <= padded.shape[0] - start:
            rows.append(padded[start : start + STACKED_FRAME_COUNT].reshape(-1))
        else:
            window = padded[start:]
            missing = STACKED_FRAME_COUNT - window.shape[0]
            rows.append(
                np.vstack([window, np.tile(padded[-1:], (missing, 1))]).reshape(-1)
            )
    return np.vstack(rows)


@pytest.mark.parametrize("sample_count", [250, 400, 1200, 16000 * 3 + 37])
def test_log_fbank_matches_torchaudio_kaldi(sample_count: int) -> None:
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    samples = noise(sample_count)
    reference = kaldi.fbank(
        torch.from_numpy(samples).unsqueeze(0) * (1 << 15),
        num_mel_bins=MEL_BIN_COUNT,
        frame_length=min(25.0, sample_count / 16000 * 1000),
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=16000,
    ).numpy()
    ours = log_fbank(samples)
    assert ours.shape == reference.shape
    np.testing.assert_allclose(ours, reference, atol=FBANK_TOLERANCE, rtol=0)


@pytest.mark.parametrize("frame_count", [1, 5, 6, 7, 12, 13, 298])
def test_low_frame_rate_stacking_matches_funasr(frame_count: int) -> None:
    fbank = np.random.default_rng(frame_count).standard_normal(
        (frame_count, MEL_BIN_COUNT)
    )
    np.testing.assert_array_equal(stack_low_frame_rate(fbank), reference_lfr(fbank))


def test_audio_features_shape() -> None:
    features = audio_features(noise(16000 * 2))
    assert features.shape == (33, STACKED_FRAME_COUNT * MEL_BIN_COUNT)
    assert features.dtype == np.float32


@pytest.mark.parametrize(
    ("stacked_frames", "tokens"), [(1, 1), (8, 1), (9, 2), (34, 5), (500, 63)]
)
def test_token_count_halves_three_times(stacked_frames: int, tokens: int) -> None:
    assert token_count(stacked_frames) == tokens


def test_audio_shorter_than_two_samples_is_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        log_fbank(np.zeros(1, dtype=np.float32))
