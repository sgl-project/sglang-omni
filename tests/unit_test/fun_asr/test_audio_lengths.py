import random

import numpy as np
import pytest

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoFeatureExtractor
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
    fun_asr_num_audio_tokens,
)


@pytest.mark.parametrize(
    ("num_samples", "expected_tokens"),
    [
        (2, 1),
        (400, 1),
        (8079, 1),
        (8080, 2),
        (16000, 3),
        (480000, 63),
    ],
)
def test_fun_asr_num_audio_tokens_matches_hand_checked_boundaries(
    num_samples: int, expected_tokens: int
) -> None:
    assert (
        fun_asr_num_audio_tokens(
            num_samples,
            frame_length_samples=400,
            frame_shift_samples=160,
            lfr_n=6,
        )
        == expected_tokens
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"num_samples": 16000, "frame_length_samples": 0},
        {"num_samples": 16000, "frame_shift_samples": 0},
        {"num_samples": 16000, "lfr_n": 0},
    ],
)
def test_fun_asr_num_audio_tokens_rejects_invalid_lengths(kwargs: dict) -> None:
    params = {
        "num_samples": 16000,
        "frame_length_samples": 400,
        "frame_shift_samples": 160,
        "lfr_n": 6,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match="must be positive"):
        fun_asr_num_audio_tokens(**params)


def test_fun_asr_num_audio_tokens_matches_real_feature_extractor() -> None:
    extractor = FunAsrNanoFeatureExtractor()
    sample_counts = [2, 80, 159, 160, 399, 400, 559, 560, 8079, 8080]
    sample_counts.extend(random.Random(20260812).sample(range(80, 480001), 12))

    for num_samples in sample_counts:
        extracted = extractor(
            np.zeros(num_samples, dtype=np.float32),
            sampling_rate=extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="longest",
        )
        lfr_frames = int(extracted["attention_mask"].sum().item())
        expected_tokens = fun_asr_low_frame_rate_length(lfr_frames)

        assert fun_asr_num_audio_tokens(
            num_samples,
            frame_length_samples=extractor.n_fft,
            frame_shift_samples=extractor.hop_length,
            lfr_n=extractor.lfr_n,
        ) == int(expected_tokens), num_samples
