# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from sglang_omni.models.fun_asr.configuration_fun_asr import (
    FunAsrNanoConfig,
    FunAsrNanoFeatureExtractor,
)


@pytest.mark.parametrize(
    "num_samples",
    [
        2,
        3,
        159,
        160,
        399,
        400,
        401,
        559,
        560,
        561,
        1359,
        1360,
        1361,
        8079,
        8080,
        8081,
        479999,
        480000,
    ],
)
@pytest.mark.parametrize(
    "sampling_rate,frame_length,frame_shift,stride_lfr",
    [(16000, 25, 10, 6), (22050, 25.1, 7.5, 3)],
)
def test_predicted_lfr_frames_match_real_frontend(
    num_samples: int,
    sampling_rate: int,
    frame_length: float,
    frame_shift: float,
    stride_lfr: int,
) -> None:
    """Early cache validation must use the same length as actual fbank/LFR."""
    extractor = FunAsrNanoFeatureExtractor(
        sampling_rate=sampling_rate,
        frame_length=frame_length,
        frame_shift=frame_shift,
        stride_lfr=stride_lfr,
    )
    audio = np.random.default_rng(7).normal(0, 0.01, num_samples).astype(np.float32)
    extracted = extractor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",
    )
    assert extractor.num_lfr_frames(num_samples) == int(
        extracted["attention_mask"].sum()
    )


@pytest.mark.parametrize("num_samples", [0, 1])
def test_invalid_window_cannot_hit_early_cache(num_samples: int) -> None:
    """Invalid short inputs must still reach the original frontend error."""
    extractor = FunAsrNanoFeatureExtractor()
    assert extractor.num_lfr_frames(num_samples) is None
    with pytest.raises(AssertionError):
        extractor(np.zeros(num_samples, dtype=np.float32), return_tensors="pt")


def test_fun_asr_config_uses_current_checkpoint_field_names() -> None:
    config = FunAsrNanoConfig(
        audio_config={
            "model_type": "fun_asr_nano_encoder",
            "num_mel_bins": 40,
            "num_stacked_frames": 3,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_timestamp_prediction_layers": 1,
            "fsmn_kernel_size": 5,
        },
        text_config={
            "model_type": "qwen3",
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 12,
            "vocab_size": 128,
        },
        audio_token_id=123,
        adaptor_config={
            "hidden_size": 24,
            "intermediate_size": 6,
            "projector_hidden_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "hidden_act": "relu",
        },
    )

    assert config.audio_config.input_size == 120
    assert config.audio_config.hidden_size == 16
    assert config.audio_config.num_hidden_layers == 4
    assert config.text_config.hidden_size == 24
    assert config.audio_token_id == 123
    assert config.adaptor_config.projector_hidden_size == 48
    assert config.adaptor_config.num_hidden_layers == 1
    assert config.adaptor_config.num_attention_heads == 2
    assert not hasattr(config, "encoder_config")
