# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoConfig


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
