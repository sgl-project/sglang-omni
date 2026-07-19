# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoConfig


def test_fun_asr_config_uses_current_checkpoint_field_names() -> None:
    config = FunAsrNanoConfig(
        encoder_config={
            "model_type": "fun_asr_nano_encoder",
            "num_mel_bins": 40,
            "num_stacked_frames": 3,
            "d_model": 16,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "encoder_layers": 3,
            "num_timestamp_prediction_blocks": 1,
            "kernel_size": 5,
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
        adaptor_intermediate_size=48,
        adaptor_num_hidden_layers=1,
        adaptor_num_attention_heads=2,
        activation_function="relu",
    )

    assert config.encoder_config.input_size == 120
    assert config.encoder_config.d_model == 16
    assert config.encoder_config.encoder_layers == 3
    assert config.text_config.hidden_size == 24
    assert config.audio_token_id == 123
    assert config.adaptor_intermediate_size == 48
    assert config.adaptor_num_hidden_layers == 1
    assert config.adaptor_num_attention_heads == 2
    assert not hasattr(config, "adaptor_config")
