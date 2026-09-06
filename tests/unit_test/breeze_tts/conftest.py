# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def tiny_config():
    return {
        "architectures": ["BreezeForConditionalGeneration"],
        "model_type": "breeze",
        "backbone_model_type": "qwen3",
        "text_encoder_proj_type": "linear",
        "tie_codebooks_embeddings": True,
        "hidden_size": 16,
        "audio_embed_size": 16,
        "vocab_size": 11,
        "num_codebooks": 4,
        "codebook_eos_token_id": 0,
        "rope_theta": 500000,
        "rms_norm_eps": 1e-5,
        "backbone_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
            "rope_theta": 1000000,
            "rms_norm_eps": 1e-6,
        },
        "depth_decoder_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 11,
            "audio_embed_size": 16,
            "backbone_hidden_size": 16,
            "num_codebooks": 4,
        },
        "text_encoder_config": {
            "model_type": "t5gemma2_text",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 64,
            "sliding_window": 4,
            "max_position_embeddings": 1024,
            "layer_types": ["sliding_attention", "full_attention"],
            "dropout_rate": 0.0,
        },
        "codec_config": {"codebook_size": 8},
    }
