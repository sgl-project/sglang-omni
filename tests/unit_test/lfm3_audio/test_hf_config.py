# SPDX-License-Identifier: Apache-2.0
"""LFM3 config parsing for both checkpoint layouts: LFM3-Audio's export and LFM3-Omni's Transformers config."""

from __future__ import annotations

import json

from sglang_omni.models.lfm3_audio import hf_config

TEXT = {
    "vocab_size": 128000,
    "hidden_size": 2048,
    "intermediate_size": 7680,
    "num_hidden_layers": 4,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 7,
    "layer_types": ["conv", "full_attention", "conv", "sliding_attention"],
    "sliding_window": 256,
    "use_sandwich_norm": True,
    "tie_word_embeddings": True,
    "pad_token_id": 124893,
    "bos_token_id": 124894,
    "eos_token_id": 124900,
    "rope_parameters": {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 1e6,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
    },
}
DEPTH = {"hidden_size": 1024, "num_hidden_layers": 3}


def write_checkpoint(path, config: dict, generation: dict) -> str:
    (path / "config.json").write_text(json.dumps(config))
    (path / "generation_config.json").write_text(json.dumps(generation))
    return str(path)


def test_audio_export_layout(tmp_path) -> None:
    checkpoint = write_checkpoint(
        tmp_path,
        {
            "lfm": TEXT,
            "depthformer": DEPTH,
            "codebooks": 12,
            "audio_vocab_size": 2017,
            "maskgit_steps": 3,
            "interleaved_n_text": 6,
            "interleaved_n_audio": 12,
        },
        {
            "eos_token_id": [124900, 125014],
            "audio_temperature": 1.0,
            "audio_top_k": 128,
            "audio_repetition_penalty": 1.1,
            "audio_repetition_window": 4096,
        },
    )
    config = hf_config.lfm3_config_from_checkpoint(checkpoint)
    assert config.depth_config == DEPTH and config.sliding_window == 256
    assert config.full_attention_layer_ids == [1, 3]
    tokens = hf_config.lfm3_tokens_from_checkpoint(checkpoint)
    assert tokens.placeholder_token_ids == (125031, 125032)
    assert tokens.maskgit_schedule == [4, 4, 4] and tokens.audio_eos_code == 2016
    speech = hf_config.speech_defaults_from_checkpoint(checkpoint)
    assert (speech.temperature, speech.top_k, speech.repetition_window) == (
        1.0,
        128,
        4096,
    )


def test_omni_transformers_layout(tmp_path) -> None:
    checkpoint = write_checkpoint(
        tmp_path,
        {
            "text_config": TEXT,
            "depth_config": DEPTH,
            "num_codebooks": 12,
            "codebook_size": 2017,
            "audio_eos_code": 2016,
            "maskgit_schedule": [4, 4, 4],
            "interleaved_n_text": 6,
            "interleaved_n_audio": 12,
            "image_token_id": 125030,
            "video_token_id": 125029,
        },
        {
            "eos_token_id": [124900, 125014],
            "audio_temperature": 0.4,
            "audio_top_k": None,
            "audio_repetition_penalty": 1.2,
            "audio_repetition_window": 64,
        },
    )
    config = hf_config.lfm3_config_from_checkpoint(checkpoint)
    assert config.depth_config == DEPTH and config.num_codebooks == 12
    assert config.rope_for("full_attention")["partial_rotary_factor"] == 0.5
    tokens = hf_config.lfm3_tokens_from_checkpoint(checkpoint)
    assert tokens.placeholder_token_ids == (125031, 125032, 125030, 125029)
    assert tokens.maskgit_schedule == [4, 4, 4]
    speech = hf_config.speech_defaults_from_checkpoint(checkpoint)
    assert (
        speech.temperature,
        speech.top_k,
        speech.repetition_penalty,
        speech.repetition_window,
    ) == (0.4, None, 1.2, 64)
