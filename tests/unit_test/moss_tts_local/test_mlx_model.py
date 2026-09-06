# SPDX-License-Identifier: Apache-2.0
"""Focused native-MLX coverage for MOSS-TTS Local."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from sglang_omni.models.moss_tts_local.mlx.config import ModelConfig
from sglang_omni.models.moss_tts_local.mlx.model import MossTTSLocalModel
from sglang_omni.models.moss_tts_local.mlx.runner import _sample


def _tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "model_type": "moss_tts_local",
            "n_vq": 2,
            "audio_vocab_size": 16,
            "audio_pad_code": 16,
            "audio_assistant_slot_token_id": 60,
            "audio_end_token_id": 61,
            "qwen3_config": {
                "model_type": "qwen3",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 64,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000,
            },
            "gpt2_config": {
                "n_embd": 32,
                "n_layer": 1,
                "n_head": 4,
                "n_inner": 64,
                "activation_function": "silu",
                "position_embedding_type": "rope",
                "rope_base": 10_000,
            },
        }
    )


def test_mlx_model_generates_one_complete_codec_row() -> None:
    model = MossTTSLocalModel(_tiny_config())
    prompt = mx.array([[[1, 16, 16], [2, 3, 4]]], dtype=mx.int32)
    hidden = model.backbone(prompt, model.make_cache())
    row = model.decode_frame(
        hidden[:, -1, :],
        sample_text=lambda logits: mx.argmax(logits, axis=-1),
        sample_audio=lambda logits, _channel: mx.argmax(logits, axis=-1),
    )
    mx.eval(row)

    assert hidden.shape == (1, 2, 32)
    assert row.shape == (1, 3)
    assert row[0, 0].item() in {60, 61}
    assert all(0 <= code < 16 for code in row[0, 1:].tolist())


def test_mlx_module_names_match_the_official_checkpoint_layout() -> None:
    keys = {
        name for name, _ in tree_flatten(MossTTSLocalModel(_tiny_config()).parameters())
    }

    assert "transformer.layers.0.self_attn.q_proj.weight" in keys
    assert "local_transformer.h.0.attn.c_attn.weight" in keys
    assert "audio_embeddings.1.weight" in keys
    assert "audio_lm_heads.1.weight" in keys
    assert "local_text_lm_head.weight" in keys


def test_seeded_sampling_is_position_stable() -> None:
    logits = mx.array([[0.1, 0.2, 0.3, 0.4]], dtype=mx.float32)
    kwargs = {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 3,
        "seed": 1234,
        "position": 7,
    }

    first = _sample(logits, **kwargs)
    second = _sample(logits, **kwargs)
    mx.eval(first, second)

    assert first.item() == second.item()


def test_model_rejects_non_local_checkpoint() -> None:
    config = _tiny_config()
    config.model_type = "moss_tts_delay"
    with pytest.raises(ValueError, match="moss_tts_local"):
        MossTTSLocalModel(config)
