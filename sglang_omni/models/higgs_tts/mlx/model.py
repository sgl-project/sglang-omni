# SPDX-License-Identifier: Apache-2.0
"""MLX Qwen3 language model construction and strict Higgs weight loading."""
from __future__ import annotations

from pathlib import Path

from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config


def load_mlx_language_model(checkpoint: str, *, dtype=None):
    import mlx.core as mx
    from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

    config = HiggsMultimodalQwen3Config.from_pretrained(checkpoint).get_text_config()
    rope = config.rope_parameters
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Higgs MLX currently requires default RoPE")
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=rope["rope_theta"],
        head_dim=config.head_dim,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    model = Qwen3Model(args)
    prefixes = {
        "tied.embedding.text_embedding.": "embed_tokens.",
        "body.layers.": "layers.",
        "body.norm.": "norm.",
    }
    weights = {}
    for path in sorted(Path(checkpoint).glob("*.safetensors")):
        for key, value in mx.load(str(path)).items():
            for prefix, target in prefixes.items():
                if key.startswith(prefix):
                    name = target + key[len(prefix) :]
                    if name in weights:
                        raise ValueError(f"Duplicate Higgs language weight: {name}")
                    weights[name] = value.astype(dtype or mx.bfloat16)
                    break
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
