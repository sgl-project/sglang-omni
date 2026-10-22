# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS text decoder and checkpoint loading in MLX."""

from __future__ import annotations

import glob
import re

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from sglang_omni_mlx.voxtral_tts.acoustic_head import (
    SEMANTIC_VOCAB_ALIGNMENT,
    SPECIAL_TOKEN_COUNT,
    FlowMatchingAudioTransformer,
    padded_size,
)
from sglang_omni_mlx.voxtral_tts.audio_tokenizer import AudioTokenizerDecoder
from sglang_omni_mlx.voxtral_tts.audio_tokenizer_loader import load_audio_tokenizer
from sglang_omni_mlx.voxtral_tts.config import TextConfig, VoxtralConfig

LAYER_KEY_PATTERN = re.compile(r"^layers\.(\d+)\.(.+)$")
LAYER_KEY_MAP: dict[str, str] = {
    "attention.wq.weight": "self_attn.q_proj.weight",
    "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}
GLOBAL_KEY_MAP = {
    "norm.weight": "norm.weight",
    "mm_audio_embeddings.tok_embeddings.weight": "embed_tokens.weight",
}
AUDIO_EMBEDDING_KEY = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"


def interleave_qk_weight(weight: mx.array, head_count: int, head_dim: int) -> mx.array:
    half_head_dim = head_dim // 2
    return (
        weight.reshape(head_count, half_head_dim, 2, -1)
        .transpose(0, 2, 1, 3)
        .reshape(head_count * head_dim, -1)
    )


def remap_text_weight_key(checkpoint_key: str) -> str | None:
    if checkpoint_key in GLOBAL_KEY_MAP:
        return GLOBAL_KEY_MAP[checkpoint_key]
    else:
        pass
    layer_match = LAYER_KEY_PATTERN.match(checkpoint_key)
    if layer_match is None or layer_match.group(2) not in LAYER_KEY_MAP:
        return None
    else:
        return f"layers.{layer_match.group(1)}.{LAYER_KEY_MAP[layer_match.group(2)]}"


class VoxtralAttention(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.head_count = config.n_heads
        self.kv_head_count = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.dim, self.head_count * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.dim, self.kv_head_count * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.dim, self.kv_head_count * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.head_count * self.head_dim, config.dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, hidden_states: mx.array, cache: KVCache) -> mx.array:
        batch_size, token_count, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch_size, token_count, self.head_count, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            batch_size, token_count, self.kv_head_count, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            batch_size, token_count, self.kv_head_count, self.head_dim
        )
        queries = self.rope(queries.transpose(0, 2, 1, 3), offset=cache.offset)
        keys = self.rope(keys.transpose(0, 2, 1, 3), offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values.transpose(0, 2, 1, 3))
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.head_dim**-0.5,
            mask="causal" if token_count > 1 else None,
        )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(batch_size, token_count, -1)
        )


class VoxtralTextLayer(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.self_attn = VoxtralAttention(config)
        self.input_layernorm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = VoxtralMLP(config)

    def __call__(self, hidden_states: mx.array, cache: KVCache) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class VoxtralMLP(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(
            nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class VoxtralTextModel(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [VoxtralTextLayer(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(self, embeddings: mx.array, caches: list[KVCache]) -> mx.array:
        hidden_states = embeddings
        for layer, cache in zip(self.layers, caches):
            hidden_states = layer(hidden_states, cache)
        return self.norm(hidden_states)[:, -1, :]


class VoxtralMLXModel(nn.Module):
    def __init__(self, config: VoxtralConfig) -> None:
        super().__init__()
        self.text_model = VoxtralTextModel(config.text)
        self.acoustic_head = FlowMatchingAudioTransformer(config.audio)
        codebook_sizes = [
            config.audio.semantic_codebook_size + SPECIAL_TOKEN_COUNT,
            *[
                config.audio.acoustic_codebook_size + SPECIAL_TOKEN_COUNT
                for _ in range(config.audio.n_acoustic_codebook)
            ],
        ]
        self.audio_codebook_offsets = mx.array(
            [
                0,
                *[
                    sum(codebook_sizes[:index])
                    for index in range(1, len(codebook_sizes))
                ],
            ],
            dtype=mx.int32,
        )
        self.audio_code_embeddings = nn.Embedding(
            padded_size(sum(codebook_sizes), SEMANTIC_VOCAB_ALIGNMENT), config.text.dim
        )
        self.audio_tokenizer: AudioTokenizerDecoder | None = None

    def new_caches(self) -> list[KVCache]:
        return self.text_model.new_caches()

    def embed_audio_codes(self, audio_codes: mx.array) -> mx.array:
        shifted_codes = audio_codes + self.audio_codebook_offsets[None, :]
        return self.audio_code_embeddings(shifted_codes).sum(axis=1)


def load_voxtral_model(config: VoxtralConfig) -> VoxtralMLXModel:
    shard_paths = sorted(glob.glob(str(config.checkpoint_directory / "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(
            f"Voxtral checkpoint has no safetensors files: {config.checkpoint_directory}"
        )
    else:
        pass
    model = VoxtralMLXModel(config)
    text_weights: list[tuple[str, mx.array]] = []
    acoustic_weights: list[tuple[str, mx.array]] = []
    audio_tokenizer_weights: dict[str, mx.array] = {}
    audio_code_embeddings: mx.array | None = None
    for shard_path in shard_paths:
        for checkpoint_key, checkpoint_weight in mx.load(shard_path).items():
            mlx_key = remap_text_weight_key(checkpoint_key)
            if checkpoint_key.startswith("audio_tokenizer."):
                audio_tokenizer_weights[checkpoint_key] = checkpoint_weight
            elif checkpoint_key == AUDIO_EMBEDDING_KEY:
                audio_code_embeddings = checkpoint_weight
            elif checkpoint_key.startswith("acoustic_transformer."):
                acoustic_key = checkpoint_key.removeprefix("acoustic_transformer.")
                if acoustic_key == "time_embedding.inv_freq":
                    pass
                else:
                    acoustic_weights.append((acoustic_key, checkpoint_weight))
            elif mlx_key is not None:
                if (
                    checkpoint_key.startswith("layers.")
                    and ".attention.wq." in checkpoint_key
                ):
                    checkpoint_weight = interleave_qk_weight(
                        checkpoint_weight, config.text.n_heads, config.text.head_dim
                    )
                elif (
                    checkpoint_key.startswith("layers.")
                    and ".attention.wk." in checkpoint_key
                ):
                    checkpoint_weight = interleave_qk_weight(
                        checkpoint_weight, config.text.n_kv_heads, config.text.head_dim
                    )
                else:
                    pass
                text_weights.append((mlx_key, checkpoint_weight))
            else:
                pass
    if audio_code_embeddings is None:
        raise RuntimeError(f"Voxtral checkpoint is missing {AUDIO_EMBEDDING_KEY}")
    else:
        pass
    model.text_model.load_weights(text_weights, strict=True)
    model.acoustic_head.load_weights(acoustic_weights, strict=True)
    expected_embedding_shape = model.audio_code_embeddings.weight.shape
    if audio_code_embeddings.shape != expected_embedding_shape:
        raise ValueError("Voxtral audio code embedding has an unexpected shape")
    else:
        pass
    model.audio_code_embeddings.weight = audio_code_embeddings
    model.audio_tokenizer = load_audio_tokenizer(config, audio_tokenizer_weights)
    mx.eval(model.parameters())
    return model
