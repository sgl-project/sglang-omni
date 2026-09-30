# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""Native MLX text and reference-audio conditioning for AuK."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open
from transformers import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
)

from sglang_omni.models.auk.mlx.conditioning_audio import AuKMlxAudioEncoder
from sglang_omni.utils.checkpoint import resolve_checkpoint


def apply_half_rope(value: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate with Qwen's half-split layout, unlike the DiT's interleaved rope."""
    half = value.shape[-1] // 2
    rotated = mx.concatenate([-value[..., half:], value[..., :half]], axis=-1)
    return (value * cos + rotated * sin).astype(value.dtype)


class TextAttention(nn.Module):
    def __init__(self, config: Qwen2_5OmniTextConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def __call__(
        self, hidden: mx.array, mask: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        compute_dtype = self.q_proj.weight.dtype
        hidden = hidden.astype(compute_dtype)
        batch, length, _ = hidden.shape
        query, key, value = (
            projection(hidden)
            .reshape(batch, length, heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            for projection, heads in (
                (self.q_proj, self.num_heads),
                (self.k_proj, self.num_key_value_heads),
                (self.v_proj, self.num_key_value_heads),
            )
        )
        query, key = apply_half_rope(query, cos, sin), apply_half_rope(key, cos, sin)
        hidden = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.head_dim**-0.5, mask=mask
        )
        return self.o_proj(hidden.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class TextMLP(nn.Module):
    def __init__(self, config: Qwen2_5OmniTextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = hidden.astype(self.gate_proj.weight.dtype)
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniTextConfig) -> None:
        super().__init__()
        self.self_attn = TextAttention(config)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self, hidden: mx.array, mask: mx.array, cos: mx.array, sin: mx.array
    ) -> mx.array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), mask, cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class AuKMlxTextEncoder(nn.Module):
    def __init__(self, config: Qwen2_5OmniTextConfig) -> None:
        super().__init__()
        if (
            config.hidden_act != "silu"
            or config.rope_parameters["rope_type"] != "default"
        ):
            raise ValueError("AuK's text conditioner requires SiLU and default RoPE")
        else:
            pass
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs_embeds: mx.array, attention_mask: mx.array) -> mx.array:
        """Return embedding, intermediate, and final normalized hidden states."""
        # note (Codex): Autocast preserves FP32 residuals; AuK consumes every layer.
        inputs_embeds = inputs_embeds.astype(mx.float32)
        length = inputs_embeds.shape[1]
        positions = mx.where(attention_mask, mx.cumsum(attention_mask, axis=-1) - 1, 1)
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        inv_freq = 1.0 / (
            self.config.rope_parameters["rope_theta"]
            ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        angles = positions[:, :, None] * inv_freq[None, None, :]
        angles = mx.concatenate([angles, angles], axis=-1)[:, None]
        cos, sin = mx.cos(angles).astype(inputs_embeds.dtype), mx.sin(angles).astype(
            inputs_embeds.dtype
        )
        indices = mx.arange(length)
        causal = indices[:, None] >= indices[None, :]
        mask = causal[None, None, :, :] & attention_mask[:, None, None, :].astype(
            mx.bool_
        )
        hidden = inputs_embeds
        states = []
        for layer, layer_type in zip(self.layers, self.config.layer_types):
            states.append(hidden)
            if layer_type == "sliding_attention":
                layer_mask = mask & (
                    indices[:, None] - indices[None, :] < self.config.sliding_window
                )
            else:
                layer_mask = mask
            hidden = layer(hidden, layer_mask, cos, sin)
            mx.eval(hidden)
        states.append(self.norm(hidden))
        return mx.stack(states, axis=1)


class AuKMlxConditionModel(nn.Module):
    def __init__(self, config: Qwen2_5OmniThinkerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AuKMlxTextEncoder(config.text_config)
        self.audio_tower = AuKMlxAudioEncoder(config.audio_config)

    def __call__(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        input_features: np.ndarray | None = None,
        feature_attention_mask: np.ndarray | None = None,
    ) -> mx.array:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Token IDs and attention masks must be matching matrices")
        elif np.any(
            (input_ids == self.config.image_token_id)
            | (input_ids == self.config.video_token_id)
        ):
            raise ValueError("AuK conditioning supports text and audio only")
        else:
            pass
        embeddings = self.model.embed_tokens(mx.array(input_ids))
        audio_rows, audio_columns = np.where(input_ids == self.config.audio_token_id)
        if input_features is not None:
            if feature_attention_mask is None:
                raise ValueError("Audio features require their attention mask")
            elif input_features.ndim != 3 or feature_attention_mask.shape != (
                input_features.shape[0],
                input_features.shape[2],
            ):
                raise ValueError(
                    "Audio features and attention masks have incompatible shapes"
                )
            else:
                pass
            valid = feature_attention_mask.astype(bool)
            lengths = valid.sum(axis=-1).tolist()
            features = np.concatenate(
                [sample[:, mask] for sample, mask in zip(input_features, valid)], axis=1
            )
            audio = self.audio_tower(mx.array(features), lengths).astype(
                embeddings.dtype
            )
            if len(audio_rows) != audio.shape[0]:
                raise ValueError(
                    "Audio features and placeholder token counts do not match"
                )
            else:
                embeddings[mx.array(audio_rows), mx.array(audio_columns)] = audio
        elif len(audio_rows):
            raise ValueError("Audio placeholder tokens require reference audio")
        else:
            pass
        return self.model(embeddings, mx.array(attention_mask))


class AuKMlxConditionEncoder:
    def __init__(self, model_path: str, *, dtype: mx.Dtype) -> None:
        if dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("AuK MLX conditioning supports float32 and bfloat16")
        else:
            pass
        path = Path(resolve_checkpoint(model_path))
        config = Qwen2_5OmniConfig.from_pretrained(path).thinker_config
        self.processor = Qwen2_5OmniProcessor.from_pretrained(path)
        self.model = AuKMlxConditionModel(config)
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = sorted(
                {
                    filename
                    for name, filename in weight_map.items()
                    if name.startswith(("thinker.model.", "thinker.audio_tower."))
                }
            )
        else:
            files = ["model.safetensors"]
        weights = []
        for filename in files:
            with safe_open(path / filename, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if not name.startswith(
                        ("thinker.model.", "thinker.audio_tower.")
                    ) or name.endswith(
                        ("audio_bos_eos_token.weight", "rotary_emb.inv_freq")
                    ):
                        continue
                    else:
                        tensor = checkpoint.get_tensor(name)
                        value = mx.array(tensor.float().numpy()).astype(dtype)
                        value = (
                            value.transpose(0, 2, 1)
                            if name.endswith(("conv1.weight", "conv2.weight"))
                            else value
                        )
                        mx.eval(value)
                        weights.append((name.removeprefix("thinker."), value))
        self.model.load_weights(weights, strict=True)
        del weights
        self.model.eval()
        mx.eval(self.model.parameters())

    def encode_batch(
        self,
        messages: list[list[dict[str, object]]],
        audios: list[np.ndarray | None],
    ) -> list[tuple[mx.array, mx.array]]:
        if len(messages) != len(audios) or not messages:
            raise ValueError(
                "Conditioning requires equally sized nonempty message and audio batches"
            )
        else:
            pass
        formatted = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        references = [audio for audio in audios if audio is not None]
        inputs = self.processor(
            text=formatted, audio=references or None, padding=True, return_tensors="np"
        )
        hidden = self.model(
            np.asarray(inputs["input_ids"]),
            np.asarray(inputs["attention_mask"]),
            inputs.get("input_features"),
            inputs.get("feature_attention_mask"),
        )
        mx.eval(hidden)
        results = []
        for item, mask in zip(hidden, inputs["attention_mask"]):
            valid = np.flatnonzero(mask)
            unpadded = item if valid.size == mask.size else item[:, mx.array(valid)]
            results.append((unpadded, mx.ones((len(valid),), dtype=mx.bool_)))
        return results
