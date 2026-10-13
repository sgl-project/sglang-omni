"""Native multimodal positions and captures over MLX-LM decoder primitives."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_moe import Attention, ModelArgs, Qwen3MoeDecoderLayer
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)


def text_args(
    config: Qwen3OmniMoeTextConfig | Qwen3OmniMoeTalkerTextConfig,
) -> ModelArgs:
    raw = config.to_dict()
    rope = config.rope_parameters
    raw.update(
        head_dim=config.head_dim,
        num_experts=config.num_experts,
        rope_theta=rope["rope_theta"],
        rope_scaling=None,
        model_type="qwen3_moe",
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
    )
    return ModelArgs.from_dict(raw)


class MultimodalAttention(Attention):
    """Explicit interleaved temporal, height and width rotary positions."""

    def __init__(
        self,
        args: ModelArgs,
        sections: list[int],
        layer_index: int,
        attention_bias: bool,
    ) -> None:
        super().__init__(args, layer_index)
        if attention_bias:
            for projection in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                projection.bias = mx.zeros((projection.weight.shape[0],))
        else:
            pass
        if len(sections) != 3 or sum(sections) != args.head_dim // 2:
            raise ValueError("M-RoPE sections must cover half the attention head")
        else:
            pass
        self.sections = sections
        self.rope_theta = args.rope_theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        if position_ids is None or position_ids.shape != (3, batch, length):
            raise ValueError("M-RoPE positions must cover all three prompt axes")
        else:
            pass
        queries = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, -1)
        values = values.transpose(0, 2, 1, 3)
        head_dim = queries.shape[-1]
        frequencies = self.rope_theta ** (
            -mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
        )
        angles = position_ids.astype(mx.float32)[..., None] * frequencies
        interleaved = angles[0]
        interleaved[..., 1 : 3 * self.sections[1] : 3] = angles[
            1, ..., 1 : 3 * self.sections[1] : 3
        ]
        interleaved[..., 2 : 3 * self.sections[2] : 3] = angles[
            2, ..., 2 : 3 * self.sections[2] : 3
        ]
        angles = mx.concatenate([interleaved, interleaved], axis=-1)[:, None]
        cosine, sine = mx.cos(angles), mx.sin(angles)
        rotated = []
        for tensor in (queries, keys):
            left, right = mx.split(tensor, 2, axis=-1)
            rotated.append(
                tensor * cosine.astype(tensor.dtype)
                + mx.concatenate([-right, left], axis=-1) * sine.astype(tensor.dtype)
            )
        queries, keys = rotated
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        else:
            pass
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MultimodalDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        args: ModelArgs,
        sections: list[int],
        layer_index: int,
        attention_bias: bool,
    ) -> None:
        super().__init__(args, layer_index)
        self.self_attn = MultimodalAttention(
            args, sections, layer_index, attention_bias
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), mask, cache, position_ids
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


@dataclass
class LanguageOutput:
    logits: mx.array
    hidden_states: mx.array | None


class LanguageModel(nn.Module):
    def __init__(
        self, text_config: Qwen3OmniMoeTextConfig, config: Qwen3OmniMoeThinkerConfig
    ) -> None:
        super().__init__()
        self.args = text_args(text_config)
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            self.args.vocab_size, self.args.hidden_size
        )
        self.model.norm = nn.RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps)
        sections = text_config.rope_parameters["mrope_section"]
        self.model.layers = [
            MultimodalDecoderLayer(
                self.args, sections, index, text_config.attention_bias
            )
            for index in range(self.args.num_hidden_layers)
        ]
        if not self.args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.args.hidden_size, self.args.vocab_size, bias=False
            )
        else:
            pass

    @property
    def layers(self) -> list[MultimodalDecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        input_ids: mx.array,
        *,
        inputs_embeds: mx.array | None = None,
        cache: list[KVCache] | None = None,
        position_ids: mx.array,
        deepstack_visual_embeds: mx.array | None = None,
        output_hidden_state_idx: int | None = None,
    ) -> LanguageOutput:
        if output_hidden_state_idx is not None and not (
            0 <= output_hidden_state_idx < len(self.layers)
        ):
            raise ValueError("Accepted hidden layer is outside the thinker")
        else:
            pass
        hidden = (
            self.model.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        cache = self.make_cache() if cache is None else cache
        mask = create_attention_mask(hidden, cache[0])
        captured = None
        for index, layer in enumerate(self.layers):
            if output_hidden_state_idx == index:
                captured = hidden
            else:
                pass
            hidden = layer(hidden, mask, cache[index], position_ids)
            if (
                deepstack_visual_embeds is not None
                and index < deepstack_visual_embeds.shape[2]
            ):
                hidden = hidden + deepstack_visual_embeds[:, :, index, :]
            else:
                pass
        hidden = self.model.norm(hidden)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return LanguageOutput(logits, captured)
