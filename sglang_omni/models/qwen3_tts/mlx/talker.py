# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-TTS (Copyright 2025 Prince Canuma and contributors).
"""MLX Qwen3-TTS talker, adapted to SGLang's MLX attention contract.

Two deliberate departures from the mlx-audio reference:

*Plain RoPE instead of external MRoPE.*  mlx-audio computes interleaved MRoPE
in ``TalkerRotaryEmbedding`` and threads ``(cos, sin)`` through every layer, so
its attention takes ``position_embeddings`` and owns no ``rope``.  SGLang's
``MLXAttentionWrapper`` instead requires ``rope`` on the attention module and
calls it with per-request offsets, which is what makes generic batched decode
work.  The two are numerically identical here: Qwen3-TTS drives the talker with
``position_ids = stack([pos, pos, pos])`` (and SGLang's CUDA path likewise
builds three identical M-RoPE rows), and interleaved MRoPE over three equal
rows collapses to its temporal row -- i.e. ordinary 1-D RoPE.  That collapse is
pinned by ``tests/unit_test/qwen3_tts/test_mlx_talker.py``.  Positions therefore
come only from cache offsets; this port accepts no ``position_ids``.

*No padded-batch attention mask.*  mlx-audio left-pads a batch into one shared
cache and passes an ``attention_mask``.  SGLang keeps one cache per request and
batches through the wrapper, so there is nothing to pad.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

from .config import CodePredictorConfig, TalkerConfig


def _rope_safe(rope: nn.RoPE, x: mx.array, offset: int) -> mx.array:
    """Apply RoPE, working around an mx.fast.rope bug.

    For a 4D tensor (B, heads, L, dim) with L == 1 and B > 1, mx.fast.rope
    (used by nn.RoPE) corrupts every batch row except the first on Metal. This
    only bites a shared-cache batched decode; SGLang's per-request decode goes
    through MLXAttentionWrapper with array offsets instead. Padding the
    sequence to length 2 and slicing keeps the fast kernel while producing the
    exact correct result. Mirrors the same helper in the Qwen3-ASR MLX port.
    """
    if x.ndim == 4 and x.shape[0] > 1 and x.shape[2] == 1:
        x = mx.concatenate([x, mx.zeros_like(x)], axis=2)
        return rope(x, offset=offset)[:, :, :1, :]
    return rope(x, offset=offset)


class _QKNormAttention(nn.Module):
    """Grouped-query attention with Q/K norms and SGLang-compatible RoPE.

    Satisfies SGLang's MLX attention contract: ``q_proj``/``k_proj``/``v_proj``/
    ``o_proj``/``rope`` plus ``scale`` and head counts, called as
    ``(x, mask=..., cache=...)``.
    """

    def __init__(
        self,
        config: Union[TalkerConfig, CodePredictorConfig],
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[Union[str, mx.array]] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        queries = self.q_proj(hidden_states).reshape(
            B, L, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = _rope_safe(self.rope, queries, offset)
            keys = _rope_safe(self.rope, keys, offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TalkerAttention(_QKNormAttention):
    """Talker self-attention (1-D RoPE; see the module docstring)."""


class CodePredictorAttention(_QKNormAttention):
    """Code-predictor self-attention."""


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ResizeMLP(nn.Module):
    """Two-layer projection between the text and talker hidden sizes."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = {
            "silu": nn.silu,
            "gelu": nn.gelu,
            "relu": nn.relu,
        }.get(hidden_act, nn.silu)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class _DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer."""

    def __init__(
        self,
        config: Union[TalkerConfig, CodePredictorConfig],
        layer_idx: int,
        attention_cls: type,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attention_cls(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[Union[str, mx.array]] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask, cache=cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class TalkerDecoderLayer(_DecoderLayer):
    def __init__(self, config: TalkerConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx, TalkerAttention)


class CodePredictorDecoderLayer(_DecoderLayer):
    def __init__(self, config: CodePredictorConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx, CodePredictorAttention)


class Qwen3TTSTalkerModel(nn.Module):
    """Talker transformer trunk.

    This is the container SGLang discovers (``model.layers``), so its KV is the
    persistent, pool-backed cache. Attribute names are deliberately free of
    ``sliding_window``/``window_size``/``layer_types``: the talker uses full
    attention, and those names would make SGLang treat it as sliding-window.
    """

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )
        self.layers = [
            TalkerDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
        mask: Optional[Union[str, mx.array]] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(inputs_embeds, cache[0])

        hidden_states = inputs_embeds
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)

    def make_cache(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.layers]


class CodePredictorModel(nn.Module):
    """Code-predictor trunk, matching the ``code_predictor.model.*`` weights."""

    def __init__(self, config: CodePredictorConfig, talker_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        # Groups 1..N-1 embed in talker hidden size; group 0 reuses the
        # talker's own codec_embedding.
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, talker_hidden_size)
            for _ in range(config.num_code_groups - 1)
        ]
        self.layers = [
            CodePredictorDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
        mask: Optional[Union[str, mx.array]] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(inputs_embeds, cache[0])

        hidden_states = inputs_embeds
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)

    def make_cache(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.layers]


def reset_code_cache(cache: List[Any]) -> None:
    """Clear a code-predictor cache in place for the next codec frame.

    The predictor restarts from position 0 on every frame, so its KV is
    frame-local. Resetting in place keeps the per-frame allocation out of the
    decode hot loop.
    """
    for entry in cache:
        entry.keys = None
        entry.values = None
        entry.offset = 0


class Qwen3TTSCodePredictor(nn.Module):
    """Predicts codec groups 1..N-1 from group 0 and the talker hidden state."""

    def __init__(self, config: CodePredictorConfig, talker_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.num_code_groups = config.num_code_groups
        self.talker_hidden_size = talker_hidden_size

        if config.hidden_size != talker_hidden_size:
            self.small_to_mtp_projection = nn.Linear(
                talker_hidden_size, config.hidden_size, bias=True
            )
        else:
            self.small_to_mtp_projection = None

        self.model = CodePredictorModel(config, talker_hidden_size)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ]

    @property
    def codec_embedding(self) -> List[nn.Embedding]:
        return self.model.codec_embedding

    def project_input(self, hidden_states: mx.array) -> mx.array:
        if self.small_to_mtp_projection is None:
            return hidden_states
        return self.small_to_mtp_projection(hidden_states)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
        generation_step: int = 0,
    ) -> mx.array:
        hidden_states = self.model(self.project_input(inputs_embeds), cache=cache)
        return self.lm_head[generation_step](hidden_states)

    def make_cache(self) -> List[Any]:
        return self.model.make_cache()


def greedy_code_sampler(logits: mx.array, group_index: int) -> mx.array:
    """Default code sampler: argmax over ``[B, vocab]``, returning ``[B, 1]``."""
    del group_index
    return mx.argmax(logits, axis=-1, keepdims=True)


class Qwen3TTSTalkerForConditionalGeneration(nn.Module):
    """Full Qwen3-TTS talker: trunk + codec head + code predictor."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3TTSTalkerModel(config)
        self.text_projection = ResizeMLP(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            config.hidden_act,
            bias=True,
        )
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = Qwen3TTSCodePredictor(
            config.code_predictor_config,
            config.hidden_size,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        return self.model.text_embedding

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[List[Any]] = None,
        mask: Optional[Union[str, mx.array]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Return ``(codec logits, hidden states)`` for ``inputs_embeds``."""
        hidden_states = self.model(inputs_embeds, cache=cache, mask=mask)
        return self.codec_head(hidden_states), hidden_states

    def codec_embeds_for_codes(self, codes: List[mx.array]) -> mx.array:
        """Sum the per-group codec embeddings of one frame's codes.

        ``codes[0]`` uses the talker's own codec embedding; groups 1..N-1 use
        the predictor's. The result is the codec half of the next talker input.
        """
        summed = self.get_input_embeddings()(codes[0])
        for index, code in enumerate(codes[1:]):
            summed = summed + self.code_predictor.codec_embedding[index](code)
        return summed

    def predict_codes(
        self,
        first_code: mx.array,
        talker_hidden: mx.array,
        *,
        cache: Optional[List[Any]] = None,
        sampler: Callable[[mx.array, int], mx.array] = greedy_code_sampler,
    ) -> Tuple[mx.array, mx.array]:
        """Expand codec group 0 into a full frame of ``num_code_groups`` codes.

        ``first_code`` is ``[B, 1]`` and ``talker_hidden`` is ``[B, L, H]``
        (only its last position is read). ``sampler`` receives ``[B, vocab]``
        logits plus the 0-based group offset, so callers own group-specific
        sampling parameters; it defaults to greedy.

        Returns ``(codes [B, num_code_groups], summed codec embedding
        [B, 1, H])``. Everything stays lazy: no ``mx.eval`` here.
        """
        if cache is None:
            cache = self.code_predictor.make_cache()
        else:
            reset_code_cache(cache)

        codes = [first_code]
        code_hidden = talker_hidden[:, -1:, :]
        num_groups = self.config.code_predictor_config.num_code_groups

        for group_index in range(num_groups - 1):
            if group_index == 0:
                # Prime the frame with the talker hidden state followed by the
                # group-0 embedding, matching the reference two-token prefill.
                predictor_input = mx.concatenate(
                    [code_hidden, self.get_input_embeddings()(first_code)],
                    axis=1,
                )
            else:
                predictor_input = self.code_predictor.codec_embedding[group_index - 1](
                    codes[-1]
                )

            logits = self.code_predictor(
                predictor_input,
                cache=cache,
                generation_step=group_index,
            )
            codes.append(sampler(logits[:, -1, :], group_index))

        return mx.concatenate(codes, axis=1), self.codec_embeds_for_codes(codes)

    def make_cache(self) -> List[Any]:
        return self.model.make_cache()

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Keep the talker subtree of a Qwen3-TTS checkpoint, unprefixed.

        MLX ``nn.Linear`` uses the same ``[out, in]`` layout as PyTorch, so no
        transposition is needed. Checkpoints that are already talker-scoped
        pass through unchanged.
        """
        if not any(key.startswith("talker.") for key in weights):
            return dict(weights)
        return {
            key[len("talker.") :]: value
            for key, value in weights.items()
            if key.startswith("talker.")
        }
