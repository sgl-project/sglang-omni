# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni thinker (text decoder only).

The module implements the Transformers 5.12.1 ``Qwen3OmniMoeThinkerTextModel``
stack -- interleaved three-axis M-RoPE attention with per-head Q/K norms, sparse
MoE (or dense SwiGLU) MLPs, DeepStack visual residuals, and the LM head -- on top
of the reviewed primitives in :mod:`sglang_omni.models.qwen3_omni.mlx.common`.

Scope and contracts:

* **Encoders stay outside.** The vision and audio towers run in their own
  pipeline stages; this model consumes the merged embeddings they produce.
  :func:`merge_thinker_input_embeddings` and :func:`visual_placeholder_mask`
  reproduce the placeholder ordering of the existing Torch/CUDA thinker path
  (``sglang_omni/model_runner/thinker_model_runner.py``): the *k*-th placeholder
  row of a modality receives the *k*-th row of that modality's embeddings, and
  the DeepStack mask is the prompt-ordered union of image and video
  placeholders.
* **Hidden capture.** ``"embed"`` is the input to layer 0; an integer key ``N``
  is the input to layer ``N`` (the output of layer ``N-1``, *including* any
  DeepStack rows added after layer ``N-1``). Layer 0 is never emitted as a
  duplicate integer key. Prefill keeps every row as ``[sequence, hidden]``
  because the CUDA stream normalizer forwards the first prefill row into
  ``TalkerPrefillBuilder``; decode naturally yields a single row.
* **Batch-1, single device.** MLX serving is batch-1 by design, so positions are
  a ``(3, sequence)`` M-RoPE tensor and the sole batch row is asserted.
* **Attention shape.** The thinker attention is fully causal: the reference
  ``Qwen3OmniMoeThinkerTextAttention`` hard-codes ``sliding_window = None``
  regardless of the text config, so no sliding-window branch exists here.
* **No shared expert.** ``Qwen3OmniMoeThinkerTextSparseMoeBlock`` has routed
  experts only (unlike the talker block), so the shared-expert branch of
  :class:`SparseMoeBlock` is never enabled for the thinker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from sglang_omni.models.qwen3_omni.mlx.common import (
    SparseMoeBlock,
    _SwiGLUMLP,
    apply_multimodal_rope,
    sanitize_qwen3_omni_weights,
)
from sglang_omni.models.qwen3_omni.mlx.config import MoeTextConfig, Qwen3OmniMlxConfig

__all__ = [
    "MlxThinkerStep",
    "Qwen3OmniMlxThinker",
    "add_visual_rows",
    "merge_thinker_input_embeddings",
    "visual_placeholder_mask",
]

# Only these two namespaces of the official checkpoint belong to this module.
_THINKER_TEXT_PREFIXES = ("model.", "lm_head.")
# Non-persistent HF buffers that some exports still carry.
_DROPPED_SUFFIXES = ("rotary_emb.inv_freq", "rotary_emb.original_inv_freq")
_VISUAL_MODALITIES = ("image", "video")


@dataclass(slots=True)
class MlxThinkerStep:
    """One thinker step: sampling logits, captured hidden rows, and the cache."""

    logits: mx.array
    hidden_states: dict[int | str, mx.array] = field(default_factory=dict)
    cache: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Modality merging helpers (stage-facing)
# ---------------------------------------------------------------------------


def _rows_by_placeholder_slot(mask: mx.array, rows: mx.array) -> mx.array:
    """Broadcast ``rows`` so slot ``k`` lands on the ``k``-th masked position.

    ``mask`` is ``[batch, sequence]``; the exclusive running count of masked
    positions gives each masked token the index of its own embedding row, which
    keeps the mapping prompt-ordered without a host round-trip. Values under
    unmasked positions are meaningless and are always discarded by the caller.
    """

    slot = mx.cumsum(mask.astype(mx.int32), axis=-1) - 1
    return rows[mx.maximum(slot, 0)]


def merge_thinker_input_embeddings(
    text_embeddings: mx.array,
    input_ids: mx.array,
    *,
    modality_embeddings: Mapping[str, mx.array | None],
    placeholder_token_ids: Mapping[str, int],
) -> mx.array:
    """Scatter stage-provided modality embeddings onto their placeholder rows.

    Args:
        text_embeddings: ``[batch, sequence, hidden]`` embedding lookup output.
        input_ids: ``[batch, sequence]`` prompt token ids.
        modality_embeddings: Per-modality ``[rows, hidden]`` embeddings; ``None``
            or an absent modality is skipped.
        placeholder_token_ids: Modality name to placeholder token id.

    Returns:
        The merged ``[batch, sequence, hidden]`` embeddings.

    Raises:
        ValueError: If a modality is unknown, its hidden size disagrees with the
            text embeddings, or its row count does not match the number of
            placeholders in the prompt. The count check is the only host
            synchronization, and it runs once per modality per prefill.
    """

    if text_embeddings.ndim != 3:
        raise ValueError(
            "text_embeddings must be [batch, sequence, hidden], got shape "
            f"{text_embeddings.shape}"
        )
    if input_ids.shape != text_embeddings.shape[:2]:
        raise ValueError(
            f"input_ids shape {input_ids.shape} does not match embeddings "
            f"shape {text_embeddings.shape[:2]}"
        )

    merged = text_embeddings
    hidden_size = text_embeddings.shape[-1]
    for modality, embeddings in modality_embeddings.items():
        if embeddings is None:
            continue
        if modality not in placeholder_token_ids:
            raise ValueError(
                f"no placeholder token id configured for modality '{modality}'"
            )
        if embeddings.ndim != 2 or embeddings.shape[-1] != hidden_size:
            raise ValueError(
                f"{modality} embeddings must be [rows, {hidden_size}], got shape "
                f"{embeddings.shape}"
            )
        mask = input_ids == placeholder_token_ids[modality]
        placeholder_count = int(mask.sum().item())
        if placeholder_count != embeddings.shape[0]:
            raise ValueError(
                f"{modality} embeddings supply {embeddings.shape[0]} rows but the "
                f"prompt holds {placeholder_count} '{modality}' placeholders"
            )
        if placeholder_count == 0:
            continue
        gathered = _rows_by_placeholder_slot(mask, embeddings.astype(merged.dtype))
        merged = mx.where(mask[..., None], gathered, merged)
    return merged


def visual_placeholder_mask(
    input_ids: mx.array, *, placeholder_token_ids: Mapping[str, int]
) -> mx.array:
    """Prompt-ordered ``[batch, sequence]`` mask of image and video placeholders.

    DeepStack rows are indexed by this mask, so it must be the union of the image
    and video placeholder positions in prompt order -- exactly the sorted
    concatenation the Torch path builds -- and must never include audio rows.
    """

    mask = mx.zeros(input_ids.shape, dtype=mx.bool_)
    for modality in _VISUAL_MODALITIES:
        token_id = placeholder_token_ids.get(modality)
        if token_id is None:
            continue
        mask = mx.logical_or(mask, input_ids == token_id)
    return mask


def add_visual_rows(
    hidden: mx.array, visual_mask: mx.array | None, visual_embeds: mx.array
) -> mx.array:
    """Add one DeepStack layer's rows to the masked visual positions."""

    if visual_mask is None:
        raise ValueError("deepstack_visual_embeds requires a visual_mask")
    if visual_mask.shape != hidden.shape[:2]:
        raise ValueError(
            f"visual_mask shape {visual_mask.shape} does not match hidden shape "
            f"{hidden.shape[:2]}"
        )
    visual_count = int(visual_mask.sum().item())
    if visual_count != visual_embeds.shape[0]:
        raise ValueError(
            f"deepstack layer supplies {visual_embeds.shape[0]} rows but the mask "
            f"selects {visual_count} visual positions"
        )
    if visual_count == 0:
        return hidden
    gathered = _rows_by_placeholder_slot(
        visual_mask, visual_embeds.astype(hidden.dtype)
    )
    return hidden + mx.where(visual_mask[..., None], gathered, mx.zeros_like(gathered))


# ---------------------------------------------------------------------------
# Decoder stack
# ---------------------------------------------------------------------------


class ThinkerAttention(nn.Module):
    """M-RoPE attention with per-head Q/K RMS norms (reference layout)."""

    def __init__(self, config: MoeTextConfig, *, attention_bias: bool):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.mrope_section = config.mrope_section
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        positions: mx.array,
        mask: Any,
        cache: Any,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape

        queries = self.q_norm(
            self.q_proj(hidden_states).reshape(batch, length, self.num_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(hidden_states).reshape(batch, length, self.num_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(hidden_states)
            .reshape(batch, length, self.num_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )

        queries, keys = apply_multimodal_rope(
            queries,
            keys,
            positions,
            sections=self.mrope_section,
            base=self.rope_theta,
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attention = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attention)


class ThinkerDecoderLayer(nn.Module):
    """Pre-norm decoder layer with a sparse or dense MLP, as the config dictates."""

    def __init__(
        self, config: MoeTextConfig, layer_index: int, *, attention_bias: bool
    ):
        super().__init__()
        self.self_attn = ThinkerAttention(config, attention_bias=attention_bias)
        if is_sparse_layer(config, layer_index):
            # The thinker sparse block is routed-experts-only: unlike the talker
            # block it never carries a shared expert.
            self.mlp: nn.Module = SparseMoeBlock(
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
                shared_expert_intermediate_size=None,
            )
        else:
            # Reuse the Task 4 dense SwiGLU: identical gate/up/down weight names.
            self.mlp = _SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        positions: mx.array,
        mask: Any,
        cache: Any,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            positions=positions,
            mask=mask,
            cache=cache,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def is_sparse_layer(config: MoeTextConfig, layer_index: int) -> bool:
    """Reference rule for choosing the sparse MoE block over the dense MLP."""

    return (
        layer_index not in config.mlp_only_layers
        and config.num_experts > 0
        and (layer_index + 1) % config.decoder_sparse_step == 0
    )


class ThinkerTextModel(nn.Module):
    """Embedding table, decoder layers, and the final norm."""

    def __init__(self, config: MoeTextConfig, *, attention_bias: bool):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            ThinkerDecoderLayer(config, index, attention_bias=attention_bias)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3OmniMlxThinker(nn.Module):
    """Self-contained MLX thinker holding ``model.*`` and ``lm_head.*`` weights."""

    def __init__(self, config: MoeTextConfig, *, attention_bias: bool | None = None):
        super().__init__()
        self.config = config
        # ``attention_bias`` defaults to the value parsed from the checkpoint's
        # thinker text config (false for every published Qwen3-Omni release);
        # an explicit keyword still wins so unit fixtures can pin either wiring.
        self.attention_bias = (
            bool(config.attention_bias)
            if attention_bias is None
            else bool(attention_bias)
        )
        self.model = ThinkerTextModel(config, attention_bias=self.attention_bias)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_omni_config(
        cls, config: Qwen3OmniMlxConfig, **kwargs: Any
    ) -> "Qwen3OmniMlxThinker":
        return cls(config.thinker.text_config, **kwargs)

    # -- accessors ---------------------------------------------------------

    @property
    def embed_tokens(self) -> nn.Module:
        return self.model.embed_tokens

    @property
    def layers(self) -> list[ThinkerDecoderLayer]:
        return self.model.layers

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    def make_cache(self) -> list[Any]:
        """A fresh per-layer KV cache; ``cache[i].offset`` tracks stored tokens."""

        return [KVCache() for _ in range(self.num_layers)]

    # -- weights -----------------------------------------------------------

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        """Map official checkpoint weights onto this self-contained module.

        Keeps only ``thinker.model.*`` and ``thinker.lm_head.*``; the thinker's
        own vision/audio towers, the talker, and code2wav are dropped. Weights
        that are already in converted MLX layout pass through unchanged (so the
        call is idempotent and stays compatible with the quantized pre-load
        conversion in :func:`quantize_converted_module`). A tied model has no
        ``lm_head`` module, so a tied checkpoint's ``lm_head`` copy is dropped.
        """

        stripped = sanitize_qwen3_omni_weights(weights, component="thinker")
        sanitized: dict[str, mx.array] = {}
        for key, value in stripped.items():
            if not key.startswith(_THINKER_TEXT_PREFIXES):
                continue
            if key.endswith(_DROPPED_SUFFIXES):
                continue
            if self.config.tie_word_embeddings and key.startswith("lm_head."):
                continue
            sanitized[key] = value
        return sanitized

    # -- forward -----------------------------------------------------------

    def _apply_lm_head(self, hidden_states: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            # The MLX/mlx-lm idiom: project through the live embedding weight so
            # no later weight load can silently break the tie.
            return self.model.embed_tokens.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def _validate_capture_layers(
        self, capture_layers: Sequence[int]
    ) -> tuple[int, ...]:
        layers = tuple(int(layer) for layer in capture_layers)
        if len(set(layers)) != len(layers):
            raise ValueError(f"capture_layers contains duplicates: {layers}")
        out_of_range = [
            layer for layer in layers if layer < 0 or layer >= self.num_layers
        ]
        if out_of_range:
            raise ValueError(
                f"capture layers {out_of_range} are outside [0, {self.num_layers})"
            )
        return layers

    def _validate_inputs(self, embeddings: mx.array, positions: mx.array) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must be [batch, sequence, hidden], got {embeddings.shape}"
            )
        if embeddings.shape[0] != 1:
            raise ValueError(
                "the MLX thinker serves one request at a time, got batch "
                f"{embeddings.shape[0]}"
            )
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"mrope_positions must be [3, sequence], got {positions.shape}"
            )
        if positions.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"mrope_positions covers {positions.shape[1]} tokens but the step "
                f"holds {embeddings.shape[1]}"
            )

    def _forward(
        self,
        embeddings: mx.array,
        *,
        positions: mx.array,
        cache: list[Any],
        capture_layers: tuple[int, ...],
        deepstack_visual_embeds: list[mx.array] | None,
        visual_mask: mx.array | None,
    ) -> MlxThinkerStep:
        hidden = embeddings
        captured: dict[int | str, mx.array] = {"embed": hidden[0]}
        mask = create_attention_mask(hidden, cache[0])
        for layer_index, layer in enumerate(self.model.layers):
            if layer_index in capture_layers and layer_index != 0:
                captured[layer_index] = hidden[0]
            hidden = layer(
                hidden, positions=positions, mask=mask, cache=cache[layer_index]
            )
            if deepstack_visual_embeds is not None and layer_index < len(
                deepstack_visual_embeds
            ):
                hidden = add_visual_rows(
                    hidden, visual_mask, deepstack_visual_embeds[layer_index]
                )
        hidden = self.model.norm(hidden)
        logits = self._apply_lm_head(hidden[:, -1:, :])
        return MlxThinkerStep(logits=logits, hidden_states=captured, cache=cache)

    def prefill(
        self,
        input_ids: mx.array,
        *,
        input_embeddings: mx.array | None = None,
        mrope_positions: mx.array,
        deepstack_visual_embeds: list[mx.array] | None = None,
        visual_mask: mx.array | None = None,
        cache: list[Any] | None = None,
        capture_layers: tuple[int, ...] = (),
    ) -> MlxThinkerStep:
        """Run a prefill step over the merged prompt embeddings.

        ``input_embeddings`` carries the stage-merged text/image/audio rows (see
        :func:`merge_thinker_input_embeddings`); when omitted the prompt is
        text-only and is embedded here. ``deepstack_visual_embeds`` holds one
        ``[visual_rows, hidden]`` tensor per DeepStack layer, added after the
        matching decoder layer, so its effect first appears in the *next*
        captured layer.
        """

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        embeddings = (
            self.model.embed_tokens(input_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if input_ids.shape != embeddings.shape[:2]:
            raise ValueError(
                f"input_ids shape {input_ids.shape} does not match input_embeddings "
                f"shape {embeddings.shape[:2]}"
            )
        self._validate_inputs(embeddings, mrope_positions)
        layers = self._validate_capture_layers(capture_layers)
        if deepstack_visual_embeds is not None and len(deepstack_visual_embeds) > (
            self.num_layers
        ):
            raise ValueError(
                f"{len(deepstack_visual_embeds)} deepstack layers exceed the "
                f"{self.num_layers} decoder layers"
            )
        return self._forward(
            embeddings,
            positions=mrope_positions,
            cache=self.make_cache() if cache is None else cache,
            capture_layers=layers,
            deepstack_visual_embeds=deepstack_visual_embeds or None,
            visual_mask=visual_mask,
        )

    def decode(
        self,
        input_ids: mx.array,
        *,
        mrope_positions: mx.array,
        cache: list[Any],
        capture_layers: tuple[int, ...] = (),
    ) -> MlxThinkerStep:
        """Run one autoregressive step against an existing KV cache."""

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if len(cache) != self.num_layers:
            raise ValueError(
                f"cache holds {len(cache)} entries but the model has "
                f"{self.num_layers} layers"
            )
        embeddings = self.model.embed_tokens(input_ids)
        self._validate_inputs(embeddings, mrope_positions)
        layers = self._validate_capture_layers(capture_layers)
        return self._forward(
            embeddings,
            positions=mrope_positions,
            cache=cache,
            capture_layers=layers,
            deepstack_visual_embeds=None,
            visual_mask=None,
        )
