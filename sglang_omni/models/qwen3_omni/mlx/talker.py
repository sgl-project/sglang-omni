# SPDX-License-Identifier: Apache-2.0
"""Native MLX Qwen3-Omni talker backbone and per-step code predictor.

The module implements the Transformers 5.12.1 talker stack
(``Qwen3OmniMoeTalkerForConditionalGeneration``) on top of the reviewed
primitives in :mod:`sglang_omni.models.qwen3_omni.mlx.common` and the attention
block already validated for the thinker in
:mod:`sglang_omni.models.qwen3_omni.mlx.thinker`:

* :class:`TalkerResizeMLP` -- the ``text_projection`` / ``hidden_projection``
  resize MLPs that lift thinker-space rows into talker space.
* :class:`Qwen3OmniMlxTalker` -- codec embedding, interleaved three-axis M-RoPE
  decoder layers with per-head Q/K norms, sparse MoE with the talker's gated
  shared expert, final norm, and the ``codec_head`` that produces layer-0 codec
  logits.
* :class:`Qwen3OmniMlxCodePredictor` -- the residual RVQ predictor that expands
  one layer-0 codec token into ``num_code_groups`` ordered codes and the single
  summed feedback row the next talker step consumes.

Scope and contracts:

* **Greedy only.** Layer-0 selection is ``argmax`` over the suppressed codec
  logits and every code group is ``argmax`` over its own head, matching the
  Apple profile (``talker_temperature=0.0``, ``talker_top_k=-1``) and the
  existing Torch talker's ``_sample_code_predictor_token``.
* **Suppression happens before the layer-0 argmax**, exactly as
  ``Qwen3OmniMoeTalkerForConditionalGeneration.generate`` (``suppress_tokens``)
  and ``Qwen3OmniMoeTalkerTextModel._sample_decode_tokens`` do. The codec EOS is
  never part of the official suppression set, so it stays selectable.
* **Cache lifetime.** Only the outer talker KV cache lives across decode steps;
  the predictor allocates and drops its own cache inside every
  :meth:`Qwen3OmniMlxCodePredictor.generate_groups` call, so no residual
  group-expansion state can leak into the next talker token.
* **Decode recurrence.** The next talker input row is
  ``feedback + next_text_row`` -- the summed codec embeddings of the previous
  step plus the next projected thinker text row (or the TTS pad row once the
  text queue drains). This is HF's
  ``prepare_inputs_for_generation`` recurrence and the production Torch path in
  ``sglang_omni/models/qwen3_omni/talker_model_runner.py``
  (``_combine_feedback_with_next_text``).
* **All talker layers are sparse.** ``Qwen3OmniMoeTalkerDecoderLayer``
  overwrites its MLP with ``Qwen3OmniMoeTalkerTextSparseMoeBlock``
  unconditionally, so -- unlike the thinker -- ``mlp_only_layers`` and
  ``decoder_sparse_step`` never produce a dense talker layer.
* **Batch-1, single device, eager.** MLX serving is batch-1 by design, so
  positions are a ``(3, sequence)`` M-RoPE tensor and the sole batch row is
  asserted.
* **Scope.** No scheduler integration, streaming queue, or code2wav lives here;
  those belong to the wiring task.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from sglang_omni.models.qwen3_omni.mlx.common import (
    SparseMoeBlock,
    sanitize_qwen3_omni_weights,
)
from sglang_omni.models.qwen3_omni.mlx.config import (
    CodePredictorConfig,
    MoeTextConfig,
    Qwen3OmniMlxConfig,
    TalkerConfig,
)
from sglang_omni.models.qwen3_omni.mlx.thinker import ThinkerAttention

__all__ = [
    "MlxTalkerStep",
    "Qwen3OmniMlxCodePredictor",
    "Qwen3OmniMlxTalker",
    "TalkerResizeMLP",
    "build_suppress_mask",
    "mask_suppressed_logits",
]

# Only these namespaces of the official checkpoint belong to this module.
_TALKER_PREFIXES = (
    "model.",
    "codec_head.",
    "text_projection.",
    "hidden_projection.",
    "code_predictor.",
)
# Non-persistent HF buffers that some exports still carry.
_DROPPED_SUFFIXES = ("rotary_emb.inv_freq", "rotary_emb.original_inv_freq")


@dataclass(slots=True)
class MlxTalkerStep:
    """One talker step: the layer-0 token, its hidden row, codes, and feedback.

    ``codes`` holds ``num_code_groups`` integer codes in official group order
    (group 0 is the layer-0 codec token). ``feedback`` is the single summed code
    embedding row that the *next* talker step adds to its text contribution.
    ``cache`` is the outer talker KV cache and is the only state that survives
    the step.

    ``layer0_token`` is int32 at this boundary: the scheduler publishes it as a
    codec token id, so the model -- not its caller -- owns the cast away from
    ``argmax``'s platform-dependent index dtype.
    """

    layer0_token: mx.array
    hidden: mx.array
    codes: mx.array
    feedback: mx.array
    cache: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer-0 suppression
# ---------------------------------------------------------------------------


def build_suppress_mask(vocab_size: int, suppress_tokens: Iterable[int]) -> mx.array:
    """Build the additive ``[vocab_size]`` mask for suppressed codec tokens.

    The mask is ``-inf`` on suppressed ids and ``0`` elsewhere, so adding it to
    the codec logits removes those ids from the greedy argmax. Building it once
    per request (rather than per step) is the reason this is a public helper.

    Raises:
        ValueError: If any id falls outside ``[0, vocab_size)``, or if the set
            would suppress the entire vocabulary (which would make the argmax
            meaningless rather than merely constrained).
    """

    unique = sorted({int(token_id) for token_id in suppress_tokens})
    invalid = [token_id for token_id in unique if not 0 <= token_id < vocab_size]
    if invalid:
        raise ValueError(f"suppress token ids {invalid} are outside [0, {vocab_size})")
    if len(unique) >= vocab_size:
        raise ValueError("refusing to suppress every token of the codec vocabulary")
    mask = mx.zeros((vocab_size,), dtype=mx.float32)
    if unique:
        mask[mx.array(unique, dtype=mx.int32)] = mx.array(-math.inf, dtype=mx.float32)
    return mask


def mask_suppressed_logits(
    logits: mx.array, suppress: Sequence[int] | mx.array | None
) -> mx.array:
    """Apply codec suppression to logits *before* the layer-0 argmax.

    ``suppress`` is either ``None``, a sequence of token ids, or an already
    built additive mask from :func:`build_suppress_mask` (the cheap per-step
    path). Returns ``logits`` unchanged when nothing is suppressed.
    """

    if suppress is None:
        return logits
    vocab_size = logits.shape[-1]
    if isinstance(suppress, mx.array):
        if suppress.shape != (vocab_size,):
            raise ValueError(
                f"suppress mask shape {suppress.shape} does not match the codec "
                f"vocabulary ({vocab_size},)"
            )
        mask = suppress
    else:
        if len(suppress) == 0:
            return logits
        mask = build_suppress_mask(vocab_size, suppress)
    return logits + mask.astype(logits.dtype)


# ---------------------------------------------------------------------------
# Resize MLPs
# ---------------------------------------------------------------------------


class TalkerResizeMLP(nn.Module):
    """``Qwen3OmniMoeTalkerResizeMLP``: thinker hidden size -> talker hidden size.

    Both projections carry biases in the official checkpoint, and the activation
    is the config's ``hidden_act`` (SiLU for every published Qwen3-Omni release).
    """

    def __init__(
        self, *, thinker_hidden_size: int, intermediate_size: int, hidden_size: int
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(thinker_hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear_fc2(nn.silu(self.linear_fc1(hidden_states)))


# ---------------------------------------------------------------------------
# Talker backbone
# ---------------------------------------------------------------------------


class TalkerDecoderLayer(nn.Module):
    """Pre-norm decoder layer: M-RoPE attention plus the gated shared-expert MoE.

    ``Qwen3OmniMoeTalkerDecoderLayer`` reuses the thinker's attention block
    verbatim (interleaved three-axis M-RoPE, per-head Q/K RMS norms, no sliding
    window) and then *unconditionally* assigns
    ``Qwen3OmniMoeTalkerTextSparseMoeBlock`` -- the dense branch it appears to
    select first is dead code upstream. The talker therefore has no dense
    layers, which is why this class takes no layer index.
    """

    def __init__(self, config: MoeTextConfig, *, attention_bias: bool):
        super().__init__()
        self.self_attn = ThinkerAttention(config, attention_bias=attention_bias)
        self.mlp = SparseMoeBlock(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
        )
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


class TalkerTextModel(nn.Module):
    """Codec embedding table, decoder layers, and the final norm."""

    def __init__(self, config: MoeTextConfig, *, attention_bias: bool):
        super().__init__()
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TalkerDecoderLayer(config, attention_bias=attention_bias)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# ---------------------------------------------------------------------------
# Code predictor
# ---------------------------------------------------------------------------


class CodePredictorAttention(nn.Module):
    """Standard (single-axis) rotary attention with per-head Q/K RMS norms.

    ``Qwen3OmniMoeTalkerCodePredictorAttention`` uses
    ``Qwen3OmniMoeRotaryEmbedding`` -- the plain default rope, *not* the
    interleaved three-axis M-RoPE of the talker backbone -- over the predictor's
    own short sequence (the talker hidden row, the layer-0 code embedding, then
    one row per generated group).
    """

    def __init__(self, config: CodePredictorConfig, *, attention_bias: bool):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

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
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, hidden_states: mx.array, *, mask: Any, cache: Any) -> mx.array:
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

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attention = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(attention)


class CodePredictorMLP(nn.Module):
    """``Qwen3OmniMoeMLP``: the predictor's dense SwiGLU block."""

    def __init__(self, config: CodePredictorConfig):
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

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CodePredictorDecoderLayer(nn.Module):
    """Pre-norm decoder layer of the code predictor stack."""

    def __init__(self, config: CodePredictorConfig, *, attention_bias: bool):
        super().__init__()
        self.self_attn = CodePredictorAttention(config, attention_bias=attention_bias)
        self.mlp = CodePredictorMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, hidden_states: mx.array, *, mask: Any, cache: Any) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), mask=mask, cache=cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class CodePredictorModel(nn.Module):
    """Per-group codec embeddings, decoder layers, and the final norm.

    ``codec_embedding`` holds ``num_code_groups - 1`` tables: group ``g`` (for
    ``g >= 1``) is embedded by table ``g - 1``, matching upstream's
    ``codec_embedding[generation_steps - 1]`` indexing. Group 0 is embedded by
    the *talker's* ``model.codec_embedding``, not by this list.
    """

    def __init__(self, config: CodePredictorConfig, *, attention_bias: bool):
        super().__init__()
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ]
        self.layers = [
            CodePredictorDecoderLayer(config, attention_bias=attention_bias)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3OmniMlxCodePredictor(nn.Module):
    """Greedy residual-RVQ predictor for one layer-0 codec token.

    Mirrors ``Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration``:
    the predictor prompt is ``[talker_hidden_row, layer0_code_embedding]``, head
    ``g`` reads the last hidden row to emit group ``g + 1``, and each emitted
    code is fed back through ``codec_embedding[g]``.
    """

    def __init__(self, config: CodePredictorConfig, *, attention_bias: bool = False):
        super().__init__()
        self.config = config
        self.model = CodePredictorModel(config, attention_bias=attention_bias)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ]

    @property
    def num_code_groups(self) -> int:
        return self.config.num_code_groups

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    def make_cache(self) -> list[Any]:
        """A fresh per-layer KV cache for one group expansion."""

        return [KVCache() for _ in range(self.num_layers)]

    def _forward(self, embeddings: mx.array, *, cache: list[Any]) -> mx.array:
        hidden = embeddings
        mask = create_attention_mask(hidden, cache[0])
        for index, layer in enumerate(self.model.layers):
            hidden = layer(hidden, mask=mask, cache=cache[index])
        return self.model.norm(hidden)

    def generate_groups(
        self,
        *,
        layer0_token: mx.array,
        layer0_embed: mx.array,
        talker_hidden: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Expand one layer-0 token into ordered codes and one feedback row.

        Args:
            layer0_token: ``[1]`` greedy codec token of the current talker step.
            layer0_embed: ``[1, 1, hidden]`` embedding of that token taken from
                the *talker's* codec embedding table (upstream's
                ``last_id_hidden``).
            talker_hidden: ``[1, 1, hidden]`` last talker hidden row (upstream's
                ``past_hidden``).

        Returns:
            ``(codes, feedback)`` with ``codes`` shaped ``[1, num_code_groups]``
            in official group order and ``feedback`` shaped ``[1, hidden]``:
            the sum of the selected code embeddings across every group.

        The KV cache is created here and dropped on return, so the predictor
        carries no state between outer talker tokens.
        """

        hidden_size = self.config.hidden_size
        if layer0_token.ndim != 1 or layer0_token.shape[0] != 1:
            raise ValueError(
                f"layer0_token must be [1], got shape {layer0_token.shape}"
            )
        for name, value in (
            ("layer0_embed", layer0_embed),
            ("talker_hidden", talker_hidden),
        ):
            if value.shape != (1, 1, hidden_size):
                raise ValueError(
                    f"{name} must be [1, 1, {hidden_size}], got shape {value.shape}"
                )

        cache = self.make_cache()
        # Upstream prefills both prompt rows in one call; the predictor's
        # attention is causal, so this is the same computation the incremental
        # Torch path performs one row at a time.
        prompt = mx.concatenate(
            [talker_hidden, layer0_embed.astype(talker_hidden.dtype)], axis=1
        )
        last_hidden = self._forward(prompt, cache=cache)[:, -1:, :]

        codes = [layer0_token.reshape(1, 1).astype(mx.int32)]
        feedback = layer0_embed[:, 0, :]
        for group in range(self.num_code_groups - 1):
            logits = self.lm_head[group](last_hidden)
            code = mx.argmax(logits[:, -1, :], axis=-1)
            codes.append(code.reshape(1, 1).astype(mx.int32))
            embed = self.model.codec_embedding[group](code)[:, None, :]
            feedback = feedback + embed[:, 0, :]
            if group < self.num_code_groups - 2:
                # The final group is never fed back through the stack: upstream
                # only embeds it (``last_residual_hidden``) for the sum.
                last_hidden = self._forward(embed, cache=cache)
        return mx.concatenate(codes, axis=1), feedback


# ---------------------------------------------------------------------------
# Talker
# ---------------------------------------------------------------------------


class Qwen3OmniMlxTalker(nn.Module):
    """Self-contained MLX talker: backbone, projections, codec head, predictor."""

    def __init__(
        self,
        config: TalkerConfig,
        *,
        thinker_hidden_size: int,
        attention_bias: bool | None = None,
    ):
        super().__init__()
        self.config = config
        text_config = config.text_config
        predictor_config = config.code_predictor_config
        # Group 0 is emitted by this module's codec head and embedded by the
        # talker's own codec table; groups 1..N-1 belong to the predictor's
        # per-group tables and heads. A disagreement would build a talker that
        # emits a different number of codes than the predictor can expand.
        if int(config.num_code_groups) != int(predictor_config.num_code_groups):
            raise ValueError(
                f"talker num_code_groups={int(config.num_code_groups)} disagrees "
                "with code_predictor_config.num_code_groups="
                f"{int(predictor_config.num_code_groups)}"
            )
        self.thinker_hidden_size = int(thinker_hidden_size)
        # ``attention_bias`` defaults to the value parsed from the checkpoint's
        # talker text config (false for every published Qwen3-Omni release); an
        # explicit keyword still wins so unit fixtures can pin either wiring.
        self.attention_bias = (
            bool(text_config.attention_bias)
            if attention_bias is None
            else bool(attention_bias)
        )
        self.model = TalkerTextModel(text_config, attention_bias=self.attention_bias)
        self.text_projection = TalkerResizeMLP(
            thinker_hidden_size=self.thinker_hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_size=text_config.hidden_size,
        )
        self.hidden_projection = TalkerResizeMLP(
            thinker_hidden_size=self.thinker_hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_size=text_config.hidden_size,
        )
        self.codec_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.code_predictor = Qwen3OmniMlxCodePredictor(
            config.code_predictor_config, attention_bias=self.attention_bias
        )

    @classmethod
    def from_omni_config(
        cls, config: Qwen3OmniMlxConfig, **kwargs: Any
    ) -> "Qwen3OmniMlxTalker":
        """Build from the nested Omni config.

        The resize MLPs are sized from ``talker_config.thinker_hidden_size``,
        which is the field the published checkpoints actually declare for them.
        A config predating that field falls back to the parsed thinker text
        hidden size (the two are equal in every published release), and a
        config that declares a value contradicting the thinker it is paired
        with is rejected rather than silently sizing the projections wrong.
        """

        declared = config.talker.thinker_hidden_size
        thinker_hidden_size = config.thinker.text_config.hidden_size
        if declared is not None:
            if int(declared) != int(thinker_hidden_size):
                raise ValueError(
                    f"talker_config.thinker_hidden_size={int(declared)} disagrees "
                    "with thinker_config.text_config.hidden_size="
                    f"{int(thinker_hidden_size)}"
                )
            thinker_hidden_size = int(declared)
        kwargs.setdefault("thinker_hidden_size", thinker_hidden_size)
        return cls(config.talker, **kwargs)

    # -- accessors ---------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def num_code_groups(self) -> int:
        return self.config.num_code_groups

    @property
    def vocab_size(self) -> int:
        return self.config.text_config.vocab_size

    def make_cache(self) -> list[Any]:
        """A fresh per-layer talker KV cache; ``cache[i].offset`` counts tokens."""

        return [KVCache() for _ in range(self.num_layers)]

    # -- weights -----------------------------------------------------------

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        """Map official checkpoint weights onto this self-contained module.

        Accepts both the official ``talker.*`` keys and already stripped
        converted keys, drops the thinker/code2wav namespaces and the
        non-persistent rotary buffers, fuses the published per-expert MoE
        linears into the quantizable ``SwitchLinear`` stacks, and leaves
        quantized ``.scales``/``.biases`` untouched so a converted 4-bit
        checkpoint loads. The mapping is idempotent.
        """

        stripped = sanitize_qwen3_omni_weights(weights, component="talker")
        sanitized: dict[str, mx.array] = {}
        for key, value in stripped.items():
            if not key.startswith(_TALKER_PREFIXES):
                continue
            if key.endswith(_DROPPED_SUFFIXES):
                continue
            sanitized[key] = value
        return sanitized

    # -- projections -------------------------------------------------------

    def project_prompt(
        self,
        input_embeddings: mx.array,
        *,
        multimodal_hidden: mx.array | None = None,
        multimodal_mask: mx.array | None = None,
    ) -> mx.array:
        """Lift thinker-space prompt rows into talker space.

        Text rows go through ``text_projection`` and multimodal rows through
        ``hidden_projection``, matching ``_get_talker_user_parts`` and the Torch
        path's ``prepare_input_embeds``. ``multimodal_hidden`` carries the
        thinker's ``accept_hidden_layer`` rows.
        """

        if input_embeddings.ndim != 3:
            raise ValueError(
                "input_embeddings must be [batch, sequence, hidden], got shape "
                f"{input_embeddings.shape}"
            )
        if input_embeddings.shape[-1] != self.thinker_hidden_size:
            raise ValueError(
                f"unprojected prompt rows must carry the thinker hidden size "
                f"{self.thinker_hidden_size}, got {input_embeddings.shape[-1]}"
            )
        projected = self.text_projection(input_embeddings)
        if multimodal_mask is None:
            if multimodal_hidden is not None:
                raise ValueError(
                    "multimodal_hidden requires a multimodal_mask selecting its rows"
                )
            return projected
        if multimodal_hidden is None:
            raise ValueError(
                "multimodal_mask requires multimodal_hidden rows to project"
            )
        if multimodal_hidden.shape != input_embeddings.shape:
            raise ValueError(
                f"multimodal_hidden shape {multimodal_hidden.shape} does not match "
                f"input_embeddings shape {input_embeddings.shape}"
            )
        if multimodal_mask.shape != input_embeddings.shape[:2]:
            raise ValueError(
                f"multimodal_mask shape {multimodal_mask.shape} does not match "
                f"[batch, sequence] {input_embeddings.shape[:2]}"
            )
        return mx.where(
            multimodal_mask[..., None],
            self.hidden_projection(multimodal_hidden),
            projected,
        )

    # -- forward -----------------------------------------------------------

    def _validate_step(self, embeddings: mx.array, positions: mx.array) -> None:
        hidden_size = self.config.text_config.hidden_size
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must be [batch, sequence, hidden], got {embeddings.shape}"
            )
        if embeddings.shape[0] != 1:
            raise ValueError(
                "the MLX talker serves one request at a time, got batch "
                f"{embeddings.shape[0]}"
            )
        if embeddings.shape[-1] != hidden_size:
            raise ValueError(
                f"projected rows must carry the talker hidden size {hidden_size}, "
                f"got {embeddings.shape[-1]}"
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

    def _run_backbone(
        self, embeddings: mx.array, *, positions: mx.array, cache: list[Any]
    ) -> mx.array:
        hidden = embeddings
        mask = create_attention_mask(hidden, cache[0])
        for index, layer in enumerate(self.model.layers):
            hidden = layer(hidden, positions=positions, mask=mask, cache=cache[index])
        return self.model.norm(hidden)

    def _step(
        self,
        embeddings: mx.array,
        *,
        positions: mx.array,
        cache: list[Any],
        suppress_tokens: Sequence[int] | mx.array | None,
    ) -> MlxTalkerStep:
        self._validate_step(embeddings, positions)
        if len(cache) != self.num_layers:
            raise ValueError(
                f"cache holds {len(cache)} entries but the talker has "
                f"{self.num_layers} layers"
            )
        hidden = self._run_backbone(embeddings, positions=positions, cache=cache)[
            :, -1:, :
        ]
        logits = mask_suppressed_logits(self.codec_head(hidden), suppress_tokens)
        layer0_token = mx.argmax(logits[:, -1, :], axis=-1)
        layer0_embed = self.model.codec_embedding(layer0_token[:, None])
        codes, feedback = self.code_predictor.generate_groups(
            layer0_token=layer0_token,
            layer0_embed=layer0_embed,
            talker_hidden=hidden,
        )
        return MlxTalkerStep(
            layer0_token=layer0_token.astype(mx.int32),
            hidden=hidden,
            codes=codes,
            feedback=feedback,
            cache=cache,
        )

    def prefill(
        self,
        input_embeddings: mx.array,
        *,
        mrope_positions: mx.array,
        input_embeddings_are_projected: bool = False,
        multimodal_hidden: mx.array | None = None,
        multimodal_mask: mx.array | None = None,
        suppress_tokens: Sequence[int] | mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> MlxTalkerStep:
        """Run the talker prompt and expand the first code frame.

        ``input_embeddings_are_projected`` distinguishes the two production
        prompt modes: the prompt builder emits rows that are *already* in talker
        space (projected text rows plus codec special-token embeddings), whereas
        a raw thinker-hidden prompt is projected here -- exactly once.
        """

        projected = (
            input_embeddings
            if input_embeddings_are_projected
            else self.project_prompt(
                input_embeddings,
                multimodal_hidden=multimodal_hidden,
                multimodal_mask=multimodal_mask,
            )
        )
        if input_embeddings_are_projected and (
            multimodal_hidden is not None or multimodal_mask is not None
        ):
            raise ValueError(
                "an already projected prompt cannot also carry multimodal rows "
                "to project"
            )
        return self._step(
            projected,
            positions=mrope_positions,
            cache=self.make_cache() if cache is None else cache,
            suppress_tokens=suppress_tokens,
        )

    def decode(
        self,
        *,
        feedback: mx.array,
        next_text_row: mx.array | None,
        mrope_positions: mx.array,
        cache: list[Any],
        suppress_tokens: Sequence[int] | mx.array | None = None,
    ) -> MlxTalkerStep:
        """Run one autoregressive talker step against an existing KV cache.

        The input row is ``feedback + next_text_row``: the previous step's
        summed code embeddings plus the next projected thinker text row, or the
        projected TTS pad row once the text queue drains. Both rows are already
        in talker space; the caller owns the queue policy, exactly as
        ``QwenTalkerModelRunner._combine_feedback_with_next_text`` does.
        """

        if next_text_row is None:
            raise ValueError(
                "decode requires a next_text_row: the talker step consumes the "
                "next projected text row, or the TTS pad row once the text "
                "queue is drained"
            )
        row = self._as_step_row(feedback, name="feedback")
        text = self._as_step_row(next_text_row, name="next_text_row")
        return self._step(
            row + text,
            positions=mrope_positions,
            cache=cache,
            suppress_tokens=suppress_tokens,
        )

    def _as_step_row(self, value: mx.array, *, name: str) -> mx.array:
        """Normalize a single talker-space row onto ``[1, 1, hidden]``."""

        hidden_size = self.config.text_config.hidden_size
        if value.ndim == 1:
            value = value[None, None, :]
        elif value.ndim == 2:
            value = value[:, None, :]
        if value.shape != (1, 1, hidden_size):
            raise ValueError(
                f"{name} must be one talker-space row of size {hidden_size}, got "
                f"shape {value.shape}"
            )
        return value
