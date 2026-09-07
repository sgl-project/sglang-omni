# SPDX-License-Identifier: Apache-2.0
"""Native MLX primitives for Qwen3-Omni thinker/talker stacks.

This module implements the correctness-critical foundations shared by the later
thinker and talker models:

* :func:`apply_multimodal_rope` -- interleaved three-axis (temporal/height/width)
  M-RoPE that matches the Transformers 5.12.1 reference rotary embedding.
* :class:`SparseMoeBlock` -- routed SwiGLU experts with optional talker shared
  expert, matching the reference MoE math (softmax router, top-k selection,
  optional top-k normalization, scatter-add of expert contributions).
* :func:`sanitize_qwen3_omni_weights` -- HF -> MLX weight sanitation that strips
  the selected component prefix, transposes only convolution kernels, and
  normalizes the fused MoE expert stacks onto their quantizable modules.
* :func:`tie_lm_head_weights` -- genuine module-level embedding tying.
* :func:`quantize_converted_module` -- quantizes only the linear layers that the
  converted checkpoint actually represents, rejecting incomplete groups.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchLinear

from sglang_omni.models.qwen3_omni.mlx.config import QuantizationConfig

_COMPONENT_PREFIXES = ("thinker.", "talker.", "code2wav.")
# The fused MoE expert stacks are stored as one 3-D tensor per projection in the
# HF checkpoint. They live on quantizable ``SwitchLinear`` modules here, so a
# converted 4-bit checkpoint can carry ``.scales``/``.biases`` beside them.
_EXPERT_STACK_NAMES = ("gate_up_proj", "down_proj")
# Published checkpoints serialize one linear per expert instead of the fused
# stack the module holds; ``stack_legacy_expert_weights`` folds them together.
_LEGACY_EXPERT_KEY = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<index>\d+)"
    r"\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_MLX_VLM_EXPERT_KEY = re.compile(
    r"^(?P<prefix>.*)\.switch_mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<suffix>weight|scales|biases)$"
)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _interleave_mrope(freqs: mx.array, sections: tuple[int, ...]) -> mx.array:
    """Reorganize chunked ``[TTT..HHH..WWW]`` freqs into interleaved ``[THWTHW..]``.

    ``freqs`` has shape ``(3, seq, head_dim // 2)`` (temporal/height/width rows).
    The result ``(seq, head_dim // 2)`` starts from the temporal row and then
    overwrites the height and width columns exactly as the reference
    ``Qwen3OmniMoeThinkerTextRotaryEmbedding.apply_interleaved_mrope`` does.

    The reference writes ``freqs_t[..., slice(offset, section * 3, 3)]``. A
    Python slice clamps its stop, so a section whose interleaved columns run
    past ``head_dim // 2`` writes only the columns that fit and the tail stays
    temporal. Clamping the column range here reproduces that instead of
    indexing off the end.
    """

    half = freqs.shape[-1]
    axis_of_col = [0] * half
    for col in range(1, min(sections[1] * 3, half), 3):
        axis_of_col[col] = 1
    for col in range(2, min(sections[2] * 3, half), 3):
        axis_of_col[col] = 2

    # (seq, half, 3) so each column can select its temporal/height/width source.
    per_axis = freqs.transpose(1, 2, 0)
    selector = mx.array(axis_of_col, dtype=mx.int32)
    selector = mx.broadcast_to(selector[None, :, None], (per_axis.shape[0], half, 1))
    return mx.take_along_axis(per_axis, selector, axis=2)[:, :, 0]


def apply_multimodal_rope(
    q: mx.array,
    k: mx.array,
    positions: mx.array,
    *,
    sections: tuple[int, ...],
    base: float,
) -> tuple[mx.array, mx.array]:
    """Apply interleaved three-axis M-RoPE to query and key tensors.

    Args:
        q: Query tensor shaped ``(batch, heads, seq, head_dim)``.
        k: Key tensor shaped ``(batch, kv_heads, seq, head_dim)``.
        positions: Integer position ids shaped ``(3, seq)`` holding the
            temporal, height, and width position rows.
        sections: The ``mrope_section`` widths (summing to ``head_dim // 2``).
        base: The rotary ``rope_theta``.

    Returns:
        The rotated ``(q, k)`` tensors with the same shapes as the inputs.
    """

    head_dim = q.shape[-1]
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    pos = positions.astype(mx.float32)  # (3, seq)
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # (3, seq, head_dim // 2)
    freqs_t = _interleave_mrope(freqs, sections)  # (seq, head_dim // 2)
    emb = mx.concatenate([freqs_t, freqs_t], axis=-1)  # (seq, head_dim)
    cos = mx.cos(emb)[None, None, :, :]
    sin = mx.sin(emb)[None, None, :, :]

    # Preserve each input's dtype: the reference casts its float32 cos/sin back
    # to the query/key dtype, so low-precision (fp16/bf16) Q/K must not be
    # silently upcast to float32 here.
    q_embed = q * cos.astype(q.dtype) + _rotate_half(q) * sin.astype(q.dtype)
    k_embed = k * cos.astype(k.dtype) + _rotate_half(k) * sin.astype(k.dtype)
    return q_embed, k_embed


class _Experts(nn.Module):
    """Stacked expert projections, one quantizable module per HF 3-D tensor.

    The HF checkpoint stores ``experts.gate_up_proj`` as a single
    ``(num_experts, 2 * intermediate, hidden)`` parameter and
    ``experts.down_proj`` as ``(num_experts, hidden, intermediate)`` -- exactly
    the ``mlx-lm`` :class:`SwitchLinear` ``(experts, out, in)`` layout. Holding
    them as ``SwitchLinear`` modules (rather than raw arrays) is what makes a
    converted 4-bit checkpoint loadable: ``SwitchLinear.to_quantized()`` swaps in
    ``QuantizedSwitchLinear``, whose ``weight``/``scales``/``biases`` parameters
    are exactly the tensors such a checkpoint ships. A raw array could never
    consume the expert ``scales``.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            hidden_size, 2 * intermediate_size, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(
            intermediate_size, hidden_size, num_experts, bias=False
        )


class _SwiGLUMLP(nn.Module):
    """Dense SwiGLU MLP used by the talker shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Routed SwiGLU MoE block with an optional talker shared expert.

    The parameter names (``gate.weight``, ``experts.gate_up_proj``,
    ``experts.down_proj``, ``shared_expert.*``, ``shared_expert_gate.weight``)
    mirror the Transformers 5.12.1 sparse MoE blocks so converted checkpoints
    load without renaming.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        shared_expert_intermediate_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = _Experts(num_experts, hidden_size, moe_intermediate_size)

        self.has_shared_expert = bool(shared_expert_intermediate_size)
        if self.has_shared_expert:
            self.shared_expert = _SwiGLUMLP(
                hidden_size, int(shared_expert_intermediate_size)
            )
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def _route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        router_logits = self.gate(x)
        router_probs = mx.softmax(router_logits.astype(mx.float32), axis=-1)
        top_indices = mx.argpartition(-router_probs, kth=self.top_k - 1, axis=-1)[
            :, : self.top_k
        ]
        top_weights = mx.take_along_axis(router_probs, top_indices, axis=-1)
        if self.norm_topk_prob:
            top_weights = top_weights / mx.maximum(
                top_weights.sum(axis=-1, keepdims=True),
                mx.array(1e-12, dtype=top_weights.dtype),
            )
        return top_indices, top_weights.astype(x.dtype)

    def _dispatch_experts(self, x: mx.array, indices: mx.array) -> mx.array:
        """Route each token through only its selected top-k experts.

        ``indices`` has shape ``(tokens, top_k)``. The ``SwitchLinear`` stacks
        gather the expert weight matrix for each ``(token, expert-slot)`` pair by
        index (``mx.gather_mm``, or ``mx.gather_qmm`` once quantized) instead of
        computing every expert over every token, so total expert compute is
        exactly ``tokens * top_k`` regardless of ``num_experts`` -- no
        non-selected token row is ever multiplied by a non-routed expert's
        weights.
        """

        xs = mx.expand_dims(x, (-2, -3))  # (tokens, 1, 1, hidden)
        gate_up = self.experts.gate_up_proj(
            xs, indices
        )  # (tokens, top_k, 1, 2 * intermediate)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = nn.silu(gate) * up
        expert_out = self.experts.down_proj(
            hidden, indices
        )  # (tokens, top_k, 1, hidden)
        return expert_out.squeeze(-2)  # (tokens, top_k, hidden)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)  # (tokens, hidden)

        top_indices, top_weights = self._route(x)

        expert_out = self._dispatch_experts(x, top_indices)  # (tokens, top_k, hidden)
        # Every top-k slot for a token already maps only to that token's own
        # row, so summing across the slot axis is exactly the scatter-add of
        # each selected expert's weighted contribution back into the token's
        # output position -- including tokens routed to multiple experts.
        output = (expert_out * top_weights[..., None]).sum(axis=-2)

        if self.has_shared_expert:
            shared_out = self.shared_expert(x)
            shared_out = mx.sigmoid(self.shared_expert_gate(x)) * shared_out
            output = output + shared_out

        return output.reshape(batch_size, sequence_length, hidden_dim)


def _is_convolution_weight(key: str, value: mx.array) -> bool:
    """Identify HF convolution kernels by module name, not by rank alone.

    Rank-3 expert stacks (``experts.gate_up_proj`` / ``experts.down_proj``) must
    not be treated as ``Conv1d`` kernels, so the decision is name-based.
    """

    if not key.endswith(".weight"):
        return False
    if value.ndim not in (3, 4):
        return False
    return any(part.startswith("conv") for part in key.split("."))


def normalize_expert_stack_key(key: str) -> str:
    """Map a fused MoE expert stack onto its ``SwitchLinear`` weight name.

    HF stores the stack as the bare parameter ``...mlp.experts.gate_up_proj``;
    a converter that quantizes that parameter in place keeps the same name for
    the packed tensor and adds ``...experts.gate_up_proj.scales`` /
    ``.biases`` beside it. Both spellings therefore have to land on this
    module's ``...experts.gate_up_proj.weight``. Keys that already carry an
    explicit ``.weight``/``.scales``/``.biases`` suffix pass through unchanged,
    which keeps the mapping idempotent.
    """

    parts = key.split(".")
    if len(parts) >= 2 and parts[-2] == "experts" and parts[-1] in _EXPERT_STACK_NAMES:
        return f"{key}.weight"
    return key


def stack_legacy_expert_weights(
    weights: Mapping[str, mx.array],
) -> dict[str, mx.array]:
    """Fuse per-expert MoE linears into the stacked expert layout.

    Published Qwen3-Omni checkpoints serialize each expert as its own trio of
    linears -- ``...mlp.experts.<e>.gate_proj.weight`` ``(intermediate, hidden)``,
    ``...up_proj.weight`` ``(intermediate, hidden)`` and ``...down_proj.weight``
    ``(hidden, intermediate)`` -- even though the Transformers module holds one
    fused ``gate_up_proj`` parameter. They are stacked here in expert order, with
    ``gate`` above ``up`` so ``chunk(2, dim=-1)`` of the fused projection matches
    the reference split.

    A group missing an expert index or one of the three projections raises
    naming the layer, so a truncated checkpoint can never load as a silently
    smaller MoE.
    """

    grouped: dict[str, dict[str, dict[int, mx.array]]] = {}
    stacked: dict[str, mx.array] = {}
    for key, value in weights.items():
        match = _LEGACY_EXPERT_KEY.match(key)
        if match is None:
            stacked[key] = value
            continue
        projections = grouped.setdefault(match.group("prefix"), {})
        projections.setdefault(match.group("projection"), {})[
            int(match.group("index"))
        ] = value

    for prefix, projections in grouped.items():
        missing = sorted({"gate_proj", "up_proj", "down_proj"} - set(projections))
        if missing:
            raise ValueError(
                f"per-expert MoE weights for '{prefix}' are missing {missing}"
            )
        indices = sorted(projections["down_proj"])
        expected = list(range(len(indices)))
        for projection, entries in projections.items():
            if sorted(entries) != expected:
                raise ValueError(
                    f"per-expert MoE weights for '{prefix}.{projection}' cover "
                    f"expert indices {sorted(entries)}, expected {expected}"
                )
        stacked[f"{prefix}.gate_up_proj.weight"] = mx.stack(
            [
                mx.concatenate(
                    [projections["gate_proj"][i], projections["up_proj"][i]], axis=0
                )
                for i in expected
            ]
        )
        stacked[f"{prefix}.down_proj.weight"] = mx.stack(
            [projections["down_proj"][i] for i in expected]
        )
    return stacked


def fuse_mlx_vlm_expert_weights(
    weights: Mapping[str, mx.array],
) -> dict[str, mx.array]:
    """Map MLX-VLM split expert stacks onto the fused SwitchLinear layout."""

    grouped: dict[str, dict[str, dict[str, mx.array]]] = {}
    fused: dict[str, mx.array] = {}
    for key, value in weights.items():
        match = _MLX_VLM_EXPERT_KEY.match(key)
        if match is None:
            fused[key] = value
            continue
        suffixes = grouped.setdefault(match.group("prefix"), {})
        suffixes.setdefault(match.group("suffix"), {})[
            match.group("projection")
        ] = value

    for prefix, suffixes in grouped.items():
        for suffix, projections in suffixes.items():
            missing = sorted({"gate_proj", "up_proj", "down_proj"} - set(projections))
            if missing:
                raise ValueError(
                    f"MLX-VLM expert weights for '{prefix}.{suffix}' are "
                    f"missing {missing}"
                )
            gate = projections["gate_proj"]
            up = projections["up_proj"]
            if gate.shape != up.shape:
                raise ValueError(
                    f"MLX-VLM expert gate/up shapes for '{prefix}.{suffix}' "
                    f"do not match: {gate.shape} != {up.shape}"
                )
            fused[f"{prefix}.experts.gate_up_proj.{suffix}"] = mx.concatenate(
                [gate, up], axis=1
            )
            fused[f"{prefix}.experts.down_proj.{suffix}"] = projections["down_proj"]
    return fused


def sanitize_qwen3_omni_weights(
    weights: Mapping[str, mx.array],
    *,
    component: Literal["thinker", "talker"],
) -> dict[str, mx.array]:
    """Convert HF checkpoint weights into the native MLX layout for a component.

    * Distinguishes HF source layout (keys carry a ``thinker.``/``talker.``/
      ``code2wav.`` prefix) from an already-converted MLX checkpoint.
    * For HF source, keeps only the selected component's tensors and strips that
      prefix; tensors of other components are dropped.
    * Transposes convolution kernels from HF ``(out, in, *k)`` layout to MLX
      ``(out, *k, in)`` layout, and only when converting from HF source.
    * Fuses per-expert MoE linears (``...experts.<e>.gate_proj.weight`` and
      friends) into the stacked expert layout.
    * Fuses MLX-VLM's separate stacked ``switch_mlp.gate_proj`` /
      ``up_proj`` tensors, including packed quantization metadata, into the
      native ``experts.gate_up_proj`` stack.
    * Normalizes fused MoE expert stacks (``...experts.gate_up_proj`` /
      ``...experts.down_proj``) onto the ``SwitchLinear`` weight name so a
      converted checkpoint's expert ``scales``/``biases`` land on the same
      module. This runs for both HF and already-converted sources and is
      idempotent.
    """

    prefix = f"{component}."
    is_hf_source = any(key.startswith(_COMPONENT_PREFIXES) for key in weights.keys())

    sanitized: dict[str, mx.array] = {}
    for key, value in weights.items():
        if is_hf_source:
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
            if component == "thinker" and key.startswith("language_model."):
                key = key[len("language_model.") :]
            if _is_convolution_weight(key, value):
                if value.ndim == 4:
                    value = value.transpose(0, 2, 3, 1)
                else:  # Conv1d: (out, in, k) -> (out, k, in)
                    value = value.transpose(0, 2, 1)
        sanitized[normalize_expert_stack_key(key)] = value

    return fuse_mlx_vlm_expert_weights(stack_legacy_expert_weights(sanitized))


def tie_lm_head_weights(lm_head: nn.Module, embedding: nn.Module) -> None:
    """Genuinely tie an output projection to the input embedding weight.

    The projection ends up referencing the *same* ``mx.array`` object as the
    embedding, so the two remain a single logical weight at the module level
    rather than a one-time numeric copy.
    """

    lm_head.weight = embedding.weight


def quantize_converted_module(
    model: nn.Module,
    weights: Mapping[str, mx.array],
    *,
    quantization: QuantizationConfig,
) -> None:
    """Quantize the linear layers a converted checkpoint actually represents.

    ``quantization`` is the metadata parsed from the checkpoint's ``quantization``
    block (bits / group_size / mode), so this pre-load path is driven by real
    configuration rather than hard-coded literals.

    A layer is quantized only when its ``<path>.scales`` tensor exists in the
    converted ``weights``. Both incomplete affine directions are rejected with a
    ``ValueError`` naming the full layer path -- ``<path>.scales`` without
    ``<path>.biases`` and ``<path>.biases`` without ``<path>.scales`` -- so an
    incomplete quantized group never silently falls back to dense weights.

    Must be called before ``model.load_weights`` so the module types already
    match the packed tensors on disk.
    """

    group_size = quantization.group_size
    bits = quantization.bits
    mode = quantization.mode

    def class_predicate(path: str, module: nn.Module):
        has_scales = f"{path}.scales" in weights
        has_biases = f"{path}.biases" in weights

        if mode == "affine" and has_biases and not has_scales:
            raise ValueError(
                f"incomplete quantized group for layer '{path}': "
                f"'{path}.biases' present but '{path}.scales' missing"
            )
        if not has_scales:
            return False
        if not hasattr(module, "to_quantized"):
            raise ValueError(
                f"quantized weights present for non-quantizable layer '{path}'"
            )
        if mode == "affine" and not has_biases:
            raise ValueError(
                f"incomplete quantized group for layer '{path}': "
                f"'{path}.scales' present but '{path}.biases' missing"
            )
        return True

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=class_predicate,
    )
