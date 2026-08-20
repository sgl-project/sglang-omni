# SPDX-License-Identifier: Apache-2.0
"""Fused QK-norm plus RoPE for the Qwen3-Omni thinker's attention layers.

The thinker's rotary is MRoPE, so upstream Qwen3MoeAttention keeps its fused
branch off: the kernel takes one position per token while MRoPE carries three.
Those three rows only differ for image and video tokens. Where they agree,
MRoPE selects the same cos/sin entry for every frequency pair, which is the
rotation the kernel applies, so a batch carrying no multimodal input can be
fused and one carrying any falls back whole.

The decision is a Python branch, so it must not be frozen into a replayed
graph; installation is refused while prefill graphs are enabled. It is also
refused off XPU, so CUDA and ROCm keep the unfused path they are tuned on.

It installs by default on XPU. The cache-consuming kernel beats the unfused pair
at every prefill length measured, so there is no size for which an operator would
want it off, and the gates below already refuse the cases it cannot serve.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch

from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)

_FUSABLE_HEAD_DIMS = (64, 128, 256)


class ThinkerFusedRopeGate:
    """Per-forward decision plus the 1-D positions the kernel needs."""

    __slots__ = ("enabled", "positions")

    def __init__(self) -> None:
        self.enabled = False
        self.positions: torch.Tensor | None = None

    def evaluate(self, positions: torch.Tensor, forward_batch: Any) -> None:
        """Decide once per forward, before any layer runs.

        Absence of multimodal input is read off the raw list rather than
        contains_mm_inputs(), which answers per item type: a type this build
        does not recognize would read as "no image" and wrongly admit fusion.
        mm_inputs is a declared ForwardBatch field, so read it directly: reaching
        it defensively would turn a caller that omits it into a text-only batch.
        """
        self.enabled = False
        self.positions = None

        mm_inputs = forward_batch.mm_inputs
        text_only = not mm_inputs or all(item is None for item in mm_inputs)
        # Decode stays unfused so a captured decode graph can never replay a
        # frozen decision; prefill is where the per-layer launches are worth it.
        prefill = forward_batch.forward_mode.is_extend()
        if not (text_only and prefill) or positions is None or positions.dim() != 2:
            return
        if positions.shape[0] != 3:
            return
        self.enabled = True
        # _compute_mrope_positions builds all three rows from one range for a
        # text-only batch, so the temporal row carries the position. Confirming
        # that per forward costs two device syncs, so the gate above is the
        # guarantee; a test drives the real builder to keep it honest.
        self.positions = positions[0].contiguous()


def _fused_apply_qk_norm_rope(
    attn: Any,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: Any,
    *,
    gate: ThinkerFusedRopeGate,
    kernel: Any,
    cos_sin_cache: torch.Tensor,
):
    if not gate.enabled or qkv.dtype != torch.bfloat16:
        return attn._omni_unfused_apply_qk_norm_rope(qkv, positions, forward_batch)

    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    tokens = qkv.shape[0]
    # The kernel writes q and k in place and wants them per head and contiguous,
    # which a slice of the packed projection is not.
    q = q.reshape(tokens, attn.num_heads, attn.head_dim).contiguous()
    k = k.reshape(tokens, attn.num_kv_heads, attn.head_dim).contiguous()
    kernel(
        q,
        k,
        attn.q_norm.weight,
        attn.k_norm.weight,
        cos_sin_cache,
        gate.positions,
        attn.rotary_emb.is_neox_style,
        attn.q_norm.variance_epsilon,
    )
    # forward_core reads this to decide save_kv_cache: the fused path writes no
    # KV of its own, so the attention call has to.
    attn._used_fused_qk_norm_rope_last_call = True
    return (
        q.reshape(tokens, attn.q_size),
        k.reshape(tokens, attn.kv_size),
        v,
    )


def _prefill_graph_enabled() -> bool:
    """Whether prefill runs under a graph that would freeze a Python decision.

    Construction outside a serving process has no global server args to read;
    answer yes there so the fused path stays off rather than guessing.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend

    from sglang_omni.vendor.sglang.server_args import get_global_server_args

    try:
        prefill = get_global_server_args().cuda_graph_config.prefill
    except ValueError:
        return True
    return prefill.backend != Backend.DISABLED


def install_thinker_fused_rope(
    model: Any,
    config: Any,
    *,
    kernel_provider: Any = None,
    prefill_graph_enabled: bool | None = None,
) -> ThinkerFusedRopeGate | None:
    """Route eligible thinker attention layers through the fused kernel.

    Every gate is evaluated here, cheapest first, so a platform that cannot use
    this does no work at model init. In particular the kernel is not acquired
    until the gates pass: a platform hook may import sgl_kernel eagerly and raise
    where the build lacks it, which must not break an unaffected model.

    Returns the gate the caller must evaluate once per forward, or None when
    nothing was patched.
    """
    if not current_platform.is_xpu():
        # Only XPU opts in. Every other platform keeps upstream's choice to leave
        # an MRoPE model unfused: on CUDA and ROCm that unfused path is the tuned
        # one, and no platform should be admitted here without being measured.
        return None
    if prefill_graph_enabled is None:
        prefill_graph_enabled = _prefill_graph_enabled()
    if prefill_graph_enabled:
        logger.info(
            "Qwen3-Omni thinker: fused QK-norm-RoPE stays off because a replayed "
            "prefill graph would freeze the per-batch multimodal decision"
        )
        return None

    provider = kernel_provider or current_platform.get_fused_qk_norm_rope
    kernel = provider()
    if kernel is None:
        return None

    from sglang.srt.models.qwen3_moe import compute_yarn_parameters

    gate = ThinkerFusedRopeGate()
    cos_sin_cache = None
    patched = 0
    skipped = 0
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "apply_qk_norm_rope"):
            continue
        if attn.head_dim not in _FUSABLE_HEAD_DIMS:
            skipped += 1
            continue
        if compute_yarn_parameters(attn.config)[0] != 1.0:
            # This kernel takes a cos/sin table and no YaRN parameters, so it
            # cannot express a scaled rotary.
            skipped += 1
            continue
        if hasattr(attn, "_omni_unfused_apply_qk_norm_rope"):
            # A second install would save the wrapper as its own fallback and
            # recurse until the stack blows.
            skipped += 1
            continue
        if cos_sin_cache is None:
            # The kernel requires float32 while the rotary keeps its table in the
            # query dtype. Upcasting the rotary's own table, once and shared by
            # every layer, keeps the frequencies identical to the unfused path.
            cos_sin_cache = attn.rotary_emb.cos_sin_cache.float().contiguous()

        attn._omni_unfused_apply_qk_norm_rope = attn.apply_qk_norm_rope

        def _bound(
            attn_self,
            qkv,
            positions,
            forward_batch,
            _gate=gate,
            _kernel=kernel,
            _cache=cos_sin_cache,
        ):
            return _fused_apply_qk_norm_rope(
                attn_self,
                qkv,
                positions,
                forward_batch,
                gate=_gate,
                kernel=_kernel,
                cos_sin_cache=_cache,
            )

        attn.apply_qk_norm_rope = MethodType(_bound, attn)
        patched += 1

    if not patched:
        logger.info(
            "Qwen3-Omni thinker: no attention layer was patched for fused "
            "QK-norm-RoPE (%d skipped)",
            skipped,
        )
        return None
    return gate


__all__ = [
    "ThinkerFusedRopeGate",
    "install_thinker_fused_rope",
]
