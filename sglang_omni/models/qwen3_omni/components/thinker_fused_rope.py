# SPDX-License-Identifier: Apache-2.0
"""Fused QK-norm plus RoPE for the Qwen3-Omni thinker's attention layers.

MRoPE's three position rows differ only for image and video tokens, so a
text-only batch selects the same rotation the 1-D kernel applies and can be
fused, while any multimodal batch falls back whole. Installs by default on XPU.
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
        """Decide once per forward, before any layer runs."""
        self.enabled = False
        self.positions = None

        mm_inputs = forward_batch.mm_inputs
        text_only = not mm_inputs or all(item is None for item in mm_inputs)
        prefill = forward_batch.forward_mode.is_extend()
        if not (text_only and prefill) or positions is None or positions.dim() != 2:
            return
        if positions.shape[0] != 3:
            return
        self.enabled = True
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
    if not gate.enabled or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return attn._omni_unfused_apply_qk_norm_rope(qkv, positions, forward_batch)

    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    tokens = qkv.shape[0]
    # Views, not copies: the kernel needs only head_dim contiguous and updates
    # the packed projection in place.
    kernel(
        q.view(tokens, attn.num_heads, attn.head_dim),
        k.view(tokens, attn.num_kv_heads, attn.head_dim),
        attn.q_norm.weight,
        attn.k_norm.weight,
        cos_sin_cache,
        gate.positions,
        attn.rotary_emb.is_neox_style,
        attn.q_norm.variance_epsilon,
    )
    attn._used_fused_qk_norm_rope_last_call = True
    return q, k, v


def _prefill_graph_enabled() -> bool:
    """Whether prefill runs under a graph that would freeze a Python decision."""
    from sglang.srt.model_executor.cuda_graph_config import Backend

    from sglang_omni.vendor.sglang.server_args import get_global_server_args

    try:
        prefill = get_global_server_args().cuda_graph_config.prefill
    except ValueError:
        return True
    return prefill.backend != Backend.DISABLED


def install_thinker_fused_rope(
    model: Any,
    *,
    kernel_provider: Any = None,
    prefill_graph_enabled: bool | None = None,
) -> ThinkerFusedRopeGate | None:
    """Route eligible thinker attention layers through the fused kernel.

    Returns the gate the caller evaluates once per forward, or None when nothing
    was patched. Gates are ordered cheapest first, and the kernel is acquired
    only once they pass.
    """
    if not current_platform.is_xpu():
        return None
    if prefill_graph_enabled is None:
        prefill_graph_enabled = _prefill_graph_enabled()
    if prefill_graph_enabled:
        logger.info(
            "Qwen3-Omni thinker: fused QK-norm-RoPE stays off because a replayed "
            "prefill graph would freeze the per-batch multimodal decision"
        )
        return None

    provider = (
        kernel_provider or current_platform.get_fused_qk_norm_rope_with_cos_sin_cache
    )
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
            # No YaRN parameters in this ABI, so a scaled rotary is unreachable.
            skipped += 1
            continue
        if hasattr(attn, "_omni_unfused_apply_qk_norm_rope"):
            # A second install would make the wrapper its own fallback.
            skipped += 1
            continue
        if cos_sin_cache is None:
            # The kernel reads float32; the rotary keeps its table in query dtype.
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
