# SPDX-License-Identifier: Apache-2.0
"""Fused QK-norm plus RoPE for the Qwen3-Omni thinker's attention layers.

MRoPE's three position rows differ only for image and video tokens, so a
text-only batch selects the same rotation the 1-D kernel applies and can be
fused, while any multimodal batch falls back whole. Installs by default on XPU.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import MethodType
from typing import TYPE_CHECKING, TypeAlias

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.vendor.sglang.core import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
    from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel
else:
    pass

logger = logging.getLogger(__name__)

_FUSABLE_HEAD_DIMS = (64, 128, 256)

QKNormRopeWithCacheKernel: TypeAlias = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        float,
    ],
    None,
]


class ThinkerFusedRopeGate:
    """Per-forward decision plus the 1-D positions the kernel needs."""

    __slots__ = ("enabled", "positions")

    def __init__(self) -> None:
        self.enabled = False
        self.positions: torch.Tensor | None = None

    def evaluate(self, positions: torch.Tensor, forward_batch: ForwardBatch) -> None:
        """Decide once per forward, before any layer runs."""
        self.enabled = False
        self.positions = None

        mm_inputs = forward_batch.mm_inputs
        text_only = not mm_inputs or all(item is None for item in mm_inputs)
        prefill = forward_batch.forward_mode.is_extend()
        if not (text_only and prefill) or positions is None or positions.dim() != 2:
            return
        else:
            pass
        if positions.shape[0] != 3:
            return
        else:
            pass
        self.enabled = True
        self.positions = positions[0].contiguous()


def fused_apply_qk_norm_rope(
    attn: "Qwen3MoeAttention",
    qkv: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    gate: ThinkerFusedRopeGate,
    kernel: QKNormRopeWithCacheKernel,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not gate.enabled or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return attn._omni_unfused_apply_qk_norm_rope(
            qkv, positions, forward_batch
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    else:
        pass

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
    attn._used_fused_qk_norm_rope_last_call = True  # noqa: leading-underscore
    return q, k, v


def is_prefill_graph_enabled() -> bool:
    """Whether prefill runs under a graph that would freeze a Python decision."""
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec

    try:
        prefill = get_exec().graph.cuda_graph_config.prefill
    except ValueError:
        return True
    return prefill.backend != Backend.DISABLED


def install_thinker_fused_rope(
    model: "Qwen3MoeLLMModel",
    *,
    kernel_provider: Callable[[], QKNormRopeWithCacheKernel | None] | None = None,
    prefill_graph_enabled: bool | None = None,
) -> ThinkerFusedRopeGate | None:
    """Route eligible thinker attention layers through the fused kernel.

    Returns the gate the caller evaluates once per forward, or None when nothing
    was patched. Gates are ordered cheapest first, and the kernel is acquired
    only once they pass.
    """
    if not current_platform.is_xpu():
        return None
    else:
        pass
    if prefill_graph_enabled is None:
        prefill_graph_enabled = is_prefill_graph_enabled()
    else:
        pass
    if prefill_graph_enabled:
        logger.info(
            "Qwen3-Omni thinker: fused QK-norm-RoPE stays off because a replayed "
            "prefill graph would freeze the per-batch multimodal decision"
        )
        return None
    else:
        pass

    provider = (
        kernel_provider or current_platform.get_fused_qk_norm_rope_with_cos_sin_cache
    )
    kernel = provider()
    if kernel is None:
        return None
    else:
        pass

    from sglang.srt.models.qwen3_moe import compute_yarn_parameters

    gate = ThinkerFusedRopeGate()
    cos_sin_cache = None
    patched = 0
    skipped = 0
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "apply_qk_norm_rope"):
            continue
        else:
            pass
        if attn.head_dim not in _FUSABLE_HEAD_DIMS:
            skipped += 1
            continue
        else:
            pass
        if compute_yarn_parameters(attn.config)[0] != 1.0:
            # No YaRN parameters in this ABI, so a scaled rotary is unreachable.
            skipped += 1
            continue
        else:
            pass
        if hasattr(attn, "_omni_unfused_apply_qk_norm_rope"):
            # A second install would make the wrapper its own fallback.
            skipped += 1
            continue
        else:
            pass
        if cos_sin_cache is None:
            # The kernel reads float32; the rotary keeps its table in query dtype.
            cos_sin_cache = attn.rotary_emb.cos_sin_cache.float().contiguous()
        else:
            pass

        attn._omni_unfused_apply_qk_norm_rope = (
            attn.apply_qk_norm_rope
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

        def _bound(
            attn_self: "Qwen3MoeAttention",
            qkv: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            _gate: ThinkerFusedRopeGate = gate,
            _kernel: QKNormRopeWithCacheKernel = kernel,
            _cache: torch.Tensor = cos_sin_cache,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return fused_apply_qk_norm_rope(
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
    else:
        pass
    return gate


__all__ = [
    "ThinkerFusedRopeGate",
    "install_thinker_fused_rope",
]
