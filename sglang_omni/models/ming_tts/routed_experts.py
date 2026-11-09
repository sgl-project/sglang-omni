# SPDX-License-Identifier: Apache-2.0
"""Serve Ming-TTS's bf16 routed experts from a bounded GPU cache.

Every routed expert moves to pinned host memory and each MoE layer keeps S of them
in GPU slots; SGLang's Triton fused MoE runs over the slots in decode and over a
gathered scratch layer in prefill.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from torch import nn

from sglang_omni.models.fishaudio_drama3.expert_cache import (
    LayerExpertCache,
    RoutedExpertCache,
    pinned_host_copy,
)

# The bf16 tensors one routed expert owns in an unquantized FusedMoE layer.
BF16_EXPERT_TENSOR_NAMES = ("w13_weight", "w2_weight")

logger = logging.getLogger(__name__)

__all__ = [
    "BF16_EXPERT_TENSOR_NAMES",
    "CachedUnquantizedFusedMoEMethod",
    "attach_routed_expert_cache",
]


class CachedUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """The unquantized Triton MoE method over a layer's routed expert cache."""

    routed_expert_cache: LayerExpertCache

    def apply(
        self, layer: torch.nn.Module, dispatch_output: StandardDispatchOutput
    ) -> CombineInput:
        cached_experts = self.routed_expert_cache.cached_experts(
            dispatch_output.topk_output
        )
        return self.runner.run(
            StandardDispatchOutput(
                hidden_states=dispatch_output.hidden_states,
                hidden_states_scale=dispatch_output.hidden_states_scale,
                topk_output=cached_experts.topk_output,
                hidden_states_pre_quant=dispatch_output.hidden_states_pre_quant,
            ),
            TritonMoeQuantInfo(
                w13_weight=cached_experts.weights["w13_weight"],
                w2_weight=cached_experts.weights["w2_weight"],
                fuse_swiglu_interleaved=self.w13_swiglu_interleaved,
            ),
        )


def attach_routed_expert_cache(
    model: nn.Module,
    *,
    slot_count: int,
    max_decode_rows: int,
    top_k: int,
    device: torch.device,
) -> RoutedExpertCache:
    """Move every bf16 routed expert to pinned host memory and serve the model's
    FusedMoE layers from S GPU slots each; the full layers are freed before any slot
    is allocated."""
    expert_layers = [
        module for module in model.modules() if isinstance(module, FusedMoE)
    ]
    if not expert_layers:
        raise ValueError("routed_expert_cache_slots needs a model with FusedMoE layers")
    else:
        pass
    expert_count = expert_layers[0].num_experts
    smallest_slot_count = min(expert_count, max_decode_rows * top_k)
    for layer in expert_layers:
        quant_method = layer.quant_method
        if type(quant_method) is not UnquantizedFusedMoEMethod:
            raise ValueError(
                "routed_expert_cache_slots needs bf16 experts, got "
                f"{type(quant_method).__name__}"
            )
        elif not quant_method.runner.runner_backend.is_triton():
            raise ValueError(
                "routed_expert_cache_slots needs the triton MoE runner backend, got "
                f"{quant_method.runner.runner_backend}"
            )
        elif layer.num_local_experts != expert_count:
            raise ValueError("routed_expert_cache_slots needs every expert local")
        else:
            pass
    if not 0 < slot_count <= expert_count:
        raise ValueError(
            f"routed_expert_cache_slots={slot_count} must be in 1..{expert_count}"
        )
    elif slot_count < smallest_slot_count:
        raise ValueError(
            f"routed_expert_cache_slots={slot_count} cannot hold the experts of one "
            f"decode step: {max_decode_rows} rows x top_k {top_k} need at least "
            f"{smallest_slot_count}"
        )
    else:
        pass
    free_bytes_before = torch.cuda.mem_get_info(device)[0]
    layer_host_weights = []
    for layer in expert_layers:
        layer_host_weights.append(
            pinned_host_copy(
                {name: getattr(layer, name) for name in BF16_EXPERT_TENSOR_NAMES}
            )
        )
        for name in BF16_EXPERT_TENSOR_NAMES:
            setattr(
                layer,
                name,
                nn.Parameter(getattr(layer, name).new_empty((0,)), requires_grad=False),
            )
        torch.cuda.empty_cache()
    routed_expert_cache = RoutedExpertCache(layer_host_weights, slot_count, device)
    for layer, layer_cache in zip(
        expert_layers, routed_expert_cache.layers, strict=True
    ):
        for name in BF16_EXPERT_TENSOR_NAMES:
            setattr(
                layer,
                name,
                nn.Parameter(layer_cache.slot_weights[name], requires_grad=False),
            )
        # note (minyi-tang): FusedMoE built its method before the weights loaded;
        # retyping the instance keeps the runner it already configured.
        layer.quant_method.__class__ = CachedUnquantizedFusedMoEMethod
        layer.quant_method.routed_expert_cache = layer_cache
    freed_bytes = torch.cuda.mem_get_info(device)[0] - free_bytes_before
    logger.info(
        f"Ming-TTS routed expert cache: {slot_count} of {expert_count} experts per "
        f"layer on the GPU over {len(expert_layers)} layers, "
        f"{freed_bytes / 2**30:.2f} GiB freed, prefill scratch included"
    )
    return routed_expert_cache
