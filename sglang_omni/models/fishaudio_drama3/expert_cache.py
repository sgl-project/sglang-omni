# SPDX-License-Identifier: Apache-2.0
"""A bounded GPU cache of the int4 routed experts, the rest in pinned host memory.

Each MoE layer keeps S expert slots on the GPU, refilled by least recent use inside
the decode CUDA graph, and Marlin reads them by slot id. A prefill gathers its
routed experts into one shared full-width scratch layer and leaves the cache alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

from sglang_omni.models.fishaudio_drama3.expert_cache_kernels import (
    HAS_TRITON,
    copy_missing_kernel,
    decode_policy_kernel,
    gather_needed_kernel,
    triton,
)

# The Marlin tensors one expert owns; its qzeros and g_idx are empty or unread.
EXPERT_TENSOR_NAMES = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
COPY_BLOCK_WORDS = 4096
KERNEL_WARPS = 8
# Rows of the statistics tensor; the policy kernel writes the first two.
STATISTIC_LOOKUPS = 0
STATISTIC_DECODE_MISSES = 1
STATISTIC_PREFILL_MISSES = 2
STATISTIC_COUNT = 3

__all__ = [
    "EXPERT_TENSOR_NAMES",
    "CacheMode",
    "ExpertCacheStatistics",
    "LayerExpertCache",
    "MarlinInputs",
    "RoutedExpertCache",
    "pinned_host_copy",
]


class CacheMode(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True, kw_only=True)
class ExpertCacheStatistics:
    """Totals since startup; a lookup is one distinct expert one layer needs."""

    lookups: int
    decode_misses: int
    prefill_misses: int
    bytes_per_expert: int

    @property
    def decode_hit_rate(self) -> float:
        if self.lookups == 0:
            return 1.0
        else:
            return 1.0 - self.decode_misses / self.lookups


class MarlinInputs(NamedTuple):
    dispatch_output: StandardDispatchOutput
    quant_info: MarlinMoeQuantInfo


def expert_words(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor as one row of int32 words per expert."""
    assert tensor.is_contiguous() and tensor[0].numel() * tensor.element_size() % 4 == 0
    return tensor.view(torch.int32).view(tensor.shape[0], -1)


def pinned_host_copy(
    expert_weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The layer's expert tensors copied into pinned host memory."""
    host_weights = {}
    for name in EXPERT_TENSOR_NAMES:
        weight = expert_weights[name]
        host_weight = torch.empty(
            weight.shape, dtype=weight.dtype, device="cpu", pin_memory=True
        )
        host_weight.copy_(weight)
        host_weights[name] = host_weight
    return host_weights


class LayerExpertCache:
    """One MoE layer: every expert in pinned host memory, S of them in GPU slots."""

    def __init__(
        self,
        host_weights: dict[str, torch.Tensor],
        shared_cache: RoutedExpertCache,
        slot_count: int,
        device: torch.device,
    ) -> None:
        self.shared_cache = shared_cache
        self.expert_count = int(host_weights[EXPERT_TENSOR_NAMES[0]].shape[0])
        self.slot_count = slot_count
        self.host_weights = host_weights
        self.slot_weights = {
            name: host_weights[name][:slot_count].to(device)
            for name in EXPERT_TENSOR_NAMES
        }
        self.expert_to_slot = torch.full(
            (self.expert_count,), -1, dtype=torch.int32, device=device
        )
        self.expert_to_slot[:slot_count] = torch.arange(
            slot_count, dtype=torch.int32, device=device
        )
        self.slot_to_expert = torch.arange(slot_count, dtype=torch.int32, device=device)
        self.slot_last_used = torch.zeros(slot_count, dtype=torch.int64, device=device)
        self.clock = torch.ones(1, dtype=torch.int64, device=device)
        self.miss_experts = torch.zeros(slot_count, dtype=torch.int32, device=device)
        self.miss_slots = torch.zeros(slot_count, dtype=torch.int32, device=device)
        self.miss_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.needed_flags = torch.zeros(
            self.expert_count, dtype=torch.int32, device=device
        )
        self.victim_order = torch.zeros(
            triton.next_power_of_2(slot_count), dtype=torch.int64, device=device
        )
        self.empty_g_idx = torch.empty(
            (slot_count, 0), dtype=torch.int32, device=device
        )

    def marlin_inputs(
        self,
        dispatch_output: StandardDispatchOutput,
        *,
        weight_bits: int,
        is_k_full: bool,
    ) -> MarlinInputs:
        """The Marlin call over cached experts that stands in for the full layer's."""
        topk_output = dispatch_output.topk_output
        assert type(topk_output) is StandardTopKOutput
        if self.shared_cache.mode == CacheMode.DECODE:
            weights = self.slot_weights
            g_idx = self.empty_g_idx
            dispatch_output = StandardDispatchOutput(
                hidden_states=dispatch_output.hidden_states,
                hidden_states_scale=dispatch_output.hidden_states_scale,
                topk_output=StandardTopKOutput(
                    topk_weights=topk_output.topk_weights,
                    topk_ids=self.decode_slot_ids(topk_output.topk_ids),
                    router_logits=topk_output.router_logits,
                ),
                hidden_states_pre_quant=dispatch_output.hidden_states_pre_quant,
            )
        else:
            weights = self.gather_prefill(topk_output.topk_ids)
            g_idx = self.shared_cache.empty_g_idx
        return MarlinInputs(
            dispatch_output=dispatch_output,
            quant_info=MarlinMoeQuantInfo(
                w13_qweight=weights["w13_qweight"],
                w2_qweight=weights["w2_qweight"],
                w13_scales=weights["w13_scales"],
                w2_scales=weights["w2_scales"],
                w13_g_idx=g_idx,
                w2_g_idx=g_idx,
                w13_g_idx_sort_indices=g_idx,
                w2_g_idx_sort_indices=g_idx,
                weight_bits=weight_bits,
                is_k_full=is_k_full,
            ),
        )

    def decode_slot_ids(self, routed_experts: torch.Tensor) -> torch.Tensor:
        """Bring every routed expert into a slot; the routed ids in slot space."""
        routed_count = routed_experts.numel()
        slot_ids = torch.empty_like(routed_experts)
        slot_block = triton.next_power_of_2(self.slot_count)
        decode_policy_kernel[(1,)](
            routed_experts,
            slot_ids,
            self.expert_to_slot,
            self.slot_to_expert,
            self.slot_last_used,
            self.clock,
            self.miss_experts,
            self.miss_slots,
            self.miss_count,
            self.shared_cache.statistics,
            self.needed_flags,
            self.victim_order,
            routed_count,
            EXPERT_COUNT=self.expert_count,
            SLOT_COUNT=self.slot_count,
            ROUTED_BLOCK=triton.next_power_of_2(routed_count),
            EXPERT_BLOCK=triton.next_power_of_2(self.expert_count),
            SLOT_BLOCK=slot_block,
            SLOT_BITS=(slot_block - 1).bit_length(),
            num_warps=KERNEL_WARPS,
        )
        # A step misses at most as many experts as it routes to.
        max_miss_count = min(routed_count, self.slot_count)
        for name in EXPERT_TENSOR_NAMES:
            host_words = expert_words(self.host_weights[name])
            words_per_expert = host_words.shape[1]
            copy_missing_kernel[
                (max_miss_count, triton.cdiv(words_per_expert, COPY_BLOCK_WORDS))
            ](
                host_words,
                expert_words(self.slot_weights[name]),
                self.miss_experts,
                self.miss_slots,
                self.miss_count,
                WORDS_PER_EXPERT=words_per_expert,
                BLOCK_WORDS=COPY_BLOCK_WORDS,
                num_warps=KERNEL_WARPS,
            )
        return slot_ids

    def gather_prefill(self, routed_experts: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every routed expert, at its own row of the shared scratch layer."""
        is_needed = torch.zeros(
            self.expert_count, dtype=torch.bool, device=routed_experts.device
        )
        is_needed.index_fill_(0, routed_experts.reshape(-1).long(), True)
        self.shared_cache.statistics[STATISTIC_PREFILL_MISSES] += (
            is_needed & (self.expert_to_slot < 0)
        ).sum()
        scratch_weights = self.shared_cache.scratch_weights
        for name in EXPERT_TENSOR_NAMES:
            host_words = expert_words(self.host_weights[name])
            words_per_expert = host_words.shape[1]
            gather_needed_kernel[
                (self.expert_count, triton.cdiv(words_per_expert, COPY_BLOCK_WORDS))
            ](
                host_words,
                expert_words(self.slot_weights[name]),
                expert_words(scratch_weights[name]),
                is_needed,
                self.expert_to_slot,
                WORDS_PER_EXPERT=words_per_expert,
                BLOCK_WORDS=COPY_BLOCK_WORDS,
                num_warps=KERNEL_WARPS,
            )
        return scratch_weights


class RoutedExpertCache:
    """Every MoE layer's cache, the prefill scratch layer they share, the mode the
    next forward runs in, and the miss totals."""

    def __init__(
        self,
        layer_host_weights: list[dict[str, torch.Tensor]],
        slot_count: int,
        device: torch.device,
    ) -> None:
        """The host weights come from pinned_host_copy; the GPU copies should be
        freed first, so the slots never coexist with the full layers."""
        if not HAS_TRITON:
            raise RuntimeError("the routed expert cache needs Triton")
        elif not layer_host_weights:
            raise ValueError("the model has no int4 routed expert layers to cache")
        else:
            pass
        first_layer = layer_host_weights[0]
        expert_count = int(first_layer[EXPERT_TENSOR_NAMES[0]].shape[0])
        if not 0 < slot_count <= expert_count:
            raise ValueError(
                f"routed_expert_cache_slots={slot_count} must be in 1..{expert_count}"
            )
        else:
            pass
        self.slot_count = slot_count
        self.expert_count = expert_count
        self.mode = CacheMode.PREFILL
        self.statistics = torch.zeros(STATISTIC_COUNT, dtype=torch.int64, device=device)
        self.scratch_weights = {
            name: torch.empty_like(first_layer[name], device=device)
            for name in EXPERT_TENSOR_NAMES
        }
        self.empty_g_idx = torch.empty(
            (expert_count, 0), dtype=torch.int32, device=device
        )
        self.bytes_per_expert = sum(
            first_layer[name][0].numel() * first_layer[name].element_size()
            for name in EXPERT_TENSOR_NAMES
        )
        self.layers = [
            LayerExpertCache(host_weights, self, slot_count, device)
            for host_weights in layer_host_weights
        ]

    def read_statistics(self) -> ExpertCacheStatistics:
        """Synchronizes with the device."""
        lookups, decode_misses, prefill_misses = self.statistics.tolist()
        return ExpertCacheStatistics(
            lookups=lookups,
            decode_misses=decode_misses,
            prefill_misses=prefill_misses,
            bytes_per_expert=self.bytes_per_expert,
        )
