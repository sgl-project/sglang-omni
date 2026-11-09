# SPDX-License-Identifier: Apache-2.0
"""A bounded GPU cache of quantized routed experts, the rest in pinned host memory.

Decode refills S slots per layer by least recent use inside its CUDA graph; prefill
reads the cache into a shared scratch layer without changing it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

from sglang_omni.models.fishaudio_drama3.expert_cache_kernels import (
    HAS_TRITON,
    copy_missing_kernel,
    decode_policy_kernel,
    gather_needed_kernel,
    triton,
)

COPY_BLOCK_WORDS = 4096
KERNEL_WARPS = 8
# Rows of the statistics tensor; the policy kernel writes the first two.
STATISTIC_LOOKUPS = 0
STATISTIC_DECODE_MISSES = 1
STATISTIC_PREFILL_MISSES = 2
STATISTIC_COUNT = 3

__all__ = [
    "CacheMode",
    "CachedExperts",
    "ExpertCacheStatistics",
    "LayerExpertCache",
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


class CachedExperts(NamedTuple):
    """What the MoE kernel runs over in place of the full layer: the dispatch with
    expert ids in this weight space, the per-expert weights by tensor name, and an
    empty int32 index with one row per expert for kernels that take one."""

    dispatch_output: StandardDispatchOutput
    weights: dict[str, torch.Tensor]
    empty_rows: torch.Tensor


def expert_words(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor as one row of int32 words per expert."""
    assert tensor.is_contiguous() and tensor[0].numel() * tensor.element_size() % 4 == 0
    return tensor.view(torch.int32).view(tensor.shape[0], -1)


def pinned_host_copy(
    expert_weights: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The layer's expert tensors, one row per expert, copied into pinned host memory."""
    host_weights = {}
    for name, weight in expert_weights.items():
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
        self.expert_count = int(next(iter(host_weights.values())).shape[0])
        self.slot_count = slot_count
        self.host_weights = host_weights
        self.slot_weights = {
            name: host_weight[:slot_count].to(device)
            for name, host_weight in host_weights.items()
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
        self.empty_rows = torch.empty((slot_count, 0), dtype=torch.int32, device=device)

    def cached_experts(self, dispatch_output: StandardDispatchOutput) -> CachedExperts:
        """Decode reads the slots by slot id; prefill reads the scratch layer."""
        topk_output = dispatch_output.topk_output
        assert type(topk_output) is StandardTopKOutput
        if self.shared_cache.mode == CacheMode.DECODE:
            weights = self.slot_weights
            empty_rows = self.empty_rows
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
            empty_rows = self.shared_cache.empty_rows
        return CachedExperts(
            dispatch_output=dispatch_output, weights=weights, empty_rows=empty_rows
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
        for name, host_weight in self.host_weights.items():
            host_words = expert_words(host_weight)
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
        for name, host_weight in self.host_weights.items():
            host_words = expert_words(host_weight)
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
            raise ValueError("the model has no quantized routed expert layers to cache")
        else:
            pass
        first_layer = layer_host_weights[0]
        expert_count = int(next(iter(first_layer.values())).shape[0])
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
            name: torch.empty_like(host_weight, device=device)
            for name, host_weight in first_layer.items()
        }
        self.empty_rows = torch.empty(
            (expert_count, 0), dtype=torch.int32, device=device
        )
        self.bytes_per_expert = sum(
            host_weight[0].numel() * host_weight.element_size()
            for host_weight in first_layer.values()
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
