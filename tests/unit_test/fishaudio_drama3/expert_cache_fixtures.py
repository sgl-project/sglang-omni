# SPDX-License-Identifier: Apache-2.0
"""Routed expert cache fixtures shared by the int4 and FP8 cache tests."""

from __future__ import annotations

import torch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

from sglang_omni.models.fishaudio_drama3.expert_cache import (
    LayerExpertCache,
    RoutedExpertCache,
    pinned_host_copy,
)

# The Drama 3 MoE layer.
EXPERT_COUNT = 256
TOP_K = 8
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 512


def reference_decode_policy(
    routed_experts: torch.Tensor,
    expert_to_slot: torch.Tensor,
    slot_to_expert: torch.Tensor,
    slot_last_used: torch.Tensor,
    clock: int,
) -> torch.Tensor:
    """The policy the decode kernel runs, in place on CPU maps; the routed ids' slots."""
    is_needed = torch.zeros(expert_to_slot.shape[0], dtype=torch.bool)
    is_needed[routed_experts.reshape(-1).long()] = True
    missing_experts = torch.nonzero(is_needed & (expert_to_slot < 0)).reshape(-1)
    holds_needed = is_needed[slot_to_expert.long()]
    candidate_slots = sorted(
        torch.nonzero(~holds_needed).reshape(-1).tolist(),
        key=lambda slot: (int(slot_last_used[slot]), slot),
    )
    assert missing_experts.numel() <= len(candidate_slots)
    victim_slots = torch.tensor(
        candidate_slots[: missing_experts.numel()], dtype=torch.long
    )
    expert_to_slot[slot_to_expert[victim_slots].long()] = -1
    expert_to_slot[missing_experts] = victim_slots.to(torch.int32)
    slot_to_expert[victim_slots] = missing_experts.to(torch.int32)
    slot_last_used[expert_to_slot[is_needed].long()] = clock
    return expert_to_slot[routed_experts.long()]


def dispatch_of(
    hidden_states: torch.Tensor, routed_experts: torch.Tensor
) -> StandardDispatchOutput:
    router_weights = torch.softmax(
        torch.linspace(0, 1, routed_experts.numel(), device="cuda").view(
            routed_experts.shape
        ),
        dim=-1,
    )
    return StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=router_weights,
            topk_ids=routed_experts,
            router_logits=torch.zeros(
                routed_experts.shape[0], EXPERT_COUNT, device="cuda"
            ),
        ),
    )


def routed_experts_of(
    generator: torch.Generator, row_count: int, hot_expert_count: int
) -> torch.Tensor:
    """Distinct experts per row, mostly from a hot window that starts anywhere, so a
    step both hits resident experts and misses."""
    scores = torch.rand(row_count, EXPERT_COUNT, generator=generator)
    window_start = int(torch.randint(0, EXPERT_COUNT, (1,), generator=generator))
    scores[:, (window_start + torch.arange(hot_expert_count)) % EXPERT_COUNT] += 0.6
    return scores.topk(TOP_K, dim=-1).indices.to(torch.int32).cuda()


def random_hidden_states(generator: torch.Generator, row_count: int) -> torch.Tensor:
    return (
        (torch.randn(row_count, HIDDEN_SIZE, generator=generator) * 0.1)
        .to(torch.bfloat16)
        .cuda()
    )


def cpu_maps(layer_cache: LayerExpertCache) -> list[torch.Tensor]:
    return [
        layer_cache.expert_to_slot.cpu(),
        layer_cache.slot_to_expert.cpu(),
        layer_cache.slot_last_used.cpu(),
    ]


def assert_slots_hold_their_experts(
    layer_cache: LayerExpertCache, expert_weights: dict[str, torch.Tensor]
) -> None:
    held_experts = layer_cache.slot_to_expert.long()
    for name, weight in expert_weights.items():
        assert torch.equal(layer_cache.slot_weights[name], weight[held_experts]), name


def assert_scratch_holds_the_routed_experts(
    cache: RoutedExpertCache,
    expert_weights: dict[str, torch.Tensor],
    routed_experts: torch.Tensor,
) -> None:
    routed = torch.unique(routed_experts).long()
    for name, weight in expert_weights.items():
        assert torch.equal(cache.scratch_weights[name][routed], weight[routed]), name


def single_layer_cache(
    expert_weights: dict[str, torch.Tensor], slot_count: int
) -> RoutedExpertCache:
    return RoutedExpertCache(
        [pinned_host_copy(expert_weights)], slot_count, torch.device("cuda")
    )
