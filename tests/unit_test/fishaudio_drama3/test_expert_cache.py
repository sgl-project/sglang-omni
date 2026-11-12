# SPDX-License-Identifier: Apache-2.0
"""The routed expert cache over int4 experts: its LRU decode policy, the experts it
holds, and Marlin over them."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

from sglang_omni.models.fishaudio_drama3.expert_cache import (
    CachedExperts,
    CacheMode,
    LayerExpertCache,
    RoutedExpertCache,
)
from tests.unit_test.fishaudio_drama3.expert_cache_fixtures import (
    EXPERT_COUNT,
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    assert_scratch_holds_the_routed_experts,
    assert_slots_hold_their_experts,
    cpu_maps,
    dispatch_of,
    random_hidden_states,
    reference_decode_policy,
    routed_experts_of,
    single_layer_cache,
)

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

SCALE_GROUP_SIZE = 128
INT4_WEIGHT_BITS = 4
# note (zengjiahua): Marlin's cross-SM reduction order follows where experts sit
# and, beyond about a hundred rows, timing; seeded runs on an L40S put a layer
# over slots within 2 bf16 ulps of the full-resident layer, which itself
# differs from its own rerun by as much.
MARLIN_RELATIVE_TOLERANCE = 2**-6
MARLIN_ABSOLUTE_TOLERANCE = 1e-4


def random_expert_weights(
    generator: torch.Generator, expert_count: int
) -> dict[str, torch.Tensor]:
    """Marlin-layout tensors of random int4 experts at the real layer's shapes."""
    word_shapes = {
        "w13_qweight": (HIDDEN_SIZE // 16, 4 * MOE_INTERMEDIATE_SIZE),
        "w2_qweight": (MOE_INTERMEDIATE_SIZE // 16, 2 * HIDDEN_SIZE),
    }
    scale_shapes = {
        "w13_scales": (HIDDEN_SIZE // SCALE_GROUP_SIZE, 2 * MOE_INTERMEDIATE_SIZE),
        "w2_scales": (MOE_INTERMEDIATE_SIZE // SCALE_GROUP_SIZE, HIDDEN_SIZE),
    }
    expert_weights = {
        name: torch.randint(
            -(2**31), 2**31 - 1, (expert_count, *shape), generator=generator
        )
        .to(torch.int32)
        .cuda()
        for name, shape in word_shapes.items()
    }
    for name, shape in scale_shapes.items():
        expert_weights[name] = (
            (torch.rand((expert_count, *shape), generator=generator) * 0.01)
            .to(torch.bfloat16)
            .cuda()
        )
    return expert_weights


def run_marlin(experts: CachedExperts) -> torch.Tensor:
    """What SGLang's Marlin MoE runner computes over these experts."""
    hidden_states = experts.dispatch_output.hidden_states
    topk_output = experts.dispatch_output.topk_output
    return fused_marlin_moe(
        hidden_states=hidden_states,
        w1=experts.weights["w13_qweight"],
        w2=experts.weights["w2_qweight"],
        w1_scale=experts.weights["w13_scales"],
        w2_scale=experts.weights["w2_scales"],
        gating_output=topk_output.router_logits,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        g_idx1=experts.empty_rows,
        g_idx2=experts.empty_rows,
        sort_indices1=experts.empty_rows,
        sort_indices2=experts.empty_rows,
        workspace=marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4),
        num_bits=INT4_WEIGHT_BITS,
        is_k_full=True,
    )


def full_resident_output(
    expert_weights: dict[str, torch.Tensor], dispatch_output: StandardDispatchOutput
) -> torch.Tensor:
    return run_marlin(
        CachedExperts(
            dispatch_output=dispatch_output,
            weights=expert_weights,
            empty_rows=torch.empty((EXPERT_COUNT, 0), dtype=torch.int32, device="cuda"),
        )
    )


def cached_output(
    layer_cache: LayerExpertCache, dispatch_output: StandardDispatchOutput
) -> torch.Tensor:
    return run_marlin(layer_cache.cached_experts(dispatch_output))


def test_decode_evicts_the_least_recently_used_unneeded_slot() -> None:
    expert_weights = random_expert_weights(torch.Generator().manual_seed(0), 8)
    cache = single_layer_cache(expert_weights, 4)
    layer_cache = cache.layers[0]
    cache.mode = CacheMode.DECODE
    # Experts 0..3 start in slots 0..3, all last used at 0; each row is one step.
    steps = [
        ([[2, 0]], [[2, 0]], [0, 1, 2, 3]),
        # Expert 6 misses and takes the oldest slot holding no needed expert.
        ([[6, 0]], [[1, 0]], [0, 6, 2, 3]),
        # Slots 0 and 2 hold needed experts; the lower missing expert takes the
        # older of the free slots 3 and 1.
        ([[5, 2], [7, 0]], [[3, 2], [1, 0]], [0, 7, 2, 5]),
        # Hits only: slots 1 and 3 are stamped, slots 0 and 2 stay at the last step.
        ([[7, 5]], [[1, 3]], [0, 7, 2, 5]),
        # Slots 0 and 2 are the oldest free slots; the lower one goes first.
        ([[4, 7]], [[0, 1]], [4, 7, 2, 5]),
        # Slot 2, unused since the third step, is the oldest slot left free.
        ([[6, 4]], [[2, 0]], [4, 7, 6, 5]),
    ]
    for routed_rows, expected_slot_ids, expected_held_experts in steps:
        slot_ids = layer_cache.decode_slot_ids(
            torch.tensor(routed_rows, dtype=torch.int32, device="cuda")
        )
        assert slot_ids.tolist() == expected_slot_ids, routed_rows
        assert layer_cache.slot_to_expert.tolist() == expected_held_experts
        assert_slots_hold_their_experts(layer_cache, expert_weights)


def cached_layer(
    slot_count: int, seed: int
) -> tuple[torch.Generator, dict[str, torch.Tensor], RoutedExpertCache]:
    generator = torch.Generator().manual_seed(seed)
    expert_weights = random_expert_weights(generator, EXPERT_COUNT)
    return generator, expert_weights, single_layer_cache(expert_weights, slot_count)


@pytest.mark.parametrize(("slot_count", "row_count"), [(192, 1), (192, 8), (128, 16)])
def test_decode_follows_the_reference_policy_and_marlin_over_the_slots(
    slot_count: int, row_count: int
) -> None:
    generator, expert_weights, cache = cached_layer(slot_count, slot_count + row_count)
    layer_cache = cache.layers[0]
    cache.mode = CacheMode.DECODE
    expected_maps = cpu_maps(layer_cache)
    for clock in range(1, 41):
        routed_experts = routed_experts_of(generator, row_count, slot_count // 2)
        dispatch_output = dispatch_of(
            random_hidden_states(generator, row_count), routed_experts
        )
        output = cached_output(layer_cache, dispatch_output)
        reference_decode_policy(routed_experts.cpu(), *expected_maps, clock=clock)
        for cache_map, expected_map in zip(
            cpu_maps(layer_cache), expected_maps, strict=True
        ):
            assert torch.equal(cache_map, expected_map), clock
        assert_slots_hold_their_experts(layer_cache, expert_weights)
        torch.testing.assert_close(
            output,
            full_resident_output(expert_weights, dispatch_output),
            rtol=MARLIN_RELATIVE_TOLERANCE,
            atol=MARLIN_ABSOLUTE_TOLERANCE,
        )
    statistics = cache.read_statistics()
    assert 0 < statistics.decode_misses < statistics.lookups
    assert statistics.prefill_misses == 0


@pytest.mark.parametrize("slot_count", [192, 128])
def test_prefill_gathers_the_routed_experts_and_leaves_the_cache_alone(
    slot_count: int,
) -> None:
    generator, expert_weights, cache = cached_layer(slot_count, slot_count)
    layer_cache = cache.layers[0]
    cache.mode = CacheMode.DECODE
    for _ in range(6):
        layer_cache.decode_slot_ids(routed_experts_of(generator, 4, slot_count // 2))
    maps_before = cpu_maps(layer_cache)
    cache.mode = CacheMode.PREFILL
    for row_count in (1, 37, 300):
        routed_experts = routed_experts_of(generator, row_count, slot_count // 2)
        dispatch_output = dispatch_of(
            random_hidden_states(generator, row_count), routed_experts
        )
        output = cached_output(layer_cache, dispatch_output)
        assert_scratch_holds_the_routed_experts(cache, expert_weights, routed_experts)
        torch.testing.assert_close(
            output,
            full_resident_output(expert_weights, dispatch_output),
            rtol=MARLIN_RELATIVE_TOLERANCE,
            atol=MARLIN_ABSOLUTE_TOLERANCE,
        )
    for cache_map, map_before in zip(cpu_maps(layer_cache), maps_before, strict=True):
        assert torch.equal(cache_map, map_before)
    assert cache.read_statistics().prefill_misses > 0


def test_a_captured_decode_step_replays_the_policy() -> None:
    slot_count, row_count = 128, 4
    generator, expert_weights, cache = cached_layer(slot_count, 7)
    layer_cache = cache.layers[0]
    cache.mode = CacheMode.DECODE
    static_routed_experts = routed_experts_of(generator, row_count, slot_count)
    static_hidden_states = torch.zeros(
        row_count, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    static_dispatch_output = dispatch_of(static_hidden_states, static_routed_experts)
    # Compile the kernels outside the graph, then restore the cache they changed.
    cached_output(layer_cache, static_dispatch_output)
    torch.cuda.synchronize()
    layer_cache.expert_to_slot.fill_(-1)
    layer_cache.expert_to_slot[:slot_count] = torch.arange(
        slot_count, dtype=torch.int32, device="cuda"
    )
    layer_cache.slot_to_expert.copy_(torch.arange(slot_count, dtype=torch.int32))
    layer_cache.slot_last_used.zero_()
    layer_cache.clock.fill_(1)
    for name, slot_weight in layer_cache.slot_weights.items():
        slot_weight.copy_(expert_weights[name][:slot_count])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = cached_output(layer_cache, static_dispatch_output)
    expected_maps = cpu_maps(layer_cache)
    for clock in range(1, 9):
        static_routed_experts.copy_(
            routed_experts_of(generator, row_count, slot_count // 2)
        )
        static_hidden_states.copy_(random_hidden_states(generator, row_count))
        graph.replay()
        torch.cuda.synchronize()
        reference_decode_policy(
            static_routed_experts.cpu(), *expected_maps, clock=clock
        )
        for cache_map, expected_map in zip(
            cpu_maps(layer_cache), expected_maps, strict=True
        ):
            assert torch.equal(cache_map, expected_map), clock
        assert_slots_hold_their_experts(layer_cache, expert_weights)
        torch.testing.assert_close(
            static_output,
            full_resident_output(expert_weights, static_dispatch_output),
            rtol=MARLIN_RELATIVE_TOLERANCE,
            atol=MARLIN_ABSOLUTE_TOLERANCE,
        )
