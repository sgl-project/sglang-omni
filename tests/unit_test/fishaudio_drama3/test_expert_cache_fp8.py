# SPDX-License-Identifier: Apache-2.0
"""The routed expert cache over block-FP8 experts and the Triton fused MoE."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_context, get_parallel

from sglang_omni.models.fishaudio_drama3.expert_cache import CachedExperts, CacheMode
from sglang_omni.models.fishaudio_drama3.quantization import (
    EXPERT_WEIGHT_BLOCK_SIZE,
    FP8_CACHED_TENSOR_NAMES,
)
from tests.unit_test.fishaudio_drama3.expert_cache_fixtures import (
    EXPERT_COUNT,
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    TOP_K,
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

BLOCK_SIZE = EXPERT_WEIGHT_BLOCK_SIZE[0]
FP8_RELATIVE_TOLERANCE = 2**-6
FP8_ABSOLUTE_TOLERANCE = 1e-4


@pytest.fixture(autouse=True)
def published_server_args() -> Iterator[None]:
    """The Triton MoE reads its kernel configs from the published server args and
    allocates its output through the tensor-parallel group."""
    with (
        get_context().override_server_args(),
        get_parallel().override(tp_rank=0, tp_size=1, tp_group=None),
    ):
        yield


def random_fp8_expert_weights(
    generator: torch.Generator, expert_count: int
) -> dict[str, torch.Tensor]:
    """Random block-FP8 experts at the real layer's shapes and the FP8 method's names."""
    weight_shapes = {
        "w13_weight": (2 * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "w2_weight": (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
    }
    expert_weights = {
        name: (torch.rand((expert_count, *shape), generator=generator) * 2 - 1)
        .to(torch.float8_e4m3fn)
        .cuda()
        for name, shape in weight_shapes.items()
    }
    for name, (rows, columns) in weight_shapes.items():
        expert_weights[f"{name}_scale_inv"] = (
            torch.rand(
                (expert_count, rows // BLOCK_SIZE, columns // BLOCK_SIZE),
                generator=generator,
            )
            * 0.01
        ).cuda()
    assert tuple(expert_weights) == FP8_CACHED_TENSOR_NAMES
    return expert_weights


def triton_runner() -> MoeRunner:
    return MoeRunner(
        MoeRunnerBackend.TRITON,
        MoeRunnerConfig(
            num_experts=EXPERT_COUNT,
            num_local_experts=EXPERT_COUNT,
            hidden_size=HIDDEN_SIZE,
            intermediate_size_per_partition=MOE_INTERMEDIATE_SIZE,
            layer_id=0,
            top_k=TOP_K,
            params_dtype=torch.bfloat16,
            inplace=False,
        ),
    )


def run_triton(runner: MoeRunner, experts: CachedExperts) -> torch.Tensor:
    """What the FP8 method hands the Triton runner over these experts."""
    return runner.run(
        experts.dispatch_output,
        TritonMoeQuantInfo(
            w13_weight=experts.weights["w13_weight"],
            w2_weight=experts.weights["w2_weight"],
            use_fp8_w8a8=True,
            w13_scale=experts.weights["w13_weight_scale_inv"],
            w2_scale=experts.weights["w2_weight_scale_inv"],
            block_shape=EXPERT_WEIGHT_BLOCK_SIZE,
        ),
    ).hidden_states


def full_resident_output(
    runner: MoeRunner,
    expert_weights: dict[str, torch.Tensor],
    dispatch_output: StandardDispatchOutput,
) -> torch.Tensor:
    return run_triton(
        runner,
        CachedExperts(
            dispatch_output=dispatch_output,
            weights=expert_weights,
            empty_rows=torch.empty((EXPERT_COUNT, 0), dtype=torch.int32, device="cuda"),
        ),
    )


@pytest.mark.parametrize(("slot_count", "row_count"), [(192, 1), (128, 16)])
def test_fp8_decode_follows_the_reference_policy_over_the_slots(
    slot_count: int, row_count: int
) -> None:
    generator = torch.Generator().manual_seed(slot_count + row_count)
    expert_weights = random_fp8_expert_weights(generator, EXPERT_COUNT)
    cache = single_layer_cache(expert_weights, slot_count)
    layer_cache = cache.layers[0]
    cache.mode = CacheMode.DECODE
    runner = triton_runner()
    expected_maps = cpu_maps(layer_cache)
    for clock in range(1, 41):
        routed_experts = routed_experts_of(generator, row_count, slot_count // 2)
        dispatch_output = dispatch_of(
            random_hidden_states(generator, row_count), routed_experts
        )
        output = run_triton(runner, layer_cache.cached_experts(dispatch_output))
        reference_decode_policy(routed_experts.cpu(), *expected_maps, clock=clock)
        for cache_map, expected_map in zip(
            cpu_maps(layer_cache), expected_maps, strict=True
        ):
            assert torch.equal(cache_map, expected_map), clock
        assert_slots_hold_their_experts(layer_cache, expert_weights)
        torch.testing.assert_close(
            output,
            full_resident_output(runner, expert_weights, dispatch_output),
            rtol=FP8_RELATIVE_TOLERANCE,
            atol=FP8_ABSOLUTE_TOLERANCE,
        )
    statistics = cache.read_statistics()
    assert 0 < statistics.decode_misses < statistics.lookups


def test_fp8_prefill_gathers_the_routed_experts_and_leaves_the_cache_alone() -> None:
    slot_count = 192
    generator = torch.Generator().manual_seed(slot_count)
    expert_weights = random_fp8_expert_weights(generator, EXPERT_COUNT)
    cache = single_layer_cache(expert_weights, slot_count)
    layer_cache = cache.layers[0]
    runner = triton_runner()
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
        output = run_triton(runner, layer_cache.cached_experts(dispatch_output))
        assert_scratch_holds_the_routed_experts(cache, expert_weights, routed_experts)
        torch.testing.assert_close(
            output,
            full_resident_output(runner, expert_weights, dispatch_output),
            rtol=FP8_RELATIVE_TOLERANCE,
            atol=FP8_ABSOLUTE_TOLERANCE,
        )
    for cache_map, map_before in zip(cpu_maps(layer_cache), maps_before, strict=True):
        assert torch.equal(cache_map, map_before)
