# SPDX-License-Identifier: Apache-2.0
"""Ming-TTS's bf16 routed experts served from the expert cache match the resident layer."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_context, get_parallel

from sglang_omni.models.fishaudio_drama3.expert_cache import (
    CacheMode,
    RoutedExpertCache,
    pinned_host_copy,
)
from sglang_omni.models.ming_tts.routed_experts import BF16_EXPERT_TENSOR_NAMES

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

# Ming-omni-tts-16.8B-A3B's MoE layer, narrowed so the test layer stays small.
EXPERT_COUNT = 64
TOP_K = 6
HIDDEN_SIZE = 512
MOE_INTERMEDIATE_SIZE = 256
SLOT_COUNT = 32
DECODE_STEP_COUNT = 40
# note (minyi-tang): the Triton kernel tiles each expert the same way wherever it
# sits, so slot and full-layer outputs agree to bf16 rounding.
RELATIVE_TOLERANCE = 2**-7
ABSOLUTE_TOLERANCE = 1e-3


@pytest.fixture(autouse=True)
def single_rank_runtime() -> Iterator[None]:
    """The Triton fused MoE reads the server config and the tensor-parallel group."""
    with (
        get_context().override_server_args(),
        get_parallel().override(tp_group=None, tp_rank=0, tp_size=1),
    ):
        yield


def random_expert_weights(generator: torch.Generator) -> dict[str, torch.Tensor]:
    shapes = {
        "w13_weight": (EXPERT_COUNT, 2 * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "w2_weight": (EXPERT_COUNT, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE),
    }
    return {
        name: (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16).cuda()
        for name, shape in shapes.items()
    }


def routed_rows(
    generator: torch.Generator, row_count: int
) -> tuple[torch.Tensor, StandardTopKOutput]:
    hidden_states = (
        torch.randn((row_count, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)
    ).cuda()
    routed_experts = torch.stack(
        [
            torch.randperm(EXPERT_COUNT, generator=generator)[:TOP_K]
            for _ in range(row_count)
        ]
    ).to(device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.rand((row_count, TOP_K), generator=generator), dim=-1
    ).cuda()
    return hidden_states, StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=routed_experts,
        router_logits=torch.zeros((row_count, EXPERT_COUNT), device="cuda"),
    )


def run_experts(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
    topk_output: StandardTopKOutput,
) -> torch.Tensor:
    return fused_experts(
        hidden_states,
        weights["w13_weight"],
        weights["w2_weight"],
        topk_output,
        MoeRunnerConfig(inplace=False),
    )


def build_cache(
    expert_weights: dict[str, torch.Tensor],
) -> RoutedExpertCache:
    return RoutedExpertCache(
        [pinned_host_copy(expert_weights)], SLOT_COUNT, torch.device("cuda")
    )


def test_decode_reads_slots_that_hold_their_experts() -> None:
    generator = torch.Generator().manual_seed(0)
    expert_weights = random_expert_weights(generator)
    routed_expert_cache = build_cache(expert_weights)
    layer_cache = routed_expert_cache.layers[0]
    routed_expert_cache.mode = CacheMode.DECODE
    for step in range(DECODE_STEP_COUNT):
        row_count = 1 + step % (SLOT_COUNT // TOP_K)
        hidden_states, topk_output = routed_rows(generator, row_count)
        cached_experts = layer_cache.cached_experts(topk_output)
        torch.testing.assert_close(
            run_experts(
                hidden_states, cached_experts.weights, cached_experts.topk_output
            ),
            run_experts(hidden_states, expert_weights, topk_output),
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
        slot_to_expert = layer_cache.slot_to_expert.long()
        for name in BF16_EXPERT_TENSOR_NAMES:
            assert torch.equal(
                layer_cache.slot_weights[name], expert_weights[name][slot_to_expert]
            )
        assert torch.equal(
            layer_cache.expert_to_slot[slot_to_expert].long(),
            torch.arange(SLOT_COUNT, device="cuda"),
        )
    statistics = routed_expert_cache.read_statistics()
    assert 0 < statistics.decode_misses < statistics.lookups


@pytest.mark.parametrize("row_count", [1, 37, 300])
def test_prefill_reads_the_scratch_and_leaves_the_cache(row_count: int) -> None:
    generator = torch.Generator().manual_seed(row_count)
    expert_weights = random_expert_weights(generator)
    routed_expert_cache = build_cache(expert_weights)
    layer_cache = routed_expert_cache.layers[0]
    expert_to_slot_before = layer_cache.expert_to_slot.clone()
    routed_expert_cache.mode = CacheMode.PREFILL
    hidden_states, topk_output = routed_rows(generator, row_count)
    cached_experts = layer_cache.cached_experts(topk_output)
    torch.testing.assert_close(
        run_experts(hidden_states, cached_experts.weights, cached_experts.topk_output),
        run_experts(hidden_states, expert_weights, topk_output),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    assert torch.equal(layer_cache.expert_to_slot, expert_to_slot_before)


def test_captured_decode_step_replays_against_new_routing() -> None:
    generator = torch.Generator().manual_seed(7)
    expert_weights = random_expert_weights(generator)
    routed_expert_cache = build_cache(expert_weights)
    layer_cache = routed_expert_cache.layers[0]
    routed_expert_cache.mode = CacheMode.DECODE
    row_count = SLOT_COUNT // TOP_K
    hidden_states, topk_output = routed_rows(generator, row_count)
    # Warm the Triton kernels outside the graph before capturing.
    run_experts(
        hidden_states,
        layer_cache.cached_experts(topk_output).weights,
        layer_cache.cached_experts(topk_output).topk_output,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cached_experts = layer_cache.cached_experts(topk_output)
        graph_output = run_experts(
            hidden_states, cached_experts.weights, cached_experts.topk_output
        )
    for _ in range(8):
        next_hidden_states, next_topk_output = routed_rows(generator, row_count)
        hidden_states.copy_(next_hidden_states)
        topk_output.topk_ids.copy_(next_topk_output.topk_ids)
        topk_output.topk_weights.copy_(next_topk_output.topk_weights)
        graph.replay()
        torch.testing.assert_close(
            graph_output,
            run_experts(hidden_states, expert_weights, topk_output),
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
