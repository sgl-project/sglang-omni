# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.patch_cache import (
    RESIDENT_CACHE_BYTES,
    PatchSolveGraph,
    PersistentCacheDiTRunner,
    ResidentCacheBudget,
    apply_rotary_embedding,
)
from sglang_omni.models.dots_tts.patch_graph import GraphedDiTSolver
from tests.unit_test.fixtures.dots_tts import (
    ODE_STEPS,
    UNIT_TOKENS,
    graphed_solver,
    tiny_flow_head,
)

import_dots_tts()

from dots_tts.modules.backbone.dit_inference import (  # noqa: E402
    DiTInferenceContext,
    DiTKvCache,
)
from dots_tts.modules.backbone.layers import rotate_half  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_concat_free_rotation_matches_rotate_half_bit_for_bit(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(3)
    values = torch.randn((2, 4, 10, 64), generator=generator).to(dtype)
    angles = torch.randn((1, 1, 10, 64), generator=generator) * 30.0
    cos, sin = angles.cos(), angles.sin()

    expected = (values * cos + rotate_half(values) * sin).to(dtype=values.dtype)

    assert torch.equal(apply_rotary_embedding(values, cos, sin), expected)


def test_the_persistent_runner_refuses_the_flex_backend(tmp_path: Path) -> None:
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)

    with pytest.raises(ValueError, match="sdpa"):
        PersistentCacheDiTRunner(
            context=DiTInferenceContext.from_core(flow),
            capacity_patches=64,
            capacity_tokens=64 * UNIT_TOKENS,
            dtype=torch.float32,
            device=torch.device("cpu"),
            attn_backend="flex",
            compile_step=False,
            branch_multiplier=2,
            budget=ResidentCacheBudget(RESIDENT_CACHE_BYTES),
        )


def test_a_reused_cache_starts_zeroed_like_a_fresh_one(tmp_path: Path) -> None:
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)
    solver = graphed_solver(flow, object())
    runner = solver._get_cached_runner(  # noqa: leading-underscore  # DiTSolver hook
        capacity_patches=64, device=torch.device("cpu"), dtype=torch.float32
    )
    assert isinstance(runner, PersistentCacheDiTRunner)

    first = runner.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    first.cache_k.fill_(3.0)
    first.cache_v.fill_(-3.0)
    first.valid_tokens = 9
    second = runner.allocate_cache(batch_size=1, nfe=ODE_STEPS)

    assert second.cache_k.data_ptr() == first.cache_k.data_ptr()
    assert second.cache_v.data_ptr() == first.cache_v.data_ptr()
    assert second.valid_tokens == 0
    assert not second.cache_k.any() and not second.cache_v.any()
    other = runner.allocate_cache(batch_size=1, nfe=ODE_STEPS + 1)
    assert other.cache_k.data_ptr() != first.cache_k.data_ptr()


class RecordingGraph:
    def __init__(self) -> None:
        self.resets = 0

    def replay(self) -> None:
        raise AssertionError("an evicted patch graph replayed")

    def reset(self) -> None:
        self.resets += 1


def cpu_runner(
    solver: GraphedDiTSolver, capacity_patches: int
) -> PersistentCacheDiTRunner:
    return solver._get_cached_runner(  # noqa: leading-underscore  # DiTSolver hook
        capacity_patches=capacity_patches,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def install_graph(runner: PersistentCacheDiTRunner, steps: int) -> RecordingGraph:
    graph = RecordingGraph()
    runner.patch_graphs[(1, steps)] = PatchSolveGraph(graph, {}, torch.zeros(1))
    return graph


def test_resident_caches_share_one_byte_budget_across_buckets(tmp_path: Path) -> None:
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)
    probe = cpu_runner(graphed_solver(flow, object()), 64)
    small = probe.cache_bytes(batch_size=1, nfe=ODE_STEPS)
    solver = graphed_solver(flow, object(), resident_cache_bytes=3 * small)
    short, long = cpu_runner(solver, 64), cpu_runner(solver, 128)

    short.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    short_graph = install_graph(short, ODE_STEPS)
    stale = long.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    long_graph = install_graph(long, ODE_STEPS)
    short.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    short.allocate_cache(batch_size=1, nfe=ODE_STEPS + 1)

    assert list(solver.cache_budget.entries) == [
        (short, (1, ODE_STEPS)),
        (short, (1, ODE_STEPS + 1)),
    ]
    assert solver.cache_budget.resident_bytes <= 3 * small
    assert not long.persistent_caches and not long.patch_graphs
    assert long_graph.resets == 1 and short_graph.resets == 0
    assert (
        solver.patch_graph(long, stale, {}, ode_steps=ODE_STEPS, batch_size=1) is None
    )


def test_a_cache_larger_than_the_budget_is_request_scoped(tmp_path: Path) -> None:
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)
    probe = cpu_runner(graphed_solver(flow, object()), 128)
    solver = graphed_solver(
        flow,
        object(),
        resident_cache_bytes=probe.cache_bytes(batch_size=1, nfe=ODE_STEPS) - 1,
    )
    runner = cpu_runner(solver, 128)

    first = runner.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    second = runner.allocate_cache(batch_size=1, nfe=ODE_STEPS)

    assert first.cache_k.data_ptr() != second.cache_k.data_ptr()
    assert not runner.persistent_caches and not solver.cache_budget.entries
    assert (
        solver.patch_graph(runner, first, {}, ode_steps=ODE_STEPS, batch_size=1) is None
    )


def test_a_failed_cache_allocation_releases_its_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)
    probe = cpu_runner(graphed_solver(flow, object()), 64)
    small = probe.cache_bytes(batch_size=1, nfe=ODE_STEPS)
    solver = graphed_solver(flow, object(), resident_cache_bytes=small)
    runner = cpu_runner(solver, 64)
    allocate = runner.new_cache

    def out_of_memory(*, batch_size: int, nfe: int) -> DiTKvCache:
        raise torch.OutOfMemoryError("device out of memory")

    monkeypatch.setattr(runner, "new_cache", out_of_memory)
    with pytest.raises(torch.OutOfMemoryError):
        runner.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    assert not solver.cache_budget.entries and not runner.persistent_caches

    monkeypatch.setattr(runner, "new_cache", allocate)
    runner.allocate_cache(batch_size=1, nfe=ODE_STEPS)
    runner.allocate_cache(batch_size=1, nfe=ODE_STEPS - 1)

    assert list(solver.cache_budget.entries) == [(runner, (1, ODE_STEPS - 1))]
    assert list(runner.persistent_caches) == [(1, ODE_STEPS - 1)]
