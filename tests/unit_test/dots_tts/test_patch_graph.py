# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead
from sglang_omni.models.dots_tts.patch_cache import (
    PatchSolveGraph,
    PersistentCacheDiTRunner,
    TailBufferNextPatchStep,
)
from sglang_omni.models.dots_tts.patch_graph import GraphedDiTSolver
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform
from tests.unit_test.fixtures.accelerator import require_device_streams
from tests.unit_test.fixtures.dots_tts import (
    FM_HIDDEN,
    ODE_STEPS,
    UNIT_TOKENS,
    graphed_solver,
    tiny_flow_head,
)

import_dots_tts()

from dots_tts.modules.backbone.dit_inference import (  # noqa: E402
    DiTInferenceContext,
    DiTKvCache,
    DiTSolver,
    DiTSolverState,
)


def eager_solver(flow: DotsTTSFlowHead) -> DiTSolver:
    return DiTSolver(
        DiTInferenceContext.from_core(flow),
        optimize=True,
        bucket_resolver=flow.bucket,
        meanflow=flow.mode == "meanflow",
    )


class FlowSequence:
    def __init__(
        self, flow: DotsTTSFlowHead, device: torch.device, dtype: torch.dtype
    ) -> None:
        generator = torch.Generator(device="cpu").manual_seed(7)
        shape = (1, 200 * UNIT_TOKENS, FM_HIDDEN)
        self.sequence = (torch.randn(shape, generator=generator) * 0.5).to(
            device=device, dtype=dtype
        )
        self.cfg_sequence = self.sequence.clone()
        self.cfg_sequence[:, ::UNIT_TOKENS] = 0.1
        self.null_g_cond = torch.zeros((1, FM_HIDDEN), device=device, dtype=dtype)
        self.g_cond = (torch.randn((1, FM_HIDDEN), generator=generator) * 0.1).to(
            device=device, dtype=dtype
        )
        self.meanflow = flow.mode == "meanflow"
        self.device = device

    def decode(
        self, solver: DiTSolver, state: DiTSolverState, flow_tokens: int, seed: int
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        if self.device.type != "cpu":
            torch.get_device_module(self.device).manual_seed(seed)
        else:
            pass
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.sequence.dtype,
                enabled=self.device.type != "cpu",
            ),
        ):
            return solver.decode_next(
                state,
                sequence=self.sequence,
                cfg_sequence=None if self.meanflow else self.cfg_sequence,
                fm_seq_len=flow_tokens,
                null_g_cond=self.null_g_cond,
                g_cond=self.g_cond,
                nfe=ODE_STEPS,
                ode_method="euler",
                guidance_scale=None if self.meanflow else 1.2,
            )


def test_only_xpu_replays_dots_patch_solves_as_graphs() -> None:
    assert XPUOmniPlatform().enable_dots_tts_patch_graph() is True
    assert CUDAOmniPlatform().enable_dots_tts_patch_graph() is False
    assert OmniPlatform().enable_dots_tts_patch_graph() is False


@pytest.mark.parametrize("platform_asks", [True, False])
def test_flow_head_builds_the_graphed_solver_only_when_the_platform_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform_asks: bool
) -> None:
    graph_backend = object()
    monkeypatch.setattr(
        current_platform, "enable_dots_tts_patch_graph", lambda: platform_asks
    )
    monkeypatch.setattr(
        current_platform, "get_device_graph_backend", lambda device: graph_backend
    )
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)

    solver = flow.solver()

    if platform_asks:
        assert isinstance(solver, GraphedDiTSolver)
        assert solver.graph_backend is graph_backend
    else:
        assert type(solver) is DiTSolver


def test_flow_head_keeps_the_eager_solver_without_optimize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(current_platform, "enable_dots_tts_patch_graph", lambda: True)
    monkeypatch.setattr(
        current_platform, "get_device_graph_backend", lambda device: object()
    )
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True)
    flow.optimize = False

    assert type(flow.solver()) is DiTSolver


@pytest.mark.parametrize("meanflow", [False, True], ids=["flow_matching", "meanflow"])
def test_a_refused_capture_solves_eagerly_and_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meanflow: bool
) -> None:
    flow = tiny_flow_head(tmp_path, meanflow=meanflow, optimize=True)
    flow_sequence = FlowSequence(flow, torch.device("cpu"), torch.float32)
    eager = eager_solver(flow)
    graphed = graphed_solver(flow, object())
    refused_buckets: list[int] = []

    def refuse(
        runner: PersistentCacheDiTRunner,
        kv_cache: DiTKvCache,
        patch_inputs: dict[str, torch.Tensor],
        *,
        ode_steps: int,
        batch_size: int,
    ) -> PatchSolveGraph:
        refused_buckets.append(int(runner.capacity_patches))
        raise RuntimeError("capture refused")

    monkeypatch.setattr(graphed, "capture", refuse)
    eager_state, graphed_state = DiTSolverState(), DiTSolverState()
    flow_tokens = 60 * UNIT_TOKENS + 1
    for patch in range(8):
        expected = flow_sequence.decode(eager, eager_state, flow_tokens, seed=patch)
        actual = flow_sequence.decode(graphed, graphed_state, flow_tokens, seed=patch)
        assert torch.equal(actual, expected), patch
        flow_tokens += UNIT_TOKENS

    assert refused_buckets == [64, 128]
    assert graphed.declined == {(64, 1, ODE_STEPS), (128, 1, ODE_STEPS)}
    assert graphed.replays == 0


def graph_backend_or_skip(device: torch.device) -> DeviceGraphBackend:
    graph_backend = current_platform.get_device_graph_backend(device)
    if graph_backend is None or not current_platform.enable_dots_tts_patch_graph():
        pytest.skip(f"{device.type} does not replay dots.tts patch solves as graphs")
    else:
        pass
    return graph_backend


def decode_against_eager(
    flow_sequence: FlowSequence,
    eager: DiTSolver,
    graphed: GraphedDiTSolver,
    starts: tuple[int, ...],
) -> int:
    patches = 0
    for request, start_patches in enumerate(starts):
        eager_state, graphed_state = DiTSolverState(), DiTSolverState()
        flow_tokens = start_patches * UNIT_TOKENS + 1
        for patch in range(8):
            seed = 100 * request + patch
            expected = flow_sequence.decode(eager, eager_state, flow_tokens, seed)
            actual = flow_sequence.decode(graphed, graphed_state, flow_tokens, seed)
            assert torch.equal(actual, expected), (request, patch)
            flow_tokens += UNIT_TOKENS
            patches += 1
        valid_tokens = eager_state.cache.valid_tokens
        assert graphed_state.cache.valid_tokens == valid_tokens
        assert graphed_state.cache.capacity_patches == (
            eager_state.cache.capacity_patches
        )
        for name in ("cache_k", "cache_v"):
            assert torch.equal(
                getattr(graphed_state.cache, name)[..., :valid_tokens, :],
                getattr(eager_state.cache, name)[..., :valid_tokens, :],
            )
    return patches


class CaptureFailingBackend:
    def __init__(self, backend: DeviceGraphBackend) -> None:
        self.backend = backend
        self.capturing = False

    def graph_pool_handle(self) -> DeviceGraphPool:
        return self.backend.graph_pool_handle()

    @contextmanager
    def capture(
        self,
        *,
        pool: DeviceGraphPool | None = None,
        stream: torch.Stream | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[ReplayableGraph]:
        with self.backend.capture(
            pool=pool, stream=stream, thread_local_errors=thread_local_errors
        ) as graph:
            self.capturing = True
            try:
                yield graph
            finally:
                self.capturing = False


@pytest.mark.accelerator
@pytest.mark.parametrize("meanflow", [False, True], ids=["flow_matching", "meanflow"])
def test_graph_replay_matches_the_eager_solve_bit_for_bit(
    tmp_path: Path, meanflow: bool
) -> None:
    device = require_device_streams()
    graph_backend = graph_backend_or_skip(device)
    flow = tiny_flow_head(tmp_path, meanflow=meanflow, optimize=True).to(
        device=device, dtype=torch.bfloat16
    )
    flow_sequence = FlowSequence(flow, device, torch.bfloat16)
    graphed = graphed_solver(flow, graph_backend)

    patches = decode_against_eager(
        flow_sequence, eager_solver(flow), graphed, starts=(60, 20)
    )

    assert graphed.declined == set()
    assert graphed.replays == patches


@pytest.mark.accelerator
def test_a_graph_captures_after_every_resident_graph_was_evicted(
    tmp_path: Path,
) -> None:
    device = require_device_streams()
    graph_backend = graph_backend_or_skip(device)
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True).to(
        device=device, dtype=torch.bfloat16
    )
    flow_sequence = FlowSequence(flow, device, torch.bfloat16)
    one_long_cache = (
        graphed_solver(flow, graph_backend)
        ._get_cached_runner(  # noqa: leading-underscore  # DiTSolver hook
            capacity_patches=128, device=device, dtype=torch.bfloat16
        )
        .cache_bytes(batch_size=1, nfe=ODE_STEPS)
    )
    graphed = graphed_solver(flow, graph_backend, resident_cache_bytes=one_long_cache)

    patches = decode_against_eager(
        flow_sequence, eager_solver(flow), graphed, starts=(60, 20)
    )

    assert graphed.declined == set()
    assert graphed.replays == patches
    assert [
        (owner.capacity_patches, shape) for owner, shape in graphed.cache_budget.entries
    ] == [(64, (1, ODE_STEPS))]
    torch.randn(4, device=device)


@pytest.mark.accelerator
def test_an_exception_inside_a_real_capture_falls_back_and_keeps_rng_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = require_device_streams()
    backend = CaptureFailingBackend(graph_backend_or_skip(device))
    run_blocks = (
        TailBufferNextPatchStep._run_cached_blocks  # noqa: leading-underscore  # NextPatchStep hook
    )
    refused: list[int] = []

    def refuse_inside_the_first_capture(
        self: TailBufferNextPatchStep, x: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if backend.capturing and not refused:
            refused.append(int(x.shape[1]))
            raise RuntimeError("refused inside capture")
        else:
            pass
        return run_blocks(self, x, **kwargs)

    monkeypatch.setattr(
        TailBufferNextPatchStep,
        "_run_cached_blocks",  # noqa: leading-underscore  # NextPatchStep hook
        refuse_inside_the_first_capture,
    )
    flow = tiny_flow_head(tmp_path, meanflow=False, optimize=True).to(
        device=device, dtype=torch.bfloat16
    )
    flow_sequence = FlowSequence(flow, device, torch.bfloat16)
    graphed = graphed_solver(flow, backend)

    patches = decode_against_eager(
        flow_sequence, eager_solver(flow), graphed, starts=(60,)
    )

    assert len(refused) == 1
    assert graphed.declined == {(64, 1, ODE_STEPS)}
    assert 0 < graphed.replays < patches
    seeded = torch.randn(
        4, device=device, generator=torch.Generator(device=device).manual_seed(5)
    )
    unseeded = torch.randn(4, device=device)
    assert torch.isfinite(seeded).all() and torch.isfinite(unseeded).all()
