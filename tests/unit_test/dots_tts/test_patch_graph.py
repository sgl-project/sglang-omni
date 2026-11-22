# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead
from sglang_omni.models.dots_tts.patch_graph import (
    GraphedDiTSolver,
    PatchSolveGraph,
    PersistentCacheDiTRunner,
    apply_rotary_embedding,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.device_graph import DeviceGraphBackend
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.xpu import XPUOmniPlatform
from tests.unit_test.fixtures.accelerator import require_device_streams

import_dots_tts()

from dots_tts.modules.backbone.dit_inference import (  # noqa: E402
    DiTInferenceContext,
    DiTKvCache,
    DiTSolver,
    DiTSolverState,
)
from dots_tts.modules.backbone.layers import rotate_half  # noqa: E402

LLM_HIDDEN = 48
FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
UNIT_TOKENS = 1 + PATCH_SIZE
ODE_STEPS = 2


def flow_head(tmp_path: Path, *, meanflow: bool) -> DotsTTSFlowHead:
    torch.save(
        {"mean": torch.zeros(LATENT_DIM), "var": torch.ones(LATENT_DIM)},
        tmp_path / "latent_stats.pt",
    )
    config = {
        "latent_dim": LATENT_DIM,
        "patch_size": PATCH_SIZE,
        "PatchEncoder": {
            "num_layers": 1,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "causal": True,
        },
        "DiT": {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "modulation": True,
            "qk_norm": True,
            "rotary_bias": True,
        },
        "vocoder": {"sample_rate": 48000},
        "meanflow": (
            {"enabled": True, "use_duration_embedding": True} if meanflow else None
        ),
    }
    flow = DotsTTSFlowHead(
        config,
        llm_hidden_size=LLM_HIDDEN,
        latent_stats_path=str(tmp_path / "latent_stats.pt"),
        optimize=True,
    )
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.normal_(0.0, 0.2)
    return flow.eval()


def eager_solver(flow: DotsTTSFlowHead) -> DiTSolver:
    return DiTSolver(
        DiTInferenceContext.from_core(flow),
        optimize=True,
        bucket_resolver=flow.bucket,
        meanflow=flow.mode == "meanflow",
    )


def graphed_solver(
    flow: DotsTTSFlowHead, graph_backend: DeviceGraphBackend
) -> GraphedDiTSolver:
    return GraphedDiTSolver(
        DiTInferenceContext.from_core(flow),
        optimize=True,
        bucket_resolver=flow.bucket,
        meanflow=flow.mode == "meanflow",
        graph_backend=graph_backend,
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
    flow = flow_head(tmp_path, meanflow=False)

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
    flow = flow_head(tmp_path, meanflow=False)
    flow.optimize = False

    assert type(flow.solver()) is DiTSolver


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
    flow = flow_head(tmp_path, meanflow=False)

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
        )


def test_a_reused_cache_starts_zeroed_like_a_fresh_one(tmp_path: Path) -> None:
    flow = flow_head(tmp_path, meanflow=False)
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


@pytest.mark.parametrize("meanflow", [False, True], ids=["flow_matching", "meanflow"])
def test_a_refused_capture_solves_eagerly_and_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meanflow: bool
) -> None:
    flow = flow_head(tmp_path, meanflow=meanflow)
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


@pytest.mark.accelerator
@pytest.mark.parametrize("meanflow", [False, True], ids=["flow_matching", "meanflow"])
def test_graph_replay_matches_the_eager_solve_bit_for_bit(
    tmp_path: Path, meanflow: bool
) -> None:
    device = require_device_streams()
    graph_backend = current_platform.get_device_graph_backend(device)
    if graph_backend is None or not current_platform.enable_dots_tts_patch_graph():
        pytest.skip(f"{device.type} does not replay dots.tts patch solves as graphs")
    else:
        pass
    flow = flow_head(tmp_path, meanflow=meanflow).to(
        device=device, dtype=torch.bfloat16
    )
    flow_sequence = FlowSequence(flow, device, torch.bfloat16)
    eager = eager_solver(flow)
    graphed = graphed_solver(flow, graph_backend)

    patches = 0
    for request, start_patches in enumerate((60, 20)):
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
        assert torch.equal(
            graphed_state.cache.cache_k[..., :valid_tokens, :],
            eager_state.cache.cache_k[..., :valid_tokens, :],
        )
        assert torch.equal(
            graphed_state.cache.cache_v[..., :valid_tokens, :],
            eager_state.cache.cache_v[..., :valid_tokens, :],
        )

    assert graphed.declined == set()
    assert graphed.replays == patches
