# SPDX-License-Identifier: Apache-2.0
"""Flow graph replay, bucket padding, and eager fallbacks."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.flow_graph import (
    euler_flow_loop,
    flow_t_span,
    prepare_flow_dit,
)
from tests.unit_test.fixtures.firered_tts3 import (
    N_TIMESTEPS,
    PATCH_SIZE,
    REDAE_DIM,
    eager_patch,
    make_inputs,
    tiny_runner,
)

pytestmark = pytest.mark.usefixtures("disable_flow_optimization_environment")


@pytest.mark.accelerator
def test_graph_replay_bit_exact_eager(tiny_dit: DiT, device: torch.device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(7)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    for _ in range(3):
        history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
        with torch.inference_mode():
            actual = runner.run(
                history_latents,
                noise_latents,
                conditioning,
                guidance_scale,
                N_TIMESTEPS,
            )
            expected = eager_patch(
                tiny_dit,
                history_latents,
                noise_latents,
                conditioning,
                guidance_scale,
                device,
            )
        assert actual is not None
        assert torch.equal(actual, expected)
    stats = runner.stats()
    assert stats["calls"] == 3 and stats["captures"] == 1 and stats["hits"] == 2


@pytest.mark.accelerator
def test_cfg_placeholder_shares_one_graph(tiny_dit: DiT, device: torch.device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(11)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    for scale in (2.0, 1.2, 0.0):
        guidance_scale = torch.full((1, 1, 1), scale, device=device)
        with torch.inference_mode():
            actual = runner.run(
                history_latents,
                noise_latents,
                conditioning,
                guidance_scale,
                N_TIMESTEPS,
            )
            expected = eager_patch(
                tiny_dit,
                history_latents,
                noise_latents,
                conditioning,
                guidance_scale,
                device,
            )
        assert torch.equal(actual, expected), f"cfg={scale} diverged"
    assert runner.stats()["captures"] == 1


@pytest.mark.accelerator
def test_bucket_exact_hit_bit_exact(tiny_dit: DiT, device: torch.device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(13)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 2)
    guidance_scale = torch.full((2, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        actual = runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        expected = eager_patch(
            tiny_dit,
            history_latents,
            noise_latents,
            conditioning,
            guidance_scale,
            device,
        )
    assert torch.equal(actual, expected)
    assert runner.stats()["padded"] == 0


@pytest.mark.accelerator
def test_bucket_zero_pad_up(tiny_dit: DiT, device: torch.device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(17)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 3)
    guidance_scale = torch.full((3, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        actual = runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        expected = eager_patch(
            tiny_dit,
            history_latents,
            noise_latents,
            conditioning,
            guidance_scale,
            device,
        )
    assert actual.shape == (3, PATCH_SIZE, REDAE_DIM)
    # Cross-batch GEMM tiling keeps the padded replay off the bit-exact
    # contract; the drift must stay at fp32 roundoff scale.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    stats = runner.stats()
    assert stats["padded"] == 1 and stats["captures"] == 1


@pytest.mark.accelerator
def test_capture_failure_disables_runner(device: torch.device) -> None:
    class FailingDiT(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("capture probe")

    runner = tiny_runner(FailingDiT(), device)
    generator = torch.Generator(device=device).manual_seed(19)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    assert (
        runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        is None
    )
    assert not runner.enabled
    assert (
        runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        is None
    )
    stats = runner.stats()
    assert stats["eager"] == 2 and stats["captures"] == 0 and stats["calls"] == 2


@pytest.mark.accelerator
def test_foreign_stream_capture_falls_back(
    tiny_dit: DiT, device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(23)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    assert (
        runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        is None
    )
    stats = runner.stats()
    assert stats["eager"] == 1 and stats["captures"] == 0


@pytest.mark.accelerator
def test_n_timesteps_mismatch_falls_back(tiny_dit: DiT, device: torch.device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(29)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    assert (
        runner.run(
            history_latents,
            noise_latents,
            conditioning,
            guidance_scale,
            N_TIMESTEPS + 5,
        )
        is None
    )
    assert runner.stats()["eager"] == 1


@pytest.mark.accelerator
def test_lru_eviction(tiny_dit: DiT, device: torch.device) -> None:
    runner = tiny_runner(tiny_dit, device, max_graphs=1, capture_sizes=(1,))
    generator = torch.Generator(device=device).manual_seed(31)
    for batch_size in (1, 3, 1, 3):
        history_latents, noise_latents, conditioning = make_inputs(
            device, generator, batch_size
        )
        guidance_scale = torch.full((batch_size, 1, 1), 2.0, device=device)
        assert (
            runner.run(
                history_latents,
                noise_latents,
                conditioning,
                guidance_scale,
                N_TIMESTEPS,
            )
            is not None
        )
    stats = runner.stats()
    assert stats["cache_size"] == 1 and stats["captures"] == 4


@pytest.mark.accelerator
def test_bf16_mode_graph_bit_exact_vs_bf16_eager(
    tiny_dit: DiT, device: torch.device
) -> None:
    run_dit, _ = prepare_flow_dit(tiny_dit, "bf16")
    runner = tiny_runner(run_dit, device, mode="bf16")
    generator = torch.Generator(device=device).manual_seed(41)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        actual = runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        expected = euler_flow_loop(
            run_dit,
            torch.cat([history_latents, noise_latents], dim=1),
            conditioning,
            flow_t_span(N_TIMESTEPS, device),
            guidance_scale,
            N_TIMESTEPS,
            PATCH_SIZE,
            dit_dtype=torch.bfloat16,
        )
    assert actual is not None
    # Same stack, exact bucket hit: graph replay is bit-exact vs the eager
    # loop, and the blend/integration state stayed fp32.
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert next(tiny_dit.parameters()).dtype == torch.float32


@pytest.mark.accelerator
def test_compile_failure_degrades_to_uncompiled_stack(
    tiny_dit: DiT, device: torch.device
) -> None:
    class BrokenCompiledDiT(torch.nn.Module):
        def __init__(self, inner: DiT) -> None:
            super().__init__()
            self.inner: DiT = inner

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("inductor probe")

    _, fallback_dit = prepare_flow_dit(tiny_dit, "compile_fp32")
    runner = tiny_runner(
        BrokenCompiledDiT(tiny_dit),
        device,
        mode="compile_fp32",
        dit_fallback=fallback_dit,
    )
    generator = torch.Generator(device=device).manual_seed(43)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        actual = runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        expected = eager_patch(
            tiny_dit,
            history_latents,
            noise_latents,
            conditioning,
            guidance_scale,
            device,
        )
    assert actual is not None
    assert runner.dit is runner.dit_fallback
    assert torch.equal(actual, expected)
    stats = runner.stats()
    assert stats["captures"] == 1 and stats["eager"] == 0


@pytest.mark.accelerator
def test_compile_fp32_graph_captures_and_runs(
    tiny_dit: DiT, device: torch.device
) -> None:
    run_dit, fallback_dit = prepare_flow_dit(tiny_dit, "compile_fp32")
    runner = tiny_runner(
        run_dit,
        device,
        mode="compile_fp32",
        dit_fallback=fallback_dit,
        capture_sizes=(1,),
    )
    generator = torch.Generator(device=device).manual_seed(47)
    history_latents, noise_latents, conditioning = make_inputs(device, generator, 1)
    guidance_scale = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        actual = runner.run(
            history_latents, noise_latents, conditioning, guidance_scale, N_TIMESTEPS
        )
        expected = eager_patch(
            tiny_dit,
            history_latents,
            noise_latents,
            conditioning,
            guidance_scale,
            device,
        )
    assert actual is not None
    assert runner.dit is run_dit  # no degradation: compile captured cleanly
    # Inductor reassociates fp32 math; the drift stays far inside the
    # batched-GEMM roundoff envelope (~5e-3) the plan budgets for c>=2.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-4)
