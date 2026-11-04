# SPDX-License-Identifier: Apache-2.0
"""Optimized flow gates, fallback paths, and graph resource lifetimes."""

from __future__ import annotations

import collections
import gc
import weakref
from unittest.mock import create_autospec

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sglang_omni.models.firered_tts3.flow_graph as flow_graph_module
from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.flow_graph import euler_flow_loop, prepare_flow_dit
from sglang_omni.models.firered_tts3.flow_time_cache import (
    TimeConditionCache,
    TimeGemmMode,
)
from tests.unit_test.fixtures.firered_tts3 import (
    N_TIMESTEPS,
    PATCH_SIZE,
    make_inputs,
    tiny_runner,
)

pytestmark = pytest.mark.usefixtures("disable_flow_optimization_environment")


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "corrupted_stage,atol,conv_enabled,accepted",
    [
        ("conv", 0.0, True, False),
        ("conv", 0.5, True, True),
        ("conv", 0.125, True, False),
        ("observed", 10.0, True, False),
        ("time", 10.0, False, False),
        ("time", 10.0, True, False),
    ],
)
def test_flow_tolerance_only_relaxes_convolution_gate(
    flow_dispatch_dit: DiT,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    corrupted_stage: str,
    atol: float,
    conv_enabled: bool,
    accepted: bool,
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", str(atol))
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv(
        "FIRERED_TTS3_PRECOMPUTE_TIME", "1" if corrupted_stage == "time" else "0"
    )
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT",
        "conv2d_channels_last" if conv_enabled else "conv1d",
    )
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    assert isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    original_loop = flow_graph_module.euler_flow_loop
    corrupted_calls = 0
    injected_delta = 0.25

    def perturb_selected_flow(
        dit: torch.nn.Module | CompiledDiTExecutor,
        xt: torch.Tensor,
        cond: torch.Tensor,
        t_span: torch.Tensor,
        cfg: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        *,
        dit_dtype: torch.dtype = torch.float32,
        executor: CompiledDiTExecutor | None = None,
        dispatch_modes: tuple[TorchDispatchMode, ...] = (),
        time_mode: TimeGemmMode | None = None,
    ) -> torch.Tensor:
        nonlocal corrupted_calls
        patch = original_loop(
            dit,
            xt,
            cond,
            t_span,
            cfg,
            n_timesteps,
            patch_size,
            dit_dtype=dit_dtype,
            executor=executor,
            dispatch_modes=dispatch_modes,
            time_mode=time_mode,
        )
        modes = dispatch_modes
        has_time = any(isinstance(mode, TimeGemmMode) for mode in modes)
        has_conv = any(isinstance(mode, Conv1dWeightCache) for mode in modes)
        should_corrupt = (
            (corrupted_stage == "observed" and executor is not None and not modes)
            or (corrupted_stage == "time" and has_time and not has_conv)
            or (corrupted_stage == "conv" and has_conv)
        )
        if should_corrupt:
            corrupted_calls += 1
            patch.add_(injected_delta)
        else:
            pass
        return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", perturb_selected_flow)
    generator = torch.Generator(device=device).manual_seed(71)
    hist, noise, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 1.2, device=device)
    with torch.inference_mode():
        got = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        calls_after_capture = corrupted_calls
        repeated = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        expected = original_loop(
            execution.original,
            torch.cat([hist, noise], dim=1),
            cond,
            runner.t_span,
            cfg,
            N_TIMESTEPS,
            PATCH_SIZE,
            dit_dtype=torch.bfloat16,
        )
        if accepted:
            expected = expected + injected_delta
        else:
            pass
    assert calls_after_capture > 0 and corrupted_calls == calls_after_capture
    assert got is not None and torch.equal(got, expected)
    assert torch.equal(got, repeated)
    assert runner.enabled and runner.dit is execution and runner.dit is not fallback
    stats = runner.stats()
    assert stats["optimized_captures"] == int(accepted)
    assert stats["optimization_fallbacks"] == int(not accepted)
    assert stats["captures"] == 1 and stats["eager"] == 0
    entry = next(iter(runner.cache.values()))
    if accepted:
        assert any(isinstance(mode, Conv1dWeightCache) for mode in entry.dispatch_modes)
        assert "Flow optimization rejected" not in caplog.text
    else:
        assert entry.dispatch_modes == () and entry.time_table is None
        assert "Flow optimization rejected" in caplog.text


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "time_cache,conv_cache", [(True, False), (False, True), (True, True)]
)
def test_optimized_flow_hits_dispatch_and_retains_time_table(
    flow_dispatch_dit: DiT,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    time_cache: bool,
    conv_cache: bool,
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", str(int(time_cache)))
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last" if conv_cache else "conv1d"
    )
    dispatch = create_autospec(
        TimeGemmMode.__torch_dispatch__, side_effect=TimeGemmMode.__torch_dispatch__
    )
    monkeypatch.setattr(TimeGemmMode, "__torch_dispatch__", dispatch)
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    assert isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    generator = torch.Generator(device=device).manual_seed(73)
    hist, noise, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        expected = euler_flow_loop(
            execution.original,
            torch.cat([hist, noise], dim=1),
            cond,
            runner.t_span,
            cfg,
            N_TIMESTEPS,
            PATCH_SIZE,
            dit_dtype=torch.bfloat16,
        )
    assert got is not None and torch.equal(got, expected)
    assert runner.dit is execution and runner.enabled
    assert runner.stats()["optimized_captures"] == 1
    assert runner.stats()["optimization_fallbacks"] == 0
    assert runner.stats()["eager"] == 0
    entry = next(iter(runner.cache.values()))
    assert sum(isinstance(mode, TimeGemmMode) for mode in entry.dispatch_modes) == int(
        time_cache
    )
    assert sum(
        isinstance(mode, Conv1dWeightCache) for mode in entry.dispatch_modes
    ) == int(conv_cache)
    cached_time_hits = collections.Counter(
        id(call.args[0])
        for call in dispatch.call_args_list
        if call.args[0].phase == "cached"
        and call.args[1] is torch.ops.aten.addmm.out
        and call.args[3][2].data_ptr() in call.args[0].weights
    )
    dispatch.reset_mock()
    if time_cache:
        time_mode = next(
            mode for mode in entry.dispatch_modes if isinstance(mode, TimeGemmMode)
        )
        assert time_mode.table is entry.time_table and entry.time_table is not None
        assert len(entry.time_table.steps) == N_TIMESTEPS
        assert cached_time_hits[id(time_mode)] > 0
        table_ref = weakref.ref(entry.time_table)
        value_ref = weakref.ref(next(iter(entry.time_table.steps[0].values())).output)
        del time_mode
        gc.collect()
        assert table_ref() is entry.time_table and value_ref() is not None
    else:
        assert entry.time_table is None
    if conv_cache:
        assert (
            next(
                mode
                for mode in entry.dispatch_modes
                if isinstance(mode, Conv1dWeightCache)
            ).hits
            > 0
        )
    else:
        pass

    def reject_time_weight_scan(self: TimeConditionCache) -> None:
        raise AssertionError("Time weights must not be scanned during replay")

    monkeypatch.setattr(TimeConditionCache, "validate_weights", reject_time_weight_scan)
    with torch.inference_mode():
        repeated = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
    assert torch.equal(got, repeated)
    assert dispatch.call_count == 0
    assert runner.stats()["captures"] == 1 and runner.stats()["hits"] == 1
    torch.cuda.synchronize(device)
    runner.cache.clear()
    del entry
    gc.collect()
    if time_cache:
        assert table_ref() is None and value_ref() is None
    else:
        pass


@pytest.mark.accelerator
def test_time_tables_follow_graph_lru_eviction(
    flow_dispatch_dit: DiT, device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
        max_graphs=1,
    )
    generator = torch.Generator(device=device).manual_seed(97)
    stream = torch.cuda.Stream(device=device)
    table_refs = []
    with torch.inference_mode():
        for batch in (1, 2, 3):
            hist, noise, cond = make_inputs(device, generator, batch)
            actual = runner.run(
                hist, noise, cond, torch.ones(batch, device=device), N_TIMESTEPS
            )
            assert actual is not None and torch.equal(actual, noise)
            entry = next(iter(runner.cache.values()))
            table_refs.append(weakref.ref(entry.time_table))
            del entry
            gc.collect()
            assert table_refs[-1]() is not None
            assert all(ref() is None for ref in table_refs[:-1])
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            replayed = runner.run(
                hist, noise, cond, torch.ones(batch, device=device), N_TIMESTEPS
            )
        runner.cache.clear()
        gc.collect()
        assert all(ref() is None for ref in table_refs)
    stream.synchronize()
    assert torch.equal(replayed, noise)
