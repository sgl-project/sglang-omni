# SPDX-License-Identifier: Apache-2.0
"""Optimized flow gates, fallback paths, and graph resource lifetimes."""

from __future__ import annotations

import collections
import gc
import weakref
from typing import Literal
from unittest.mock import create_autospec

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import sglang_omni.models.firered_tts3.flow_graph_capture as flow_graph_module
from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.ffn_gemm import FfnGemmMode
from sglang_omni.models.firered_tts3.flow_execution import (
    euler_flow_loop,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.flow_time_cache import (
    TimeConditionCache,
    TimeGemmMode,
)
from sglang_omni.models.firered_tts3.weights import FfnWeight
from tests.unit_test.fixtures.firered_tts3 import (
    N_TIMESTEPS,
    PATCH_SIZE,
    make_inputs,
    tiny_runner,
)

pytestmark = pytest.mark.usefixtures("disable_flow_optimization_environment")


@pytest.mark.parametrize("mode_name", ["convolution", "ffn", "time"])
def test_dispatch_modes_preserve_unrelated_operator_returns(
    mode_name: Literal["convolution", "ffn", "time"],
) -> None:
    source = torch.nn.Identity()
    if mode_name == "convolution":
        mode = Conv1dWeightCache(source)
    elif mode_name == "ffn":
        mode = FfnGemmMode(source)
        mode.enabled = True
    else:
        mode = TimeGemmMode({}, phase="collect", n_timesteps=1)
    values = torch.tensor([3, 1, 2, 4])
    with mode:
        chunks = torch.ops.aten.split.Tensor(values, 2)
        maximum, maximum_index = torch.ops.aten.max.dim(values, 0)
        nonzero = torch.ops.aten.is_nonzero.default(values[0])
    assert len(chunks) == 2
    assert torch.equal(chunks[0], torch.tensor([3, 1]))
    assert torch.equal(chunks[1], torch.tensor([2, 4]))
    assert maximum.item() == 4 and maximum_index.item() == 3
    assert nonzero is True


def test_dispatch_modes_preserve_custom_operator_values_and_errors() -> None:
    library = torch.library.Library("firered_dispatch_boundary", "DEF")
    library.define("roundtrip(Tensor input, Any payload, *, str tag) -> Any")

    def roundtrip(
        inputs: torch.Tensor, payload: dict[str, list[list[int]]], *, tag: str
    ) -> dict[str, list[list[int]]]:
        assert inputs.numel() == 1 and tag == "reference"
        return payload

    library.impl("roundtrip", roundtrip, "CPU")
    inputs = torch.ones(1)
    payload = {"blocks": [[1, 2], [3, 4]]}
    operation = torch.ops.firered_dispatch_boundary.roundtrip.default
    expected = operation(inputs, payload, tag="reference")
    with pytest.raises(RuntimeError) as expected_error:
        torch.ops.aten.add.Tensor(torch.ones(2), torch.ones(3))
    ffn_mode = FfnGemmMode(torch.nn.Identity())
    ffn_mode.enabled = True
    modes = (
        Conv1dWeightCache(torch.nn.Identity()),
        ffn_mode,
        TimeGemmMode({}, phase="collect", n_timesteps=1),
    )
    for mode in modes:
        with mode:
            actual = operation(inputs, payload, tag="reference")
            with pytest.raises(RuntimeError) as actual_error:
                torch.ops.aten.add.Tensor(torch.ones(2), torch.ones(3))
        assert actual == expected == payload
        assert str(actual_error.value) == str(expected_error.value)


def test_ffn_dispatch_preserves_addmm_keyword_fallback() -> None:
    layer = torch.nn.Linear(3, 4, bias=True)
    assert layer.bias is not None
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weight = layer.weight.t()
    expected = torch.addmm(layer.bias, inputs, weight, alpha=2)
    output = torch.empty_like(expected)
    mode = FfnGemmMode(layer)
    mode.enabled = True
    mode.weights[weight.data_ptr()] = FfnWeight.from_linear("linear", layer)
    with torch.inference_mode(), mode:
        result = torch.ops.aten.addmm.out(
            self=layer.bias, mat1=inputs, mat2=weight, alpha=2, out=output
        )
    assert result is output
    torch.testing.assert_close(output, expected)
    assert mode.hits == 0 and mode.misses == 1


@pytest.mark.parametrize("argument_style", ["positional", "keywords"])
def test_ffn_dispatch_uses_real_addmm_inputs(
    argument_style: Literal["positional", "keywords"],
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(801)
        layer = torch.nn.Linear(1024, 3072, dtype=torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(802)
    assert layer.bias is not None
    inputs = torch.randn(24, 1024, dtype=torch.bfloat16, generator=generator)
    weight = layer.weight.t()
    mode = FfnGemmMode(layer)
    mode.enabled = True
    mode.weights[weight.data_ptr()] = FfnWeight.from_linear("linear", layer)

    def cpu_gemm(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        config: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        # note (kermit): The CPU reference does not use GPU launch geometry.
        return torch.addmm(bias, inputs, weight, out=out)

    mode.gemm_out = cpu_gemm
    with torch.inference_mode():
        expected = torch.addmm(layer.bias, inputs, weight)
        output = torch.empty_like(expected)
        if argument_style == "positional":
            result = mode.__torch_dispatch__(
                torch.ops.aten.addmm.out,
                (),
                (layer.bias, inputs, weight),
                {"out": output},
            )
        else:
            result = mode.__torch_dispatch__(
                torch.ops.aten.addmm.out,
                (),
                (),
                {"self": layer.bias, "mat1": inputs, "mat2": weight, "out": output},
            )
    assert result is output
    assert torch.equal(output, expected)
    assert mode.hits == 1 and mode.misses == 0


@pytest.mark.parametrize("operator", ["conv1d", "convolution"])
def test_convolution_dispatch_accepts_keyword_arguments(
    operator: Literal["conv1d", "convolution"],
) -> None:
    source = torch.nn.Conv1d(2, 2, 3, padding=1).eval()
    inputs = torch.arange(24, dtype=torch.float32).reshape(1, 2, 12)
    cache = Conv1dWeightCache(source)
    with torch.inference_mode():
        if operator == "conv1d":
            operation = torch.ops.aten.conv1d.default
            expected = operation(inputs, source.weight, source.bias, [1], [1], [1], 1)
            actual = cache.__torch_dispatch__(
                operation,
                (),
                (inputs,),
                {"weight": source.weight, "bias": source.bias, "padding": [1]},
            )
        else:
            operation = torch.ops.aten.convolution.default
            expected = operation(
                inputs, source.weight, source.bias, [1], [1], [1], False, [0], 1
            )
            actual = cache.__torch_dispatch__(
                operation,
                (),
                (),
                {
                    "input": inputs,
                    "weight": source.weight,
                    "bias": source.bias,
                    "stride": [1],
                    "padding": [1],
                    "dilation": [1],
                    "transposed": False,
                    "output_padding": [0],
                    "groups": 1,
                },
            )
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
    assert cache.hits == 1 and cache.misses == 0


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
    corrupted_stage: Literal["conv", "observed", "time"],
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
        flow_state: torch.Tensor,
        conditioning: torch.Tensor,
        t_span: torch.Tensor,
        guidance_scales: torch.Tensor,
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
            flow_state,
            conditioning,
            t_span,
            guidance_scales,
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
    history_latents, initial_noise, conditioning = make_inputs(device, generator, 1)
    guidance_scales = torch.full((1, 1, 1), 1.2, device=device)
    with torch.inference_mode():
        got = runner.run(
            history_latents, initial_noise, conditioning, guidance_scales, N_TIMESTEPS
        )
        calls_after_capture = corrupted_calls
        repeated = runner.run(
            history_latents, initial_noise, conditioning, guidance_scales, N_TIMESTEPS
        )
        expected = original_loop(
            execution.original,
            torch.cat([history_latents, initial_noise], dim=1),
            conditioning,
            runner.t_span,
            guidance_scales,
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
    history_latents, initial_noise, conditioning = make_inputs(device, generator, 1)
    guidance_scales = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(
            history_latents, initial_noise, conditioning, guidance_scales, N_TIMESTEPS
        )
        expected = euler_flow_loop(
            execution.original,
            torch.cat([history_latents, initial_noise], dim=1),
            conditioning,
            runner.t_span,
            guidance_scales,
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
        repeated = runner.run(
            history_latents, initial_noise, conditioning, guidance_scales, N_TIMESTEPS
        )
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
            history_latents, initial_noise, conditioning = make_inputs(
                device, generator, batch
            )
            actual = runner.run(
                history_latents,
                initial_noise,
                conditioning,
                torch.ones(batch, device=device),
                N_TIMESTEPS,
            )
            assert actual is not None and torch.equal(actual, initial_noise)
            entry = next(iter(runner.cache.values()))
            table_refs.append(weakref.ref(entry.time_table))
            del entry
            gc.collect()
            assert table_refs[-1]() is not None
            assert all(ref() is None for ref in table_refs[:-1])
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            replayed = runner.run(
                history_latents,
                initial_noise,
                conditioning,
                torch.ones(batch, device=device),
                N_TIMESTEPS,
            )
        runner.cache.clear()
        gc.collect()
        assert all(ref() is None for ref in table_refs)
    stream.synchronize()
    assert torch.equal(replayed, initial_noise)
