# SPDX-License-Identifier: Apache-2.0
"""CPU flow execution policy, numerical acceptance, and immutable fallback behavior."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.compiled_dit import DIT_RECOMPILE_LIMIT
from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.flow_graph import (
    FlowGraphRunner,
    flow_t_span,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.flow_time_cache import (
    FlowPrecisionGate,
    TimeCacheUnavailable,
    TimeGemmMode,
    TimeGemmTable,
    finite_byte_equal,
)
from sglang_omni.models.firered_tts3.weights import PackedQkvModel
from tests.unit_test.fixtures.firered_tts3 import (
    HISTORY_LENGTH,
    N_TIMESTEPS,
    PATCH_SIZE,
    tiny_cpu_dit,
)

pytestmark = pytest.mark.usefixtures("disable_flow_optimization_environment")


def test_flow_mode_validation() -> None:
    dit = tiny_cpu_dit()
    with pytest.raises(ValueError, match="unknown flow mode"):
        prepare_flow_dit(dit, "compile-bf16")
    with pytest.raises(ValueError, match="unknown flow mode"):
        FlowGraphRunner(
            dit,
            flow_t_span(N_TIMESTEPS, torch.device("cpu")),
            n_timesteps=N_TIMESTEPS,
            patch_size=PATCH_SIZE,
            history_length=HISTORY_LENGTH,
            mode="fp16",
        )
    with pytest.raises(ValueError, match="flow_mode"):
        pipeline = FireRedTTS3PipelineConfig(model_path="model")
        FireRedTTS3EngineBuilder(
            max_running_requests=pipeline.generation_admission_defaults()[
                "max_running_requests"
            ],
            **{**pipeline.stage_factory_kwargs("talker"), "flow_mode": "fast"},
        )


def test_prepare_flow_dit_bf16_deepcopy_isolation() -> None:
    source = tiny_cpu_dit()
    source_weight = next(source.parameters())
    original = source_weight.detach().clone()
    run_dit, fallback_dit = prepare_flow_dit(source, "bf16")
    assert run_dit is fallback_dit and run_dit is not source
    assert next(run_dit.parameters()).dtype == torch.bfloat16
    # The caller's fp32 DiT (the eager reference path) is never mutated.
    assert source_weight.dtype == torch.float32
    assert torch.equal(source_weight.detach(), original)
    with torch.no_grad():
        source_weight.add_(1.0)
    assert not torch.equal(next(run_dit.parameters()).float(), source_weight)


@pytest.mark.parametrize("inherited_limit", [2, 96])
@pytest.mark.parametrize("should_raise", [False, True])
def test_explicit_time_cache_protects_original_compiled_call(
    monkeypatch: pytest.MonkeyPatch, inherited_limit: int, should_raise: bool
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    config = (
        torch._dynamo.config
    )  # noqa: SLF001  # leading-underscore: PyTorch compiler API
    observed: list[tuple[int, bool, bool]] = []

    def config_state() -> tuple[int, bool, bool]:
        return (
            config.recompile_limit,
            config.fail_on_recompile_limit_hit,
            config.suppress_errors,
        )

    class CompiledProbe(torch.nn.Module):
        def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            observed.append(config_state())
            if should_raise:
                raise RuntimeError("compiled probe failure")
            else:
                return x + t

    runner = FlowGraphRunner(
        CompiledProbe(),
        torch.tensor([0.0, 1.0]),
        n_timesteps=1,
        patch_size=1,
        history_length=0,
        mode="compile_fp32",
        dit_fallback=torch.nn.Identity(),
        precompute_time_conditions=True,
    )
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t = torch.ones_like(x)
    initial = config_state()
    with config.patch(
        recompile_limit=inherited_limit,
        fail_on_recompile_limit_hit=False,
        suppress_errors=True,
    ):
        if should_raise:
            with pytest.raises(RuntimeError, match="compiled probe failure"):
                runner.dit(x=x, t=t)
        else:
            assert torch.equal(runner.dit(x=x, t=t), x + t)
        assert observed == [(max(DIT_RECOMPILE_LIMIT, inherited_limit), True, False)]
        assert config_state() == (inherited_limit, False, True)
    assert config_state() == initial


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
@pytest.mark.parametrize("switch", ["time", "conv"])
def test_flow_caches_require_compiled_execution(
    monkeypatch: pytest.MonkeyPatch, mode: str, switch: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1" if switch == "time" else "0")
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT",
        "conv2d_channels_last" if switch == "conv" else "conv1d",
    )
    with pytest.raises(ValueError, match="require a compile flow mode"):
        prepare_flow_dit(tiny_cpu_dit(), mode)


@pytest.mark.parametrize(
    "time_switch,conv_switch,message",
    [
        ("2", "conv1d", "FIRERED_TTS3_PRECOMPUTE_TIME"),
        ("0", "conv2d", "FIREREDTTS3_CONV_LAYOUT"),
    ],
)
def test_invalid_flow_cache_switches_fail(
    monkeypatch: pytest.MonkeyPatch, time_switch: str, conv_switch: str, message: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", time_switch)
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", conv_switch)
    with pytest.raises(ValueError, match=message):
        prepare_flow_dit(tiny_cpu_dit(), "compile_bf16")


def test_private_model_keeps_source_and_forward_unchanged() -> None:
    source = DiT(
        in_channels=20,
        out_channels=4,
        hidden_size=32,
        num_heads=4,
        mlp_ratio=3,
        depth=2,
    ).eval()
    values = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    random_state = torch.get_rng_state().clone()
    packed = PackedQkvModel(source)
    assert torch.equal(torch.get_rng_state(), random_state)
    assert packed.source is source and packed.model is not source
    assert all(
        torch.equal(source.state_dict()[name], tensor)
        for name, tensor in values.items()
    )
    x, time = torch.randn(2, 12, 20), torch.ones(2, 1, 1)
    with torch.inference_mode():
        expected = source(x=x, t=time)
        actual = packed.model(x=x, t=time)
        unchanged = source(x=x, t=time)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(unchanged, expected)
    packed.validate_weights()


@pytest.mark.parametrize("operator", ["conv1d", "convolution_3d", "convolution_4d"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv_weight_cache_dispatch_contract(operator: str, with_bias: bool) -> None:
    device = torch.device("cpu")
    source = torch.nn.Conv1d(4, 4, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.zero_()
        source.bias.fill_(3.0)
        for channel in range(4):
            source.weight[channel, channel, 1] = 0.5
            source.weight[channel, (channel + 1) % 4, 0] = 0.25
    inputs = ((torch.arange(96, device=device) % 8) - 4).reshape(2, 4, 12)
    inputs = inputs.to(dtype=torch.bfloat16)
    cache = Conv1dWeightCache(source)
    bias = source.bias if with_bias else None

    def call(weight: torch.Tensor) -> torch.Tensor:
        if operator == "conv1d":
            return torch.ops.aten.conv1d.default(inputs, weight, bias, padding=[1])
        elif operator == "convolution_3d":
            return torch.ops.aten.convolution.default(
                inputs, weight, bias, [1], [1], [1], False, [0], 1
            )
        else:
            return torch.ops.aten.convolution.default(
                inputs.unsqueeze(2),
                weight.unsqueeze(2),
                bias,
                [1, 1],
                [0, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            )

    with torch.inference_mode():
        want = call(source.weight)
        with cache:
            got = call(source.weight)
        assert cache.hits == 1 and cache.misses == 0
        unrelated_weight = source.weight.clone()
        with cache:
            unhandled = call(unrelated_weight)
    assert got.shape == want.shape and got.stride() == want.stride()
    assert got.dtype == want.dtype == torch.bfloat16
    assert torch.equal(got, want)
    assert torch.equal(unhandled, want)
    assert cache.hits == 1 and cache.misses == 1
    cache.validate_weights()


def test_time_gemm_cache_copies_snapshot_and_rejects_wrong_signature() -> None:
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias, output = torch.ones(4), torch.empty(2, 4)
    weights = {weight.data_ptr(): "time"}
    collector = TimeGemmMode(weights, phase="collect", n_timesteps=1)
    with collector, collector.step(0):
        torch.addmm(bias, inputs, weight, out=output)
    expected = output.clone()
    table = TimeGemmTable(
        model_id=id(weight), n_timesteps=1, steps=(collector.collected[0],)
    )
    cached = TimeGemmMode(weights, phase="cached", n_timesteps=1, table=table)
    output.fill_(-1)
    with cached, cached.step(0):
        returned = torch.addmm(bias, inputs, weight, out=output)
    assert returned is output
    assert finite_byte_equal(output, expected)
    changed_stride = inputs.t().contiguous().t()
    with pytest.raises(TimeCacheUnavailable, match="signature changed"):
        with cached, cached.step(0):
            torch.addmm(bias, changed_stride, weight, out=output)


@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_flow_precision_explicit_exact_checks_remain_byte_exact(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    gate = (
        FlowPrecisionGate()
        if source == "constructor"
        else FlowPrecisionGate.from_environment()
    )
    reference = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    exact = gate.compare(reference.t().contiguous().t(), reference)
    assert exact.accepted and exact.byte_exact
    assert not gate.compare(torch.tensor([0.0]), torch.tensor([-0.0])).accepted
    changed = torch.nextafter(reference, torch.full_like(reference, float("inf")))
    assert not gate.compare(changed, reference).accepted


def test_flow_precision_default_budget_accepts_only_small_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_FLOW_ATOL", raising=False)
    monkeypatch.delenv("FIRERED_TTS3_FLOW_RTOL", raising=False)
    gate = FlowPrecisionGate.from_environment()
    reference = torch.tensor([[0.0, 1.0]])
    close = torch.tensor([[0.0005, 1.005]])
    assert gate.compare(close, reference).accepted
    assert not FlowPrecisionGate().compare(close, reference).accepted
    assert not gate.compare(torch.tensor([[0.002, 1.0]]), reference).accepted
    assert not gate.compare(torch.tensor([[0.0, 1.02]]), reference).accepted


@pytest.mark.parametrize(
    "variable", ["FIRERED_TTS3_FLOW_ATOL", "FIRERED_TTS3_FLOW_RTOL"]
)
def test_flow_precision_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    for value in ("-1", "nan", "inf", "-inf", "invalid"):
        monkeypatch.setenv(variable, value)
        with pytest.raises(ValueError, match=variable):
            FlowPrecisionGate.from_environment()


def test_flow_precision_rejects_nonfinite_values() -> None:
    gate = FlowPrecisionGate(atol=100.0, rtol=100.0)
    for value in (float("nan"), float("inf"), -float("inf")):
        finite = torch.ones(2, 3)
        nonfinite = finite.clone()
        nonfinite[1, 2] = value
        for actual, expected in (
            (nonfinite, finite),
            (finite, nonfinite),
            (nonfinite, nonfinite),
        ):
            result = gate.compare(actual, expected)
            assert not result.accepted and not result.finite


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_flow_precision_rejects_incompatible_tensors(mismatch: str) -> None:
    reference = torch.ones(2, 3)
    if mismatch == "shape":
        actual = reference.flatten()
    elif mismatch == "dtype":
        actual = reference.double()
    else:
        actual = torch.empty_like(reference, device="meta")
    result = FlowPrecisionGate(atol=100.0, rtol=100.0).compare(actual, reference)
    assert not result.accepted and not result.schema_valid


@pytest.mark.parametrize(
    "reference,actual,atol,rtol,accepted",
    [
        (0.0, 0.125, 0.125, 0.0, True),
        (0.0, 0.125 + 2**-26, 0.125, 0.0, False),
        (2.0, 2.25, 0.0, 0.125, True),
        (2.0, 2.25 + 2**-22, 0.0, 0.125, False),
        (-2.0, -2.25, 0.0, 0.125, True),
        (2.0, 2.375, 0.125, 0.125, True),
        (2.0, 2.375 + 2**-22, 0.125, 0.125, False),
        (0.0, 2**-24, 0.0, 100.0, False),
        (2**-24, 2**-23, 0.0, 0.5, False),
    ],
)
def test_flow_precision_elementwise_tolerance_boundaries(
    reference: float,
    actual: float,
    atol: float,
    rtol: float,
    accepted: bool,
) -> None:
    result = FlowPrecisionGate(atol=atol, rtol=rtol).compare(
        torch.tensor([actual]), torch.tensor([reference])
    )
    assert result.accepted == accepted
    assert result.out_of_bounds == int(not accepted)


def test_flow_precision_rejects_one_outlier_in_one_batch_row() -> None:
    reference = torch.ones(8, 4, 64)
    actual = reference.clone()
    actual[-1, -1, -1] += 0.25
    result = FlowPrecisionGate(atol=0.125, rtol=0.0).compare(actual, reference)
    assert not result.accepted and result.out_of_bounds == 1
    assert result.rmse < 0.125 and result.max_abs == 0.25
