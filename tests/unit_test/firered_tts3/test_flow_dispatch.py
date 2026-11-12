# SPDX-License-Identifier: Apache-2.0
"""Dispatch modes preserve operator behavior and select supported optimizations."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.ffn_gemm import (
    FFN_CONFIGS,
    QKV_CONFIGS,
    FfnGemmMode,
)
from sglang_omni.models.firered_tts3.flow_time_cache import TimeGemmMode
from sglang_omni.models.firered_tts3.weights import FfnWeight

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
        inputs: torch.Tensor,
        payload: dict[str, list[list[int]]] | complex | torch.device | None,
        *,
        tag: str,
    ) -> dict[str, list[list[int]]] | complex | torch.device | None:
        if tag != "reference":
            raise ValueError(f"unsupported tag: {tag}")
        else:
            pass
        assert inputs.numel() == 1
        return payload

    library.impl("roundtrip", roundtrip, "CPU")
    inputs = torch.ones(1)
    operation = torch.ops.firered_dispatch_boundary.roundtrip.default
    with pytest.raises(RuntimeError) as expected_error:
        torch.ops.aten.add.Tensor(torch.ones(2), torch.ones(3))
    ffn_mode = FfnGemmMode(torch.nn.Identity())
    ffn_mode.enabled = True
    modes = (
        Conv1dWeightCache(torch.nn.Identity()),
        ffn_mode,
        TimeGemmMode({}, phase="collect", n_timesteps=1),
    )
    payloads = (
        {"blocks": [[1, 2], [3, 4]]},
        complex(1, 2),
        torch.device("cpu"),
        None,
    )
    for payload in payloads:
        expected = operation(inputs, payload, tag="reference")
        for mode in modes:
            with mode:
                actual = operation(inputs, payload, tag="reference")
            assert actual == expected == payload
    with pytest.raises(ValueError) as expected_custom_error:
        operation(inputs, None, tag="unsupported")
    for mode in modes:
        with mode:
            with pytest.raises(RuntimeError) as actual_error:
                torch.ops.aten.add.Tensor(torch.ones(2), torch.ones(3))
            with pytest.raises(ValueError) as actual_custom_error:
                operation(inputs, None, tag="unsupported")
        assert str(actual_error.value) == str(expected_error.value)
        assert str(actual_custom_error.value) == str(expected_custom_error.value)


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


@pytest.mark.parametrize(
    "rows,argument_style,override_config",
    [
        (24, "positional", False),
        (48, "positional", False),
        (96, "positional", False),
        (192, "positional", False),
        (384, "positional", False),
        (24, "positional", True),
        (24, "keywords", True),
    ],
)
def test_ffn_dispatch_uses_real_addmm_inputs(
    rows: int,
    argument_style: Literal["positional", "keywords"],
    override_config: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(801)
        layer = torch.nn.Linear(1024, 3072, dtype=torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(802)
    assert layer.bias is not None
    inputs = torch.randn(rows, 1024, dtype=torch.bfloat16, generator=generator)
    weight = layer.weight.t()
    mode = FfnGemmMode(layer)
    mode.enabled = True
    if rows in FFN_CONFIGS:
        mode.weights[weight.data_ptr()] = FfnWeight.from_linear("linear", layer)
        configs = FFN_CONFIGS
    else:
        mode.packed_weights[weight.data_ptr()] = FfnWeight.from_linear("linear", layer)
        configs = QKV_CONFIGS
    if override_config:
        launch_config = (32, 32, 128, 4, 2)
        monkeypatch.setitem(configs, rows, launch_config)
    else:
        launch_config = configs[rows]
    selected_configs: list[tuple[int, int, int, int, int]] = []

    def cpu_gemm(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        config: tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        # note (kermit): The CPU reference does not use GPU launch geometry.
        selected_configs.append(config)
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
    assert len(selected_configs) == 1 and selected_configs[0] is launch_config
    assert type(selected_configs[0]) is tuple


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
