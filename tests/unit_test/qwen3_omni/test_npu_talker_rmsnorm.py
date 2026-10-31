# SPDX-License-Identifier: Apache-2.0
"""Analytic NPU RMSNorm cases that distinguish casting before weight multiplication."""

from __future__ import annotations

import math
from typing import Literal

import pytest
import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.utils.offloader import OffloaderV1
from torch import nn

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalkerNpuRMSNorm

INPUT_VALUE = 2**-10
EPSILON = 2**-20

pytestmark = pytest.mark.accelerator


@pytest.fixture
def npu_device() -> torch.device:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    else:
        torch.npu.set_device(0)
        return torch.device("npu:0")


@pytest.mark.parametrize("hidden_width", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "mode", ["plain", "residual", "residual_post", "post_only", "quant_linear"]
)
def test_talker_preserves_the_cast_contract_for_supported_npu_inputs(
    npu_device: torch.device,
    hidden_width: int,
    dtype: torch.dtype,
    mode: Literal["plain", "residual", "residual_post", "post_only", "quant_linear"],
) -> None:
    reference_input = torch.full((2, hidden_width), INPUT_VALUE, dtype=torch.float32)
    variance = reference_input.square().mean(-1, keepdim=True)
    assert torch.equal(variance, torch.full((2, 1), EPSILON))
    normalized = reference_input * torch.rsqrt(variance + EPSILON)
    weight = (1 + (torch.arange(hidden_width, dtype=torch.float64) % 129) / 128).to(
        dtype
    )
    expected = normalized.to(dtype) * weight
    analytic_normalized = torch.full(
        (2, hidden_width), 1 / math.sqrt(2), dtype=torch.float64
    )
    analytic_expected = (analytic_normalized.to(dtype).double() * weight.double()).to(
        dtype
    )
    assert torch.equal(expected, analytic_expected)
    mul_before_cast = (normalized * weight.float()).to(dtype)

    if mode == "residual":
        cpu_input = torch.full((2, hidden_width), INPUT_VALUE / 2, dtype=dtype)
        cpu_residual = cpu_input.clone()
        cpu_post = None
    elif mode == "residual_post":
        cpu_input = torch.full((2, hidden_width), INPUT_VALUE / 4, dtype=dtype)
        cpu_residual = cpu_input.clone()
        cpu_post = torch.full_like(cpu_input, INPUT_VALUE / 2)
    else:
        cpu_input = reference_input.to(dtype)
        cpu_residual = None
        cpu_post = (
            torch.full_like(cpu_input, INPUT_VALUE / 2) if mode == "post_only" else None
        )
    quant_linear = nn.Identity() if mode == "quant_linear" else None

    product = Qwen3OmniTalkerNpuRMSNorm(hidden_width, eps=EPSILON).to(
        npu_device, dtype=dtype
    )
    old = RMSNorm(hidden_width, eps=EPSILON, cast_x_before_out_mul=True).to(
        npu_device, dtype=dtype
    )
    with torch.no_grad():
        product.weight.copy_(weight.to(npu_device))
        old.weight.copy_(weight.to(npu_device))
    device_input = cpu_input.to(npu_device)
    device_residual = cpu_residual.to(npu_device) if cpu_residual is not None else None
    device_post = cpu_post.to(npu_device) if cpu_post is not None else None
    with torch.inference_mode():
        output = product(device_input, device_residual, device_post, quant_linear)
        old_output = old(
            cpu_input.to(npu_device).clone(),
            cpu_residual.to(npu_device).clone() if cpu_residual is not None else None,
            cpu_post.to(npu_device).clone() if cpu_post is not None else None,
            quant_linear,
        )
    if cpu_residual is None:
        assert isinstance(output, torch.Tensor)
        assert isinstance(old_output, torch.Tensor)
        actual = output.cpu()
        legacy = old_output.cpu()
    else:
        assert isinstance(output, tuple) and len(output) == 2
        assert isinstance(old_output, tuple) and len(old_output) == 2
        actual = output[0].cpu()
        legacy = old_output[0].cpu()
        assert torch.equal(output[1].cpu(), torch.full_like(cpu_input, INPUT_VALUE))
        assert torch.equal(output[1], old_output[1])
        assert torch.equal(device_residual.cpu(), cpu_residual)
    assert torch.equal(device_input.cpu(), cpu_input)
    if dtype == torch.float32:
        assert torch.equal(actual, legacy)
        torch.testing.assert_close(
            actual, expected, rtol=4 * torch.finfo(dtype).eps, atol=0
        )
    elif cpu_residual is None:
        assert not torch.equal(expected, mul_before_cast)
        assert torch.equal(actual, expected)
    else:
        assert torch.equal(actual, legacy)
        assert torch.equal(actual, expected)
    assert "weight" in product.state_dict()
    assert "unit_weight" not in product.state_dict()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offload", [False, True])
def test_talker_norm_preserves_cast_order_during_npu_graph_replay(
    npu_device: torch.device, dtype: torch.dtype, offload: bool
) -> None:
    hidden_width = 128
    product = Qwen3OmniTalkerNpuRMSNorm(hidden_width, eps=EPSILON).to(
        device=npu_device, dtype=dtype
    )
    weight = (1 + (torch.arange(hidden_width, dtype=torch.float64) % 129) / 128).to(
        dtype
    )
    with torch.no_grad():
        product.weight.copy_(weight.to(npu_device))
    if offload:
        product = OffloaderV1(cpu_offload_max_bytes=1024**2).maybe_offload_to_cpu(
            product
        )
        assert product.weight.device.type == "cpu"
    else:
        assert product.weight.device == npu_device
    assert product.unit_weight.device == npu_device
    inputs = torch.full((2, hidden_width), INPUT_VALUE, device=npu_device, dtype=dtype)
    expected = torch.full((2, hidden_width), 1 / math.sqrt(2), dtype=dtype) * weight
    with torch.inference_mode():
        for _ in range(3):
            torch.testing.assert_close(product(inputs).cpu(), expected, rtol=0, atol=0)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = product(inputs)
        for _ in range(3):
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(output.cpu(), expected)
    assert product.weight.device.type == ("cpu" if offload else "npu")
    assert "unit_weight" not in product.state_dict()
