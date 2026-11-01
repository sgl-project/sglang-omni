"""FP8 feed forward kernel on SM120."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.packed_dit import uses_fa4

pytestmark = pytest.mark.accelerator

HIDDEN = 1024
INNER = 2048


def make_inputs(rows: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(5)
    in_weight = torch.randn(INNER, HIDDEN, device="cuda", dtype=torch.bfloat16) / 32
    out_weight = torch.randn(HIDDEN, INNER, device="cuda", dtype=torch.float32) / 45
    out_scale = out_weight.abs().max() / 448.0
    return {
        "x": torch.randn(1, rows, HIDDEN, device="cuda", dtype=torch.bfloat16),
        "in_weight": in_weight,
        "in_bias": torch.randn(INNER, device="cuda", dtype=torch.bfloat16) / 32,
        "out_weight": out_weight.to(torch.bfloat16),
        "out_weight_fp8": (out_weight / out_scale)
        .to(torch.float8_e4m3fn)
        .t()
        .contiguous(),
        "out_bias": torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) / 32,
        "out_weight_scale": out_scale,
    }


@pytest.fixture
def fp8_module():
    if not torch.cuda.is_available() or not uses_fa4(torch.device("cuda")):
        pytest.skip("requires SM120")
    else:
        pass
    from sglang_omni.models.fun_cosyvoice3 import fp8_feed_forward

    return fp8_feed_forward


def run_op(fp8_module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    gelu_scale = torch.tensor(fp8_module.GELU_SCALE, device="cuda", dtype=torch.float32)
    return fp8_module.fp8_feed_forward(
        inputs["x"],
        inputs["in_weight"],
        inputs["in_bias"],
        inputs["out_weight"],
        inputs["out_weight_fp8"],
        inputs["out_bias"],
        inputs["out_weight_scale"],
        gelu_scale,
    )


@pytest.mark.parametrize("rows", [1, 63, 65, 840, 1300])
def test_fp8_matches_bf16_at_every_row_count(
    fp8_module, rows: int
) -> None:
    inputs = make_inputs(rows)
    with torch.inference_mode():
        hidden = F.gelu(
            F.linear(inputs["x"], inputs["in_weight"], inputs["in_bias"]),
            approximate="tanh",
        )
        expected = F.linear(hidden, inputs["out_weight"], inputs["out_bias"]).float()
        output = run_op(fp8_module, inputs).float()
    assert output.shape == expected.shape
    assert (output - expected).norm() / expected.norm() < 0.05


def test_gelu_above_the_static_limit_stays_finite(fp8_module) -> None:
    inputs = make_inputs(128)
    inputs["x"] = torch.full_like(inputs["x"], 1000.0)
    with torch.inference_mode():
        output = run_op(fp8_module, inputs)
    assert torch.isfinite(output.float()).all()
