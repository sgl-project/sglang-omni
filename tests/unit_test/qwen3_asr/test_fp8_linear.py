# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn

from sglang_omni.platforms import current_platform

if (
    not torch.cuda.is_available()
    or not current_platform.is_cuda()
    or torch.cuda.get_device_capability()[0] != 12
):
    pytest.skip("FP8 linear kernels need NVIDIA SM120", allow_module_level=True)
else:
    pass

# note (emre570): the kernel modules import CuTe DSL, so they load only after the platform skip.
from sglang_omni.models.qwen3_asr.fp8_linear import (  # noqa: E402
    is_fp8_supported,
    set_fp8_weight,
)

pytestmark = pytest.mark.accelerator

# (N, K): the four Qwen3-ASR-1.7B decoder linears, the TP4 lm_head shard, and N smaller than one tile
SHAPES = [
    (4096, 2048),
    (2048, 2048),
    (12288, 2048),
    (2048, 6144),
    (37984, 2048),
    (40, 2048),
]
# Weight-only rows (<= 64) and W8A8 rows (> 64), with partial M tiles
ROW_COUNTS = [1, 7, 16, 33, 64, 65, 300, 1100]
# Worst measured max-error ratio over these cases: 0.0034 weight-only, 0.0295 W8A8
WEIGHT_ONLY_TOLERANCE = 1e-2
W8A8_TOLERANCE = 5e-2


def dequantized_reference(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    weight = module.weight.float() * module.weight_scale
    return x.float() @ weight.t()


@pytest.mark.parametrize("shape", SHAPES)
def test_fp8_linear_matches_dequantized_weight(shape: tuple[int, int]) -> None:
    """Every output column is written, also when N does not fill the last N tile."""
    N, K = shape
    torch.manual_seed(0)
    module = nn.Module()
    set_fp8_weight(
        module, torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    )

    for M in ROW_COUNTS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        y = module.quant_method.apply(module, x)
        reference = dequantized_reference(x, module)
        assert y.shape == (M, N)
        assert y.dtype == torch.bfloat16
        error_ratio = (
            (y.float() - reference).abs().amax() / reference.abs().amax()
        ).item()
        if M <= 64:
            assert error_ratio < WEIGHT_ONLY_TOLERANCE, (M, error_ratio)
        else:
            assert error_ratio < W8A8_TOLERANCE, (M, error_ratio)


@pytest.mark.parametrize(
    "dtype, shape, expected",
    [
        (torch.bfloat16, (2048, 2048), True),
        (torch.float16, (2048, 2048), False),
        (torch.bfloat16, (2048, 2080), False),
    ],
)
def test_is_fp8_supported(
    dtype: torch.dtype, shape: tuple[int, int], expected: bool
) -> None:
    """FP16 weights and K that does not fill the K tiles keep the original linear."""
    weight = torch.empty(shape, device="cuda", dtype=dtype)
    assert is_fp8_supported(weight) == expected


def test_fp8_linear_keeps_leading_dims_and_bias() -> None:
    N, K = 2048, 2048
    module = nn.Module()
    set_fp8_weight(
        module, torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    x = torch.randn(2, 5, K, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    y = module.quant_method.apply(module, x, bias)
    y_no_bias = module.quant_method.apply(module, x)

    assert y.shape == (2, 5, N)
    torch.testing.assert_close(y, y_no_bias + bias, rtol=1e-2, atol=1e-2)


def test_fp8_linear_empty_batch() -> None:
    module = nn.Module()
    set_fp8_weight(module, torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16))
    y = module.quant_method.apply(
        module, torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16)
    )
    assert y.shape == (0, 2048)
