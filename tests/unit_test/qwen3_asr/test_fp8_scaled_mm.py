# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn

from sglang_omni.platforms import current_platform

if (
    not torch.cuda.is_available()
    or not current_platform.is_cuda()
    or torch.cuda.get_device_capability()[0] != 9
):
    pytest.skip("scaled_mm FP8 linears need NVIDIA SM90", allow_module_level=True)
else:
    pass

from sglang_omni.models.qwen3_asr.fp8_scaled_mm import (  # noqa: E402
    is_scaled_mm_supported,
    scaled_mm_w8a8,
    set_scaled_mm_fp8_weight,
)

pytestmark = pytest.mark.accelerator

# (N, K): lm_head, gate_up_proj, and the shapes _scaled_mm must decline.
SHAPES = [
    (151936, 2048),
    (12288, 2048),
    (2048, 2048),
    (40, 2048),
    (2048, 2050),
]
ROW_COUNTS = [1, 7, 16, 33, 64, 65, 300, 1100]
# Worst measured max-error ratio over these cases: 0.032 (weight + activation
# quantization against the dequantized-weight reference), the SM120 W8A8 level.
W8A8_TOLERANCE = 5e-2


def dequantized_reference(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    weight = module.weight.float() * module.weight_scale
    return x.float() @ weight.t()


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_scaled_mm_matches_dequantized_weight(shape: tuple[int, int]) -> None:
    """Every output column is written, also when N does not fill an MMA tile."""
    N, K = shape
    torch.manual_seed(0)
    module = nn.Module()
    set_scaled_mm_fp8_weight(
        module, torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    )

    for M in ROW_COUNTS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        y = module.quant_method.apply(module, x)
        reference = dequantized_reference(x, module)
        error_ratio = (y.float() - reference).abs().max() / reference.abs().max()
        assert error_ratio < W8A8_TOLERANCE, f"M={M} error_ratio={error_ratio}"
        assert y.shape == (M, N)
        assert y.dtype == torch.bfloat16


def test_scaled_mm_support_gate() -> None:
    """bf16 with 16-aligned N and K converts; other dtypes and odd K decline."""
    good = torch.randn(2048, 2048, dtype=torch.bfloat16)
    odd_k = torch.randn(2048, 2050, dtype=torch.bfloat16)
    odd_n = torch.randn(40, 2048, dtype=torch.bfloat16)
    fp32 = torch.randn(2048, 2048, dtype=torch.float32)
    assert is_scaled_mm_supported(good)
    assert not is_scaled_mm_supported(odd_k)
    assert not is_scaled_mm_supported(odd_n)
    assert not is_scaled_mm_supported(fp32)


def test_scaled_mm_empty_rows() -> None:
    """M=0 returns an empty bf16 result without touching cuBLASLt."""
    module = nn.Module()
    set_scaled_mm_fp8_weight(
        module, torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    )
    y = scaled_mm_w8a8(
        torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16),
        module.weight,
        module.weight_scale_col,
    )
    assert y.shape == (0, 64)
    assert y.dtype == torch.bfloat16


def test_scaled_mm_garbage_pad_rows_leave_real_rows_intact() -> None:
    """Under CUDA graph replay, bucket padding holds garbage; rowwise scales
    must keep one garbage row from poisoning every real row."""
    torch.manual_seed(1)
    module = nn.Module()
    set_scaled_mm_fp8_weight(
        module, torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    x = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)
    y_clean = module.quant_method.apply(module, x)
    padded = torch.cat(
        [x, torch.full((5, 2048), 3.0e4, device="cuda", dtype=torch.bfloat16)], 0
    )
    y_padded = module.quant_method.apply(module, padded)
    assert torch.equal(y_padded[:3], y_clean)
