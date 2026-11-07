# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="FP8 linear kernels are built for SM120",
)

# Qwen3-ASR-1.7B decoder shapes (N, K): qkv, o_proj, gate_up, down
DECODER_SHAPES = [(4096, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
# Decode (weight-only kernel) and prefill (W8A8 kernel), with partial tiles
ROW_COUNTS = [1, 7, 16, 33, 64, 65, 300, 1100]


def dequantized_reference(x, module):
    weight = module.weight.float() * module.weight_scale
    return x.float() @ weight.t()


@requires_sm120
@pytest.mark.parametrize("shape", DECODER_SHAPES)
def test_fp8_linear_matches_dequantized_weight(shape):
    from sglang_omni.models.qwen3_asr.fp8_linear import set_fp8_weight

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
        # Decode rows: X stays bf16. Prefill rows: X is FP8 too, so allow its error.
        relative_error = ((y.float() - reference).norm() / reference.norm()).item()
        if M <= 64:
            assert relative_error < 1e-2, (M, relative_error)
        else:
            assert relative_error < 5e-2, (M, relative_error)


@requires_sm120
def test_fp8_linear_keeps_leading_dims_and_bias():
    from sglang_omni.models.qwen3_asr.fp8_linear import set_fp8_weight

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


@requires_sm120
def test_fp8_linear_empty_batch():
    from sglang_omni.models.qwen3_asr.fp8_linear import set_fp8_weight

    module = nn.Module()
    set_fp8_weight(module, torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16))
    y = module.quant_method.apply(
        module, torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16)
    )
    assert y.shape == (0, 2048)
