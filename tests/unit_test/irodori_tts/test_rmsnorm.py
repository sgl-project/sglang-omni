import pytest
import torch

from sglang_omni.models.irodori_tts.dit import RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_preserves_per_head_semantics(dtype):
    torch.manual_seed(0)
    x = torch.randn(2, 7, 4, 16, dtype=dtype)
    norm = RMSNorm((4, 16), eps=1e-5).to(dtype=dtype)

    x_fp32 = x.float()
    expected = (
        x_fp32
        * torch.rsqrt((x_fp32 * x_fp32).mean(dim=-1, keepdim=True) + norm.eps)
        * norm.weight
    ).to(dtype)

    torch.testing.assert_close(norm(x), expected, rtol=1e-3, atol=1e-3)
