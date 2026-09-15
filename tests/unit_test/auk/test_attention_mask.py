# SPDX-License-Identifier: Apache-2.0
"""Key-mask broadcasting preserves full attention without quadratic masks."""

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.auk.dit import Attention, _attention_bias


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_mask_layout_and_output_match_expanded_reference(monkeypatch, enabled, masked):
    torch.manual_seed(42)
    attention = Attention(32, heads=2, dim_head=16)
    q = torch.randn(2, 2, 7, 16)
    k, v = torch.randn(2, 2, 11, 16), torch.randn(2, 2, 11, 16)
    mask = torch.ones(2, 11, dtype=torch.bool) if masked else None
    if masked:
        mask[0, -3:] = False
        mask[1] = False
    sdpa = F.scaled_dot_product_attention
    expanded = (
        mask[:, None, None, :].expand(2, 2, 7, 11) if masked and enabled else None
    )
    expected = sdpa(q, k, v, attn_mask=expanded).transpose(1, 2).reshape(2, 7, 32)

    def record(*args, **kwargs):
        actual_mask = kwargs["attn_mask"]
        if masked and enabled:
            assert actual_mask.shape == (2, 1, 1, 11)
        else:
            assert actual_mask is None
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    torch.testing.assert_close(
        attention._attend(
            q, k, v, _attention_bias(mask, q.dtype) if masked and enabled else None
        ),
        expected,
        rtol=0,
        atol=0,
    )
