# SPDX-License-Identifier: Apache-2.0
"""The fused positional conv against the module's grouped Conv1d and Mish."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.accelerator


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.double() - expected)
        / torch.linalg.vector_norm(expected)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("frames", [31, 32, 95, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_group_conv_mish_is_as_close_to_float64_as_the_module(
    frames: int, dtype: torch.dtype
) -> None:
    from sglang_omni.models.fun_cosyvoice3.causal_conv import (
        group_conv_mish,
        pack_group_conv_weight,
    )

    torch.manual_seed(0)
    conv = torch.nn.Conv1d(1024, 1024, 31, groups=16).cuda().to(dtype)
    with torch.no_grad():
        conv.bias.normal_(0, 0.5)
    x = torch.randn(2, frames, 1024, device="cuda").to(dtype)

    with torch.inference_mode():
        fused = group_conv_mish(x, pack_group_conv_weight(conv), conv.bias)
        module = F.mish(conv(x.transpose(1, 2))).transpose(1, 2)
        float64 = F.mish(
            F.conv1d(
                x.double().transpose(1, 2),
                conv.weight.double(),
                conv.bias.double(),
                groups=16,
            )
        ).transpose(1, 2)

    assert fused.shape == (2, frames - 30, 1024) and fused.dtype == dtype
    assert relative_error(fused, float64) <= relative_error(module, float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_padded_dit_runs_its_positional_convs_on_the_fused_kernel() -> None:
    cosyvoice_dit = pytest.importorskip("cosyvoice.flow.DiT.dit")
    from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT

    torch.manual_seed(0)
    dit = cosyvoice_dit.DiT(
        dim=256,
        depth=1,
        heads=4,
        dim_head=64,
        mel_dim=80,
        mu_dim=80,
        spk_dim=80,
        out_channels=80,
        static_chunk_size=50,
        num_decoding_left_chunks=-1,
    ).cuda()
    conv_pos_embed = dit.input_embed.conv_pos_embed
    float64 = type(conv_pos_embed)(256).cuda().double()
    float64.load_state_dict(conv_pos_embed.state_dict())
    for module in dit.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    PackedDiT(dit, device="cuda")
    x = torch.randn(2, 77, 256, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        fused = conv_pos_embed(x)
        module = type(conv_pos_embed).forward(conv_pos_embed, x)
        reference = float64(x.double())

    assert not torch.equal(fused, module)
    assert relative_error(fused, reference) <= relative_error(module, reference)
