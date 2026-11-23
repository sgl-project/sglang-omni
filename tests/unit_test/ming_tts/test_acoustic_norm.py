# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming-TTS acoustic RMSNorm backend selection."""

from functools import partial

import pytest
import torch

from sglang_omni.models.ming_omni.talker.talker_module.aggregator import Aggregator
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT
from sglang_omni.models.ming_omni.talker.talker_module.execution import (
    TalkerExecutionConfig,
)
from sglang_omni.models.ming_omni.talker.talker_module.modules import (
    RMSNorm as MingRMSNorm,
)
from sglang_omni.vendor.sglang.layers import RMSNorm as SGLangRMSNorm


def test_ming_tts_acoustic_rms_norm_preserves_bf16_semantics() -> None:
    hidden_size = 8
    legacy = MingRMSNorm(hidden_size, 1e-6).to(dtype=torch.bfloat16)
    optimized = SGLangRMSNorm(
        hidden_size,
        eps=1e-6,
        cast_x_before_out_mul=True,
    ).to(dtype=torch.bfloat16)
    weight = torch.linspace(0.5, 1.5, hidden_size).to(torch.bfloat16)
    inputs = torch.linspace(-3.0, 3.0, 3 * 5 * hidden_size)
    inputs = inputs.reshape(3, 5, hidden_size).to(torch.bfloat16)

    with torch.no_grad():
        legacy.weight.copy_(weight)
        optimized.weight.copy_(weight)
        expected = legacy(inputs)
        actual = optimized(inputs)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("component", "component_kwargs"),
    [
        (Aggregator, {"llm_input_dim": 6}),
        (DiT, {"llm_cond_dim": 6, "cfg_dropout_prob": 0}),
    ],
)
def test_acoustic_components_select_norm_layer(
    component: type[Aggregator] | type[DiT],
    component_kwargs: dict[str, int | float],
) -> None:
    common_kwargs = {
        "in_channels": 4,
        "hidden_size": 8,
        "depth": 1,
        "num_heads": 2,
        "qk_norm": "rms_norm",
        **component_kwargs,
    }
    native = component(
        **common_kwargs,
        execution_config=TalkerExecutionConfig(attn_backend="torch"),
    )
    optimized = component(
        **common_kwargs,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch",
            norm_layer=partial(
                SGLangRMSNorm,
                cast_x_before_out_mul=True,
            ),
        ),
    )

    native_norms = (
        native.blocks[0].norm1,
        native.blocks[0].norm2,
        native.final_layer.norm_final,
    )
    optimized_norms = (
        optimized.blocks[0].norm1,
        optimized.blocks[0].norm2,
        optimized.final_layer.norm_final,
    )
    assert all(type(norm) is MingRMSNorm for norm in native_norms)
    assert all(type(norm) is SGLangRMSNorm for norm in optimized_norms)
    assert type(optimized.blocks[0].attn.q_norm) is MingRMSNorm
    assert type(optimized.blocks[0].attn.k_norm) is MingRMSNorm
    optimized.load_state_dict(native.state_dict(), strict=True)


@pytest.mark.parametrize("component", [Aggregator, DiT])
def test_fused_residual_norm_matches_the_unfused_block_chain(
    component: type[Aggregator] | type[DiT],
) -> None:
    torch.manual_seed(0)
    if component is DiT:
        component_kwargs = {"llm_cond_dim": 6, "cfg_dropout_prob": 0}
        inputs = (
            torch.randn(3, 4, 4),
            torch.rand(3),
            torch.randn(3, 1, 6),
            torch.randn(3, 2, 4),
        )
    else:
        component_kwargs = {"llm_input_dim": 6}
        inputs = (torch.randn(3, 4, 4),)
    norm_layer = partial(SGLangRMSNorm, cast_x_before_out_mul=True)
    unfused = component(
        in_channels=4,
        hidden_size=8,
        depth=3,
        num_heads=2,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch", norm_layer=norm_layer
        ),
        **component_kwargs,
    ).eval()
    fused = component(
        in_channels=4,
        hidden_size=8,
        depth=3,
        num_heads=2,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch", norm_layer=norm_layer, fuse_residual_norm=True
        ),
        **component_kwargs,
    ).eval()
    with torch.no_grad():
        for parameter in unfused.parameters():
            parameter.normal_(std=0.5)
    fused.load_state_dict(unfused.state_dict(), strict=True)

    with torch.no_grad():
        expected = unfused(*inputs)
        actual = fused(*inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.parametrize("component", [Aggregator, DiT])
def test_fused_residual_norm_kernel_matches_the_unfused_block_chain_in_bf16(
    monkeypatch: pytest.MonkeyPatch,
    component: type[Aggregator] | type[DiT],
) -> None:
    """The CUDA fused kernel updates x and residual in place; the block chain must
    still match the unfused math and leave the caller's inputs untouched."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    else:
        pass
    import sglang.srt.layers.layernorm as layernorm_module

    kernel_calls = 0
    real_kernel = (
        layernorm_module._jit_fused_add_rmsnorm  # noqa: leading-underscore  # sglang API
    )

    def counting_kernel(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(layernorm_module, "_jit_fused_add_rmsnorm", counting_kernel)
    torch.manual_seed(0)
    device = torch.device("cuda")
    # The JIT kernel needs a hidden size that is a multiple of 16 and at least 32.
    hidden_size, depth = 64, 3
    if component is DiT:
        component_kwargs = {"llm_cond_dim": 6, "cfg_dropout_prob": 0}
        inputs = (
            torch.randn(3, 4, 4),
            torch.rand(3),
            torch.randn(3, 1, 6),
            torch.randn(3, 2, 4),
        )
    else:
        component_kwargs = {"llm_input_dim": 6}
        inputs = (torch.randn(3, 4, 4),)
    inputs = tuple(tensor.to(device=device, dtype=torch.bfloat16) for tensor in inputs)
    norm_layer = partial(SGLangRMSNorm, cast_x_before_out_mul=True)

    def build(fuse_residual_norm: bool) -> torch.nn.Module:
        return component(
            in_channels=4,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=2,
            execution_config=TalkerExecutionConfig(
                attn_backend="torch",
                norm_layer=norm_layer,
                fuse_residual_norm=fuse_residual_norm,
            ),
            **component_kwargs,
        ).eval()

    unfused = build(fuse_residual_norm=False)
    with torch.no_grad():
        for parameter in unfused.parameters():
            parameter.normal_(std=0.5)
    unfused = unfused.to(device=device, dtype=torch.bfloat16)
    fused = build(fuse_residual_norm=True).to(device=device, dtype=torch.bfloat16)
    fused.load_state_dict(unfused.state_dict(), strict=True)
    snapshots = tuple(tensor.clone() for tensor in inputs)

    with torch.no_grad():
        expected = unfused(*inputs)
        assert kernel_calls == 0
        actual = fused(*inputs)

    # Each block fuses two adds, except the first block's norm1, plus the final layer.
    assert kernel_calls == 2 * depth
    for tensor, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    # The kernel rounds the residual sum to bf16 at a different point than the
    # native chain; two bf16 ulps at |out| ~ 4-8, while a fusion bug moves it by O(1).
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=2**-4)
