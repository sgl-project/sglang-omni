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
