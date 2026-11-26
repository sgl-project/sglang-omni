# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the XPU-only quantization dispatch the MiniCPM-o AWQ recipe relies on."""

from __future__ import annotations

import copy

import pytest
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.layers.quantization.awq.schemes.awq_linear import AWQXPULinearScheme
from sglang.srt.utils import is_xpu

pytestmark = pytest.mark.skipif(
    not is_xpu(), reason="SGLang registers the XPU quantization configs only on XPU"
)

MINICPM_O_AWQ = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": [
        "vpm",
        "apm",
        "resampler",
        "audio_projection_layer",
        "audio_avg_pooler",
        "tts",
        "lm_head",
    ],
}


@pytest.mark.parametrize(
    "method, expected",
    [("awq", "AWQXPUConfig"), ("gptq", "GPTQXPUConfig")],
)
def test_int4_methods_resolve_to_the_xpu_config(method: str, expected: str) -> None:
    assert get_quantization_config(method).__name__ == expected


def test_minicpm_o_thinker_linear_takes_the_xpu_int4_scheme() -> None:
    config = get_quantization_config("awq").from_config(copy.deepcopy(MINICPM_O_AWQ))
    layer = LinearBase.__new__(LinearBase)

    config.get_quant_method(layer, prefix="model.layers.0.self_attn.qkv_proj")

    assert config.weight_bits == 4
    assert config.group_size == 128
    assert isinstance(layer.scheme, AWQXPULinearScheme)


def test_awq_moe_is_refused_rather_than_mis_applied() -> None:
    config = get_quantization_config("awq").from_config(copy.deepcopy(MINICPM_O_AWQ))
    layer = FusedMoE.__new__(FusedMoE)

    with pytest.raises(NotImplementedError, match="AWQ MoE is not yet supported"):
        config.get_quant_method(layer, prefix="model.layers.0.mlp.experts")
