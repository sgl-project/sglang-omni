# SPDX-License-Identifier: Apache-2.0
"""Pin the XPU-only quantization dispatch the MiniCPM-o AWQ recipe relies on.

SGLang builds this registry from is_xpu() at import time, so the module skips elsewhere.
"""

from __future__ import annotations

import copy

import pytest
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.utils import is_xpu

pytestmark = pytest.mark.skipif(
    not is_xpu(), reason="SGLang registers the XPU quantization configs only on XPU"
)

# The quantization_config of openbmb/MiniCPM-o-4_5-awq. Only llm.* is quantized.
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


def test_minicpm_o_awq_config_keeps_every_exclusion() -> None:
    expected_exclusions = copy.deepcopy(MINICPM_O_AWQ["modules_to_not_convert"])

    config = get_quantization_config("awq").from_config(copy.deepcopy(MINICPM_O_AWQ))

    assert config.weight_bits == 4
    assert config.group_size == 128
    assert config.zero_point
    assert config.modules_to_not_convert == expected_exclusions


def test_awq_moe_is_refused_rather_than_mis_applied() -> None:
    """AWQ on XPU is dense-only, so an MoE layer must raise instead of being built."""
    config = get_quantization_config("awq").from_config(copy.deepcopy(MINICPM_O_AWQ))
    layer = FusedMoE.__new__(FusedMoE)

    with pytest.raises(NotImplementedError, match="AWQ MoE is not yet supported"):
        config.get_quant_method(layer, prefix="llm.model.layers.0.mlp.experts")
