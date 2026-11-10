# SPDX-License-Identifier: Apache-2.0
"""Tests for the quantization methods SGLang dispatches on Intel XPU.

`openbmb/MiniCPM-o-4_5-awq` serves on a single Arc Pro B60 today, and nothing
else pins the dispatch it relies on: SGLang registers `AWQXPUConfig` and
`GPTQXPUConfig` only when `is_xpu()`, so a change to that registry would take
the recipe out without failing any test.

The registry is built at import time from `is_xpu()`, so on any other host
`awq` maps to the generic `AWQConfig` and there is nothing to pin. The module
skips there and runs in the XPU CI lane.
"""

from __future__ import annotations

import pytest
from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.utils import is_xpu

pytestmark = pytest.mark.skipif(
    not is_xpu(), reason="SGLang registers the XPU quantization configs only on XPU"
)

# openbmb/MiniCPM-o-4_5-awq, which quantizes only the `llm.*` backbone; the
# vision tower, audio tower, resampler and talker stay bf16.
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


def test_minicpm_o_awq_config_is_accepted() -> None:
    config = get_quantization_config("awq").from_config(MINICPM_O_AWQ)

    assert config.weight_bits == 4
    assert config.group_size == 128
    assert config.zero_point
    # The exclusion list is what keeps the bf16 submodules off the AWQ path.
    assert "tts" in config.modules_to_not_convert
    assert "lm_head" in config.modules_to_not_convert


def test_awq_moe_is_refused_rather_than_mis_applied() -> None:
    """AWQ on XPU is dense-only. MiniCPM-o's backbone is dense, so it is in
    scope; an MoE checkpoint has to be told, not silently mis-built."""
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    config = get_quantization_config("awq").from_config(MINICPM_O_AWQ)
    layer = FusedMoE.__new__(FusedMoE)

    with pytest.raises(NotImplementedError, match="AWQ MoE is not yet supported"):
        config.get_quant_method(layer, prefix="llm.model.layers.0.mlp.experts")
