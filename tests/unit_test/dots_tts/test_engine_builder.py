# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


def test_dots_engine_uses_shared_tts_builder() -> None:
    builder = DotsTTSEngineBuilder(optimize=True)

    assert isinstance(builder, TtsEngineBuilder)
    assert builder.optimize is True
    assert builder.generation_defaults(dtype="bfloat16")["max_running_requests"] == 16


def test_dots_engine_accepts_continuous_batching() -> None:
    DotsTTSEngineBuilder().adjust_overrides({"tp_size": 1, "max_running_requests": 16})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tp_size": 2, "max_running_requests": 16}, "does not implement TP"),
        (
            {
                "tp_size": 1,
                "max_running_requests": 16,
                "enable_torch_compile": True,
            },
            "backbone compile is disabled",
        ),
    ],
)
def test_dots_engine_rejects_unsupported_generation_modes(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTTSEngineBuilder().adjust_overrides(overrides)
