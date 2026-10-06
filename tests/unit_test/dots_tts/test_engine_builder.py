# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


@pytest.mark.parametrize("padding", [None, False, True])
def test_dots_engine_uses_shared_tts_builder(padding: bool | None) -> None:
    options = {} if padding is None else {"enable_acoustic_tail_batch_padding": padding}
    builder = DotsTTSEngineBuilder(optimize=True, **options)

    assert isinstance(builder, TtsEngineBuilder)
    assert builder.optimize is True
    assert builder.enable_acoustic_tail_batch_padding is (padding is not False)
    assert builder.generation_defaults(dtype="bfloat16")["max_running_requests"] == 16


def test_dots_engine_accepts_continuous_batching() -> None:
    DotsTTSEngineBuilder().adjust_overrides({"tp_size": 1, "max_running_requests": 16})


def test_dots_engine_rejects_string_batch_padding_flag() -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        DotsTTSEngineBuilder(enable_acoustic_tail_batch_padding="false")


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


def test_extra_scheduler_callbacks_wire_tail_shutdown_logging() -> None:
    builder = DotsTTSEngineBuilder()
    assert builder.extra_scheduler_callbacks() == {}

    calls: list[int] = []
    builder.acoustic_tail = SimpleNamespace(log_graph_counters=lambda: calls.append(1))
    callback = builder.extra_scheduler_callbacks()["shutdown_callback"]
    callback()

    assert calls == [1]
