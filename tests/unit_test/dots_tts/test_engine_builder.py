# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.dots_tts.engine_builder import DotsTTSEngineBuilder
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
)


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


def test_extra_scheduler_callbacks_wire_tail_shutdown_logging() -> None:
    builder = DotsTTSEngineBuilder()
    assert builder.extra_scheduler_callbacks() == {}

    calls: list[int] = []
    builder.acoustic_tail = SimpleNamespace(log_graph_counters=lambda: calls.append(1))
    callback = builder.extra_scheduler_callbacks()["shutdown_callback"]
    callback()

    assert calls == [1]


def test_prefill_graph_is_off_by_default_and_reaches_the_context_length() -> None:
    builder = DotsTTSEngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["cuda_graph_backend_prefill"] == CudaGraphBackend.DISABLED

    overrides = build_generation_batch_overrides(
        **defaults,
        server_args_overrides={
            "disable_cuda_graph": False,
            "cuda_graph_backend_prefill": CudaGraphBackend.BREAKABLE,
        },
    )
    builder.adjust_overrides(overrides)

    assert overrides["cuda_graph_bs_prefill"][-1] == builder.context_length
    assert overrides["chunked_prefill_size"] == 0
    assert overrides["enable_return_hidden_states"] is True
