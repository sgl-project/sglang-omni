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


def test_prefill_coalescing_is_opt_in_and_reaches_the_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.dots_tts import engine_builder, stages

    assert (
        DotsTTSEngineBuilder().extra_scheduler_kwargs()["prefill_coalesce_requests"]
        == 0
    )
    captured: dict[str, int | float] = {}

    class RecordingBuilder:
        def __init__(self, **kwargs: int | float | bool) -> None:
            captured.update(kwargs)

        def build(self, model_path: str, **kwargs: str | None) -> str:
            del model_path, kwargs
            return "engine"

    monkeypatch.setattr(stages.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_builder, "DotsTTSEngineBuilder", RecordingBuilder)
    assert (
        stages.create_sglang_latent_engine_executor(
            "model", prefill_coalesce_requests=3, prefill_coalesce_wait_ms=45.0
        )
        == "engine"
    )
    assert captured["prefill_coalesce_requests"] == 3
    assert captured["prefill_coalesce_wait_ms"] == 45.0
    extras = DotsTTSEngineBuilder(
        prefill_coalesce_requests=3, prefill_coalesce_wait_ms=45.0
    ).extra_scheduler_kwargs()
    assert extras["prefill_coalesce_requests"] == 3
    assert extras["prefill_coalesce_wait_ms"] == 45.0


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
