# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from sglang_omni.models.qwen3_tts import stages as qwen3_tts_stages
from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder


def test_qwen3_tts_prefill_coalesce_factory_defaults() -> None:
    signature = inspect.signature(qwen3_tts_stages.create_sglang_tts_engine_executor)

    assert signature.parameters["prefill_coalesce_requests"].default == 0
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 12.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is False
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert (
        signature.parameters["prefill_coalesce_after_builds_during_decode"].default
        is True
    )


def test_qwen3_tts_prefill_coalesce_defaults_forwarded_to_scheduler() -> None:
    builder = Qwen3TtsEngineBuilder()
    builder._stream_output_builder = object()

    assert builder.extra_scheduler_kwargs() == {
        "stream_output_builder": builder._stream_output_builder,
        "request_build_max_workers": 4,
        "request_build_max_pending": 16,
        "prefill_coalesce_requests": 0,
        "prefill_coalesce_wait_ms": 12.0,
        "prefill_coalesce_when_idle": False,
        "prefill_coalesce_requires_pending_builds": True,
        "prefill_coalesce_after_builds_during_decode": True,
    }


def test_qwen3_tts_prefill_coalesce_opt_in_forwarded_to_scheduler() -> None:
    builder = Qwen3TtsEngineBuilder(
        prefill_coalesce_requests=8,
        prefill_coalesce_wait_ms=12.0,
        prefill_coalesce_when_idle=False,
        prefill_coalesce_requires_pending_builds=True,
        prefill_coalesce_after_builds_during_decode=True,
    )
    builder._stream_output_builder = object()

    kwargs = builder.extra_scheduler_kwargs()
    assert kwargs["prefill_coalesce_requests"] == 8
    assert kwargs["prefill_coalesce_wait_ms"] == 12.0
    assert kwargs["prefill_coalesce_when_idle"] is False
    assert kwargs["prefill_coalesce_requires_pending_builds"] is True
    assert kwargs["prefill_coalesce_after_builds_during_decode"] is True


def test_qwen3_tts_factory_forwards_coalesce_values_to_builder(monkeypatch) -> None:
    # Guard the full factory -> builder path: if stages.py drops the
    # constructor passthrough while keeping the signature params, config
    # validation passes but coalescing silently stays off in production.
    captured: dict[str, object] = {}

    class _FakeBuilder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def build(self, *args: object, **kwargs: object) -> object:
            return object()

    monkeypatch.setattr(
        "sglang_omni.models.qwen3_tts.engine_builder.Qwen3TtsEngineBuilder",
        _FakeBuilder,
    )

    qwen3_tts_stages.create_sglang_tts_engine_executor(
        "model",
        prefill_coalesce_requests=8,
        prefill_coalesce_wait_ms=12.0,
        prefill_coalesce_when_idle=False,
        prefill_coalesce_requires_pending_builds=True,
        prefill_coalesce_after_builds_during_decode=True,
    )

    assert captured["prefill_coalesce_requests"] == 8
    assert captured["prefill_coalesce_wait_ms"] == 12.0
    assert captured["prefill_coalesce_when_idle"] is False
    assert captured["prefill_coalesce_requires_pending_builds"] is True
    assert captured["prefill_coalesce_after_builds_during_decode"] is True
