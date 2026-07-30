# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.utils.compiled_stage import CompiledStage, CompileWarmupCase


def test_compile_setup_failure_uses_eager(caplog) -> None:
    def fail_compile(_fn, **_kwargs):
        raise RuntimeError("compiler unavailable")

    with caplog.at_level(logging.WARNING):
        stage = CompiledStage(
            "setup-failure",
            lambda value: value * 2,
            compile_fn=fail_compile,
        )

    assert stage(4) == 8
    assert "setup failed" in caplog.text


def test_warmup_failure_restricts_only_failed_bucket_to_eager() -> None:
    eager_calls = []

    def eager(value):
        eager_calls.append(value)
        return value + 10

    def compile_fn(_fn, **_kwargs):
        def compiled(value):
            if value == 2:
                raise RuntimeError("bad shape")
            return value + 100

        return compiled

    stage = CompiledStage(
        "bucketed",
        eager,
        compile_fn=compile_fn,
        bucket_fn=lambda value: value,
        restrict_to_warmed_buckets=True,
    )
    stats = stage.warmup(
        [
            CompileWarmupCase("one", args=(1,), bucket=1),
            CompileWarmupCase("two", args=(2,), bucket=2),
        ]
    )

    assert stats.warmed_buckets == frozenset({1})
    assert stats.failed_buckets == frozenset({2})
    assert stats.warmup_failures == 1
    assert stage(1) == 101
    assert stage(2) == 12
    assert stage(3) == 13
    assert eager_calls == [2, 3]


def test_runtime_compile_failure_falls_back_and_sticks_per_bucket() -> None:
    compiled_calls = []
    eager_calls = []

    def eager(value):
        eager_calls.append(value)
        return -value

    def compiled(value):
        compiled_calls.append(value)
        if value == 7:
            raise RuntimeError("inductor failure")
        return value

    stage = CompiledStage(
        "runtime-failure",
        eager,
        compile_fn=lambda _fn, **_kwargs: compiled,
        bucket_fn=lambda value: value,
    )

    assert stage(7) == -7
    assert stage(7) == -7
    assert stage(8) == 8
    assert compiled_calls == [7, 8]
    assert eager_calls == [7, 7]
    assert stage.stats().runtime_fallbacks == 1


def test_compile_kwargs_and_repeat_are_forwarded() -> None:
    seen = {}
    calls = []

    def compile_fn(fn, **kwargs):
        seen.update(kwargs)
        return fn

    stage = CompiledStage(
        "kwargs",
        lambda value, *, scale: calls.append(value) or value * scale,
        compile_kwargs={"dynamic": True, "mode": "default"},
        compile_fn=compile_fn,
    )
    stage.warmup(
        [CompileWarmupCase("repeat", args=(3,), kwargs={"scale": 2}, repeat=3)]
    )

    assert seen == {"dynamic": True, "mode": "default"}
    assert calls == [3, 3, 3]


def test_warmup_case_rejects_non_positive_repeat() -> None:
    with pytest.raises(ValueError, match="repeat"):
        CompileWarmupCase("bad", repeat=0)


def test_compile_events_track_time_and_recompilations(monkeypatch) -> None:
    class FakeCallbackHandler:
        def __init__(self) -> None:
            self.starts = []
            self.ends = []
            self.start_registrations = 0
            self.end_registrations = 0

        def register_start_callback(self, callback) -> None:
            self.start_registrations += 1
            self.starts.append(callback)

        def register_end_callback(self, callback) -> None:
            self.end_registrations += 1
            self.ends.append(callback)

        def remove_start_callback(self, callback) -> None:
            self.starts.remove(callback)

        def remove_end_callback(self, callback) -> None:
            self.ends.remove(callback)

        def fire(self) -> None:
            args = SimpleNamespace(compile_id="test", callback_trigger="dynamo")
            for callback in list(self.starts):
                callback(args)
            for callback in list(self.ends):
                callback(args)

    handler = FakeCallbackHandler()
    monkeypatch.setattr(torch._dynamo, "callback_handler", handler)

    def compile_fn(fn, **_kwargs):
        def compiled(value):
            handler.fire()
            return fn(value)

        return compiled

    stage = CompiledStage(
        "stats",
        lambda value: value + 1,
        compile_fn=compile_fn,
        bucket_fn=lambda value: value,
    )

    assert stage(1) == 2
    assert stage(2) == 3
    assert stage(2) == 3
    stats = stage.stats()
    assert stats.compilation_count == 2
    assert stats.recompilation_count == 1
    assert stats.compile_time_s >= 0.0
    assert handler.start_registrations == 2
    assert handler.end_registrations == 2
