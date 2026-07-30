# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import pytest

import sglang_omni.compilation.stage_compile as stage_compile
from sglang_omni.compilation import (
    CompilePhase,
    CompilePlan,
    CompileTarget,
    CompileWarmupCase,
    StageCompileManager,
)


def _plan(*targets: CompileTarget) -> CompilePlan:
    return CompilePlan(name="test.plan", targets=targets)


def test_manager_installs_compiled_callable_once_per_phase() -> None:
    installed = []
    configured = []
    compile_calls = []

    def eager(value: int) -> str:
        return f"eager:{value}"

    def compile_fn(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return lambda value: f"compiled:{value}"

    manager = StageCompileManager(
        _plan(
            CompileTarget(
                name="target",
                eager=eager,
                install=installed.append,
                compile_kwargs={"mode": "default"},
            )
        ),
        compile_fn=compile_fn,
        configure_fn=lambda: configured.append(True),
    )

    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)
    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    assert configured == [True]
    assert compile_calls == [(eager, {"mode": "default"})]
    assert len(installed) == 1
    assert installed[0](3) == "compiled:3"
    assert manager.stats().target_count == 1


def test_default_compiler_is_resolved_after_configuration(monkeypatch) -> None:
    installed = []
    compile_calls = []

    def configured_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return lambda value: value + 10

    def configure_fn() -> None:
        monkeypatch.setattr(stage_compile.torch, "compile", configured_compile)

    manager = StageCompileManager(
        _plan(
            CompileTarget(
                name="target",
                eager=lambda value: value,
                install=installed.append,
            )
        ),
        configure_fn=configure_fn,
    )

    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    assert len(compile_calls) == 1
    assert installed[0](2) == 12


@pytest.mark.parametrize("failure_site", ["configure", "compile"])
def test_setup_failure_installs_direct_eager_callable(failure_site: str) -> None:
    installed = []

    def eager(value: int) -> int:
        return value + 1

    def configure_fn() -> None:
        if failure_site == "configure":
            raise RuntimeError("configure failed")

    def compile_fn(fn, **kwargs):
        del fn, kwargs
        if failure_site == "compile":
            raise RuntimeError("compile failed")
        raise AssertionError("compile_fn must not run after configuration failure")

    manager = StageCompileManager(
        _plan(
            CompileTarget(
                name="target",
                eager=eager,
                install=installed.append,
            )
        ),
        compile_fn=compile_fn,
        configure_fn=configure_fn,
    )

    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    assert installed == [eager]
    assert installed[0](4) == 5
    assert manager.stats().setup_failures == 1


def test_warmup_failure_falls_back_only_for_failed_bucket() -> None:
    installed = []
    compiled_calls = []
    eager_calls = []

    def eager(value: int) -> str:
        eager_calls.append(value)
        return f"eager:{value}"

    def compiled(value: int) -> str:
        compiled_calls.append(value)
        if value == 2:
            raise RuntimeError("unsupported bucket")
        return f"compiled:{value}"

    target = CompileTarget(
        name="target",
        eager=eager,
        install=installed.append,
        bucket_fn=lambda value: value,
        warmup_cases=(
            CompileWarmupCase("one", args=(1,), bucket=1),
            CompileWarmupCase("two", args=(2,), bucket=2),
        ),
    )
    manager = StageCompileManager(
        _plan(target),
        compile_fn=lambda fn, **kwargs: compiled,
        configure_fn=lambda: None,
    )

    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    assert installed[0](1) == "compiled:1"
    assert installed[0](2) == "eager:2"
    assert compiled_calls == [1, 2, 1]
    assert eager_calls == [2]
    assert manager.stats().warmup_failures == 1


def test_warmup_repeat_and_restriction_keep_unwarmed_buckets_eager() -> None:
    installed = []
    compiled_calls = []
    eager_calls = []

    def eager(value: int) -> str:
        eager_calls.append(value)
        return f"eager:{value}"

    def compiled(value: int) -> str:
        compiled_calls.append(value)
        return f"compiled:{value}"

    manager = StageCompileManager(
        _plan(
            CompileTarget(
                name="target",
                eager=eager,
                install=installed.append,
                bucket_fn=lambda value: value,
                warmup_cases=(CompileWarmupCase("one", args=(1,), repeat=3),),
                restrict_to_warmed_buckets=True,
            )
        ),
        compile_fn=lambda fn, **kwargs: compiled,
        configure_fn=lambda: None,
    )

    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    assert installed[0](1) == "compiled:1"
    assert installed[0](2) == "eager:2"
    assert compiled_calls == [1, 1, 1, 1]
    assert eager_calls == [2]


def test_runtime_failure_disables_compiled_callable_for_that_bucket() -> None:
    installed = []
    compiled_calls = []
    eager_calls = []

    def eager(value: int) -> str:
        eager_calls.append(value)
        return f"eager:{value}"

    def compiled(value: int) -> str:
        compiled_calls.append(value)
        if value == 2:
            raise RuntimeError("runtime failure")
        return f"compiled:{value}"

    manager = StageCompileManager(
        _plan(
            CompileTarget(
                name="target",
                eager=eager,
                install=installed.append,
                bucket_fn=lambda value: value,
            )
        ),
        compile_fn=lambda fn, **kwargs: compiled,
        configure_fn=lambda: None,
    )
    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    assert installed[0](2) == "eager:2"
    assert installed[0](2) == "eager:2"
    assert compiled_calls == [2]
    assert eager_calls == [2, 2]
    assert manager.stats().runtime_fallbacks == 1


def test_compile_events_are_observed_once_per_bucket(monkeypatch) -> None:
    installed = []

    @contextmanager
    def fake_observer(events):
        events.append(0.25)
        yield

    monkeypatch.setattr(stage_compile, "_observe_torch_compilations", fake_observer)
    manager = StageCompileManager(
        _plan(
            CompileTarget(
                name="target",
                eager=lambda value: value,
                install=installed.append,
                bucket_fn=lambda value: value,
            )
        ),
        compile_fn=lambda fn, **kwargs: fn,
        configure_fn=lambda: None,
    )
    manager.apply(CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH)

    installed[0](1)
    installed[0](1)
    installed[0](2)

    stats = manager.stats()
    assert stats.compilation_count == 2
    assert stats.recompilation_count == 1
    assert stats.compile_time_s == pytest.approx(0.5)
