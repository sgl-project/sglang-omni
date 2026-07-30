# SPDX-License-Identifier: Apache-2.0
"""Shared torch.compile lifecycle for host-bound pipeline stages."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileWarmupCase:
    """One model-defined input used to compile and warm a shape bucket."""

    label: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    bucket: Hashable | None = None
    repeat: int = 1

    def __post_init__(self) -> None:
        if self.repeat < 1:
            raise ValueError("compile warmup repeat must be >= 1")


@dataclass(frozen=True)
class CompiledStageStats:
    """Immutable snapshot of compile and fallback counters."""

    compilation_count: int
    recompilation_count: int
    compile_time_s: float
    warmup_time_s: float
    warmup_failures: int
    runtime_fallbacks: int
    warmed_buckets: frozenset[Hashable]
    failed_buckets: frozenset[Hashable]


class CompiledStage:
    """Callable torch.compile wrapper with warmup and eager fallback.

    ``torch.compile`` is lazy: most compile time is paid on the first call and
    on later guard misses. This wrapper observes those calls, records graph
    compilation events, and falls back to the original callable if setup,
    warmup, or a serving-time compiled invocation fails.

    When ``restrict_to_warmed_buckets`` is true, only successfully warmed
    buckets use the compiled callable. Models with exact/static shapes should
    enable it; dynamic-shape stages can leave it false and use warmup cases to
    move known specializations before readiness without rejecting new shapes.
    """

    def __init__(
        self,
        name: str,
        eager: Callable[..., Any],
        *,
        compile_kwargs: Mapping[str, Any] | None = None,
        bucket_fn: Callable[..., Hashable | None] | None = None,
        restrict_to_warmed_buckets: bool = False,
        compile_fn: Callable[..., Callable[..., Any]] | None = None,
    ) -> None:
        self.name = name
        self.eager = eager
        self.bucket_fn = bucket_fn
        self.restrict_to_warmed_buckets = bool(restrict_to_warmed_buckets)
        self._compiled: Callable[..., Any] | None = None
        self._compile_kwargs = dict(compile_kwargs or {})
        self._lock = threading.Lock()
        self._compilation_count = 0
        self._compile_trigger_count = 0
        self._compile_time_s = 0.0
        self._warmup_time_s = 0.0
        self._warmup_failures = 0
        self._runtime_fallbacks = 0
        self._warmed_buckets: set[Hashable] = set()
        self._failed_buckets: set[Hashable] = set()
        self._failed_without_bucket = False
        self._observed_buckets: set[Hashable] = set()
        self._observed_without_bucket = False

        compiler = torch.compile if compile_fn is None else compile_fn
        try:
            self._compiled = compiler(eager, **self._compile_kwargs)
        except Exception:
            self._failed_without_bucket = True
            logger.warning(
                "Compiled stage %s setup failed; using eager execution",
                name,
                exc_info=True,
            )

    @property
    def compiled(self) -> Callable[..., Any] | None:
        return self._compiled

    def stats(self) -> CompiledStageStats:
        with self._lock:
            return CompiledStageStats(
                compilation_count=self._compilation_count,
                recompilation_count=max(0, self._compile_trigger_count - 1),
                compile_time_s=self._compile_time_s,
                warmup_time_s=self._warmup_time_s,
                warmup_failures=self._warmup_failures,
                runtime_fallbacks=self._runtime_fallbacks,
                warmed_buckets=frozenset(self._warmed_buckets),
                failed_buckets=frozenset(self._failed_buckets),
            )

    def warmup(self, cases: Sequence[CompileWarmupCase]) -> CompiledStageStats:
        """Run model-defined inputs before readiness and retain good buckets."""
        if self._compiled is None:
            return self.stats()

        started = time.perf_counter()
        for case in cases:
            bucket = case.bucket
            if bucket is None:
                bucket = self._resolve_bucket(case.args, case.kwargs)
            try:
                for _ in range(case.repeat):
                    self._call_compiled(
                        case.args,
                        case.kwargs,
                        observe_compile=self._reserve_compile_observation(bucket),
                    )
            except Exception:
                with self._lock:
                    self._warmup_failures += 1
                    self._mark_failed_locked(bucket)
                logger.warning(
                    "Compiled stage %s warmup failed for %s (bucket=%r); "
                    "that bucket will run eager",
                    self.name,
                    case.label,
                    bucket,
                    exc_info=True,
                )
                continue
            if bucket is not None:
                with self._lock:
                    self._warmed_buckets.add(bucket)

        elapsed = time.perf_counter() - started
        with self._lock:
            self._warmup_time_s += elapsed
            snapshot = self.stats_unlocked()
        logger.info(
            "Compiled stage %s warmup complete: buckets=%s failures=%d "
            "compile_events=%d recompiles=%d compile_time=%.3fs warmup_time=%.3fs",
            self.name,
            sorted(snapshot.warmed_buckets, key=repr),
            snapshot.warmup_failures,
            snapshot.compilation_count,
            snapshot.recompilation_count,
            snapshot.compile_time_s,
            snapshot.warmup_time_s,
        )
        return snapshot

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bucket = self._resolve_bucket(args, kwargs)
        if not self._should_use_compiled(bucket):
            return self.eager(*args, **kwargs)

        try:
            return self._call_compiled(
                args,
                kwargs,
                observe_compile=self._reserve_compile_observation(bucket),
            )
        except Exception:
            with self._lock:
                self._runtime_fallbacks += 1
                self._mark_failed_locked(bucket)
            logger.warning(
                "Compiled stage %s failed at runtime for bucket=%r; "
                "falling back to eager execution",
                self.name,
                bucket,
                exc_info=True,
            )
            return self.eager(*args, **kwargs)

    def stats_unlocked(self) -> CompiledStageStats:
        return CompiledStageStats(
            compilation_count=self._compilation_count,
            recompilation_count=max(0, self._compile_trigger_count - 1),
            compile_time_s=self._compile_time_s,
            warmup_time_s=self._warmup_time_s,
            warmup_failures=self._warmup_failures,
            runtime_fallbacks=self._runtime_fallbacks,
            warmed_buckets=frozenset(self._warmed_buckets),
            failed_buckets=frozenset(self._failed_buckets),
        )

    def _resolve_bucket(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Hashable | None:
        if self.bucket_fn is None:
            return None
        return self.bucket_fn(*args, **kwargs)

    def _should_use_compiled(self, bucket: Hashable | None) -> bool:
        if self._compiled is None or self._failed_without_bucket:
            return False
        with self._lock:
            if bucket is not None and bucket in self._failed_buckets:
                return False
            if self.restrict_to_warmed_buckets:
                return bucket is not None and bucket in self._warmed_buckets
        return True

    def _mark_failed_locked(self, bucket: Hashable | None) -> None:
        if bucket is None:
            self._failed_without_bucket = True
        else:
            self._failed_buckets.add(bucket)
            self._warmed_buckets.discard(bucket)

    def _reserve_compile_observation(self, bucket: Hashable | None) -> bool:
        """Observe only the first compiled call for each serving shape bucket."""
        with self._lock:
            if bucket is None:
                if self._observed_without_bucket:
                    return False
                self._observed_without_bucket = True
                return True
            if bucket in self._observed_buckets:
                return False
            self._observed_buckets.add(bucket)
            return True

    def _call_compiled(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        observe_compile: bool,
    ) -> Any:
        compiled = self._compiled
        if compiled is None:
            raise RuntimeError(f"Compiled stage {self.name} is unavailable")

        if not observe_compile:
            return compiled(*args, **kwargs)

        events: list[float] = []
        try:
            with _observe_torch_compilations(events):
                result = compiled(*args, **kwargs)
        finally:
            if events:
                elapsed = sum(events)
                with self._lock:
                    self._compilation_count += len(events)
                    self._compile_trigger_count += 1
                    self._compile_time_s += elapsed
                    count = self._compilation_count
                    recompilations = max(0, self._compile_trigger_count - 1)
                    total = self._compile_time_s
                logger.info(
                    "Compiled stage %s observed %d compile event(s): total=%d "
                    "recompiles=%d compile_time=%.3fs",
                    self.name,
                    len(events),
                    count,
                    recompilations,
                    total,
                )
        return result


@contextmanager
def _observe_torch_compilations(events: list[float]):
    """Observe lazy Dynamo compile duration during one callable invocation."""
    dynamo = getattr(torch, "_dynamo", None)
    handler = getattr(dynamo, "callback_handler", None)
    if handler is None:
        yield
        return

    started: list[float] = []

    def _on_start(_args: Any) -> None:
        started.append(time.perf_counter())

    def _on_end(_args: Any) -> None:
        if started:
            events.append(time.perf_counter() - started.pop())

    handler.register_start_callback(_on_start)
    handler.register_end_callback(_on_end)
    try:
        yield
    finally:
        handler.remove_start_callback(_on_start)
        handler.remove_end_callback(_on_end)


__all__ = ["CompileWarmupCase", "CompiledStage", "CompiledStageStats"]
