# SPDX-License-Identifier: Apache-2.0
"""Compile lifecycle shared by model-owned pipeline callables."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CompilePhase(str, Enum):
    """Startup point at which a compile target must be installed."""

    BEFORE_PRIMARY_CUDA_GRAPH = "before_primary_cuda_graph"
    AFTER_PRIMARY_CUDA_GRAPH = "after_primary_cuda_graph"
    BEFORE_STAGE_READY = "before_stage_ready"


@dataclass(frozen=True)
class CompileWarmupCase:
    """One model-defined invocation used to warm a specialization."""

    label: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    bucket: Hashable | None = None
    repeat: int = 1

    def __post_init__(self) -> None:
        if self.repeat < 1:
            raise ValueError("CompileWarmupCase.repeat must be >= 1")


@dataclass(frozen=True)
class CompileTarget:
    """Declarative compile policy for one model-owned callable."""

    name: str
    eager: Callable[..., Any]
    install: Callable[[Callable[..., Any]], None]
    phase: CompilePhase = CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH
    compile_kwargs: Mapping[str, Any] = field(default_factory=dict)
    bucket_fn: Callable[..., Hashable | None] | None = None
    warmup_cases: Sequence[CompileWarmupCase] = ()
    restrict_to_warmed_buckets: bool = False


@dataclass(frozen=True)
class CompilePlan:
    """A named collection of declarative targets for one stage."""

    name: str
    targets: Sequence[CompileTarget]


@dataclass(frozen=True)
class _CompileTargetStats:
    setup_failures: int
    compilation_count: int
    recompilation_count: int
    compile_time_s: float
    warmup_time_s: float
    warmup_failures: int
    runtime_fallbacks: int


@dataclass(frozen=True)
class CompileStats:
    name: str
    target_count: int
    setup_failures: int
    compilation_count: int
    recompilation_count: int
    compile_time_s: float
    warmup_time_s: float
    warmup_failures: int
    runtime_fallbacks: int


CompilerFn = Callable[..., Callable[..., Any]]
ConfigureFn = Callable[[], None]


def configure_sglang_torch_compile() -> None:
    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

    set_torch_compile_config()


class StageCompileManager:
    """Install and observe every target in one stage compile plan."""

    def __init__(
        self,
        plan: CompilePlan,
        *,
        compile_fn: CompilerFn | None = None,
        configure_fn: ConfigureFn | None = None,
    ) -> None:
        self.plan = plan
        self._compile_fn = compile_fn
        self._configure_fn = configure_fn
        self._targets = tuple(plan.targets)
        self._handles: dict[str, _CompiledCallable] = {}
        self._applied_phases: set[CompilePhase] = set()
        self._configured = False
        self._configuration_error: Exception | None = None

    def apply(self, phase: CompilePhase) -> None:
        """Install all targets assigned to ``phase`` exactly once."""
        if phase in self._applied_phases:
            return

        targets = [target for target in self._targets if target.phase is phase]
        if targets:
            self._configure_once()
        for target in targets:
            handle = _CompiledCallable(
                target.name,
                target.eager,
                compile_kwargs=target.compile_kwargs,
                bucket_fn=target.bucket_fn,
                restrict_to_warmed_buckets=target.restrict_to_warmed_buckets,
                compile_fn=self._compile_fn,
                setup_error=self._configuration_error,
            )
            self._handles[target.name] = handle
            target.install(handle if handle.available else target.eager)
            if handle.available and target.warmup_cases:
                handle.warmup(target.warmup_cases)

        self._applied_phases.add(phase)

    def finish_startup(self) -> CompileStats:
        """Log one aggregate snapshot after the stage finishes startup work."""
        snapshot = self.stats()
        logger.info(
            "Compile plan %s startup complete: targets=%d setup_failures=%d "
            "compile_events=%d recompiles=%d compile_time=%.3fs "
            "warmup_time=%.3fs warmup_failures=%d runtime_fallbacks=%d",
            snapshot.name,
            snapshot.target_count,
            snapshot.setup_failures,
            snapshot.compilation_count,
            snapshot.recompilation_count,
            snapshot.compile_time_s,
            snapshot.warmup_time_s,
            snapshot.warmup_failures,
            snapshot.runtime_fallbacks,
        )
        return snapshot

    def stats(self) -> CompileStats:
        target_stats = [handle.stats() for handle in self._handles.values()]
        return CompileStats(
            name=self.plan.name,
            target_count=len(self._handles),
            setup_failures=sum(item.setup_failures for item in target_stats),
            compilation_count=sum(item.compilation_count for item in target_stats),
            recompilation_count=sum(item.recompilation_count for item in target_stats),
            compile_time_s=sum(item.compile_time_s for item in target_stats),
            warmup_time_s=sum(item.warmup_time_s for item in target_stats),
            warmup_failures=sum(item.warmup_failures for item in target_stats),
            runtime_fallbacks=sum(item.runtime_fallbacks for item in target_stats),
        )

    def _configure_once(self) -> None:
        if self._configured:
            return
        self._configured = True
        if self._configure_fn is None:
            return
        try:
            self._configure_fn()
        except Exception as exc:
            self._configuration_error = exc
            logger.warning(
                "Compile plan %s setup failed; using eager execution",
                self.plan.name,
                exc_info=True,
            )


def compile_callable(
    name: str,
    eager: Callable[..., Any],
    install: Callable[[Callable[..., Any]], None],
    *,
    phase: CompilePhase = CompilePhase.BEFORE_STAGE_READY,
    compile_kwargs: Mapping[str, Any] | None = None,
    bucket_fn: Callable[..., Hashable | None] | None = None,
    warmup_cases: Sequence[CompileWarmupCase] = (),
    restrict_to_warmed_buckets: bool = False,
    configure_fn: ConfigureFn | None = None,
) -> StageCompileManager:
    """Compile and install one standalone stage callable."""
    plan = CompilePlan(
        name=name,
        targets=(
            CompileTarget(
                name=name,
                eager=eager,
                install=install,
                phase=phase,
                compile_kwargs=dict(compile_kwargs or {}),
                bucket_fn=bucket_fn,
                warmup_cases=warmup_cases,
                restrict_to_warmed_buckets=restrict_to_warmed_buckets,
            ),
        ),
    )
    manager = StageCompileManager(plan, configure_fn=configure_fn)
    manager.apply(phase)
    manager.finish_startup()
    return manager


def build_module_list_compile_plan(
    name: str,
    modules: Sequence[Callable[..., Any]],
    *,
    install: Callable[[list[Callable[..., Any]]], None],
    phase: CompilePhase = CompilePhase.BEFORE_PRIMARY_CUDA_GRAPH,
    compile_kwargs: Mapping[str, Any] | None = None,
    bucket_fn: Callable[..., Hashable | None] | None = None,
) -> CompilePlan:
    """Build a plan that compiles each callable and installs a parallel list."""
    eager_modules = list(modules)
    if not eager_modules:
        raise ValueError(f"Compile module list {name!r} has no callables")

    installed_modules = list(eager_modules)
    install(installed_modules)
    options = (
        {
            "mode": os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE",
                "max-autotune-no-cudagraphs",
            )
        }
        if compile_kwargs is None
        else dict(compile_kwargs)
    )
    return CompilePlan(
        name=name,
        targets=tuple(
            CompileTarget(
                name=f"{name}.layer_{index}",
                eager=module,
                install=lambda compiled, index=index: installed_modules.__setitem__(
                    index, compiled
                ),
                phase=phase,
                compile_kwargs=options,
                bucket_fn=bucket_fn,
            )
            for index, module in enumerate(eager_modules)
        ),
    )


def tensor_dim_bucket(
    argument: str,
    *,
    dim: int = 0,
    positional_index: int = 0,
) -> Callable[..., int | None]:
    """Return a bucket function for one tensor argument dimension."""

    def _bucket(*args: Any, **kwargs: Any) -> int | None:
        value = kwargs.get(argument)
        if value is None and len(args) > positional_index:
            value = args[positional_index]
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            return None
        resolved_dim = dim if dim >= 0 else value.ndim + dim
        if resolved_dim < 0 or resolved_dim >= value.ndim:
            return None
        return int(value.shape[resolved_dim])

    return _bucket


class _CompiledCallable:
    """Private callable implementing lazy compile observation and fallback."""

    def __init__(
        self,
        name: str,
        eager: Callable[..., Any],
        *,
        compile_kwargs: Mapping[str, Any],
        bucket_fn: Callable[..., Hashable | None] | None,
        restrict_to_warmed_buckets: bool,
        compile_fn: CompilerFn | None,
        setup_error: Exception | None,
    ) -> None:
        self.name = name
        self.eager = eager
        self.bucket_fn = bucket_fn
        self.restrict_to_warmed_buckets = restrict_to_warmed_buckets
        self._compiled: Callable[..., Any] | None = None
        self._lock = threading.Lock()
        self._setup_failures = 0
        self._compilation_count = 0
        self._compile_trigger_count = 0
        self._compile_time_s = 0.0
        self._warmup_time_s = 0.0
        self._warmup_failures = 0
        self._runtime_fallbacks = 0
        self._failed_buckets: set[Hashable] = set()
        self._failed_without_bucket = False
        self._observed_buckets: set[Hashable] = set()
        self._observed_without_bucket = False
        self._warmed_buckets: set[Hashable] = set()
        self._warmed_without_bucket = False

        if setup_error is not None:
            self._setup_failures = 1
            self._failed_without_bucket = True
            return
        compiler = torch.compile if compile_fn is None else compile_fn
        try:
            self._compiled = compiler(eager, **dict(compile_kwargs))
        except Exception:
            self._setup_failures = 1
            self._failed_without_bucket = True
            logger.warning(
                "Compile target %s setup failed; using eager execution",
                name,
                exc_info=True,
            )

    @property
    def available(self) -> bool:
        return self._compiled is not None and not self._failed_without_bucket

    def warmup(self, cases: Sequence[CompileWarmupCase]) -> None:
        if not self.available:
            return

        started = time.perf_counter()
        for case in cases:
            bucket = case.bucket
            if bucket is None:
                bucket = self._resolve_bucket(case.args, case.kwargs)
            succeeded = True
            for _ in range(case.repeat):
                try:
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
                        "Compile target %s warmup failed for %s (bucket=%r); "
                        "using eager execution for that bucket",
                        self.name,
                        case.label,
                        bucket,
                        exc_info=True,
                    )
                    succeeded = False
                    break
            if succeeded:
                with self._lock:
                    if bucket is None:
                        self._warmed_without_bucket = True
                    else:
                        self._warmed_buckets.add(bucket)
        with self._lock:
            self._warmup_time_s += time.perf_counter() - started

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
                "Compile target %s failed at runtime for bucket=%r; "
                "falling back to eager execution",
                self.name,
                bucket,
                exc_info=True,
            )
            return self.eager(*args, **kwargs)

    def stats(self) -> _CompileTargetStats:
        with self._lock:
            return _CompileTargetStats(
                setup_failures=self._setup_failures,
                compilation_count=self._compilation_count,
                recompilation_count=max(0, self._compile_trigger_count - 1),
                compile_time_s=self._compile_time_s,
                warmup_time_s=self._warmup_time_s,
                warmup_failures=self._warmup_failures,
                runtime_fallbacks=self._runtime_fallbacks,
            )

    def _resolve_bucket(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Hashable | None:
        if self.bucket_fn is None:
            return None
        return self.bucket_fn(*args, **kwargs)

    def _should_use_compiled(self, bucket: Hashable | None) -> bool:
        if not self.available:
            return False
        with self._lock:
            if bucket is not None and bucket in self._failed_buckets:
                return False
            if self.restrict_to_warmed_buckets:
                if bucket is None:
                    return self._warmed_without_bucket
                return bucket in self._warmed_buckets
        return True

    def _mark_failed_locked(self, bucket: Hashable | None) -> None:
        if bucket is None:
            self._failed_without_bucket = True
        else:
            self._failed_buckets.add(bucket)

    def _reserve_compile_observation(self, bucket: Hashable | None) -> bool:
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
            raise RuntimeError(f"Compile target {self.name} is unavailable")
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
                    "Compile target %s observed %d compile event(s): total=%d "
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
