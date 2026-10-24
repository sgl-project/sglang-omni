# SPDX-License-Identifier: Apache-2.0
"""Capture the mapping/formal trace pair any omni workload needs, plus the gates.

The backend's two-trace mode wants two recordings of the same work:

* ``mapping`` -- CUDA graph off, ``with_stack=1``. Only here do kernels carry
  python stacks, so only here can a kernel be named a line of omni source.
* ``formal`` -- the real serving config (graph on, stacks off). Only here are
  the timings the ones a user would see.

A workload supplies two callables that run the same computation under those two
configs; everything else -- env flags, trace naming, warmup, the steady-state
gate -- lives here so each workload stays a few lines.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sglang_omni.profiler.torch_profiler import TorchProfiler

_WITH_STACK_ENV = "SGLANG_TORCH_PROFILER_WITH_STACK"


def _torch_profiler() -> type[TorchProfiler]:
    """Imported where used: the gate is stdlib-only and must stay importable on a
    box without the serving runtime that ``TorchProfiler`` pulls in."""
    from sglang_omni.profiler.torch_profiler import TorchProfiler

    return TorchProfiler


# Substrings that mean the trace caught one-time work rather than steady state.
# Matched against every event name, so each must be unable to appear as ordinary
# steady-state activity: ``is_torchdynamo_compiling`` is a predicate every HF
# forward calls, and bare ``torch/_inductor/`` matches the entry point of
# already-compiled code, so the path markers name compile-side subpaths only.
# Path markers are python frames and a formal trace has none; there the gate
# rests on Dynamo's timed regions (named per torch version, always suffixed
# ``(dynamo_timed)``) and on module loads.
_COMPILE_MARKERS = (
    "(dynamo_timed)",
    "entire_frame_compile",  # torch 2.13
    "backend_compile",  # torch 2.13, also an fx-graph-cache hit
    "torch/_dynamo/convert_frame",  # the tracer entry
    "torch/_inductor/compile_fx",  # inductor compile entry
    "torch/_inductor/async_compile",
    "torch/_inductor/codecache",  # codegen, and fx-graph-cache loads
    "cudaModuleLoad",  # JIT load of a freshly compiled kernel
    "cuModuleLoad",
)
# One-time work as well, but ambiguous: the profiler emits this for the first use
# of *any* kernel, not only a freshly compiled one, so a clean mapping trace holds
# a handful of tens-of-microsecond loads with no compile marker beside them.
# Where stacks are on it is also redundant -- compilation matches a path marker
# there too -- so it is reported and not fatal. A stackless trace has no path
# markers to fall back on, so there it stays fatal.
_FIRST_CALL_MARKERS = ("Lazy Function Loading",)
# Capture, not replay: cudaGraphLaunch is exactly what a formal trace should be
# full of, so it is deliberately absent here.
_CAPTURE_MARKERS = (
    "cudaStreamBeginCapture",
    "cudaStreamEndCapture",
    "cudaGraphInstantiate",
)


def await_compression(gz_path: Path, *, timeout_s: float = 300.0) -> None:
    """TorchProfiler gzips in a background subprocess; the analyzer needs the .gz.

    ``gzip -f`` creates the archive before it finishes writing and unlinks the
    source only on success, so the vanished source -- not the archive's
    existence -- is what says the file is complete.
    """
    json_path = gz_path.with_suffix("")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if gz_path.exists() and not json_path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"background gzip did not finish writing {gz_path}")


def steady_state_violations(
    trace_gz: Path, *, allow_capture: bool = False, samples: int = 3
) -> dict[str, list[str]]:
    """Map each matched marker to up to ``samples`` of the events that matched it.

    Each sample carries the event's category and timestamp: the category says
    whether the match is a python stack frame or a runtime call, and the
    timestamps say whether the one-time work sits at the start of the window or
    recurs through it.

    Every marker that matched is reported, including the ambiguous ones;
    ``assert_steady_state`` is where they are weighed.
    """
    markers = _COMPILE_MARKERS + _FIRST_CALL_MARKERS
    if not allow_capture:
        markers += _CAPTURE_MARKERS
    with gzip.open(trace_gz, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    if not events:
        raise ValueError(f"{trace_gz} has no events; the profiler recorded nothing")
    hits: dict[str, list[str]] = {}
    for event in events:
        name = str(event.get("name", ""))
        for marker in markers:
            if marker not in name:
                continue
            seen = hits.setdefault(marker, [])
            if len(seen) < samples:
                seen.append(f"{name} [cat={event.get('cat')} ts={event.get('ts')}]")
    return hits


def _format_hits(hits: dict[str, list[str]]) -> str:
    """One indented block per marker, its samples listed under it."""
    return "\n".join(
        f"  {marker}\n" + "\n".join(f"    {sample}" for sample in samples)
        for marker, samples in sorted(hits.items())
    )


def assert_steady_state(
    trace_gz: Path, *, tag: str, allow_capture: bool = False, with_stack: bool = False
) -> None:
    """Gate: refuse a trace that recorded compilation or graph capture.

    A trace with either in it attributes one-time cost to steady-state kernels,
    which is the single most common way a profiling run reaches a wrong
    conclusion. Warm up until these are gone rather than subtracting them later.

    ``with_stack`` states that this trace carries python stacks, which demotes
    ``_FIRST_CALL_MARKERS`` to a printed note. It defaults to the strict reading,
    so a trace gated by hand off the server path has to opt in.
    """
    hits = steady_state_violations(trace_gz, allow_capture=allow_capture)
    demoted = _FIRST_CALL_MARKERS if with_stack else ()

    fatal = {
        marker: samples for marker, samples in hits.items() if marker not in demoted
    }
    if fatal:
        raise RuntimeError(
            f"[{tag}] trace is not steady state: {trace_gz}\n{_format_hits(fatal)}\n"
            "Increase --warmup (and warm every shape bucket) so compile and "
            "capture finish before the profiler starts. Timestamps bunched at "
            "the window start mean one shape bucket went unwarmed; timestamps "
            "spread across it mean something recompiles every call."
        )

    # After the raise, never beside it: a note saying one marker is harmless
    # reads as an excuse for the failure it is printed next to.
    noted = {marker: samples for marker, samples in hits.items() if marker in demoted}
    if noted:
        print(
            f"[{tag}] first-call kernel loads, not a failure here:\n"
            f"{_format_hits(noted)}\n"
            "  Nothing else matched, and a compile would have matched a path "
            "marker too, so this is the first use of an ordinary kernel. Its "
            "load cost lands on the kernel that triggered it; more warmup "
            "removes it."
        )


def capture(
    *,
    output_dir: Path | str,
    tag: str,
    body: Callable[[], object],
    iters: int,
    warmup: int,
    with_stack: bool,
    allow_capture: bool = False,
) -> Path:
    """Warm up, record ``iters`` calls of ``body``, gate the trace, return its dir.

    ``output_dir`` is coerced, so an argparse string works without ``type=Path``.
    """
    for _ in range(warmup):
        body()
    torch.cuda.synchronize()

    profiler = _torch_profiler()
    run_dir = Path(output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    # TorchProfiler reads the flag in start(); restore it so a capture inside a
    # server process does not change what a later /start_profile records.
    previous_with_stack = os.environ.get(_WITH_STACK_ENV)
    os.environ[_WITH_STACK_ENV] = "1" if with_stack else "0"
    try:
        trace = Path(profiler.start(str(run_dir / tag), run_id=tag))
    finally:
        if previous_with_stack is None:
            os.environ.pop(_WITH_STACK_ENV)
        else:
            os.environ[_WITH_STACK_ENV] = previous_with_stack
    # TorchNPUProfiler returns a directory; nothing below applies to it.
    assert trace.suffix == ".gz", f"expected a gzipped chrome trace, got {trace}"
    try:
        for _ in range(iters):
            body()
        torch.cuda.synchronize()
    finally:
        # A profiler left running past an exception has crashed the interpreter
        # at exit, and blocks the next capture in the same process.
        profiler.stop(run_id=tag)

    await_compression(trace)
    assert_steady_state(
        trace, tag=tag, allow_capture=allow_capture, with_stack=with_stack
    )
    print(f"[{tag}] trace -> {trace}")
    return run_dir


def capture_pair(
    *,
    output_dir: Path | str,
    mapping_body: Callable[[], object],
    formal_body: Callable[[], object],
    iters: int = 10,
    warmup: int = 5,
) -> tuple[Path, Path]:
    """Capture mapping then formal into ``<output_dir>/{mapping,formal}``."""
    mapping_dir = capture(
        output_dir=output_dir,
        tag="mapping",
        body=mapping_body,
        iters=iters,
        warmup=warmup,
        with_stack=True,
    )
    formal_dir = capture(
        output_dir=output_dir,
        tag="formal",
        body=formal_body,
        iters=iters,
        warmup=warmup,
        with_stack=False,
    )
    return mapping_dir, formal_dir
