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
from typing import Callable

import torch

from sglang_omni.profiler.torch_profiler import TorchProfiler

# Substrings that mean the trace caught one-time work rather than steady state.
# Matched against every event name, so each must be unable to appear as ordinary
# steady-state activity: a predicate like ``is_torchdynamo_compiling`` is called
# on every forward, which is why bare "dynamo" is not on this list.
_COMPILE_MARKERS = (
    "torch/_inductor/",             # inductor codegen frames
    "torch/_dynamo/convert_frame",  # the tracer entry, not the is-compiling predicate
    "cudaModuleLoad",               # JIT load of a freshly compiled kernel
    "cuModuleLoad",
)
# Capture, not replay: cudaGraphLaunch is exactly what a formal trace should be
# full of, so it is deliberately absent here.
_CAPTURE_MARKERS = (
    "cudaStreamBeginCapture",
    "cudaStreamEndCapture",
    "cudaGraphInstantiate",
)


def _await_compression(gz_path: Path, *, timeout_s: float = 300.0) -> None:
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


def _event_names(trace_gz: Path) -> list[str]:
    with gzip.open(trace_gz, "rt") as handle:
        trace = json.load(handle)
    return [str(event.get("name", "")) for event in trace.get("traceEvents", [])]


def assert_steady_state(trace_gz: Path, *, tag: str, allow_capture: bool = False) -> None:
    """Gate: refuse a trace that recorded compilation or graph capture.

    A trace with either in it attributes one-time cost to steady-state kernels,
    which is the single most common way a profiling run reaches a wrong
    conclusion. Warm up until these are gone rather than subtracting them later.
    """
    names = _event_names(trace_gz)
    markers = _COMPILE_MARKERS if allow_capture else _COMPILE_MARKERS + _CAPTURE_MARKERS
    hits = {marker for marker in markers if any(marker in name for name in names)}
    if hits:
        raise RuntimeError(
            f"[{tag}] trace is not steady state: saw {sorted(hits)} in {trace_gz}. "
            "Increase --warmup (and warm every shape bucket) so compile and "
            "capture finish before the profiler starts."
        )


def capture(
    *,
    output_dir: Path,
    tag: str,
    body: Callable[[], object],
    iters: int,
    warmup: int,
    with_stack: bool,
    allow_capture: bool = False,
) -> Path:
    """Warm up, record ``iters`` calls of ``body``, gate the trace, return its dir."""
    for _ in range(warmup):
        body()
    torch.cuda.synchronize()

    os.environ["SGLANG_TORCH_PROFILER_WITH_STACK"] = "1" if with_stack else "0"
    run_dir = output_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    trace = Path(TorchProfiler.start(str(run_dir / tag), run_id=tag))
    for _ in range(iters):
        body()
    torch.cuda.synchronize()
    TorchProfiler.stop(run_id=tag)

    _await_compression(trace)
    assert_steady_state(trace, tag=tag, allow_capture=allow_capture)
    print(f"[{tag}] trace -> {trace}")
    return run_dir


def capture_pair(
    *,
    output_dir: Path,
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
