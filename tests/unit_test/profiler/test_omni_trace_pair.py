# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the omni-gpu-deep-dive steady-state gate.

The gate in ``.claude/skills/omni-gpu-deep-dive/scripts/omni_trace_pair.py``
stops a profiling run from charging one-time cost -- Dynamo/Inductor
compilation, CUDA graph capture -- to a steady-state kernel. Its whole value is
one distinction that is easy to break while editing the marker tuples: reject
*compiling* and *capturing*, accept *compiled* and *captured execution*.

``.claude/`` is not on the pytest path, so the module is loaded by file path.
Nothing here needs a GPU or the serving runtime.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / ".claude"
    / "skills"
    / "omni-gpu-deep-dive"
    / "scripts"
    / "omni_trace_pair.py"
)
_PROFILER_MODULE = "sglang_omni.profiler.torch_profiler"
_WITH_STACK = "SGLANG_TORCH_PROFILER_WITH_STACK"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("omni_trace_pair", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trace_pair() -> ModuleType:
    if not _SCRIPT.exists():
        pytest.skip(f"{_SCRIPT} is not present in this checkout")
    return _load_module()


def _write_trace(path: Path, event_names: list[str], *, cat: str = "cpu_op") -> Path:
    """The minimal gzipped chrome trace the gate reads, one ts per event."""
    events = [
        {"name": name, "cat": cat, "ph": "X", "ts": index, "dur": 1}
        for index, name in enumerate(event_names)
    ]
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    return path


class _FakeProfiler:
    """Stands in for ``TorchProfiler``: records call order, writes a clean trace.

    ``start`` also records the with_stack env var, where the real one reads it.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.seen_with_stack: list[str | None] = []

    def start(self, trace_path_template: str, run_id: str | None = None) -> str:
        self.calls.append(f"start:{run_id}")
        self.seen_with_stack.append(os.environ.get(_WITH_STACK))
        gz_path = Path(f"{trace_path_template}_rank0.trace.json.gz")
        _write_trace(gz_path, ["cudaLaunchKernel", "aten::mm"])
        return str(gz_path)

    def stop(self, *, run_id: str | None = None) -> None:
        self.calls.append(f"stop:{run_id}")


@pytest.fixture
def fake_profiler(
    trace_pair: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> _FakeProfiler:
    profiler = _FakeProfiler()
    monkeypatch.setattr(trace_pair, "_torch_profiler", lambda: profiler)
    monkeypatch.setattr(trace_pair.torch.cuda, "synchronize", lambda: None)
    return profiler


def test_only_capture_needs_the_serving_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is stdlib plus torch; only ``capture`` imports omni's profiler.

    A ``None`` in ``sys.modules`` makes that import raise, standing in for a box
    without the pinned CUDA stack ``sglang`` pulls in. A plain torch stage there
    captures with ``torch.profiler`` and still gates the result.
    """
    monkeypatch.setitem(sys.modules, _PROFILER_MODULE, None)
    module = _load_module()

    trace = _write_trace(tmp_path / "formal.trace.json.gz", ["cudaGraphLaunch"])
    module.assert_steady_state(trace, tag="formal")

    with pytest.raises(ImportError):
        module._torch_profiler()


def test_gate_accepts_captured_and_compiled_execution(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """A healthy formal trace replays graphs and calls the is-compiling predicate."""
    trace = _write_trace(
        tmp_path / "formal.trace.json.gz",
        [
            "cudaGraphLaunch",
            "cudaLaunchKernel",
            "is_torchdynamo_compiling",
            "triton_poi_fused_add_0",
            "aten::mm",
        ],
    )
    trace_pair.assert_steady_state(trace, tag="formal")


def test_gate_accepts_inductor_frames_that_are_not_compilation(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """The false positive that made a real run bypass this gate.

    ``output_code.py`` is how an already-compiled graph is entered, and the
    ``compile_worker`` threads sit in a blocking read for the life of the
    process, so both land in every trace. A gate that rejects them fails clean
    runs, and a gate that fails clean runs gets bypassed -- worse than none.
    """
    trace = _write_trace(
        tmp_path / "mapping.trace.json.gz",
        [
            "torch/_inductor/output_code.py(581): __call__",
            "torch/_inductor/compile_worker/subproc_pool.py(195): _read_thread",
            "torch/_inductor/runtime/autotune_cache.py(481): end_compile",
            "aten::mm",
        ],
        cat="python_function",
    )
    trace_pair.assert_steady_state(trace, tag="mapping")


def test_gate_rejects_a_trace_with_no_events(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """An empty trace matches no marker, so passing it would gate nothing."""
    trace = _write_trace(tmp_path / "formal.trace.json.gz", [])
    with pytest.raises(ValueError, match="no events"):
        trace_pair.assert_steady_state(trace, tag="formal")


@pytest.mark.parametrize(
    "event_name",
    ["torch/_inductor/compile_fx.py(1500): compile_fx", "cudaModuleLoad"],
)
def test_gate_rejects_compilation(
    trace_pair: ModuleType, tmp_path: Path, event_name: str
) -> None:
    """Compilation in the window must fail loudly, not be subtracted afterwards."""
    trace = _write_trace(tmp_path / "mapping.trace.json.gz", ["aten::mm", event_name])
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(trace, tag="mapping")


@pytest.mark.parametrize(
    "event_name",
    [
        "entire_frame_compile",  # torch 2.13
        "compile_fx_inner (dynamo_timed)",  # torch 2.8
        "Lazy Function Loading",  # first call of a kernel, any version
    ],
)
def test_gate_rejects_compilation_without_python_stacks(
    trace_pair: ModuleType, tmp_path: Path, event_name: str
) -> None:
    """A formal trace has no python stacks, so path markers never appear in it.

    A cold ``torch.compile`` inside a formal capture used to pass for that
    reason. Dynamo's timed regions are plain events, present with stacks on or
    off and absent once warmed; the names change between torch versions, the
    ``(dynamo_timed)`` suffix does not.
    """
    trace = _write_trace(
        tmp_path / "formal.trace.json.gz",
        ["cudaGraphLaunch", event_name, "aten::mm"],
        cat="user_annotation",
    )
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(trace, tag="formal")


def test_gate_failure_reports_bounded_samples_with_category_and_timestamp(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """The substring alone cannot tell a stack frame from real work, and a real
    mapping trace runs to hundreds of MB, so the report stays bounded."""
    trace = _write_trace(
        tmp_path / "mapping.trace.json.gz",
        ["aten::mm"] + ["torch/_inductor/compile_fx.py(1500): compile_fx"] * 50,
        cat="python_function",
    )
    with pytest.raises(RuntimeError) as excinfo:
        trace_pair.assert_steady_state(trace, tag="mapping")

    message = str(excinfo.value)
    assert "torch/_inductor/compile_fx.py(1500): compile_fx" in message
    assert "cat=python_function" in message
    assert "ts=1" in message

    hits = trace_pair.steady_state_violations(trace, samples=3)
    assert list(hits) == ["torch/_inductor/compile_fx"]
    assert len(hits["torch/_inductor/compile_fx"]) == 3


def test_capture_is_rejected_by_default_and_allowed_only_by_its_own_flag(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """``allow_capture`` is an escape hatch for capture, never for compilation."""
    captured = _write_trace(
        tmp_path / "captured.trace.json.gz", ["cudaStreamBeginCapture", "aten::mm"]
    )
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(captured, tag="formal")
    trace_pair.assert_steady_state(captured, tag="formal", allow_capture=True)

    compiled = _write_trace(
        tmp_path / "compiled.trace.json.gz", ["cudaModuleLoad", "aten::mm"]
    )
    with pytest.raises(RuntimeError, match="not steady state"):
        trace_pair.assert_steady_state(compiled, tag="formal", allow_capture=True)


def test_await_compression_waits_for_the_json_source_to_vanish(
    trace_pair: ModuleType, tmp_path: Path
) -> None:
    """``gzip -f`` creates the archive first and unlinks the source only on
    success, so a ``.gz`` beside a live ``.json`` is a truncated read."""
    gz_path = tmp_path / "trace.json.gz"
    json_path = tmp_path / "trace.json"
    gz_path.write_bytes(b"")
    json_path.write_bytes(b"{}")

    with pytest.raises(TimeoutError, match="background gzip did not finish"):
        trace_pair.await_compression(gz_path, timeout_s=0.5)

    json_path.unlink()
    trace_pair.await_compression(gz_path, timeout_s=0.5)


def test_capture_accepts_a_string_output_dir_and_gates_the_trace(
    trace_pair: ModuleType, tmp_path: Path, fake_profiler: _FakeProfiler
) -> None:
    """argparse hands over a str, so the API boundary must coerce it.

    Also pins the rest of the contract: warmup runs before the profiler is
    armed, ``iters`` calls are recorded, with_stack is set before ``start``.
    """
    calls = fake_profiler.calls

    run_dir = trace_pair.capture(
        output_dir=str(tmp_path / "run"),
        tag="mapping",
        body=lambda: calls.append("body"),
        iters=3,
        warmup=2,
        with_stack=True,
    )

    assert run_dir == tmp_path / "run" / "mapping"
    assert fake_profiler.seen_with_stack == ["1"]
    assert calls == ["body", "body", "start:mapping"] + ["body"] * 3 + ["stop:mapping"]


def test_capture_restores_the_with_stack_env_var(
    trace_pair: ModuleType,
    tmp_path: Path,
    fake_profiler: _FakeProfiler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture inside a server process must not change later /start_profile runs."""
    monkeypatch.setenv(_WITH_STACK, "1")

    trace_pair.capture(
        output_dir=tmp_path,
        tag="formal",
        body=lambda: None,
        iters=1,
        warmup=0,
        with_stack=False,
    )

    assert fake_profiler.seen_with_stack == ["0"]
    assert os.environ[_WITH_STACK] == "1"


def test_capture_stops_the_profiler_when_the_body_raises(
    trace_pair: ModuleType, tmp_path: Path, fake_profiler: _FakeProfiler
) -> None:
    """A profiler left active past an exception crashed the interpreter at exit.

    The error still propagates; what changes is that ``stop`` runs first.
    """
    calls = fake_profiler.calls

    def body() -> None:
        calls.append("body")
        raise RuntimeError("shape bucket missing")

    with pytest.raises(RuntimeError, match="shape bucket missing"):
        trace_pair.capture(
            output_dir=tmp_path,
            tag="formal",
            body=body,
            iters=3,
            warmup=0,
            with_stack=False,
        )

    assert calls == ["start:formal", "body", "stop:formal"]


def test_capture_refuses_a_profiler_that_returns_a_directory(
    trace_pair: ModuleType,
    tmp_path: Path,
    fake_profiler: _FakeProfiler,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TorchNPUProfiler.start`` returns a directory, which nothing here can
    gate; failing at ``start`` beats a 300 s ``await_compression`` timeout."""
    monkeypatch.setattr(fake_profiler, "start", lambda template, run_id=None: template)

    with pytest.raises(AssertionError, match="gzipped chrome trace"):
        trace_pair.capture(
            output_dir=tmp_path,
            tag="formal",
            body=lambda: None,
            iters=1,
            warmup=0,
            with_stack=False,
        )
