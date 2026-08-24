# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import stat
import sys
from types import SimpleNamespace

from sglang_omni.profiler.torch_profiler import (
    TorchProfiler,
    _make_native_trace_handler,
    _resolve_profiler_api,
)


def test_npu_profiler_uses_torch_npu_activity(monkeypatch) -> None:
    profile = object()
    activity = SimpleNamespace(CPU=object(), NPU=object())
    torch_npu = SimpleNamespace(
        profiler=SimpleNamespace(
            profile=profile,
            ProfilerActivity=activity,
            tensorboard_trace_handler=object(),
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    resolved_profile, activities, trace_handler = _resolve_profiler_api("npu")

    assert resolved_profile is profile
    assert activities == [activity.CPU, activity.NPU]
    assert trace_handler is torch_npu.profiler.tensorboard_trace_handler


def test_native_trace_handler_uses_secure_offline_analysis_dir(tmp_path) -> None:
    seen = {}
    handler = object()

    def factory(**kwargs):
        seen.update(kwargs)
        return handler

    trace_dir, resolved_handler = _make_native_trace_handler(
        factory, str(tmp_path / "trace"), rank=2
    )

    assert trace_dir == str(tmp_path / "trace_rank2")
    assert stat.S_IMODE((tmp_path / "trace_rank2").stat().st_mode) == 0o750
    assert resolved_handler is handler
    assert seen == {
        "dir_name": trace_dir,
        "worker_name": "rank2",
        "analyse_flag": False,
        "async_mode": False,
    }


def test_restarting_same_run_returns_existing_trace(monkeypatch) -> None:
    monkeypatch.setattr(TorchProfiler, "_profiler", object())
    monkeypatch.setattr(TorchProfiler, "_active_run_id", "active")
    monkeypatch.setattr(TorchProfiler, "_trace_template", "/tmp/trace")
    monkeypatch.setattr(TorchProfiler, "_get_rank", lambda: 3)

    assert (
        TorchProfiler.start("unused", run_id="active")
        == "/tmp/trace_rank3.trace.json.gz"
    )
