# SPDX-License-Identifier: Apache-2.0
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks import runtime_metrics
from benchmarks.eval import benchmark_asr_seedtts
from benchmarks.runtime_metrics import (
    ResourceMonitor,
    ResourceSample,
    summarize_resource_samples,
)


def test_summarize_resource_samples_reports_peak_and_steady_values() -> None:
    samples = [
        ResourceSample(
            elapsed_s=0.0,
            gpu_memory_used_mib=100.0,
            gpu_memory_free_mib=900.0,
            gpu_process_memory_mib=80.0,
            gpu_process_rss_mib=256.0,
            gpu_util_percent=10.0,
            power_w=20.0,
            system_cpu_percent=5.0,
            gpu_process_cpu_percent=25.0,
            gpu_process_pids=(11,),
        ),
        ResourceSample(
            elapsed_s=1.0,
            gpu_memory_used_mib=200.0,
            gpu_memory_free_mib=800.0,
            gpu_process_memory_mib=180.0,
            gpu_process_rss_mib=512.0,
            gpu_util_percent=90.0,
            power_w=120.0,
            system_cpu_percent=15.0,
            gpu_process_cpu_percent=125.0,
            gpu_process_pids=(11, 22),
        ),
    ]

    result = summarize_resource_samples(samples, interval_s=1.0)

    assert result["available"] is True
    assert result["gpu_memory_used_mib"] == {
        "min": 100.0,
        "max": 200.0,
        "end": 200.0,
        "steady_mean": 150.0,
    }
    assert result["gpu_process_memory_mib"]["max"] == 180.0
    assert result["gpu_process_rss_mib"]["max"] == 512.0
    assert result["power_w"]["max"] == 120.0
    assert result["gpu_process_cpu_percent"]["max"] == 125.0
    assert result["gpu_process_pids"] == [11, 22]


def test_summarize_resource_samples_reports_unavailable_monitor() -> None:
    result = summarize_resource_samples([], interval_s=0.2, error="NVML unavailable")

    assert result == {
        "available": False,
        "sample_interval_s": 0.2,
        "samples": 0,
        "error": "NVML unavailable",
    }


def test_provenance_labels_server_configuration_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_metrics.importlib.metadata,
        "distributions",
        lambda: [],
    )
    monkeypatch.setattr(runtime_metrics, "_command", lambda *_args: None)
    monkeypatch.setattr(
        runtime_metrics,
        "_first_prefixed_line",
        lambda *_args: None,
    )
    monkeypatch.setattr(runtime_metrics, "_package_version", lambda _name: None)

    result = runtime_metrics.collect_benchmark_provenance(
        model_id="model",
        model_revision=None,
        dataset_id="dataset",
        dataset_revision=None,
        launch_command=None,
        server_config={"dtype": "bfloat16"},
    )

    assert result["declared_server_config"] == {"dtype": "bfloat16"}
    assert "server_config" not in result
    assert result["artifacts"]["declared_model_revision"] is None
    assert "model_revision" not in result["artifacts"]
    assert result["repository"]["dirty"] is None
    assert result["dependency_inventory"] == []


def test_nvml_handle_respects_explicitly_hidden_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    pynvml = SimpleNamespace(
        nvmlDeviceGetHandleByIndex=lambda index: f"gpu-{index}",
    )

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES is empty"):
        runtime_metrics._resolve_nvml_handle(pynvml, 0)


def test_nvml_process_snapshot_marks_unsupported_getters_unavailable() -> None:
    assert runtime_metrics._nvml_compute_processes(SimpleNamespace(), object()) is None


@pytest.mark.parametrize(
    "interval_s",
    [float("nan"), float("inf"), float("-inf"), 0.0],
)
def test_resource_monitor_rejects_nonfinite_or_nonpositive_interval(
    interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="interval_s must be finite and > 0"):
        ResourceMonitor(interval_s=interval_s)


def test_resource_monitor_refuses_overlapping_nvml_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvml_session_lock = threading.Lock()
    nvml_session_lock.acquire()
    monkeypatch.setattr(
        runtime_metrics,
        "_NVML_SESSION_LOCK",
        nvml_session_lock,
        raising=False,
    )

    result = ResourceMonitor().start().stop()

    assert result["available"] is False
    assert result["error"] == "another NVML resource monitor is still active"


def test_resource_monitor_aggregates_gpu_process_host_rss() -> None:
    class FakeProcess:
        def cpu_percent(self, *, interval=None):
            assert interval is None
            return 12.5

        def memory_info(self):
            return SimpleNamespace(rss=2 * 1024 * 1024)

    monitor = ResourceMonitor()
    monitor._psutil = SimpleNamespace(
        Process=lambda _pid: FakeProcess(),
        NoSuchProcess=RuntimeError,
        AccessDenied=PermissionError,
    )

    cpu_percent, rss_mib = monitor._gpu_process_host_metrics({11, 22})

    assert cpu_percent == 25.0
    assert rss_mib == 4.0


def test_resource_monitor_keeps_nvml_calls_on_sampler_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    caller_thread = threading.get_ident()
    nvml_threads: list[int] = []

    def record(value=None):
        nvml_threads.append(threading.get_ident())
        return value

    pynvml = ModuleType("pynvml")
    pynvml.nvmlInit = lambda: record()
    pynvml.nvmlShutdown = lambda: record()
    pynvml.nvmlDeviceGetHandleByIndex = lambda _index: record("gpu")
    pynvml.nvmlDeviceGetMemoryInfo = lambda _handle: record(
        SimpleNamespace(used=1024, free=2048)
    )
    pynvml.nvmlDeviceGetComputeRunningProcesses = lambda _handle: record([])
    pynvml.nvmlDeviceGetGraphicsRunningProcesses = lambda _handle: record([])
    pynvml.nvmlDeviceGetUtilizationRates = lambda _handle: record(
        SimpleNamespace(gpu=50)
    )
    pynvml.nvmlDeviceGetPowerUsage = lambda _handle: record(1000)

    psutil = ModuleType("psutil")
    psutil.cpu_percent = lambda interval=None: 10.0
    psutil.NoSuchProcess = RuntimeError
    psutil.AccessDenied = PermissionError

    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setitem(sys.modules, "psutil", psutil)

    monitor = ResourceMonitor(interval_s=0.01).start()
    deadline = time.monotonic() + 1.0
    while not monitor.samples and time.monotonic() < deadline:
        time.sleep(0.01)
    result = monitor.stop()

    assert result["available"] is True
    assert nvml_threads
    assert caller_thread not in nvml_threads
    assert len(set(nvml_threads)) == 1


@pytest.mark.asyncio
async def test_asr_repeat_stops_resource_monitor_when_request_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = False

    class FakeMonitor:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self):
            return self

        def stop(self):
            nonlocal stopped
            stopped = True
            return {}

    async def fail_request(*_args, **_kwargs):
        raise RuntimeError("request failed")

    monkeypatch.setattr(benchmark_asr_seedtts, "ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        benchmark_asr_seedtts,
        "run_asr_seedtts_once",
        fail_request,
    )
    args = SimpleNamespace(
        disable_resource_monitor=False,
        gpu_index=0,
        monitor_interval_s=0.2,
        host="127.0.0.1",
        port=8000,
        model_path="model",
        lang="en",
        stream=False,
    )

    with pytest.raises(RuntimeError, match="request failed"):
        await benchmark_asr_seedtts._run_repeat(args, [], 1, 1)

    assert stopped is True
