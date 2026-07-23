# SPDX-License-Identifier: Apache-2.0

from benchmarks.runtime_metrics import ResourceSample, summarize_resource_samples


def test_summarize_resource_samples_reports_peak_and_steady_values() -> None:
    samples = [
        ResourceSample(
            elapsed_s=0.0,
            gpu_memory_used_mib=100.0,
            gpu_memory_free_mib=900.0,
            gpu_process_memory_mib=80.0,
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
