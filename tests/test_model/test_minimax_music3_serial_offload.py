# SPDX-License-Identifier: Apache-2.0
"""Opt-in Music3 HTTP parity and repeated serial-offload integration test."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import pynvml
import pytest
import requests
import soundfile as sf

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from sglang_omni.utils.connection import find_available_port

CHECKPOINT_ENV = "MINIMAX_MUSIC3_TEST_CHECKPOINT"
REPEATED_REQUESTS_ENV = "MINIMAX_MUSIC3_TEST_REPEATED_REQUESTS"
STARTUP_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 600
HANDOFF_TIMEOUT_SECONDS = 30
SAMPLE_RATE = 32_000
FRAME_RATE = 25
MAX_RELATIVE_RMS_DIFFERENCE = 0.005
MIN_WAVEFORM_CORRELATION = 0.9999


@pytest.mark.accelerator
def test_music3_serial_offload_http_parity_and_lifecycle(tmp_path: Path) -> None:
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    if not checkpoint:
        pytest.skip(f"Set {CHECKPOINT_ENV} to run the Music3 HTTP integration test")
    else:
        pass
    repeated_requests = int(os.environ.get(REPEATED_REQUESTS_ENV, "20"))
    payload = {
        "model": "minimax-music3",
        "input": "[Verse]\nCity lights are calling out my name",
        "instructions": "A dreamy synthwave track with analog pads and a bassline at 110 BPM",
        "seed": 42,
    }
    resident_audio: dict[int, bytes] = {}
    metrics: dict[str, list[dict[str, int | float | str]]] = {}
    memory_metrics: dict[str, list[dict[str, int | float]]] = {}
    memory_context: dict[str, dict[str, str]] = {}
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    modes = [
        ("resident-eager", False, "mmap", False, False, False),
        ("resident-graphs", False, "mmap", True, False, False),
        ("resident-compiled", False, "mmap", False, True, False),
        ("resident-accelerated", False, "mmap", True, True, False),
        ("offload-mmap-eager", True, "mmap", False, False, False),
        ("offload-ram-eager", True, "ram", False, False, False),
        ("offload-mmap-accelerated", True, "mmap", True, True, False),
        ("offload-ram-accelerated", True, "ram", True, True, False),
        ("resident-breakable", False, "mmap", True, True, True),
        ("offload-mmap-breakable", True, "mmap", True, True, True),
    ]
    selected_modes = os.environ.get("MINIMAX_MUSIC3_TEST_MODES")
    for mode, offload, source, graphs, compilation, breakable in modes:
        if selected_modes and mode not in selected_modes.split(","):
            continue
        else:
            pass
        mode_audio: dict[int, bytes] = {}
        port = find_available_port(host="127.0.0.1")
        log_path = tmp_path / f"{mode}.log"
        command = [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            checkpoint,
            "--model-name",
            payload["model"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dit_dav.process",
            "minimax_music3_ar",
            "--minimax_music3_ar.engine.max_running_requests",
            "1",
            "--minimax_music3_ar.engine.disable_cuda_graph",
            str(not graphs).lower(),
            "--minimax_music3_ar.engine.kv_cache_bytes",
            str(4 * 1024**3),
            "--dit_dav.factory.compile_acoustic",
            str(compilation).lower(),
            "--dit_dav.factory.breakable_cuda_graph",
            str(breakable).lower(),
            "--minimax_music3_ar.factory.serial_offload_source",
            source,
            "--minimax_music3_ar.factory.serial_offload_cache_dir",
            os.environ.get(
                "MINIMAX_MUSIC3_TEST_CACHE_DIR", str(tmp_path / "runtime-weights")
            ),
        ]
        if offload:
            command.extend(["--stage-offload-components", "ar,dit"])
        else:
            pass
        memory_metrics[mode] = []
        cgroup_directory = Path("/sys/fs/cgroup")
        memory_context[mode] = {
            filename: (cgroup_directory / filename).read_text()
            for filename in ("memory.max", "memory.events")
            if (cgroup_directory / filename).exists()
        }
        if (
            offload
            and source == "mmap"
            and os.environ.get("MINIMAX_MUSIC3_TEST_REQUIRE_32G_LIMIT")
        ):
            assert int(memory_context[mode]["memory.max"]) <= 32 * 1024**3
        else:
            pass
        stop_monitor = threading.Event()

        def monitor_memory() -> None:
            while not stop_monitor.is_set():
                anonymous_bytes = 0
                for process in psutil.Process().children(recursive=True):
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text()
                        match = re.search(r"^RssAnon:\s+(\d+) kB", status, re.MULTILINE)
                        anonymous_bytes += int(match[1]) * 1024 if match else 0
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        pass
                cgroup_path = Path("/sys/fs/cgroup/memory.current")
                memory_metrics[mode].append(
                    {
                        "time_seconds": time.monotonic(),
                        "anonymous_rss_bytes": anonymous_bytes,
                        "cgroup_memory_bytes": (
                            int(cgroup_path.read_text()) if cgroup_path.exists() else 0
                        ),
                        "physical_gpu_bytes": pynvml.nvmlDeviceGetMemoryInfo(
                            gpu_handle
                        ).used,
                    }
                )
                stop_monitor.wait(0.2)

        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        try:
            server = start_server_from_cmd(
                command,
                log_path,
                port,
                timeout=STARTUP_TIMEOUT_SECONDS,
                env={"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8"},
            )
        except BaseException:
            stop_monitor.set()
            monitor_thread.join()
            raise
        metrics[mode] = []

        def generate_audio(frames: int) -> bytes:
            started_at_seconds = time.perf_counter()
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/audio/speech",
                json={**payload, "max_new_tokens": frames},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            assert response.status_code == 200, response.text
            waveform, sample_rate = sf.read(
                io.BytesIO(response.content), always_2d=True
            )
            assert sample_rate == SAMPLE_RATE
            assert waveform.shape[1] == 2
            assert np.isfinite(waveform).all()
            assert np.max(np.abs(waveform)) > 0
            duration_seconds = waveform.shape[0] / sample_rate
            assert abs(duration_seconds - frames / FRAME_RATE) < 0.1
            metrics[mode].append(
                {
                    "frames": frames,
                    "duration_seconds": duration_seconds,
                    "latency_seconds": time.perf_counter() - started_at_seconds,
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                }
            )
            return response.content

        try:
            if (
                offload
                and source == "mmap"
                and os.environ.get("MINIMAX_MUSIC3_TEST_COLD_WAKE")
            ):
                cache_directory = Path(
                    command[
                        command.index(
                            "--minimax_music3_ar.factory.serial_offload_cache_dir"
                        )
                        + 1
                    ]
                )
                for shard_path in cache_directory.glob("*/*.safetensors"):
                    with shard_path.open("rb") as shard_file:
                        os.posix_fadvise(
                            shard_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                        )
                memory_context[mode]["cold_wake"] = "runtime shard page cache evicted"
            else:
                pass
            for frames in (100, 250):
                audio = generate_audio(frames)
                (tmp_path / f"{mode}-{frames}.wav").write_bytes(audio)
                mode_audio[frames] = audio
                assert generate_audio(frames) == audio
                if frames in resident_audio:
                    reference_waveform, _ = sf.read(io.BytesIO(resident_audio[frames]))
                    offload_waveform, _ = sf.read(io.BytesIO(audio))
                    assert reference_waveform.shape == offload_waveform.shape
                    difference = offload_waveform - reference_waveform
                    relative_rms_difference = float(
                        np.sqrt(np.mean(difference**2) / np.mean(reference_waveform**2))
                    )
                    waveform_correlation = float(
                        np.corrcoef(
                            reference_waveform.ravel(), offload_waveform.ravel()
                        )[0, 1]
                    )
                    metrics[mode][-1][
                        "relative_rms_difference"
                    ] = relative_rms_difference
                    metrics[mode][-1]["waveform_correlation"] = waveform_correlation
                    assert relative_rms_difference < MAX_RELATIVE_RMS_DIFFERENCE
                    assert waveform_correlation > MIN_WAVEFORM_CORRELATION
                else:
                    resident_audio[frames] = audio

            if offload:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    submitted = [executor.submit(generate_audio, 100) for _ in range(2)]
                    for future in submitted:
                        assert future.result() == mode_audio[100]
                repeated_audio = generate_audio(25)
                for _ in range(repeated_requests - 1):
                    assert generate_audio(25) == repeated_audio
                assert generate_audio(100) == mode_audio[100]
                handoff_deadline_seconds = time.monotonic() + HANDOFF_TIMEOUT_SECONDS
                while log_path.read_text().count("serial offload: AR -> GPU") < len(
                    metrics[mode]
                ):
                    assert time.monotonic() < handoff_deadline_seconds
                    time.sleep(0.1)
            else:
                pass
        finally:
            stop_server(server)
            stop_monitor.set()
            monitor_thread.join()
            (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (tmp_path / "memory-metrics.json").write_text(
                json.dumps(memory_metrics, indent=2)
            )
            if (cgroup_directory / "memory.events").exists():
                memory_context[mode]["memory.events.after"] = (
                    cgroup_directory / "memory.events"
                ).read_text()
            else:
                pass
            (tmp_path / "memory-context.json").write_text(
                json.dumps(memory_context, indent=2)
            )
            if offload:
                wake_seconds = re.findall(
                    r"residency (ar|dit/dav) -> gpu elapsed_seconds=([\d.]+)",
                    log_path.read_text(),
                )
                (tmp_path / f"{mode}-wake-latency.json").write_text(
                    json.dumps(wake_seconds, indent=2)
                )
            else:
                pass

        log_text = log_path.read_text()
        assert "Traceback" not in log_text
        if graphs:
            assert log_text.count("Capture target decode CUDA graph begin") == 1
            assert log_text.count("RVQ depth device graphs captured") == 1
        else:
            assert "RVQ depth device graphs captured" not in log_text
        if compilation:
            assert log_text.count("DIT blocks compiled in") == (0 if breakable else 1)
            assert log_text.count("DAV decoder compiled in") == 1
        else:
            pass
        if breakable:
            assert log_text.count("diffusion BCG captured") == 1
        else:
            pass
        if "memory.events.after" in memory_context[mode]:
            before_events = dict(
                line.split()
                for line in memory_context[mode]["memory.events"].splitlines()
            )
            after_events = dict(
                line.split()
                for line in memory_context[mode]["memory.events.after"].splitlines()
            )
            assert after_events["oom"] == before_events["oom"]
            assert after_events["oom_kill"] == before_events["oom_kill"]
        else:
            pass
        if offload:
            transitions = re.findall(
                r"serial offload: AR -> (CPU|GPU) .*?request=([^\)]+)", log_text
            )
            assert len(transitions) == 2 * len(metrics[mode])
            for index in range(0, len(transitions), 2):
                assert transitions[index][0] == "CPU"
                assert transitions[index + 1][0] == "GPU"
                assert transitions[index][1] == transitions[index + 1][1]
            assert "dit/dav -> gpu" in log_text
            assert "dit/dav -> host" in log_text
        else:
            pass
