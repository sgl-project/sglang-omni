# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model AuK parity, cancellation recovery and residency measurements."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile
import torch

pytest.importorskip("pynvml", reason="CUDA measurements require nvidia-ml-py")

from sglang_omni.scheduling.stage_residency import offload_environment
from sglang_omni.utils.connection import find_available_port
from tests.test_model.auk_offload_helpers import generate_audio, measured_server

REPEAT_REQUESTS = 5


@pytest.mark.accelerator
@pytest.mark.parametrize(
    ("compilation", "graphs"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["eager", "graphs", "compiled", "accelerated"],
)
def test_auk_serial_offload_http_parity_and_memory(
    tmp_path: Path, compilation: bool, graphs: bool
) -> None:
    checkpoint = os.environ.get("AUK_OFFLOAD_CHECKPOINT")
    reference = os.environ.get("AUK_OFFLOAD_REFERENCE")
    if not checkpoint or not reference:
        pytest.skip(
            "Set AUK_OFFLOAD_CHECKPOINT and AUK_OFFLOAD_REFERENCE to a checkpoint and WAV"
        )
    elif not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    selected_modes = os.environ.get("AUK_OFFLOAD_MODES", "resident,ram,mmap").split(",")
    if (
        not selected_modes
        or selected_modes[0] != "resident"
        or len(set(selected_modes)) != len(selected_modes)
        or not set(selected_modes) <= {"resident", "ram", "mmap"}
    ):
        raise ValueError(
            "AUK_OFFLOAD_MODES must start with resident and contain unique supported modes"
        )
    else:
        pass
    memory_context: dict[str, dict[str, str | int]] = {}
    reference_audio = (
        "data:audio/wav;base64,"
        + base64.b64encode(Path(reference).read_bytes()).decode()
    )
    capture_shapes = "[[1,192,0,192],[1,320,320,384]]"
    expected_audio: dict[str, bytes] = {}
    results: dict[str, dict[str, int | float | str | list[float]]] = {}
    cache_directory = os.environ.get("AUK_OFFLOAD_CACHE_DIR", str(tmp_path / "weights"))
    for mode in selected_modes:
        log_path = tmp_path / f"{mode}.log"
        port = find_available_port(host="127.0.0.1")
        command = [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            checkpoint,
            "--model-name",
            "auk",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--conditioning.factory.max_batch_size",
            "1",
            "--auk_engine.factory.max_batch_size",
            "1",
            "--decode.factory.max_batch_size",
            "1",
            "--auk_engine.factory.enable_dit_torch_compile",
            str(compilation).lower(),
            "--auk_engine.factory.enable_dit_cuda_graph",
            str(graphs).lower(),
            "--auk_engine.factory.dit_cuda_graph_capture_shapes",
            capture_shapes,
        ]
        if mode != "resident":
            command += [
                "--stage-offload-components",
                "conditioning,dit,vae",
                "--conditioning.factory.serial_offload_source",
                mode,
                "--conditioning.factory.serial_offload_cache_dir",
                cache_directory,
            ]
        else:
            pass
        try:
            with measured_server(
                command,
                output_directory=tmp_path,
                mode=mode,
                port=port,
                compilation=compilation,
                memory_context=memory_context,
            ) as measurements:
                if compilation:
                    assert "done compiler function inductor" in log_path.read_text()
                else:
                    pass
                if graphs:
                    captures = re.findall(
                        r"AuK DiT step graphs: captured (\d+)",
                        log_path.read_text(),
                    )
                    assert captures and int(captures[-1]) == len(
                        json.loads(capture_shapes)
                    ), "Accelerated test requires every declared graph to be captured"
                else:
                    pass

                if mode == "mmap" and os.environ.get("AUK_OFFLOAD_COLD_WAKE"):
                    evicted_bytes = 0
                    for shard_path in Path(cache_directory).glob("*/*.safetensors"):
                        with shard_path.open("rb") as shard:
                            os.posix_fadvise(
                                shard.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                            )
                        evicted_bytes += shard_path.stat().st_size
                    assert evicted_bytes > 0
                    memory_context[mode]["cold_wake_advised_file_bytes"] = evicted_bytes
                else:
                    pass

                generate = partial(
                    generate_audio,
                    port=port,
                    reference_audio=reference_audio,
                    latencies_seconds=measurements.latencies_seconds,
                )
                for case in ("plain", "clone", "edit"):
                    waveform = generate(case)
                    (tmp_path / f"{mode}-{case}.wav").write_bytes(waveform)
                    if mode == "resident":
                        expected_audio[case] = waveform
                    else:
                        np.testing.assert_array_equal(
                            soundfile.read(io.BytesIO(waveform))[0],
                            soundfile.read(io.BytesIO(expected_audio[case]))[0],
                            err_msg=f"{mode} {case} differs from resident output",
                        )
                    for _ in range(REPEAT_REQUESTS):
                        log_offset_bytes = log_path.stat().st_size
                        assert generate(case) == waveform
                        if mode != "resident":
                            with log_path.open("rb") as log:
                                log.seek(log_offset_bytes)
                                transitions = re.findall(
                                    r"Stage residency (auk_\w+) -> (host|gpu)",
                                    log.read().decode(),
                                )
                            assert transitions == [
                                ("auk_vae", "host"),
                                ("auk_conditioning", "gpu"),
                                ("auk_conditioning", "host"),
                                ("auk_dit", "gpu"),
                                ("auk_dit", "host"),
                                ("auk_vae", "gpu"),
                            ], f"Unexpected residency transitions: {mode} {case}"
                        else:
                            pass
                    if mode != "resident":
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            concurrent = [
                                executor.submit(generate, case) for _ in range(2)
                            ]
                            assert all(
                                request.result() == waveform for request in concurrent
                            )
                    else:
                        pass
                results[mode] = {
                    "checkpoint": checkpoint,
                    "accelerated": int(compilation and graphs),
                    "compilation": int(compilation),
                    "graphs": int(graphs),
                    "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "startup_seconds": measurements.startup_seconds,
                    "latencies_seconds": measurements.latencies_seconds,
                    "plain_sha256": hashlib.sha256(expected_audio["plain"]).hexdigest(),
                }
            for column, name in enumerate(
                ("gpu", "host_rss", "anonymous_rss", "cgroup"), start=1
            ):
                results[mode][f"sampled_{name}_peak_bytes"] = max(
                    sample[column] for sample in measurements.samples
                )
        finally:
            (tmp_path / "results.json").write_text(json.dumps(results, indent=2))


@pytest.mark.accelerator
@pytest.mark.parametrize("source", ["ram", "mmap"])
def test_auk_serial_offload_accelerated_recovery(
    tmp_path: Path, source: Literal["ram", "mmap"]
) -> None:
    """Cancellation after real Graph replay and long requests must preserve the next output."""
    if not os.environ.get("AUK_OFFLOAD_CHECKPOINT"):
        pytest.skip("Set AUK_OFFLOAD_CHECKPOINT to run the real-model recovery test")
    elif not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    log_path = tmp_path / f"{source}-recovery.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from tests.test_model.auk_offload_helpers import "
                "run_recovery_and_long_requests; "
                "run_recovery_and_long_requests(sys.argv[1], sys.argv[2])",
                source,
                str(tmp_path),
            ],
            env=offload_environment({**os.environ, "TORCH_LOGS": "dynamo"}),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    assert process.returncode == 0, log_path.read_text()
    log_text = log_path.read_text()
    assert "done compiler function inductor" in log_text
    assert re.findall(r"AuK DiT step graphs: captured (\d+)", log_text) == ["2"]
