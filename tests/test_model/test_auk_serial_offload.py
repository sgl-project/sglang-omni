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

from sglang_omni.models.auk.config import AuKPipelineConfig
from sglang_omni.models.auk.step_cuda_graph import DEFAULT_CAPTURE_SHAPES
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
    """Offloading with compilation and CUDA Graphs must preserve seeded audio."""
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
    capture_shapes = os.environ.get("AUK_OFFLOAD_CAPTURE_SHAPES")
    expected_audio: dict[str, bytes] = {}
    results: dict[str, dict[str, int | float | str | list[float]]] = {}
    cache_directory = os.environ.get("AUK_OFFLOAD_CACHE_DIR", str(tmp_path / "weights"))
    server_script = (
        "import logging, sys; "
        "from sglang_omni.models.auk.config import AuKPipelineConfig; "
        "from sglang_omni.serve.launcher import launch_server; "
        "logging.basicConfig(level=logging.INFO); "
        "launch_server(AuKPipelineConfig.model_validate_json(sys.argv[1]), "
        "host='127.0.0.1', port=int(sys.argv[2]), model_name='auk')"
    )
    for mode in selected_modes:
        log_path = tmp_path / f"{mode}.log"
        port = find_available_port(host="127.0.0.1")
        configuration = AuKPipelineConfig(model_path=checkpoint)
        for stage in configuration.stages[1:]:
            stage.factory.max_batch_size = 1
        conditioning = configuration.stage_named("conditioning")
        conditioning.factory_path = "tests.test_model.auk_offload_helpers.create_deterministic_conditioning_executor"
        encoder_checkpoint = os.environ.get("AUK_QWEN_CHECKPOINT")
        if encoder_checkpoint is not None:
            conditioning.factory.text_encoder_path = encoder_checkpoint
        else:
            pass
        engine = configuration.stage_named("auk_engine")
        engine.factory.enable_dit_torch_compile = compilation
        engine.factory.enable_dit_cuda_graph = graphs
        if capture_shapes is not None:
            engine.factory.dit_cuda_graph_capture_shapes = json.loads(capture_shapes)
            expected_graph_count = len(json.loads(capture_shapes))
        else:
            expected_graph_count = sum(
                mode == "resident" or shape.batch == 1
                for shape in DEFAULT_CAPTURE_SHAPES
            )
        if mode != "resident":
            conditioning.factory.serial_offload_source = mode
            conditioning.factory.serial_offload_cache_dir = cache_directory
            configuration = configuration.with_serial_offload(
                frozenset({"conditioning", "dit", "vae"})
            )
        else:
            pass
        command = [
            sys.executable,
            "-c",
            server_script,
            configuration.model_dump_json(serialize_as_any=True),
            str(port),
        ]
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
                    captured_graph_count = int(captures[-1]) if captures else 0
                    assert (
                        captured_graph_count == expected_graph_count
                    ), "Accelerated test requires every declared graph to be captured"
                else:
                    captured_graph_count = 0

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
                    "cudnn_deterministic": True,
                    "sdpa_backends": "efficient,math",
                    "captured_graph_count": captured_graph_count,
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
    recovery_script = (
        "import sys; from tests.test_model.auk_offload_helpers import "
        "run_recovery_and_long_requests; "
        "run_recovery_and_long_requests(sys.argv[1], sys.argv[2])"
    )
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                recovery_script,
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
