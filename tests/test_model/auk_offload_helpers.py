# SPDX-License-Identifier: Apache-2.0
"""Server measurements and audio requests for real AuK offload regression tests."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Literal
from unittest.mock import patch
from uuid import UUID

import numpy as np
import psutil
import pynvml
import pytest
import requests
import soundfile
import torch

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from sglang_omni.models.auk.offload_scheduler import AuKOffloadScheduler
from sglang_omni.models.auk.request_builders import preprocess_auk_payload
from sglang_omni.models.auk.stages import (
    create_auk_engine_executor,
    create_conditioning_executor,
    create_decode_executor,
    create_preprocessing_executor,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.protocol import (
    CreateSpeechRequest,
    GenerateResponse,
    RolloutGenerateRequest,
    RolloutSamplingParams,
)

STARTUP_TIMEOUT_SECONDS = 1800
REQUEST_TIMEOUT_SECONDS = 600
MEMORY_SAMPLE_SECONDS = 0.05


@dataclass(kw_only=True)
class ServerMeasurements:
    startup_seconds: float = 0.0
    latencies_seconds: list[float] = field(default_factory=list)
    samples: list[tuple[float, int, int, int, int]] = field(default_factory=list)


@contextmanager
def measured_server(
    command: list[str],
    *,
    output_directory: Path,
    mode: str,
    port: int,
    compilation: bool,
    memory_context: dict[str, dict[str, str | int]],
) -> Generator[ServerMeasurements, None, None]:
    """Measure startup through shutdown and persist samples even on request failure."""
    cgroup_directory = Path(os.environ.get("AUK_OFFLOAD_CGROUP_DIR", "/sys/fs/cgroup"))
    context = memory_context[mode] = {
        filename: (cgroup_directory / filename).read_text()
        for filename in ("memory.max", "memory.swap.max", "memory.events")
    }
    if os.environ.get("AUK_OFFLOAD_REQUIRE_32G_LIMIT"):
        assert context["memory.max"] != "max\n"
        assert int(context["memory.max"]) <= 32 * 1024**3
        assert int(context["memory.swap.max"]) == 0
    else:
        pass
    measurements = ServerMeasurements()
    monitor_errors: list[Exception] = []
    stop_monitor = Event()
    started_seconds = time.monotonic()
    pynvml.nvmlInit()
    try:
        device_uuid = UUID(bytes=bytes(torch.cuda.get_device_properties(0).uuid.bytes))
        gpu = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{device_uuid}")

        def monitor() -> None:
            try:
                while not stop_monitor.is_set():
                    host_rss_bytes = 0
                    anonymous_rss_bytes = 0
                    for process in psutil.Process().children(recursive=True):
                        try:
                            host_rss_bytes += process.memory_info().rss
                            status = Path(f"/proc/{process.pid}/status").read_text()
                            match = re.search(
                                r"^RssAnon:\s+(\d+) kB", status, re.MULTILINE
                            )
                            anonymous_rss_bytes += int(match[1]) * 1024 if match else 0
                        except (
                            psutil.NoSuchProcess,
                            FileNotFoundError,
                            ProcessLookupError,
                        ):
                            # note (guozhihao-224): Children may exit between process enumeration and sampling.
                            pass
                    measurements.samples.append(
                        (
                            time.monotonic() - started_seconds,
                            pynvml.nvmlDeviceGetMemoryInfo(gpu).used,
                            host_rss_bytes,
                            anonymous_rss_bytes,
                            int((cgroup_directory / "memory.current").read_text()),
                        )
                    )
                    stop_monitor.wait(MEMORY_SAMPLE_SECONDS)
            except Exception as failure:
                monitor_errors.append(failure)

        monitor_thread = Thread(target=monitor, daemon=True)
        monitor_thread.start()
        server = None
        try:
            server = start_server_from_cmd(
                command,
                output_directory / f"{mode}.log",
                port,
                timeout=STARTUP_TIMEOUT_SECONDS,
                strip_proxy=True,
                env={"TORCH_LOGS": "dynamo"} if compilation else None,
            )
            measurements.startup_seconds = time.monotonic() - started_seconds
            yield measurements
        finally:
            try:
                if server is not None:
                    stop_server(server)
                else:
                    pass
            finally:
                stop_monitor.set()
                monitor_thread.join(timeout=10)
                (output_directory / f"{mode}-memory.json").write_text(
                    json.dumps(measurements.samples)
                )
                context["memory.events.after"] = (
                    cgroup_directory / "memory.events"
                ).read_text()
                (output_directory / "memory-context.json").write_text(
                    json.dumps(memory_context, indent=2)
                )
            assert not monitor_thread.is_alive()
            assert not monitor_errors, monitor_errors
            assert measurements.samples, "Memory monitoring produced no samples"
            before_events = dict(
                line.split() for line in str(context["memory.events"]).splitlines()
            )
            after_events = dict(
                line.split()
                for line in str(context["memory.events.after"]).splitlines()
            )
            assert after_events["oom"] == before_events["oom"]
            assert after_events["oom_kill"] == before_events["oom_kill"]
    finally:
        pynvml.nvmlShutdown()


def generate_audio(
    case: Literal["plain", "clone", "edit"],
    *,
    port: int,
    reference_audio: str,
    latencies_seconds: list[float],
) -> bytes:
    if case == "edit":
        endpoint = "/generate"
        request = RolloutGenerateRequest(
            model="auk",
            prompt="Remove the background noise.",
            metadata={"tts_params": {"ref_audio": reference_audio}},
            output_modalities=["audio"],
            return_logprob=False,
            sampling_params=RolloutSamplingParams(seed=42),
            stage_params={"auk_engine": {"gen_seconds": 3.0}},
        )
    else:
        endpoint = "/v1/audio/speech"
        request = CreateSpeechRequest(
            model="auk",
            input="Welcome home. The weather is lovely today.",
            seed=42,
            response_format="wav",
            stage_params={"auk_engine": {"gen_seconds": 3.0}},
            ref_audio=reference_audio if case == "clone" else None,
            instructions="A clear, natural voice.",
        )
    before_seconds = time.perf_counter()
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"http://127.0.0.1:{port}{endpoint}",
            json=request.model_dump(exclude_none=True),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    latencies_seconds.append(time.perf_counter() - before_seconds)
    assert response.status_code == 200, response.text
    if case == "edit":
        result = GenerateResponse.model_validate_json(response.content)
        assert result.audio is not None and result.audio.data is not None
        assert result.audio.format == "wav"
        audio = base64.b64decode(result.audio.data, validate=True)
    else:
        audio = response.content
    waveform, sample_rate = soundfile.read(io.BytesIO(audio))
    assert sample_rate == 24000
    assert waveform.size > 0 and np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 0
    assert abs(waveform.shape[0] / sample_rate - 3.0) <= 0.02
    return audio


def run_recovery_and_long_requests(source: Literal["ram", "mmap"], output: str) -> None:
    """Exercise the real stages in a fresh process with the allocator preloaded."""
    logging.basicConfig(level=logging.INFO)
    checkpoint = os.environ["AUK_OFFLOAD_CHECKPOINT"]
    output_directory = Path(output)
    create_preprocessing_executor(checkpoint)
    conditioning = create_conditioning_executor(
        checkpoint,
        device="cuda",
        gpu_id=0,
        max_batch_size=1,
        enable_serial_offload=True,
        serial_offload_source=source,
        serial_offload_cache_dir=os.environ.get(
            "AUK_OFFLOAD_CACHE_DIR", str(output_directory / "weights")
        ),
    )
    engine = create_auk_engine_executor(
        checkpoint,
        device="cuda",
        gpu_id=0,
        max_batch_size=1,
        weight_dtype="bfloat16",
        enable_serial_offload=True,
        enable_dit_torch_compile=True,
        enable_dit_cuda_graph=True,
        dit_cuda_graph_capture_shapes=[[1, 192, 0, 192], [1, 320, 320, 384]],
    )
    decoder = create_decode_executor(
        checkpoint, device="cuda", gpu_id=0, enable_serial_offload=True
    )
    assert isinstance(conditioning, AuKOffloadScheduler)
    assert isinstance(engine, AuKOffloadScheduler)
    assert isinstance(decoder, AuKOffloadScheduler)
    coordinator = engine.coordinator
    gpu_used_bytes: list[int] = []

    def condition(request_id: str, seconds: float) -> StagePayload:
        request = OmniRequest(
            inputs={"text": "Welcome home. The weather is lovely today."},
            params={
                "seed": 42,
                "stage_params": {"auk_engine": {"gen_seconds": seconds}},
            },
            metadata={
                "task": "tts",
                "tts_params": {"seed": 42, "instructions": "A clear, natural voice."},
            },
        )
        return conditioning.execute(
            preprocess_auk_payload(
                StagePayload(request_id=request_id, request=request, data={})
            )
        )

    def generate(request_id: str, seconds: float) -> bytes:
        result = decoder.execute(engine.execute(condition(request_id, seconds)))
        assert decoder.claim_result(result, terminal=True)
        decoder.release_result(result, delivered=True)
        assert coordinator.owner_request_id is None
        assert not coordinator.abort_event.is_set()
        assert coordinator.resident_component == "vae"
        assert [
            name
            for name, residency in coordinator.residencies.items()
            if residency.resident
        ] == ["vae"]
        waveform_bytes = result.data["audio_waveform"]
        assert isinstance(waveform_bytes, bytes)
        waveform = np.frombuffer(waveform_bytes, dtype=np.float32)
        assert waveform.size == int(24000 * seconds)
        assert np.isfinite(waveform).all() and np.max(np.abs(waveform)) > 0
        soundfile.write(output_directory / f"{request_id}.wav", waveform, 24000)
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu_used_bytes.append(total_bytes - free_bytes)
        return waveform_bytes

    baseline = generate("baseline", 3.0)
    replay_count = 0
    original_replay = torch.cuda.CUDAGraph.replay

    def replay_then_cancel(graph: torch.cuda.CUDAGraph) -> None:
        nonlocal replay_count
        original_replay(graph)
        replay_count += 1
        engine.abort("cancelled")
        assert coordinator.owner_request_id == "cancelled"
        assert coordinator.is_computing

    cancelled = condition("cancelled", 3.0)
    with patch.object(torch.cuda.CUDAGraph, "replay", replay_then_cancel):
        with pytest.raises(InterruptedError, match="sampling cancelled"):
            engine.execute(cancelled)
    assert replay_count == 1
    assert coordinator.owner_request_id is None
    assert not coordinator.is_computing
    assert generate("recovered", 3.0) == baseline
    long_waveform = generate("long-30-seconds", 30.0)
    assert generate("long-30-seconds-repeat", 30.0) == long_waveform
    assert generate("after-long", 3.0) == baseline
    (output_directory / "results.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "source": source,
                "accelerated": True,
                "cancelled_after_real_gpu_steps": replay_count,
                "recovery_bit_exact": True,
                "long_seconds": 30,
                "long_repeat_bit_exact": True,
                "retained_component": coordinator.resident_component,
                "gpu_used_bytes_after_requests": gpu_used_bytes,
            },
            indent=2,
        )
    )
