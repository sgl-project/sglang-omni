# SPDX-License-Identifier: Apache-2.0
"""Process-replica smoke test for the Qwen3-Omni speech pipeline.

Launches a 2-GPU deployment with thinker on GPU 0, talker_ar on GPU 1,
and one code2wav replica on each GPU. Drives audio requests through it
and asserts both code2wav replicas are spawned, registered, and selected
by admission round-robin.

Requires 2 GPUs.

Usage:
    pytest tests/test_model/test_qwen3_omni_process_replicas.py -s -x
"""

from __future__ import annotations

import ast
import base64
import io
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
import yaml

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

REQUIRED_GPUS = 2

requires_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < REQUIRED_GPUS,
    reason=f"requires {REQUIRED_GPUS} GPUs",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REPLICA_CONFIG = "examples/configs/qwen3_omni_speech_code2wav_replica2_ci.yaml"
STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 300
PCM_ATOL = 2

NUM_REQUESTS = 4
REPLICATED_PROCESSES = ("code2wav",)
REPLICA_INSTANCES = (
    "code2wav@r0",
    "code2wav@r1",
)

EQUIVALENCE_PROMPT = "Please answer briefly: what is the capital of France?"
PROMPTS = [
    EQUIVALENCE_PROMPT,
    EQUIVALENCE_PROMPT,
    "Count from one to five.",
    "Name one primary color.",
]


def _start_server(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    name: str,
    config_path: Path,
):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, name) or (
        tmp_path_factory.mktemp(name) / "server.log"
    )
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        str(config_path),
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT, tee=True)
    proc.port = port  # type: ignore[attr-defined]
    proc.log_file = log_file  # type: ignore[attr-defined]
    return proc


def _single_instance_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = yaml.safe_load((PROJECT_ROOT / REPLICA_CONFIG).read_text())
    config["name"] = "qwen3-omni-speech-code2wav-single-ci"
    config["processes"] = {}
    config_path = tmp_path_factory.mktemp("single_instance_config") / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture(scope="module")
def baseline_pcm(tmp_path_factory: pytest.TempPathFactory) -> np.ndarray:
    proc = _start_server(
        tmp_path_factory,
        name="single_instance_logs",
        config_path=_single_instance_config(tmp_path_factory),
    )
    try:
        body = _post_audio_request(proc.port, EQUIVALENCE_PROMPT)
        return _decode_wav_pcm(_audio_bytes(body, request_index=0))
    finally:
        stop_server(proc)


@pytest.fixture(scope="module")
def replica_server(
    tmp_path_factory: pytest.TempPathFactory,
    baseline_pcm: np.ndarray,
):
    proc = _start_server(
        tmp_path_factory,
        name="stage_replica_logs",
        config_path=PROJECT_ROOT / REPLICA_CONFIG,
    )
    yield proc
    stop_server(proc)


def _post_audio_request(port: int, prompt: str) -> dict:
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": 123456,
        "stream": False,
    }
    with disable_proxy():
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


def _audio_bytes(body: dict, *, request_index: int) -> bytes:
    choice = body["choices"][0]
    audio = choice["message"].get("audio") or {}
    audio_b64 = audio.get("data")
    assert audio_b64, f"request {request_index}: no audio in response: {body}"
    audio_bytes = base64.b64decode(audio_b64)
    assert len(audio_bytes) > 1000, (
        f"request {request_index}: audio payload suspiciously small "
        f"({len(audio_bytes)} bytes)"
    )
    return audio_bytes


def _decode_wav_pcm(audio_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    assert pcm.size > 0
    return pcm


def _assert_pcm_equivalent(baseline: np.ndarray, replica: np.ndarray) -> None:
    assert (
        replica.shape == baseline.shape
    ), f"PCM length differs: baseline={baseline.size}, replica={replica.size}"
    delta = np.abs(replica.astype(np.int32) - baseline.astype(np.int32))
    max_delta = int(delta.max(initial=0))
    assert (
        max_delta <= PCM_ATOL
    ), f"PCM differs by {max_delta} LSB; allowed absolute tolerance is {PCM_ATOL}"


def test_pcm_equivalence_accepts_samples_within_tolerance() -> None:
    baseline = np.array([0, 10, -10], dtype=np.int16)
    replica = np.array([2, 8, -8], dtype=np.int16)

    _assert_pcm_equivalent(baseline, replica)


def test_pcm_equivalence_rejects_samples_outside_tolerance() -> None:
    baseline = np.array([0, 10, -10], dtype=np.int16)
    replica = np.array([3, 10, -10], dtype=np.int16)

    with pytest.raises(AssertionError, match="PCM differs"):
        _assert_pcm_equivalent(baseline, replica)


@requires_gpus
def test_every_replica_serves_audio(replica_server, baseline_pcm: np.ndarray):
    port: int = replica_server.port
    log_file: Path = replica_server.log_file

    for index in range(NUM_REQUESTS):
        body = _post_audio_request(port, PROMPTS[index % len(PROMPTS)])
        audio_bytes = _audio_bytes(body, request_index=index)
        if index < 2:
            _assert_pcm_equivalent(baseline_pcm, _decode_wav_pcm(audio_bytes))

    log_text = log_file.read_text()
    missing = [name for name in REPLICA_INSTANCES if name not in log_text]
    assert not missing, (
        f"replica instances never appeared in server log: {missing}; "
        "expected both code2wav instances to be spawned and registered"
    )

    admitted = re.findall(r"bindings=(\{.*?\})", log_text)
    assert (
        len(admitted) >= NUM_REQUESTS
    ), f"expected at least {NUM_REQUESTS} admission log lines, got {len(admitted)}"
    bound: dict[str, set[int]] = {}
    for raw in admitted:
        for stage, replica_id in ast.literal_eval(raw).items():
            bound.setdefault(stage, set()).add(replica_id)
    for process_name in REPLICATED_PROCESSES:
        assert bound.get(process_name) == {
            0,
            1,
        }, (
            f"{process_name} did not round-robin across both replicas: "
            f"{bound.get(process_name)}"
        )
