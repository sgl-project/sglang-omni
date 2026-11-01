# SPDX-License-Identifier: Apache-2.0
"""AuK and AuK-Flash speech served from local checkpoints on an Intel XPU."""

from __future__ import annotations

import base64
import io
import os
import sys
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pytest
import requests
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

from benchmarks.benchmarker.utils import (
    server_log_file,
    start_server_from_cmd,
    stop_server,
)
from sglang_omni.models.auk.constants import DEFAULT_TEXT_ENCODER, SAMPLE_RATE
from sglang_omni.platforms import current_platform
from sglang_omni.utils.connection import find_available_port
from sglang_omni.utils.json import JsonValue

pytestmark = pytest.mark.benchmark

CHECKPOINTS = {
    "AuK": ("AUK_CHECKPOINT", "tencent/AuK"),
    "AuK-Flash": ("AUK_FLASH_CHECKPOINT", "tencent/AuK-Flash"),
}
TEXT_ENCODER_ENV = "AUK_TEXT_ENCODER"
HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 1800
REQUEST_TIMEOUT_SECONDS = 600
GEN_SECONDS = 2
REFERENCE_AUDIO = Path(__file__).resolve().parents[1] / "data" / "query_to_cars.wav"
REFERENCE_TEXT = "How many cars are there in the picture?"
# Far below the quietest output measured on Arc Pro B60 (RMS 0.031).
SILENCE_RMS = 1e-3


class AuKServer(NamedTuple):
    base_url: str
    session: requests.Session


def local_checkpoint(model_path: str) -> str:
    """A local directory for model_path, which may be a path or a cached repo id."""
    if os.path.isdir(model_path):
        return model_path
    else:
        pass
    try:
        return snapshot_download(model_path, local_files_only=True)
    except (HFValidationError, LocalEntryNotFoundError) as error:
        pytest.fail(
            f"WEIGHTS_UNAVAILABLE: {model_path} is neither a directory nor a "
            f"cached Hugging Face snapshot: {error}"
        )


@pytest.fixture(scope="module", params=sorted(CHECKPOINTS))
def server(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[AuKServer]:
    if current_platform.device_type != "xpu":
        pytest.skip(
            f"AuK XPU CI needs an Intel XPU, found {current_platform.device_type}"
        )
    else:
        pass
    checkpoint_env, repo_id = CHECKPOINTS[request.param]
    checkpoint = local_checkpoint(os.environ.get(checkpoint_env, repo_id))
    text_encoder = local_checkpoint(
        os.environ.get(TEXT_ENCODER_ENV, DEFAULT_TEXT_ENCODER)
    )
    port = find_available_port(HOST)
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        checkpoint,
        "--conditioning.factory.text_encoder_path",
        text_encoder,
        "--host",
        HOST,
        "--port",
        str(port),
    ]
    try:
        process = start_server_from_cmd(
            command,
            server_log_file(tmp_path_factory, prefix=f"{request.param}_server"),
            port,
            timeout=STARTUP_TIMEOUT_SECONDS,
            env={
                "HF_HUB_OFFLINE": "1",
                "SGLANG_OMNI_STARTUP_TIMEOUT": str(STARTUP_TIMEOUT_SECONDS),
            },
            strip_proxy=True,
        )
    except (RuntimeError, TimeoutError) as error:
        pytest.fail(f"SERVER_STARTUP: {request.param} did not become healthy: {error}")
    session = requests.Session()
    session.trust_env = False
    yield AuKServer(f"http://{HOST}:{port}", session)
    session.close()
    stop_server(process)


def speech(server: AuKServer, payload: dict[str, JsonValue]) -> requests.Response:
    return server.session.post(
        f"{server.base_url}/v1/audio/speech",
        json={"response_format": "wav", **payload},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


@pytest.mark.parametrize("voice", ["instruct", "clone"])
def test_speech_is_a_24khz_wav_of_the_requested_length(
    server: AuKServer, voice: Literal["instruct", "clone"]
) -> None:
    payload: dict[str, JsonValue] = {
        "input": "Welcome home.",
        "seed": 1234,
        "stage_params": {"auk_engine": {"gen_seconds": GEN_SECONDS}},
    }
    if voice == "clone":
        encoded = base64.b64encode(REFERENCE_AUDIO.read_bytes()).decode()
        payload["ref_audio"] = f"data:audio/wav;base64,{encoded}"
        payload["ref_text"] = REFERENCE_TEXT
    else:
        payload["instructions"] = "A warm, relaxed voice."
    response = speech(server, payload)
    assert (
        response.status_code == 200
    ), f"INFERENCE: HTTP {response.status_code}: {response.text[:500]}"
    with wave.open(io.BytesIO(response.content), "rb") as audio:
        audio_format = (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
        )
        pcm = audio.readframes(audio.getnframes())
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    assert audio_format == (SAMPLE_RATE, 1, 2), f"OUTPUT: WAV format {audio_format}"
    assert samples.size == GEN_SECONDS * SAMPLE_RATE, f"OUTPUT: {samples.size} samples"
    assert np.sqrt(np.mean(samples**2)) > SILENCE_RMS, "OUTPUT: the audio is silent"


def test_a_rejected_duration_leaves_the_server_serving(server: AuKServer) -> None:
    response = speech(
        server,
        {"input": "Welcome home.", "stage_params": {"auk_engine": {"gen_seconds": -1}}},
    )
    assert response.status_code == 400, f"HTTP {response.status_code}: {response.text}"
    assert "gen_seconds must be positive" in response.json()["error"]["message"]
    health = server.session.get(f"{server.base_url}/health", timeout=30)
    assert health.json()["status"] == "healthy"
