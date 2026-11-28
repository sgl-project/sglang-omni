# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint HTTP serving checks of Stable Audio 3 Medium on one CUDA device."""

import concurrent.futures
import io
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import soundfile as sf

from tests.unit_test.fixtures.accelerator import require_cuda

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        not os.environ.get("STABLE_AUDIO3_MODEL_PATH"),
        reason="Set STABLE_AUDIO3_MODEL_PATH to run real-checkpoint tests",
    ),
]

SAMPLE_RATE = 44100
PROMPT = "Warm lofi piano over soft vinyl crackle, 80 BPM"
# note (ruilin): one single-chunk clip, one that pads to a longer latent bucket, and one of several decoder chunks.
DURATIONS_SECONDS = (2.0, 5.3, 14.0)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    require_cuda()
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            os.environ["STABLE_AUDIO3_MODEL_PATH"],
            "--port",
            str(port),
        ],
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            assert process.poll() is None, "Stable Audio 3 server exited during startup"
            try:
                if httpx.get(f"{base}/health", timeout=3).status_code == 200:
                    break
                else:
                    pass
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Stable Audio 3 server did not become ready")
        yield base
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=120)


def speech_request(seed: int, duration_seconds: float) -> dict[str, object]:
    return {
        "model": os.environ["STABLE_AUDIO3_MODEL_PATH"],
        "input": PROMPT,
        "seed": seed,
        "response_format": "wav",
        "stage_params": {"generation": {"duration_seconds": duration_seconds}},
    }


def synthesize(base: str, body: dict[str, object]) -> httpx.Response:
    return httpx.post(f"{base}/v1/audio/speech", json=body, timeout=600)


def test_concurrent_lengths_return_the_audio_of_one_request_at_a_time(
    server: str,
) -> None:
    bodies = [
        speech_request(seed, duration_seconds)
        for seed, duration_seconds in enumerate(DURATIONS_SECONDS, start=40)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        batched = list(pool.map(lambda body: synthesize(server, body), bodies))
    for body, response, duration_seconds in zip(
        bodies, batched, DURATIONS_SECONDS, strict=True
    ):
        assert response.status_code == 200, response.text
        audio, rate = sf.read(io.BytesIO(response.content), always_2d=True)
        assert rate == SAMPLE_RATE
        assert audio.shape == (int(duration_seconds * SAMPLE_RATE), 2)
        assert np.isfinite(audio).all() and np.abs(audio).max() > 0.01
        assert synthesize(server, body).content == response.content


def test_seed_sets_the_audio_and_invalid_requests_return_400(server: str) -> None:
    body = speech_request(42, 3.0)
    first = synthesize(server, body)
    assert first.status_code == 200, first.text
    assert synthesize(server, body).content == first.content
    assert synthesize(server, speech_request(43, 3.0)).content != first.content
    for invalid in (
        {**body, "voice": "alloy"},
        {**body, "temperature": 0.7},
        {**body, "stage_params": {"generation": {"duration_seconds": 381}}},
    ):
        assert synthesize(server, invalid).status_code == 400
    generate = httpx.post(f"{server}/generate", json={"prompt": PROMPT}, timeout=600)
    assert generate.status_code == 400
    assert synthesize(server, body).content == first.content
