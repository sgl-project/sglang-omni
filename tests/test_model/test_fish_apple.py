# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP checks against an already running Fish Apple server.

FISH_APPLE_URL=http://127.0.0.1:18170 pytest tests/test_model/test_fish_apple.py -q
"""

from __future__ import annotations

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests
import soundfile as sf

URL = os.environ.get("FISH_APPLE_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(
    not URL, reason="set FISH_APPLE_URL for live Apple tests"
)
TEXT = "Hello! This is a local speech test."


def speech(**kwargs):
    return requests.post(
        f"{URL}/v1/audio/speech",
        json={"input": TEXT, "max_new_tokens": 100, "seed": 42, **kwargs},
        timeout=180,
    )


def waveform(response):
    assert response.status_code == 200, (
        response.text[:500] if response.status_code != 200 else ""
    )
    audio, sr = sf.read(io.BytesIO(response.content))
    assert sr == 44100
    assert audio.ndim == 1 and 0.25 < len(audio) / sr < 10
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-3
    return audio, sr


@pytest.fixture(scope="module")
def baseline():
    response = speech()
    waveform(response)
    return response


def test_seeded_requests_repeat(baseline):
    repeated = speech()
    audio, _ = waveform(repeated)
    original, _ = waveform(baseline)
    np.testing.assert_array_equal(audio, original)


def test_reference_audio_cloning(baseline):
    response = speech(
        input="Welcome back. The voice reference is ready.",
        references=[
            {
                "data": base64.b64encode(baseline.content).decode("ascii"),
                "media_type": "audio/wav",
                "text": TEXT,
            }
        ],
    )
    waveform(response)


def test_streaming_pcm_is_bounded_and_complete(baseline):
    chunks = []
    with requests.post(
        f"{URL}/v1/audio/speech",
        json={
            "input": TEXT,
            "max_new_tokens": 100,
            "seed": 42,
            "stream": True,
            "response_format": "pcm",
        },
        stream=True,
        timeout=180,
    ) as response:
        assert response.status_code == 200
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                chunks.append(chunk)
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    original, sr = waveform(baseline)
    assert len(chunks) >= 2
    assert abs(len(pcm) - len(original)) <= sr * 0.1
    assert np.max(np.abs(pcm.astype(np.int32))) > 32


def test_concurrent_requests_queue_and_complete():
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda text: speech(input=text), ["Good morning.", "Good evening."]
            )
        )
    for response in responses:
        waveform(response)


def test_disconnect_recovers():
    with requests.post(
        f"{URL}/v1/audio/speech",
        json={
            "input": "This is a longer sentence. " * 12,
            "max_new_tokens": 200,
            "stream": True,
            "response_format": "pcm",
            "seed": 9,
        },
        stream=True,
        timeout=180,
    ) as response:
        assert response.status_code == 200
        assert next(response.iter_content(chunk_size=4096))
    deadline = time.monotonic() + 20
    while requests.get(f"{URL}/health", timeout=5).json()["pending_completions"]:
        assert time.monotonic() < deadline
        time.sleep(0.1)
    waveform(speech(input="Ready again."))


def test_invalid_sampling_leaves_server_healthy():
    response = speech(top_k=31)
    assert response.status_code == 400, response.text
    requests.get(f"{URL}/health", timeout=10).raise_for_status()
