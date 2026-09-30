# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP contract tests for a running native MLX AuK server."""

from __future__ import annotations

import base64
import io
import os
import wave
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest

pytestmark = pytest.mark.accelerator


@pytest.fixture(scope="module")
def http_client() -> Iterator[httpx.Client]:
    url = os.environ.get("AUK_MLX_SERVER_URL")
    if not url:
        pytest.skip("Set AUK_MLX_SERVER_URL to an already running native AuK server")
    with httpx.Client(base_url=url.rstrip("/"), timeout=1200) as client:
        yield client


@pytest.fixture(scope="module")
def reference_data_uri() -> str:
    path = Path(__file__).parents[1] / "data" / "query_to_cars.wav"
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )


def speech_payload(
    text: str, seconds: float = 3, seed: int = 1234
) -> dict[str, object]:
    return {
        "input": text,
        "seed": seed,
        "response_format": "wav",
        "stage_params": {"auk_engine": {"gen_seconds": seconds}},
    }


def validate_waveform(content: bytes, seconds: float) -> np.ndarray:
    with wave.open(io.BytesIO(content)) as audio:
        assert audio.getframerate() == 24000
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert abs(audio.getnframes() / 24000 - seconds) <= 0.021
        samples = (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                np.float32
            )
            / 32768
        )
    assert np.isfinite(samples).all()
    assert np.sqrt(np.mean(samples**2)) > 1e-5, "Generated audio is silent"
    assert np.ptp(samples) > 1e-4, "Generated audio is constant"
    return samples


@pytest.mark.parametrize(
    "text", ["Welcome home. Have a wonderful day.", "你好，欢迎回家。祝你今天愉快。"]
)
def test_http_speech_without_reference(http_client: httpx.Client, text: str) -> None:
    response = http_client.post("/v1/audio/speech", json=speech_payload(text))
    assert response.status_code == 200, response.text[:1000]
    assert response.headers["content-type"].startswith("audio/")
    validate_waveform(response.content, 3)


def test_http_cloning_is_seeded_and_repeatable(
    http_client: httpx.Client, reference_data_uri: str
) -> None:
    payload = {**speech_payload("Welcome home."), "ref_audio": reference_data_uri}
    first = http_client.post("/v1/audio/speech", json=payload)
    second = http_client.post("/v1/audio/speech", json=payload)
    for response in (first, second):
        assert response.status_code == 200, response.text[:1000]
        validate_waveform(response.content, 3)
    assert first.content == second.content


def test_http_concurrent_requests_keep_their_outputs(http_client: httpx.Client) -> None:
    payloads = [
        speech_payload("Hello there.", 2, 41),
        speech_payload("祝你今天愉快。", 3, 73),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(http_client.post, "/v1/audio/speech", json=payload)
            for payload in payloads
        ]
        responses = [future.result() for future in futures]
    outputs = []
    for response, seconds in zip(responses, (2, 3)):
        assert response.status_code == 200, response.text[:1000]
        outputs.append(validate_waveform(response.content, seconds))
    assert not np.array_equal(outputs[0], outputs[1][: len(outputs[0])])


def test_http_generate_edits_reference_audio(
    http_client: httpx.Client, reference_data_uri: str
) -> None:
    response = http_client.post(
        "/generate",
        json={
            "prompt": "Remove the background noise while preserving the speaker's voice.",
            "metadata": {"tts_params": {"ref_audio": reference_data_uri, "seed": 1234}},
            "stage_params": {"auk_engine": {"gen_seconds": 3}},
            "output_modalities": ["audio"],
            "return_logprob": False,
        },
    )
    assert response.status_code == 200, response.text[:1000]
    result = response.json()
    assert result["audio"]["format"] == "wav"
    assert result["meta_info"]["finish_reason"]["type"] in ("stop", "length")
    validate_waveform(base64.b64decode(result["audio"]["data"], validate=True), 3)


@pytest.mark.parametrize(
    "text,seconds,expected",
    [("", 3, "input"), ("Hello.", 0, "gen_seconds"), ("Hello.", -1, "gen_seconds")],
)
def test_http_invalid_speech_returns_bad_request(
    http_client: httpx.Client, text: str, seconds: float, expected: str
) -> None:
    response = http_client.post("/v1/audio/speech", json=speech_payload(text, seconds))
    assert response.status_code == 400, response.text[:1000]
    error = response.json()["error"]
    assert error["type"] == "BadRequestError"
    assert expected in error["message"]
