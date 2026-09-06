# SPDX-License-Identifier: Apache-2.0
"""Real-weight Breeze checks against an explicitly started, dedicated server.

BREEZE_TEST_BASE_URL=http://127.0.0.1:18774 \
BREEZE_TEST_FIXTURE_DIR=/path/to/seedtts-fixtures \
python -m pytest tests/test_model/test_breeze_tts.py -v

The fixture directory contains en/samples.json and zh/samples.json entries with
ref_audio, ref_text and target_text, as produced from the pinned SeedTTS dataset.
These tests do not launch servers or clean up GPUs belonging to other jobs.
"""

import base64
import io
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

BASE_URL = os.environ.get("BREEZE_TEST_BASE_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="BREEZE_TEST_BASE_URL must name a dedicated Breeze server"
)
TEXT = {
    "en": "Hello, welcome to Breeze speech synthesis.",
    "zh": "你好，欢迎使用语音合成服务。",
}
INSTRUCTIONS = {
    "en": "A warm, clear and natural voice.",
    "zh": "声音温暖、清晰，语气自然。",
}


def _payload(language="en", mode="design", *, seed=42):
    payload = {
        "input": TEXT[language],
        "instructions": INSTRUCTIONS[language],
        "cfg_scale": 4,
        "seed": seed,
        "max_new_tokens": 160,
        "response_format": "wav",
    }
    if mode != "design":
        root = Path(os.environ["BREEZE_TEST_FIXTURE_DIR"])
        sample = json.loads((root / language / "samples.json").read_text())[0]
        payload.update(ref_audio=sample["ref_audio"], ref_text=sample["ref_text"])
    if mode == "clone":
        payload.pop("instructions")
        payload.pop("cfg_scale")
    return payload


def _session():
    session = requests.Session()
    session.trust_env = False
    return session


def _generate(payload, *, streaming=False):
    payload = dict(payload)
    if streaming:
        payload.update(stream=True, response_format="pcm")
    start = time.perf_counter()
    arrivals, parts = [], []
    with _session() as session:
        with session.post(
            BASE_URL + "/v1/audio/speech", json=payload, stream=streaming, timeout=300
        ) as response:
            assert response.status_code == 200, response.text
            headers = dict(response.headers)
            if streaming:
                assert headers["content-type"].startswith("audio/pcm")
                assert headers["x-sample-rate"] == "24000"
                assert headers["x-channels"] == "1"
                assert headers["x-bit-depth"] == "16"
                for part in response.iter_content(chunk_size=None):
                    if part:
                        parts.append(part)
                        arrivals.append(time.perf_counter() - start)
                raw = b"".join(parts)
                assert len(raw) % 2 == 0
                audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            else:
                assert headers["content-type"].startswith("audio/wav")
                audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
                assert rate == 24000
    assert audio.ndim == 1
    assert len(audio) > 2400
    assert np.isfinite(audio).all()
    assert float(np.sqrt(np.mean(audio**2))) > 1e-4
    assert float(np.abs(audio).max()) <= 1
    return audio, headers, arrivals, time.perf_counter() - start


def _assert_idle(timeout=30):
    deadline = time.monotonic() + timeout
    with _session() as session:
        while time.monotonic() < deadline:
            health = session.get(BASE_URL + "/health", timeout=10).json()
            info = session.post(
                BASE_URL + "/model_info",
                json={"stages": ["tts_engine"], "timeout_s": 5},
                timeout=10,
            ).json()
            data = info["results"][0]["data"]
            if (
                health["total_requests"] == 0
                and health["pending_completions"] == 0
                and all(
                    data[key] == 0
                    for key in (
                        "running_batch_size",
                        "waiting_queue_size",
                        "request_build_pending",
                        "request_admission_pending",
                        "request_build_backlog",
                    )
                )
            ):
                return
            time.sleep(0.1)
    pytest.fail(f"Request state did not drain: health={health}, engine={data}")


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("mode", ["design", "clone", "direction"])
@pytest.mark.parametrize("streaming", [False, True])
def test_synthesis_modes(language, mode, streaming, tmp_path):
    audio, headers, arrivals, wall = _generate(
        _payload(language, mode), streaming=streaming
    )
    sf.write(tmp_path / "output.wav", audio, 24000)
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "headers": headers,
                "arrival_s": arrivals,
                "wall_s": wall,
                "duration_s": len(audio) / 24000,
            },
            indent=2,
        )
    )
    if streaming:
        assert len(arrivals) >= 2
        assert arrivals[0] < arrivals[-1]
    else:
        assert headers["x-finish-reason"] == "stop"
    _assert_idle()


@pytest.mark.parametrize("language", ["en", "zh"])
def test_streaming_and_offline_preserve_all_generated_samples(language):
    payload = _payload(language)
    offline, _, _, _ = _generate(payload)
    streaming, _, _, _ = _generate(payload, streaming=True)
    assert len(streaming) == len(offline)
    # Stateful and offline codec paths may round slightly differently, but
    # must not lose/replay frames or decode a different utterance.
    assert float(np.mean(np.abs(streaming - offline))) < 0.002


def test_seeded_requests_are_isolated_when_queued_concurrently():
    payloads = [_payload("en" if i % 2 == 0 else "zh", seed=100 + i) for i in range(4)]
    expected = [_generate(payload)[0] for payload in payloads]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda payload: _generate(payload)[0], payloads))
    for first, second in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(first, second)
    _assert_idle()


def test_disconnect_releases_cfg_pair_and_next_request_still_succeeds():
    payload = _payload()
    payload.update(
        input="Please read this slowly. " * 16,
        max_new_tokens=750,
        stream=True,
        response_format="pcm",
    )
    with _session() as session:
        with session.post(
            BASE_URL + "/v1/audio/speech", json=payload, stream=True, timeout=300
        ) as response:
            assert response.status_code == 200
            assert next(response.iter_content(chunk_size=None))
            # Close before completion, forcing the HTTP disconnect/abort path.
    _assert_idle()
    _generate(_payload())
    _assert_idle()


@pytest.mark.parametrize(
    "override",
    [
        {"cfg_scale": -1},
        {"temperature": -1},
        {"input": ""},
        {"speed": 2},
        {"language": "Japanese"},
        {"input": "a " * 1500},
    ],
)
def test_invalid_requests_do_not_poison_worker(override):
    with _session() as session:
        response = session.post(
            BASE_URL + "/v1/audio/speech", json={**_payload(), **override}, timeout=30
        )
    assert response.status_code in (400, 422), response.text
    _assert_idle()


@pytest.mark.parametrize("response_format", ["mp3", "flac", "opus"])
def test_nonstreaming_audio_formats(response_format):
    with _session() as session:
        response = session.post(
            BASE_URL + "/v1/audio/speech",
            json={**_payload(), "response_format": response_format},
            timeout=300,
        )
    assert response.status_code == 200, response.text
    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    assert rate in (24000, 48000)
    assert audio.ndim == 1 and len(audio) > rate // 10
    assert np.isfinite(audio).all() and np.abs(audio).max() > 1e-4
    _assert_idle()


def test_flat_inline_and_references_array_condition_on_the_same_audio():
    payload = _payload(mode="clone")
    expected = _generate(payload)[0]
    inline = (
        "data:audio/wav;base64,"
        + base64.b64encode(Path(payload["ref_audio"]).read_bytes()).decode()
    )
    flat = {**payload, "ref_audio": inline}
    np.testing.assert_array_equal(expected, _generate(flat)[0])
    reference = {"audio_path": inline, "text": payload.pop("ref_text")}
    payload.pop("ref_audio")
    np.testing.assert_array_equal(
        expected, _generate({**payload, "references": [reference]})[0]
    )
    _assert_idle()


def test_uploaded_voice_reference_lifecycle():
    payload = _payload(mode="clone")
    name = "breeze_test_" + uuid.uuid4().hex
    with _session() as session:
        with Path(payload["ref_audio"]).open("rb") as reference:
            response = session.post(
                BASE_URL + "/v1/audio/voices",
                data={
                    "name": name,
                    "consent": "test-only-seedtts-evaluation-fixture",
                    "ref_text": payload["ref_text"],
                },
                files={"audio_sample": ("reference.wav", reference, "audio/wav")},
                timeout=30,
            )
        assert response.status_code == 200, response.text
        try:
            listing = session.get(
                BASE_URL + "/v1/audio/voices", params={"names_only": "true"}, timeout=30
            )
            assert name in listing.json()["uploaded_voice_names"]
            payload.pop("ref_audio")
            payload.pop("ref_text")
            _generate({**payload, "voice": name})
        finally:
            deleted = session.delete(BASE_URL + "/v1/audio/voices/" + name, timeout=30)
            assert deleted.status_code == 200, deleted.text
    _assert_idle()


@pytest.mark.parametrize("override", [{"temperature": 0}, {"cfg_scale": 0}])
def test_greedy_and_zero_guidance_controls(override):
    _generate({**_payload(), **override})
    _assert_idle()


def test_single_frame_limit_flushes_short_initial_stream():
    payload = {
        **_payload(),
        "max_new_tokens": 1,
        "stream": True,
        "response_format": "pcm",
    }
    with _session() as session:
        response = session.post(BASE_URL + "/v1/audio/speech", json=payload, timeout=60)
    assert response.status_code == 200, response.text
    # One codec frame is 1920 mono samples. It must be emitted even though
    # the normal initial streaming stride waits for two frames.
    assert len(response.content) == 1920 * 2
    _assert_idle()


def test_long_form_prompt_finishes_without_truncation(tmp_path):
    payload = _payload()
    payload.update(
        input="Today we are testing a speech synthesis system. It should speak clearly and naturally, preserve every word, and keep the same voice throughout the entire message. Thank you for listening to this longer example.",
        max_new_tokens=300,
    )
    audio, headers, _, wall = _generate(payload)
    assert headers["x-finish-reason"] == "stop"
    assert len(audio) / 24000 > 8
    streamed, _, _, _ = _generate(payload, streaming=True)
    assert len(streamed) == len(audio)
    assert float(np.mean(np.abs(streamed - audio))) < 0.002
    sf.write(tmp_path / "long-form.wav", audio, 24000)
    sf.write(tmp_path / "long-form-stream.wav", streamed, 24000)
    (tmp_path / "long-form.json").write_text(
        json.dumps(
            {
                "sample_id": "long-form-en",
                "target_text": payload["input"],
                "wav_path": str(tmp_path / "long-form.wav"),
                "is_success": True,
                "latency_s": wall,
                "audio_duration_s": len(audio) / 24000,
            },
            indent=2,
        )
    )
    _assert_idle()
