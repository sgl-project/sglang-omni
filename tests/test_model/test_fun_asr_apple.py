# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP validation against an already-running Fun-ASR Apple server.

FUN_ASR_APPLE_URL=http://127.0.0.1:8000 pytest tests/test_model/test_fun_asr_apple.py -q
Start the server with SGLANG_USE_MLX=0. No server is started here.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf
from scipy.signal import resample as _scipy_resample

_URL = os.environ.get("FUN_ASR_APPLE_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(
    not _URL, reason="set FUN_ASR_APPLE_URL to a running Apple server"
)
_AUDIO = Path(__file__).resolve().parents[1] / "data" / "query_to_cars.wav"


@pytest.fixture(scope="module")
def audio():
    if not _AUDIO.exists():
        pytest.skip(f"audio fixture unavailable: {_AUDIO}")
    return _AUDIO.read_bytes()


def _transcribe(audio, **params):
    return requests.post(
        f"{_URL}/v1/audio/transcriptions",
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"language": "en", **params},
        timeout=120,
    )


def _wav(seconds):
    buffer = io.BytesIO()
    sf.write(
        buffer, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV"
    )
    return buffer.getvalue()


def _encode(samples, sample_rate, fmt):
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format=fmt)
    return buffer.getvalue()


def _to_m4a(wav_bytes, output_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to encode the M4A fixture")
    # A regular MP4/M4A container needs seekable output to finalize its header.
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-f",
            "ipod",
            str(output_path),
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
    return output_path.read_bytes()


@pytest.fixture(scope="module")
def real_speech_samples(audio):
    samples, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != 16000:
        samples = _scipy_resample(samples, int(len(samples) * 16000 / sample_rate))
    return samples.astype(np.float32)


def test_json_sse_and_queued_requests_match(audio):
    response = _transcribe(audio)
    response.raise_for_status()
    expected = response.json()["text"]
    assert "cars" in expected.lower()
    streamed = _transcribe(audio, stream="true")
    streamed.raise_for_status()
    records = [
        line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")
    ]
    assert records[-1] == "[DONE]"
    events = [json.loads(record) for record in records[:-1]]
    done = [event for event in events if event["type"] == "transcript.text.done"]
    assert len(done) == 1
    assert done[0]["text"] == expected
    assert (
        "".join(
            event["delta"]
            for event in events
            if event["type"] == "transcript.text.delta"
        )
        == expected
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(_transcribe, [audio] * 4))
    for response in responses:
        response.raise_for_status()
        assert response.json()["text"] == expected


def test_invalid_requests_leave_server_healthy(audio):
    assert _transcribe(audio, temperature="0.5").status_code == 400
    assert _transcribe(_wav(30.01)).status_code == 400
    assert _transcribe(b"not an audio file").status_code == 400
    _transcribe(audio).raise_for_status()
    requests.get(f"{_URL}/health", timeout=10).raise_for_status()


def test_audio_at_duration_limit():
    _transcribe(_wav(30), max_new_tokens="16").raise_for_status()


def test_real_speech_near_thirty_seconds_uses_default_budget(real_speech_samples):
    # The duration-limit boundary test above uses silence with a reduced
    # max_new_tokens override, which never exercises the default per-duration
    # token budget (scales up to 200 tokens at 30s) against real speech. Tile
    # the real clip with small gaps to approach the 30s VAD cap without going
    # over it, and let the server pick the budget itself.
    gap = np.zeros(int(16000 * 0.3), dtype=np.float32)
    unit = np.concatenate([real_speech_samples, gap])
    target_samples = int(16000 * 29.5)
    repeats = -(-target_samples // len(unit))
    near_thirty = np.tile(unit, repeats)[:target_samples]

    response = _transcribe(_encode(near_thirty, 16000, "WAV"))
    response.raise_for_status()
    assert "cars" in response.json()["text"].lower()


def test_mp3_upload_transcribes_correctly(audio):
    samples, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")

    mp3_response = requests.post(
        f"{_URL}/v1/audio/transcriptions",
        files={
            "file": ("audio.mp3", _encode(samples, sample_rate, "MP3"), "audio/mpeg")
        },
        data={"language": "en"},
        timeout=120,
    )
    mp3_response.raise_for_status()
    assert "cars" in mp3_response.json()["text"].lower()


def test_m4a_upload_transcribes_correctly(audio, tmp_path):
    m4a_response = requests.post(
        f"{_URL}/v1/audio/transcriptions",
        files={
            "file": ("audio.m4a", _to_m4a(audio, tmp_path / "audio.m4a"), "audio/mp4")
        },
        data={"language": "en"},
        timeout=120,
    )
    m4a_response.raise_for_status()
    assert "cars" in m4a_response.json()["text"].lower()


@pytest.mark.parametrize("sample_rate", [8000, 22050, 44100, 48000])
def test_varied_sample_rates_transcribe_correctly(real_speech_samples, sample_rate):
    resampled = _scipy_resample(
        real_speech_samples, int(len(real_speech_samples) * sample_rate / 16000)
    ).astype(np.float32)

    response = _transcribe(_encode(resampled, sample_rate, "WAV"))
    response.raise_for_status()
    assert "cars" in response.json()["text"].lower()


def test_silence_and_noise_do_not_hallucinate_or_break_the_server():
    # Real ASR models can loop into repeated hallucinated phrases when fed
    # non-speech input. Bound the output size as a coarse well-behaved check,
    # and confirm the server stays healthy afterward either way.
    silence_response = _transcribe(_wav(3), max_new_tokens="32")
    silence_response.raise_for_status()
    assert len(silence_response.json()["text"]) < 200

    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)
    noise_response = _transcribe(_encode(noise, 16000, "WAV"), max_new_tokens="32")
    noise_response.raise_for_status()
    assert len(noise_response.json()["text"]) < 200

    requests.get(f"{_URL}/health", timeout=10).raise_for_status()


def test_disconnect_releases_request_and_next_request_succeeds(audio):
    with requests.post(
        f"{_URL}/v1/audio/transcriptions",
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"language": "en", "stream": "true"},
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if b"transcript.text.delta" in line:
                break
    deadline = time.monotonic() + 10
    while True:
        health = requests.get(f"{_URL}/health", timeout=5).json()
        if health["pending_completions"] == 0:
            break
        assert time.monotonic() < deadline, health
        time.sleep(0.05)
    recovered = _transcribe(audio)
    recovered.raise_for_status()
    assert "cars" in recovered.json()["text"].lower()
