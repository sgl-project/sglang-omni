# SPDX-License-Identifier: Apache-2.0
"""Opt-in NPU TTS serving regressions, independent of GPU speed baselines."""

from __future__ import annotations

import array
import io
import json
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.environ.get("OMNI_RUN_NPU_TESTS") != "1",
    reason="Set OMNI_RUN_NPU_TESTS=1 with a reserved NPU and local model",
)


def _validate_pcm(pcm: bytes, sample_rate: int = 24000) -> float:
    assert pcm and len(pcm) % 2 == 0, "Empty or truncated PCM16 output"
    samples = array.array("h")
    samples.frombytes(pcm)
    assert any(samples), "All-silent output"
    duration = len(samples) / sample_rate
    assert 0.2 <= duration <= 30, f"Unexpected short-prompt duration: {duration}"
    return duration


def _validate_wav(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1,
            2,
            24000,
        )
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
        assert len(pcm) == frames * 2, "Truncated WAV payload"
    return _validate_pcm(pcm)


@pytest.fixture(scope="module")
def tts_server(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    from benchmarks.benchmarker.utils import managed_omni_server
    from sglang_omni.platforms import current_platform
    from tests.test_model.omni_router_utils import _find_available_port_range

    assert current_platform.is_npu() and torch.npu.is_available()
    assert torch.npu.device_count() == 1, "Reserve one visible NPU"
    model = Path(os.environ["OMNI_NPU_TTS_MODEL"]).resolve()
    config = Path(os.environ["OMNI_NPU_TTS_CONFIG"]).resolve()
    assert (model / "config.json").is_file(), "A complete local model is required"
    assert (model / "speech_tokenizer").is_dir(), "Include speech_tokenizer weights"
    assert config.is_file(), "Supply the model's NPU serving config"
    task = os.environ.get("OMNI_NPU_TTS_TASK", "CustomVoice")
    assert task in {"CustomVoice", "Base", "VoiceDesign"}
    configured_port = os.environ.get("OMNI_NPU_TTS_PORT")
    port = int(configured_port) if configured_port else _find_available_port_range(1)
    output = Path(
        os.environ.get("OMNI_NPU_TTS_OUTPUT", str(tmp_path_factory.mktemp("npu-tts")))
    )
    output.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    with managed_omni_server(
        model_path=str(model),
        server_config=str(config),
        port=port,
        host="127.0.0.1",
        log_file=output / "server.log",
        # The shared default cleanup is CUDA-specific. The context manager still
        # stops this server's process group; the CI container owns NPU cleanup.
        wait_for_gpu_release=False,
    ):
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(base_url + "/v1/models", timeout=10)
            response.raise_for_status()
            assert response.json()["data"]
        yield base_url, task, output


def _payload(task: str, language: str = "English") -> dict:
    payload = {
        "input": (
            "Hello, this is a speech synthesis test on Ascend."
            if language == "English"
            else "你好，欢迎使用语音合成服务。"
        ),
        "language": language,
        "task_type": task,
        "seed": 123456,
        "max_new_tokens": 256,
        "response_format": "wav",
    }
    if task == "CustomVoice":
        payload["voice"] = "Ryan" if language == "English" else "Vivian"
    elif task == "VoiceDesign":
        payload["instructions"] = "A warm, clear female voice with a calm delivery."
    elif task == "Base":
        reference = Path(os.environ["OMNI_NPU_TTS_REFERENCE"]).resolve()
        assert reference.is_file()
        transcript = os.environ["OMNI_NPU_TTS_REFERENCE_TEXT"]
        assert transcript.strip()
        payload["references"] = [{"audio_path": str(reference), "text": transcript}]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return payload


def _request(base_url, payload, output, name, stream=False):
    payload = dict(payload, stream=stream, response_format="pcm" if stream else "wav")
    start = time.monotonic()
    first_byte = None
    chunks = []
    # Separate sessions also keep concurrent requests independent.
    with requests.Session() as session:
        session.trust_env = False
        with session.post(
            base_url + "/v1/audio/speech", json=payload, stream=True, timeout=(10, 600)
        ) as response:
            assert response.status_code == 200, response.text[:2000]
            if stream:
                assert response.headers.get("x-sample-rate") == "24000"
                assert response.headers.get("x-channels") == "1"
                assert response.headers.get("x-bit-depth") == "16"
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    if first_byte is None:
                        first_byte = time.monotonic() - start
                    chunks.append(chunk)
    audio = b"".join(chunks)
    (output / f"{name}.{'pcm' if stream else 'wav'}").write_bytes(audio)
    duration = _validate_pcm(audio) if stream else _validate_wav(audio)
    (output / f"{name}.json").write_text(
        json.dumps(
            {
                "request": payload,
                "duration_s": duration,
                "elapsed_s": time.monotonic() - start,
                "first_byte_s": first_byte,
                "received_chunks": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@pytest.mark.parametrize("language", ["English", "Chinese"])
def test_nonstream_speech(tts_server, language):
    base_url, task, output = tts_server
    _request(base_url, _payload(task, language), output, language.lower())


def test_stream_speech(tts_server):
    base_url, task, output = tts_server
    _request(base_url, _payload(task), output, "stream", stream=True)


def test_concurrent_requests(tts_server):
    base_url, task, output = tts_server
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_request, base_url, _payload(task), output, f"concurrent-{i}")
            for i in range(2)
        ]
        for future in futures:
            future.result()


def test_valid_request_after_rejection(tts_server):
    base_url, task, output = tts_server
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(base_url + "/v1/audio/speech", json={}, timeout=10)
        assert response.status_code in {400, 422}, response.text
    _request(base_url, _payload(task), output, "after-rejection")
