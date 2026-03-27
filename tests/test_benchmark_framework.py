# SPDX-License-Identifier: Apache-2.0
"""Tests for the model-agnostic benchmark framework."""

from __future__ import annotations

import base64
import io
import json
import threading
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest

from benchmarks.adapters.request_family.speech_http import SpeechHTTPAdapter
from benchmarks.core.runner import run_from_config
from benchmarks.core.types import BenchmarkRunConfig, PreparedRequest
from benchmarks.profiles.registry import PROFILE_REGISTRY
from benchmarks.profiles.resolver import resolve_profile_selection


def _make_wav_bytes(num_frames: int = 1600, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_frames)
    return buffer.getvalue()


@contextmanager
def _serve(routes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._dispatch()

        def do_POST(self):
            self._dispatch()

        def log_message(self, format, *args):
            del format, args

        def _dispatch(self):
            key = (self.command, self.path)
            if key not in routes:
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            status_code, headers, response_body = routes[key](self, body)
            self.send_response(status_code)
            for header_name, header_value in headers.items():
                self.send_header(header_name, header_value)
            self.end_headers()
            self.wfile.write(response_body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_profile_registry_discovers_builtin_profiles() -> None:
    assert "s2_tts" in PROFILE_REGISTRY.list_profiles()


def test_resolver_uses_explicit_model_profile() -> None:
    selection = resolve_profile_selection(
        run_config=BenchmarkRunConfig(
            base_url="http://localhost:8000",
            model="s2-pro",
            model_profile="s2_tts",
            case_id="voice-cloning",
        ),
        registry=PROFILE_REGISTRY,
    )

    assert selection.model_profile_name == "s2_tts"
    assert selection.case_spec.case_id == "voice-cloning"
    assert selection.served_model_name == "s2-pro"


@pytest.mark.asyncio
async def test_speech_http_stream_adapter_parses_stream_response() -> None:
    audio_b64 = base64.b64encode(_make_wav_bytes()).decode("ascii")
    stream_payload = (
        f'data: {json.dumps({"audio": {"data": audio_b64}, "finish_reason": None})}\n\n'
        f'data: {json.dumps({"audio": None, "finish_reason": "stop", "usage": {"prompt_tokens": 3, "completion_tokens": 5, "engine_time_s": 0.5}})}\n\n'
        "data: [DONE]\n\n"
    ).encode("utf-8")

    with _serve(
        {
            ("POST", "/v1/audio/speech"): lambda _handler, body: (
                200,
                {"Content-Type": "text/event-stream"},
                stream_payload,
            )
        }
    ) as base_url:
        adapter = SpeechHTTPAdapter()
        async with aiohttp.ClientSession() as session:
            result = await adapter.execute(
                session=session,
                base_url=base_url,
                request=PreparedRequest(
                    request_id="speech-1",
                    input_preview="hello",
                    payload={
                        "model": "s2-pro",
                        "input": "hello",
                        "response_format": "wav",
                        "stream": True,
                    },
                ),
            )

    assert result.success is True
    assert result.audio_duration_s is not None and result.audio_duration_s > 0
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 5
    assert result.tok_per_s == 10.0


@pytest.mark.asyncio
async def test_run_from_config_supports_s2_profile(tmp_path) -> None:
    dataset_root = tmp_path / "seedtts"
    dataset_root.mkdir()
    meta_path = dataset_root / "meta.lst"
    ref_audio = dataset_root / "ref.wav"
    ref_audio.write_bytes(_make_wav_bytes())
    meta_path.write_text("utt-1|reference transcript|ref.wav|hello world\n")

    wav_bytes = _make_wav_bytes()

    def _speech_response(_handler, body: bytes):
        payload = json.loads(body.decode("utf-8"))
        assert payload["model"] == "custom-s2"
        assert payload["references"][0]["text"] == "reference transcript"
        return (
            200,
            {
                "Content-Type": "audio/wav",
                "X-Prompt-Tokens": "10",
                "X-Completion-Tokens": "20",
                "X-Engine-Time": "1.0",
            },
            wav_bytes,
        )

    with _serve(
        {
            ("GET", "/health"): lambda _handler, _body: (
                200,
                {"Content-Type": "application/json"},
                json.dumps({"status": "healthy", "running": True}).encode("utf-8"),
            ),
            ("POST", "/v1/audio/speech"): _speech_response,
        }
    ) as base_url:
        results = await run_from_config(
            BenchmarkRunConfig(
                base_url=base_url,
                model="custom-s2",
                model_profile="s2_tts",
                case_id="voice-cloning",
                dataset_path=str(meta_path),
                output_dir=str(tmp_path / "out-s2"),
                max_samples=1,
                warmup=0,
            )
        )

    assert results.summary.model_profile == "s2_tts"
    assert results.summary.completed_requests == 1
    assert results.summary.failed_requests == 0
    assert results.summary.prompt_tokens_total == 10
    assert results.summary.completion_tokens_total == 20
    assert results.per_request[0].success is True
    assert (tmp_path / "out-s2" / "benchmark_results.json").exists()
    assert (tmp_path / "out-s2" / "per_request.csv").exists()


def test_multi_case_profile_requires_explicit_case() -> None:
    profile = PROFILE_REGISTRY.get_by_name_or_alias("s2_tts")
    with pytest.raises(ValueError, match="--case is required"):
        profile.get_case(None)


@pytest.mark.asyncio
async def test_run_from_config_requires_model_profile(tmp_path) -> None:
    with pytest.raises(ValueError, match="--model-profile is required"):
        await run_from_config(
            BenchmarkRunConfig(
                base_url="http://localhost:8000",
                model="test",
                model_profile="",
                output_dir=str(tmp_path / "out"),
            )
        )
