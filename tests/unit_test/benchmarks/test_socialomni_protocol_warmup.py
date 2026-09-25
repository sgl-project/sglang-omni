# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
from aiohttp import web

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.eval import benchmark_omni_socialomni as entrypoint
from benchmarks.tasks.socialomni import (
    JUDGE_MAX_TOKENS,
    JUDGE_PARSE_ATTEMPTS,
    JudgeSpec,
    build_judge_prompt,
    build_level1_result_records,
    build_response_prompt,
    build_when_prompt,
    judge_payload,
    load_judge_config,
    model_payload,
    parse_choice,
    parse_judge_score,
    parse_when,
    request_chat_completion,
    run_judges,
    run_level2_model,
)






def _level1(path: str = "/tmp/video.mp4") -> SocialOmniLevel1Sample:
    return SocialOmniLevel1Sample(
        "one", path, "Who?", ("one", "two", "three", "four"), "A", "speaker_visible"
    )

def _level2(index: int = 0) -> SocialOmniLevel2Sample:
    return SocialOmniLevel2Sample(
        str(index),
        "/tmp/video.mp4",
        3.0,
        "Alex",
        "Should Alex speak now?",
        "What should Alex say?",
        "NO",
        "private reference response",
        "private reference transcript",
    )

def _config(**overrides) -> entrypoint.SocialOmniEvalConfig:
    values = {
        "dataset_root": ".",
        "model": "qwen3-omni",
        "base_url": "http://localhost:8000",
        "level": "level1",
        "judge_config": None,
        "prefix_cache_dir": "cache",
        "mini": False,
        "max_samples": None,
        "max_concurrency": 1,
        "timeout_s": 30,
        "output_dir": "results",
    }
    values.update(overrides)
    return entrypoint.SocialOmniEvalConfig(**values)

class _Response:
    def __init__(self, status: int = 400, body: str = "specific failure body"):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self) -> str:
        return self.body

class _Session:
    def __init__(self, *responses: _Response):
        self.responses = list(responses) or [_Response()]
        self.calls = 0

    def post(self, *_args, **_kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response

@pytest.mark.parametrize("status", [429, 501, 507])
@pytest.mark.asyncio
async def test_retryable_http_error_is_retried(monkeypatch, status: int) -> None:
    session = _Session(
        _Response(status, "retry"),
        _Response(
            200,
            json.dumps({"choices": [{"message": {"content": "Answer: A"}}]}),
        ),
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("benchmarks.tasks.socialomni.asyncio.sleep", no_sleep)
    result = await request_chat_completion(
        session,  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
    )
    assert result.is_success
    assert session.calls == 2

@pytest.mark.asyncio
async def test_non_retryable_http_error_is_not_retried() -> None:
    session = _Session(_Response(400, "bad request"))
    result = await request_chat_completion(
        session,  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
    )
    assert not result.is_success
    assert session.calls == 1

@pytest.mark.asyncio
async def test_malformed_success_response_does_not_escape() -> None:
    result = await request_chat_completion(
        _Session(_Response(200, "[]")),  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
        max_attempts=1,
    )
    assert not result.is_success
    assert "invalid JSON response object" in result.error

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"error": {"message": "backend failed"}},
        {"choices": []},
        {"choices": {}},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": "A"}]},
        {"choices": [{"message": {}}]},
        *[
            {"choices": [{"message": {"content": content}}]}
            for content in (None, 3, {}, [None], [{"type": "text", "text": 3}])
        ],
    ],
)
async def test_malformed_completion_is_recorded_as_failure(body) -> None:
    """HTTP 200 alone must not count as a completed model request."""
    result = await request_chat_completion(
        _Session(_Response(200, json.dumps(body))),
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="malformed",
    )
    assert not result.is_success
    assert "invalid completion response" in result.error
    assert result.completion_tokens == 0

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [(" A ", "A"), ("", ""), ([], ""), ([{"type": "text", "text": "A"}], "A")],
)
async def test_valid_completion_content(content, expected) -> None:
    result = await request_chat_completion(
        _Session(
            _Response(200, json.dumps({"choices": [{"message": {"content": content}}]}))
        ),
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="valid",
    )
    assert result.is_success
    assert result.text == expected

@pytest.mark.parametrize(
    "suffix", ["", "/", "/v1", "/v1/", "/v1/chat/completions", "/v1/chat/completions/"]
)
def test_cli_checks_server_root_and_preserves_completion_url(
    tmp_path, monkeypatch, capsys, suffix
) -> None:
    """Accepted API URLs must reach both health and completion routes through the CLI."""
    routes = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            routes.append(self.path)
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            routes.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200 if self.path == "/v1/chat/completions" else 404)
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"Answer: A"}}]}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level1_samples", lambda *_a, **_k: [_level1()]
    )
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {"metadata_matches_expected_revision": False},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "socialomni",
            "--dataset-root",
            "~/dataset",
            "--model",
            "test",
            "--model-revision",
            "weights-commit",
            "--launch-command",
            "python -m sglang_omni.cli serve --model-path /models/qwen",
            "--base-url",
            f"http://127.0.0.1:{server.server_port}{suffix}",
            "--level",
            "level1",
            "--warmup",
            "0",
            "--timeout-s",
            "2",
            "--disable-tqdm",
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )
    try:
        entrypoint.main()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    output = json.loads(capsys.readouterr().out)

    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON constant: {value}")

    saved = json.loads(
        Path(output["result"]).read_text(), parse_constant=reject_constant
    )
    assert saved["config"]["request_rate"] == "inf"
    assert saved["provenance"]["declared_server_config"]["request_rate"] == "inf"
    assert saved["provenance"]["declared_server_config"]["judge_request_rate"] == "inf"
    assert routes == ["/health", "/v1/chat/completions"]
    assert saved["per_sample"]["level1"][0]["predicted_answer"] == "A"
    assert saved["config"]["dataset_root"] == str(tmp_path / "dataset")
    assert saved["config"]["model_revision"] == "weights-commit"
    assert saved["config"]["launch_command"] == saved["provenance"]["launch_command"]
    assert saved["provenance"]["launch_command"].endswith("--model-path /models/qwen")
    assert saved["provenance"]["declared_server_config"]["trust_env"] is True
    assert (
        saved["provenance"]["artifacts"]["declared_model_revision"] == "weights-commit"
    )

@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize(
    "invalid_count", [True, False, -1, 1.5, "2", None, {"value": 1}]
)
async def test_invalid_usage_becomes_a_request_failure(field, invalid_count) -> None:
    usage = {field: invalid_count}
    attempts = []
    body = json.dumps({"choices": [{"message": {"content": "YES"}}], "usage": usage})
    result = await request_chat_completion(
        _Session(_Response(200, body)),
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="one",
        attempt_records=attempts,
    )
    assert not result.is_success
    assert result.request_id == "one"
    assert "invalid token usage" in result.error
    assert result.text == "YES"
    assert len(attempts) == 1
    assert attempts[0]["text"] == "YES"
    assert attempts[0]["is_success"] is False
