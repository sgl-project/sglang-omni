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

@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, [], [1], "", "invalid", 0, False])
async def test_non_object_usage_is_a_request_failure(usage) -> None:
    attempts = []
    result = await request_chat_completion(
        _Session(
            _Response(
                200,
                json.dumps(
                    {
                        "choices": [{"message": {"content": "A"}}],
                        "usage": usage,
                    }
                ),
            )
        ),
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="usage",
        attempt_records=attempts,
    )
    assert not result.is_success
    assert "usage must be an object" in result.error
    assert result.text == "A"
    assert len(attempts) == 1
    assert attempts[0]["text"] == "A"
    assert attempts[0]["is_success"] is False

@pytest.mark.parametrize("api_key_env", ["", " ", " KEY", "KEY ", 1])
def test_judge_config_rejects_invalid_key_names(tmp_path, api_key_env) -> None:
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": "http://localhost:8000",
                        "api_key_env": api_key_env,
                    }
                    for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="api_key_env"):
        load_judge_config(path)

@pytest.mark.asyncio
async def test_judges_preserve_raw_results(monkeypatch) -> None:
    sample = _level2()
    record = {
        "sample_id": sample.sample_id,
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {},
        "judge_results": {},
    }
    judges = [
        JudgeSpec(name, name, "http://localhost:8000", None, 1)
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    async def fake_request(*_args, request_id: str, **_kwargs):
        result = RequestResult(
            request_id=request_id, text="75", is_success=True, latency_s=0.1
        )
        _kwargs["attempt_records"].append(asdict(result))
        return result

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    requests, failures = await run_judges([sample], [record], judges, timeout_s=30)
    assert not failures
    assert record["gold_judge_scores"] == {name.name: 75 for name in judges}
    result = record["judge_results"]["gpt-4o"]
    assert set(result) == {
        "request",
        "attempts",
        "score",
        "raw_response",
        "is_success",
        "latency_s",
        "prompt_tokens",
        "completion_tokens",
        "error",
    }
    assert result["raw_response"] == "75"
    assert result["request"] == asdict(requests[0])

@pytest.mark.asyncio
@pytest.mark.parametrize("request_failed", [False, True])
async def test_judges_preserve_failure_phase_and_error(
    monkeypatch, request_failed
) -> None:
    sample = _level2()
    record = {
        "sample_id": sample.sample_id,
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {},
        "judge_results": {},
    }
    judges = [JudgeSpec("gpt-4o", "gpt-4o", "http://localhost:8000", None, 1)]

    calls = 0

    async def fake_request(*_args, request_id: str, **_kwargs):
        nonlocal calls
        calls += 1
        result = RequestResult(
            request_id=request_id,
            text="" if request_failed else "Score: 80",
            is_success=not request_failed,
            error="connection failed" if request_failed else None,
        )
        _kwargs["attempt_records"].append(asdict(result))
        return result

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    _, failures = await run_judges([sample], [record], judges, timeout_s=30)
    result = record["judge_results"]["gpt-4o"]
    assert len(failures) == 1
    assert failures[0] == {
        "request_id": result["request"]["request_id"],
        "sample_id": str(sample.sample_id),
        "judge": "gpt-4o",
        "phase": "level2_judge",
        "error": result["error"],
    }
    assert result["score"] is None
    assert result["raw_response"] == ("" if request_failed else "Score: 80")
    assert result["is_success"] is False
    assert len(result["attempts"]) == calls
    assert result["attempts"][-1]["text"] == result["raw_response"]
    if request_failed:
        assert result["error"] == "connection failed"
        assert calls == 1
    else:
        assert "invalid judge score" in result["error"]
        assert calls == JUDGE_PARSE_ATTEMPTS

@pytest.mark.asyncio
async def test_judge_parse_retry_stops_after_valid_score(monkeypatch) -> None:
    sample = _level2()
    record = {
        "sample_id": sample.sample_id,
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {},
        "judge_results": {},
    }
    judges = [JudgeSpec("gpt-4o", "gpt-4o", "http://localhost:8000", None, 1)]
    responses = iter(("not a score", "75"))
    clock = [0.0]
    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.time.perf_counter", lambda: clock[0]
    )

    async def fake_request(*_args, request_id: str, **_kwargs):
        clock[0] += 0.1
        result = RequestResult(
            request_id=request_id,
            text=next(responses),
            is_success=True,
            latency_s=0.1,
            engine_time_s=0.1,
            prompt_tokens=2,
            completion_tokens=1,
        )
        _kwargs["attempt_records"].append(asdict(result))
        return result

    async def no_sleep(_seconds: float) -> None:
        clock[0] += _seconds

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    monkeypatch.setattr("benchmarks.tasks.socialomni.asyncio.sleep", no_sleep)
    results, failures = await run_judges([sample], [record], judges, timeout_s=30)
    assert not failures
    assert record["gold_judge_scores"] == {"gpt-4o": 75}
    assert results[0].latency_s == pytest.approx(1.2)
    assert results[0].engine_time_s == pytest.approx(0.2)
    assert results[0].tok_per_s == pytest.approx(10)
    assert results[0].prompt_tokens == 4
    assert results[0].completion_tokens == 2
    attempts = record["judge_results"]["gpt-4o"]["attempts"]
    assert [attempt["text"] for attempt in attempts] == ["not a score", "75"]
    assert [attempt["completion_tokens"] for attempt in attempts] == [1, 1]
    assert [attempt["engine_time_s"] for attempt in attempts] == [0.1, 0.1]
    assert len({attempt["request_id"] for attempt in attempts}) == 2
