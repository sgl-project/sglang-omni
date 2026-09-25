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
@pytest.mark.parametrize("level", ["level1", "level2"])
@pytest.mark.parametrize("warmup", [None, 0])
async def test_model_failures_follow_shared_warmup_policy(monkeypatch, level, warmup):
    """Warmup fails fast; explicitly disabling it retains failed measured samples."""
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level1_samples", lambda *_a, **_k: [_level1()]
    )
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level2_samples", lambda *_a, **_k: [_level2()]
    )
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {"metadata_matches_expected_revision": False},
    )
    calls = []

    async def prefix(*_args):
        return Path("/tmp/prefix.mp4")

    async def failed(*_args, request_id, **_kwargs):
        calls.append(request_id)
        return RequestResult(request_id=request_id, error="server failed")

    monkeypatch.setattr("benchmarks.tasks.socialomni.create_video_prefix", prefix)
    monkeypatch.setattr("benchmarks.tasks.socialomni.request_chat_completion", failed)
    config = _config(level=level, warmup=warmup, disable_tqdm=True)
    if warmup is None:
        with pytest.raises(ValueError, match="Warmup failed"):
            await entrypoint.run_socialomni(config)
    else:
        output = await entrypoint.run_socialomni(config)
        assert len(output["per_sample"][level]) == 1
        assert len(output["failures"]) == 1
        metrics = output["summary"][level]["metrics"]
        assert (metrics if level == "level1" else metrics["when"])["total_samples"] == 1
    assert len(calls) == 1

@pytest.mark.asyncio
async def test_level2_uses_runner_warmup_without_polluting_results(
    tmp_path: Path, monkeypatch
) -> None:
    samples = [_level2(0), _level2(1)]
    active = 0
    maximum = 0
    request_ids = []
    timeline = []

    async def fake_prefix(*_args, **_kwargs) -> Path:
        return tmp_path / "prefix.mp4"

    async def fake_request(*_args, request_id: str, **_kwargs) -> RequestResult:
        nonlocal active, maximum
        request_ids.append(request_id)
        timeline.append(request_id)
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0)
        active -= 1
        return RequestResult(request_id=request_id, text="Answer: B", is_success=True)

    timer_values = iter((10.0, 12.0))

    def fake_perf_counter() -> float:
        timeline.append("timer")
        return next(timer_values)

    monkeypatch.setattr("benchmarks.tasks.socialomni.create_video_prefix", fake_prefix)
    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    monkeypatch.setattr(
        "benchmarks.benchmarker.runner.time",
        SimpleNamespace(perf_counter=fake_perf_counter),
    )

    records, measured, measured_wall_s = await run_level2_model(
        samples,
        model="qwen3-omni",
        base_url="http://localhost:8000",
        prefix_cache_dir=tmp_path,
        max_concurrency=3,
        timeout_s=30,
    )

    assert request_ids[:3] == ["0:when"] * 3
    assert timeline[:4] == ["0:when", "0:when", "0:when", "timer"]
    assert maximum == 3
    assert [result.request_id for result in measured] == ["0:when", "1:when"]
    assert len(records) == 2
    assert measured_wall_s == 2.0

@pytest.mark.asyncio
async def test_level2_keeps_failed_prefixes_and_forces_gold_responses(
    tmp_path: Path, monkeypatch
) -> None:
    samples = [replace(_level2(i), gold_when="YES") for i in range(3)]
    samples.append(_level2(3))
    calls = []
    prepared = []

    async def fake_prefix(_path, timestamp_s, _cache):
        prepared.append(timestamp_s)
        if timestamp_s == 1.0:
            raise RuntimeError("broken clip")
        return tmp_path / "prefix.mp4"

    async def fake_request(*_args, request_id, **_kwargs):
        assert len(prepared) == len(samples)
        calls.append(request_id)
        await asyncio.sleep(0)
        return RequestResult(
            request_id=request_id,
            text="Answer: B" if request_id.endswith(":when") else "candidate",
            is_success=request_id != "0:when",
            error="model failure" if request_id == "0:when" else "",
        )

    samples = [
        replace(sample, timestamp_s=float(i)) for i, sample in enumerate(samples)
    ]
    monkeypatch.setattr("benchmarks.tasks.socialomni.create_video_prefix", fake_prefix)
    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    records, requests, _ = await run_level2_model(
        samples,
        model="qwen3-omni",
        base_url="http://localhost:8000",
        prefix_cache_dir=tmp_path,
        max_concurrency=2,
        timeout_s=30,
        warmup=0,
    )
    assert [r["sample_id"] for r in records] == ["0", "1", "2", "3"]
    assert not records[0]["when_success"]
    assert records[0]["gold_response"] == "candidate"
    assert "broken clip" in records[1]["requests"][0]["error"]
    assert records[2]["predicted_when"] == "NO"
    assert records[2]["gold_response"] == "candidate"
    assert records[3]["gold_response_success"] is None
    assert set(calls) == {"0:when", "2:when", "3:when", "0:response", "2:response"}
    assert len(requests) == 6

@pytest.mark.asyncio
async def test_level2_response_warmup_is_excluded(tmp_path: Path, monkeypatch) -> None:
    samples = [replace(_level2(i), gold_when="YES") for i in range(2)]
    calls = []

    async def fake_prefix(*_args):
        return tmp_path / "prefix.mp4"

    async def fake_request(*_args, request_id, **_kwargs):
        calls.append(request_id)
        return RequestResult(request_id=request_id, text="YES", is_success=True)

    monkeypatch.setattr("benchmarks.tasks.socialomni.create_video_prefix", fake_prefix)
    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    records, requests, _ = await run_level2_model(
        samples,
        model="qwen3-omni",
        base_url="http://localhost:8000",
        prefix_cache_dir=tmp_path,
        max_concurrency=2,
        timeout_s=30,
    )
    assert calls.count("0:when") == calls.count("0:response") == 3
    assert len(requests) == 4
    assert all(len(record["requests"]) == 2 for record in records)

@pytest.mark.asyncio
async def test_judge_runners_limit_each_endpoint_without_warmup(monkeypatch) -> None:
    samples = [_level2(i) for i in range(6)]
    records = [
        {
            "sample_id": sample.sample_id,
            "gold_when": "YES",
            "gold_response": "candidate",
            "gold_response_success": True,
            "gold_judge_scores": {},
            "judge_results": {},
        }
        for sample in samples
    ]
    names = ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    judges = [
        JudgeSpec(name, name, "http://localhost:8000", None, i + 1)
        for i, name in enumerate(names)
    ]
    active = dict.fromkeys(names, 0)
    maximum = dict.fromkeys(names, 0)
    calls = []

    async def fake_request(*_args, request_id, **_kwargs):
        name = request_id.split(":judge:", 1)[1].split(":attempt:", 1)[0]
        active[name] += 1
        maximum[name] = max(maximum[name], active[name])
        calls.append(request_id)
        await asyncio.sleep(0.01)
        active[name] -= 1
        result = RequestResult(request_id=request_id, text="75", is_success=True)
        _kwargs["attempt_records"].append(asdict(result))
        return result

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    results, failures = await run_judges(samples, records, judges, timeout_s=30)
    assert maximum == dict(zip(names, (1, 2, 3), strict=True))
    assert len(calls) == len(set(calls)) == len(results) == 18
    assert not failures
    assert all(
        record["gold_judge_scores"] == dict.fromkeys(names, 75) for record in records
    )

@pytest.mark.asyncio
async def test_http_error_body_is_preserved() -> None:
    result = await request_chat_completion(
        _Session(),  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
    )
    assert not result.is_success
    assert "HTTP 400" in result.error
    assert "specific failure body" in result.error

@pytest.mark.asyncio
async def test_completion_populates_shared_speed_metrics() -> None:
    from benchmarks.metrics.performance import compute_speed_metrics

    result = await request_chat_completion(
        _Session(
            _Response(
                200,
                json.dumps(
                    {
                        "choices": [{"message": {"content": "A"}}],
                        "usage": {"completion_tokens": 10},
                    }
                ),
            )
        ),
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="speed",
    )
    assert result.is_success
    assert result.engine_time_s == result.latency_s > 0
    assert result.tok_per_s == pytest.approx(10 / result.engine_time_s)
    assert "output_tok_per_req_s" in compute_speed_metrics([result], wall_clock_s=1)
