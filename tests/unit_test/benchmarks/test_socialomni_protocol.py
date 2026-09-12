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


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [None, "score", "http"])
async def test_complete_protocol_over_http(tmp_path: Path, monkeypatch, retry) -> None:
    samples = [replace(_level2(0), gold_when="YES"), _level2(1)]
    seen = []
    runner_configs = []
    original_init = entrypoint.BenchmarkRunner.__init__

    def capture_config(self, config):
        runner_configs.append(config)
        original_init(self, config)

    monkeypatch.setattr(entrypoint.BenchmarkRunner, "__init__", capture_config)

    async def completion(request):
        payload = await request.json()
        seen.append(payload)
        limit = payload["max_tokens"]
        text = {32: "Answer: A", 8: "Answer: B", 256: "candidate", 8192: "75"}[limit]
        if (
            retry
            and payload["model"] == "gpt-4o"
            and sum(item["model"] == "gpt-4o" for item in seen) == 1
        ):
            if retry == "http":
                return web.Response(status=503, text="temporarily unavailable")
            text = "not a score"
        if limit != JUDGE_MAX_TOKENS:
            assert payload["use_audio_in_video"] is True
            assert payload["videos"]
            assert "private reference" not in json.dumps(payload)
        return web.json_response(
            {
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    base_url = f"http://127.0.0.1:{server.addresses[0][1]}"
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    config_path = tmp_path / "judges.json"
    config_path.write_text(
        json.dumps(
            {
                "judges": [
                    {"name": name, "model": name, "base_url": base_url}
                    for name in entrypoint.SOCIALOMNI_JUDGE_NAMES
                ]
            }
        ),
        encoding="utf-8",
    )

    async def prepared(*_args):
        return tmp_path / "prefix.mp4"

    monkeypatch.setattr("benchmarks.tasks.socialomni.create_video_prefix", prepared)
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level1_samples", lambda *_a, **_k: [_level1()]
    )
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level2_samples", lambda *_a, **_k: samples
    )
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {
            "metadata_matches_expected_revision": False,
        },
    )
    try:
        result = await entrypoint.run_socialomni(
            _config(
                level="both",
                base_url=base_url,
                judge_config=str(config_path),
                request_rate=10000.0,
                warmup=0,
                disable_tqdm=True,
            )
        )
    finally:
        await server.cleanup()
    assert result["summary"]["status"] == "complete"
    assert len(runner_configs) == 6
    assert all(config.request_rate == 10000.0 for config in runner_configs)
    assert result["config"]["request_rate"] == 10000.0
    assert result["provenance"]["declared_server_config"]["request_rate"] == 10000.0
    assert (
        result["provenance"]["declared_server_config"]["judge_request_rate"] == 10000.0
    )
    assert not result["failures"]
    assert len(seen) == 7 + int(bool(retry))
    speed = result["summary"]["level2"]["speed"]["judges"]
    assert speed["total_requests"] == 3 + int(bool(retry))
    assert speed["failed_requests"] == int(retry == "http")
    assert len(result["per_sample"]["level1"]) == 1
    positive, negative = result["per_sample"]["level2"]
    attempts = positive["judge_results"]["gpt-4o"]["attempts"]
    assert [attempt["text"] for attempt in attempts] == (
        ["", "75"] if retry == "http" else ["not a score", "75"] if retry else ["75"]
    )
    assert len({attempt["request_id"] for attempt in attempts}) == len(attempts)
    assert positive["predicted_when"] == "NO"
    assert positive["gold_response"] == "candidate"
    assert positive["gold_judge_scores"] == dict.fromkeys(
        entrypoint.SOCIALOMNI_JUDGE_NAMES, 75
    )
    assert negative["gold_response_success"] is None


def test_model_prompts_do_not_leak_reference_material() -> None:
    sample = _level2()
    when = build_when_prompt(sample)
    response = build_response_prompt(sample)
    for secret in (sample.reference_context, sample.reference_response):
        assert secret not in when
        assert secret not in response
    judge = build_judge_prompt(sample, "candidate")
    assert sample.reference_context in judge
    assert sample.reference_response in judge


def test_model_payload_uses_native_video_with_embedded_audio() -> None:
    payload = model_payload("qwen3-omni", "prompt", "/tmp/prefix.mp4", 8)
    assert payload["videos"] == ["/tmp/prefix.mp4"]
    assert payload["use_audio_in_video"] is True
    assert payload["modalities"] == ["text"]


def test_judge_payload_allows_reasoning_before_score() -> None:
    judge = JudgeSpec(
        "gemini-2.5-pro", "gemini-2.5-pro", "http://localhost:8000", None, 1
    )
    assert judge_payload(judge, "prompt")["max_tokens"] == JUDGE_MAX_TOKENS == 8192


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: A", "A"),
        ("\\boxed{B}", "B"),
        ("C", "C"),
        ("A or B", ""),
        ("Answer: A or B", ""),
        ("Answer: A/B", ""),
        ("Answer: A and B", ""),
        ("Answer: A. Answer: B", ""),
        ("\\boxed{A} or B", ""),
    ],
)
def test_choice_parser_is_strict(raw: str, expected: str) -> None:
    assert parse_choice(raw, ("A", "B", "C", "D")) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The speaker is on the left.\nAnswer: A\n\n", "A"),
        ("B is not the speaker.\nAnswer: A", "A"),
        ("Explanation.\nAnswer: A or B", ""),
        ("Answer: A\nThere is no final answer here.", ""),
        ("", ""),
    ],
)
def test_level1_parses_only_the_final_answer_line(raw, expected) -> None:
    """Accept the final-line format requested by the Level 1 prompt."""
    result = RequestResult(request_id="one", text=raw, is_success=True)
    record = build_level1_result_records([_level1()], [result])[0]
    assert record["predicted_answer"] == expected
    assert record["raw_response"] == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: A", "YES"),
        ("Answer: B", "NO"),
        ("YES", "YES"),
        ("maybe", ""),
        ("Answer: A or B", ""),
    ],
)
def test_when_parser(raw: str, expected: str) -> None:
    assert parse_when(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("75", 75),
        ("Score: 25", 25),
        ("75 or 100", None),
        ("-25", None),
        ("25.5", None),
        (".25", None),
        ("0.75", None),
        ("125", None),
        ("80 or 75", None),
    ],
)
def test_judge_score_parser(raw: str, expected: int | None) -> None:
    assert parse_judge_score(raw) == expected


@pytest.mark.parametrize("concurrency", [True, False, 0, -1, 1.5, "1"])
def test_judge_config_rejects_invalid_concurrency(tmp_path, concurrency) -> None:
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": "http://localhost:8000",
                        "max_concurrency": concurrency,
                    }
                    for name in entrypoint.SOCIALOMNI_JUDGE_NAMES
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="max_concurrency"):
        load_judge_config(path)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.com/v1",
        "https://secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/v1?token=secret",
        "https://example.com/v1#secret",
        "file:///secret",
        "http://localhost:99999",
        "http://localhost:not-a-port",
        "http://localhost:0",
        "http://localhost:-1",
    ],
)
def test_endpoint_credentials_are_rejected_without_echoing_url(tmp_path, url):
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {"name": name, "model": name, "base_url": url}
                    for name in entrypoint.SOCIALOMNI_JUDGE_NAMES
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="base_url") as caught:
        load_judge_config(path)
    assert "secret" not in str(caught.value)
    assert "api_key_env" not in str(caught.value)
    with pytest.raises(ValueError, match="base_url") as model_error:
        _config(base_url=url)
    assert str(model_error.value) == str(caught.value)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), -float("inf")])
def test_invalid_request_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="request_rate"):
        _config(request_rate=rate)


def test_service_timeout_is_independent_from_request_timeout(monkeypatch):
    observed = []

    def wait(url, timeout):
        observed.append((url, timeout))
        raise RuntimeError("stop before evaluation")

    monkeypatch.setattr(entrypoint, "wait_for_service", wait)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "socialomni",
            "--dataset-root",
            ".",
            "--model",
            "test",
            "--timeout-s",
            "2",
            "--server-timeout",
            "600",
            "--request-rate",
            "3",
        ],
    )
    parsed = entrypoint.SocialOmniEvalConfig(**vars(entrypoint._parser().parse_args()))
    assert parsed.timeout_s == 2
    assert parsed.request_rate == 3
    assert _config(timeout_s=2).server_timeout == 300
    with pytest.raises(RuntimeError, match="stop before evaluation"):
        entrypoint.main()
    assert observed == [("http://localhost:8000", 600)]


def test_judge_config_has_only_fixed_public_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_VALUE", "must-not-appear")
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": "http://localhost:8000",
                        "api_key_env": "SECRET_VALUE" if index == 0 else None,
                        "max_concurrency": 1,
                    }
                    for index, name in enumerate(
                        ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    public = [asdict(judge) for judge in load_judge_config(path)]
    assert "must-not-appear" not in json.dumps(public)
    assert public[0]["api_key_env"] == "SECRET_VALUE"


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


@pytest.mark.asyncio
async def test_level2_automatically_derives_first_200_view(monkeypatch) -> None:
    samples = [_level2(index) for index in range(209)]
    records = [
        {
            "sample_id": sample.sample_id,
            "gold_when": "NO",
            "predicted_when": "NO",
            "when_success": True,
            "gold_response": "",
            "gold_response_success": None,
            "gold_judge_scores": {},
            "judge_results": {},
        }
        for sample in samples
    ]
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level2_samples", lambda *_a, **_k: samples
    )
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {
            "expected_huggingface_revision": "revision",
            "verification_scope": "metadata_only",
            "metadata_sha256": {},
            "metadata_matches_expected_revision": True,
        },
    )
    provenance_args = {}

    def fake_provenance(**kwargs):
        provenance_args.update(kwargs)
        return {"repository": {"commit": "abc", "dirty": False}}

    monkeypatch.setattr(entrypoint, "collect_benchmark_provenance", fake_provenance)

    async def fake_model(*_args, **_kwargs):
        return (
            records,
            [
                RequestResult(request_id="bad:prefix", error="no encoder"),
                RequestResult(request_id="0:when", is_success=True),
                RequestResult(request_id="1:when", error="server failure"),
            ],
            1.0,
        )

    monkeypatch.setattr(entrypoint, "run_level2_model", fake_model)
    config = entrypoint.SocialOmniEvalConfig(
        dataset_root=".",
        model="qwen3-omni",
        base_url="http://localhost:8000",
        level="level2",
        judge_config=None,
        prefix_cache_dir="cache",
        mini=False,
        max_samples=None,
        max_concurrency=1,
        timeout_s=30,
        output_dir="results",
    )
    records[0].update(
        gold_when="YES", gold_response="candidate", gold_response_success=True
    )

    async def unexpected_judges(*args, **kwargs):
        pytest.fail("Model-only diagnostics must not call judges")

    monkeypatch.setattr(entrypoint, "run_judges", unexpected_judges)
    result = await entrypoint.run_socialomni(config)
    assert result["paper_core_200"]["sample_count"] == 200
    assert result["paper_core_200"]["when"]["total_samples"] == 200
    assert result["summary"]["status"] == "incomplete"
    metrics = result["summary"]["level2"]["metrics"]
    assert metrics["quality"] is None
    assert metrics["judge_status"]["configured"] is False
    assert metrics["judge_status"]["complete"] is False
    assert metrics["judge_status"]["completed_scores"] == 0
    assert metrics["judge_status"]["required_scores"] == 3
    assert result["paper_core_200"]["quality"] is None
    assert result["paper_core_200"]["judges_complete"] is False
    assert provenance_args["dataset_revision"] == entrypoint.SOCIALOMNI_DATASET_REVISION
    assert provenance_args["model_revision"] is None
    speed = result["summary"]["level2"]["speed"]["model"]
    assert speed["total_requests"] == 2
    assert speed["failed_requests"] == 1
    assert any(f["request_id"] == "bad:prefix" for f in result["failures"])


@pytest.mark.asyncio
async def test_invalid_judge_config_fails_before_level2_requests(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "judges.json"
    config_path.write_text('{"judges": []}', encoding="utf-8")
    called = False

    async def fake_model(*_args, **_kwargs):
        nonlocal called
        called = True
        return [], [], 0.0

    monkeypatch.setattr(entrypoint, "run_level2_model", fake_model)
    config = _config(level="level2", judge_config=str(config_path))
    with pytest.raises(ValueError, match="exactly three judges"):
        await entrypoint.run_socialomni(config)
    assert not called


@pytest.mark.asyncio
async def test_level2_status_requires_full_three_judge_run(monkeypatch) -> None:
    samples = [_level2(index) for index in range(209)]
    records = [
        {
            "sample_id": sample.sample_id,
            "gold_when": "YES",
            "predicted_when": "YES",
            "when_success": True,
            "when_raw_response": "Answer: A",
            "gold_response": "candidate",
            "gold_response_success": True,
            "gold_judge_scores": {},
            "judge_results": {},
        }
        for sample in samples
    ]
    judges = [
        JudgeSpec(name, name, "http://localhost:8000", None, 1)
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    monkeypatch.setattr(
        entrypoint, "load_socialomni_level2_samples", lambda *_a, **_k: samples
    )
    monkeypatch.setattr(entrypoint, "load_judge_config", lambda _path: judges)
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {
            "expected_huggingface_revision": "revision",
            "verification_scope": "metadata_only",
            "metadata_sha256": {},
            "metadata_matches_expected_revision": True,
        },
    )
    monkeypatch.setattr(
        entrypoint,
        "collect_benchmark_provenance",
        lambda **_kwargs: {"repository": {"commit": "abc", "dirty": False}},
    )

    async def fake_model(*_args, **_kwargs):
        return records, [], 1.0

    async def fake_judges(_samples, current, _judges, **_kwargs):
        for record in current:
            record["gold_judge_scores"] = {judge.name: 75 for judge in judges}
        return [], []

    monkeypatch.setattr(entrypoint, "run_level2_model", fake_model)
    monkeypatch.setattr(entrypoint, "run_judges", fake_judges)

    result = await entrypoint.run_socialomni(
        _config(level="level2", judge_config="judges.json")
    )

    assert result["summary"]["status"] == "complete"


def test_paper_core_judge_completeness_is_independent() -> None:
    scores = {name: 75 for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")}
    records = [
        {
            "gold_when": "YES",
            "gold_response": "candidate",
            "gold_response_success": True,
            "gold_judge_scores": dict(scores),
        }
        for _ in range(209)
    ]
    records[-1]["gold_judge_scores"].pop("gpt-4o")
    assert entrypoint._judges_complete(records[:200], True)
    assert not entrypoint._judges_complete(records, True)


def test_invalid_score_is_not_judge_complete() -> None:
    record = {
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {
            "gpt-4o": 75,
            "gemini-2.5-pro": 80,
            "qwen3-omni": 75,
        },
    }
    assert not entrypoint._judges_complete([record], True)
