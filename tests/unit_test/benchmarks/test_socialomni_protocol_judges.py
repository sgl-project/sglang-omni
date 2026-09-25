# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.eval import benchmark_omni_socialomni as entrypoint
from benchmarks.tasks.socialomni import JUDGE_PARSE_ATTEMPTS, JudgeSpec, run_judges


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
