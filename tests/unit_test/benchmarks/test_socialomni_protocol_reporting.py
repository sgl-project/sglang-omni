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
