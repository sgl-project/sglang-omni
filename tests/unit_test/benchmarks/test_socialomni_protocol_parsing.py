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
