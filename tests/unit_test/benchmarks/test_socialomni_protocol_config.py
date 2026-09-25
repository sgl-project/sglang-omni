# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest
from aiohttp import web

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.eval import benchmark_omni_socialomni as entrypoint
from benchmarks.tasks.socialomni import (
    JUDGE_MAX_TOKENS,
    JudgeSpec,
    build_judge_prompt,
    build_level1_result_records,
    build_response_prompt,
    build_when_prompt,
    judge_payload,
    model_payload,
    parse_choice,
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
