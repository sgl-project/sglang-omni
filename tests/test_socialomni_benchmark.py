# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SocialOmni benchmark integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import (
    SOCIALOMNI_MINI_LEVEL1,
    SOCIALOMNI_MINI_LEVEL2,
    SocialOmniLevel1Sample,
    prepare_socialomni_dataset,
)
from benchmarks.tasks.socialomni import (
    build_socialomni_level2_summary,
    compute_socialomni_level1_results,
    ensure_ffmpeg_available,
    parse_level2_timestamp_to_seconds,
    resolve_level2_outcome,
    validate_judge_specs,
)


def _write_fake_socialomni_source(root: Path) -> None:
    (root / "level_1" / "videos").mkdir(parents=True)
    (root / "level_2" / "videos").mkdir(parents=True)

    level1_rows = []
    for sample_id in list(SOCIALOMNI_MINI_LEVEL1) + [9999]:
        video_name = f"video_{sample_id}.mp4"
        (root / "level_1" / "videos" / video_name).write_bytes(b"level1")
        level1_rows.append(
            {
                "id": sample_id,
                "video_path": video_name,
                "question": f"Question {sample_id}?",
                "options": [
                    "A. option a",
                    "B. option b",
                    "C. option c",
                    "D. option d",
                ],
                "correct_answer": "A",
                "asr_content": "transcript",
                "metadata": {"consistency": "consistent"},
            }
        )
    (root / "level_1" / "dataset.json").write_text(
        json.dumps(level1_rows),
        encoding="utf-8",
    )

    level2_rows = []
    for video_id in list(SOCIALOMNI_MINI_LEVEL2) + ["video_0999"]:
        video_name = f"{video_id}.mp4"
        (root / "level_2" / "videos" / video_name).write_bytes(b"level2")
        is_yes = video_id in {SOCIALOMNI_MINI_LEVEL2[0], SOCIALOMNI_MINI_LEVEL2[1]}
        level2_rows.append(
            {
                "video_id": video_id,
                "video_file": video_name,
                "original_video_id": video_id,
                "source_dir": "source",
                "question_1": {
                    "question": "Should they speak?",
                    "timestamp": "00:17:00",
                    "option_A": "YES",
                    "option_B": "NO",
                    "correct_answer": "A" if is_yes else "B",
                },
                "question_2": {
                    "question": "What should they say?",
                    "answer": "hello there",
                },
                "metadata": {"consistency": "consistent"},
                "full_asr": "full transcript",
            }
        )
    (root / "level_2" / "annotations.json").write_text(
        json.dumps({"dataset_name": "socialomni", "data": level2_rows}),
        encoding="utf-8",
    )


def test_prepare_socialomni_mini_materializes_frozen_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "socialomni-mini"
    _write_fake_socialomni_source(source_root)

    prepare_socialomni_dataset(
        "socialomni-mini",
        local_dir=str(target_root),
        source_root=source_root,
    )

    level1_rows = json.loads((target_root / "level_1" / "dataset.json").read_text())
    assert [row["id"] for row in level1_rows] == list(SOCIALOMNI_MINI_LEVEL1)
    level2_payload = json.loads(
        (target_root / "level_2" / "annotations.json").read_text()
    )
    assert [row["video_id"] for row in level2_payload["data"]] == list(
        SOCIALOMNI_MINI_LEVEL2
    )
    assert (target_root / "mini_manifest.json").is_file()
    assert (target_root / "level_2" / "videos" / "video_0005.mp4").is_file()


def test_validate_judge_specs_requires_exact_count() -> None:
    with pytest.raises(ValueError, match="requires exactly 3"):
        validate_judge_specs(
            3, ["http://localhost:8001", "http://localhost:8002"], ["a", "b"]
        )

    specs = validate_judge_specs(1, ["http://localhost:8001/v1"], ["judge-a"])
    assert specs[0].api_url == "http://localhost:8001/v1/chat/completions"


@pytest.mark.parametrize(
    (
        "q1_answer",
        "q1_prediction",
        "judge_scores",
        "judge_errors",
        "expected_branch",
        "expected_score",
    ),
    [
        ("A", "A", [50, 75, 100], [], "yes_judged", 75.0),
        ("B", "B", [], [], "no_skipped", None),
        ("A", "B", [], [], "zeroed_wrong_q1", 0.0),
        ("A", "A", [75], ["judge failed"], "judge_failed", None),
    ],
)
def test_level2_protocol_branches(
    q1_answer: str,
    q1_prediction: str,
    judge_scores: list[int],
    judge_errors: list[str],
    expected_branch: str,
    expected_score: float | None,
) -> None:
    q1_correct, branch, q2_score = resolve_level2_outcome(
        q1_answer=q1_answer,
        q1_prediction=q1_prediction,
        judge_scores=judge_scores,
        judge_errors=judge_errors,
    )
    assert q1_correct is (q1_answer == q1_prediction)
    assert branch == expected_branch
    assert q2_score == expected_score


def test_parse_level2_timestamp_reference_semantics() -> None:
    assert parse_level2_timestamp_to_seconds("00:17:00") == 17.0
    assert parse_level2_timestamp_to_seconds("27") == 27.0


def test_level2_summary_includes_zero_scores_in_average() -> None:
    summary = build_socialomni_level2_summary(
        [
            {
                "q1_correct": True,
                "q2_score": 75.0,
                "branch_status": "yes_judged",
                "error": "",
            },
            {
                "q1_correct": False,
                "q2_score": 0.0,
                "branch_status": "zeroed_wrong_q1",
                "error": "",
            },
            {
                "q1_correct": True,
                "q2_score": None,
                "branch_status": "no_skipped",
                "error": "",
            },
        ]
    )
    assert summary["q2_count"] == 2
    assert summary["q2_avg_score"] == 37.5


def test_level1_results_include_required_fields() -> None:
    sample = SocialOmniLevel1Sample(
        sample_id="1",
        video_path="/tmp/video_1.mp4",
        question="What happened?",
        options=["A. alpha", "B. beta", "C. gamma", "D. delta"],
        correct_answer="D",
        asr_content="transcript",
        metadata={"consistency": "consistent"},
    )
    result = RequestResult(
        request_id="1", text="Answer: D", is_success=True, latency_s=1.25
    )

    summary, per_sample = compute_socialomni_level1_results([sample], [result])

    assert summary["accuracy"] == 1.0
    assert per_sample[0]["prediction"] == "D"
    assert per_sample[0]["correct_answer"] == "D"
    assert per_sample[0]["is_correct"] is True
    assert per_sample[0]["raw_response"] == "Answer: D"


import asyncio
import shutil

from aiohttp import web

from benchmarks.eval.benchmark_omni_socialomni import (
    SocialOmniEvalConfig,
    run_socialomni_level1_eval,
    run_socialomni_level2_eval,
)


def _write_level1_dataset_root(root: Path) -> Path:
    (root / "level_1" / "videos").mkdir(parents=True)
    (root / "level_1" / "videos" / "video_1.mp4").write_bytes(b"fake")
    (root / "level_1" / "dataset.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "video_path": "video_1.mp4",
                    "question": "What happened?",
                    "options": [
                        "A. alpha",
                        "B. beta",
                        "C. gamma",
                        "D. delta",
                    ],
                    "correct_answer": "D",
                    "asr_content": "transcript",
                    "metadata": {"consistency": "consistent"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return root


def _write_level2_dataset_root(root: Path) -> Path:
    (root / "level_2" / "videos").mkdir(parents=True)
    for name in ("video_yes.mp4", "video_no.mp4"):
        (root / "level_2" / "videos" / name).write_bytes(b"fake")
    (root / "level_2" / "annotations.json").write_text(
        json.dumps(
            {
                "dataset_name": "socialomni-mini",
                "data": [
                    {
                        "video_id": "video_yes",
                        "video_file": "video_yes.mp4",
                        "original_video_id": "video_yes",
                        "source_dir": "src",
                        "question_1": {
                            "question": "Should the speaker talk now?",
                            "timestamp": "00:17:00",
                            "option_A": "YES",
                            "option_B": "NO",
                            "correct_answer": "A",
                        },
                        "question_2": {
                            "question": "What should the speaker say?",
                            "answer": "reference answer",
                        },
                        "metadata": {"consistency": "consistent"},
                        "full_asr": "transcript",
                    },
                    {
                        "video_id": "video_no",
                        "video_file": "video_no.mp4",
                        "original_video_id": "video_no",
                        "source_dir": "src",
                        "question_1": {
                            "question": "Should the speaker talk now?",
                            "timestamp": "12",
                            "option_A": "YES",
                            "option_B": "NO",
                            "correct_answer": "B",
                        },
                        "question_2": {
                            "question": "What should the speaker say?",
                            "answer": "unused",
                        },
                        "metadata": {"consistency": "consistent"},
                        "full_asr": "transcript",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


async def _start_json_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_run_socialomni_level1_eval_end_to_end(tmp_path: Path) -> None:
    dataset_root = _write_level1_dataset_root(tmp_path / "dataset")

    async def handler(request):
        return web.json_response(
            {
                "choices": [{"message": {"content": "Answer: D"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            }
        )

    async def _run() -> None:
        runner, base_url = await _start_json_server(handler)
        try:
            results = await run_socialomni_level1_eval(
                SocialOmniEvalConfig(
                    model="test-model",
                    level="level1",
                    dataset_name="socialomni-mini",
                    dataset_dir=str(dataset_root),
                    base_url=base_url,
                    output_dir=str(tmp_path / "out-level1"),
                    max_concurrency=1,
                )
            )
            assert results["summary"]["accuracy"] == 1.0
            assert results["per_sample"][0]["prediction"] == "D"
            assert (
                tmp_path / "out-level1" / "socialomni_mini_level1_results.json"
            ).is_file()
        finally:
            await runner.cleanup()

    asyncio.run(_run())


def test_run_socialomni_level2_eval_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _write_level2_dataset_root(tmp_path / "dataset")

    def _fake_cut_video_prefix(
        input_video: str, timestamp_s: float, output_video: str
    ) -> None:
        Path(output_video).write_bytes(Path(input_video).read_bytes())

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.ensure_ffmpeg_available",
        lambda: None,
    )
    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.cut_video_prefix",
        _fake_cut_video_prefix,
    )

    async def primary_handler(request):
        payload = await request.json()
        prompt = payload["messages"][0]["content"]
        videos = payload.get("videos") or []
        if not videos:
            return web.json_response(
                {
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
            )
        video_name = Path(videos[0]).name
        if "Should the speaker talk now?" in prompt:
            content = "Answer: A" if video_name == "video_yes.mp4" else "Answer: B"
        else:
            content = "generated interruption"
        return web.json_response(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
        )

    async def judge_handler(request):
        return web.json_response(
            {
                "choices": [{"message": {"content": "75"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1},
            }
        )

    async def _run() -> None:
        primary_runner, primary_url = await _start_json_server(primary_handler)
        judge_runner, judge_url = await _start_json_server(judge_handler)
        try:
            results = await run_socialomni_level2_eval(
                SocialOmniEvalConfig(
                    model="test-model",
                    level="level2",
                    dataset_name="socialomni-mini",
                    dataset_dir=str(dataset_root),
                    base_url=primary_url,
                    output_dir=str(tmp_path / "out-level2"),
                    max_concurrency=1,
                    judges=1,
                    judge_base_urls=(judge_url,),
                    judge_models=("judge-model",),
                )
            )
            assert results["summary"]["q1_accuracy"] == 1.0
            assert results["summary"]["q2_avg_score"] == 75.0
            assert results["summary"]["branch_counts"]["yes_judged"] == 1
            assert results["summary"]["branch_counts"]["no_skipped"] == 1
            assert results["judge_metrics"]["requested_judges"] == 1
            assert results["per_sample"][0]["judge_scores"][0]["score"] == 75
            assert (
                tmp_path / "out-level2" / "socialomni_mini_level2_results.json"
            ).is_file()
        finally:
            await primary_runner.cleanup()
            await judge_runner.cleanup()

    asyncio.run(_run())


def test_level2_ffmpeg_missing_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        ensure_ffmpeg_available()


def test_level2_judge_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _write_level2_dataset_root(tmp_path / "dataset-preflight")

    async def primary_handler(request):
        return web.json_response(
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        )

    async def bad_judge_handler(request):
        return web.Response(status=500, text="judge down")

    async def _run() -> None:
        primary_runner, primary_url = await _start_json_server(primary_handler)
        judge_runner, judge_url = await _start_json_server(bad_judge_handler)
        try:
            with pytest.raises(RuntimeError, match="judge endpoint"):
                await run_socialomni_level2_eval(
                    SocialOmniEvalConfig(
                        model="test-model",
                        level="level2",
                        dataset_name="socialomni-mini",
                        dataset_dir=str(dataset_root),
                        base_url=primary_url,
                        output_dir=str(tmp_path / "out-preflight"),
                        max_concurrency=1,
                        judges=1,
                        judge_base_urls=(judge_url,),
                        judge_models=("judge-model",),
                    )
                )
        finally:
            await primary_runner.cleanup()
            await judge_runner.cleanup()

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.ensure_ffmpeg_available", lambda: None
    )
    asyncio.run(_run())
