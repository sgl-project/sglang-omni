# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_SCRIPT = REPO_ROOT / ".github/scripts/omni_ci_metric_audit.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("omni_ci_metric_audit", AUDIT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_audit_extracts_stable_metrics(tmp_path, monkeypatch):
    audit = load_audit_module()
    first = tmp_path / "pytest-0/videoamme_audio0/videoamme_results.json"
    second = tmp_path / "pytest-1/videoamme_audio0/videoamme_results.json"
    payload = {
        "summary": {"accuracy": 0.5, "total_samples": 10},
        "speed": {
            "latency_mean_s": 180.27,
            "rtf_mean": 13.5138,
            "throughput_qps": 0.055,
        },
        "config": {"max_samples": 10, "max_tokens": 256},
        "per_sample": [{"idx": 0}],
    }
    write_json(first, payload)
    write_json(
        second, payload | {"speed": payload["speed"] | {"latency_mean_s": 185.539}}
    )
    matches = tmp_path / "matches.txt"
    matches.write_text(f"{first}\n{second}\n", encoding="utf-8")

    monkeypatch.setenv("GITHUB_REPOSITORY", "sgl-project/sglang-omni")
    monkeypatch.setenv("GITHUB_WORKFLOW", "Omni CI")
    monkeypatch.setenv("GITHUB_JOB", "stage-10")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(REPO_ROOT))

    event = audit.collect_audit(
        stage_label="stage 10",
        matches_file=str(matches),
        artifact_search_root=str(tmp_path),
        artifact_path_globs="*/videoamme_results.json",
    )

    assert len(event["artifact_files"]) == 2
    assert event["artifact_files"][0]["summary"]["per_sample"] == {
        "omitted": True,
        "count": 1,
    }
    latest = audit.latest_metrics_by_id(event)
    assert (
        latest["stage 10|videoamme_results.json|speed.latency_mean_s"]["value"]
        == 185.539
    )
    assert (
        latest["stage 10|videoamme_results.json|speed.latency_mean_s"]["direction"]
        == "lower"
    )
    assert (
        latest["stage 10|videoamme_results.json|speed.throughput_qps"]["direction"]
        == "higher"
    )
    assert "stage 10|videoamme_results.json|config.max_samples" not in latest


def test_compare_event_detects_lower_and_higher_regressions():
    audit = load_audit_module()
    event = {
        "repository": "sgl-project/sglang-omni",
        "workflow": {"run_id": "2"},
        "stage": {"label": "stage 10"},
        "pull_request": {"number": 911},
        "metrics": [
            {
                "metric_id": "stage 10|videoamme_results.json|speed.latency_mean_s",
                "stage_label": "stage 10",
                "artifact_id": "videoamme_results.json",
                "json_path": "speed.latency_mean_s",
                "value": 112.0,
                "direction": "lower",
                "attempt_ordinal": 1,
            },
            {
                "metric_id": "stage 10|videoamme_results.json|speed.throughput_qps",
                "stage_label": "stage 10",
                "artifact_id": "videoamme_results.json",
                "json_path": "speed.throughput_qps",
                "value": 8.8,
                "direction": "higher",
                "attempt_ordinal": 1,
            },
        ],
    }
    history = [
        {
            "metrics": [
                {
                    "metric_id": "stage 10|videoamme_results.json|speed.latency_mean_s",
                    "value": 100.0,
                },
                {
                    "metric_id": "stage 10|videoamme_results.json|speed.throughput_qps",
                    "value": 10.0,
                },
            ]
        }
    ]

    alert = audit.compare_event(
        event,
        history,
        threshold=0.10,
        min_baseline_count=1,
    )

    by_path = {item["json_path"]: item for item in alert["regressions"]}
    assert by_path["speed.latency_mean_s"]["regression_ratio"] == pytest.approx(0.12)
    assert by_path["speed.throughput_qps"]["regression_ratio"] == pytest.approx(0.12)


def test_history_record_dedupes_retry_metrics():
    audit = load_audit_module()
    event = {
        "created_at": "2026-07-02T00:00:00Z",
        "repository": "sgl-project/sglang-omni",
        "workflow": {"run_id": "1", "run_attempt": "1", "job": "stage-10"},
        "stage": {"label": "stage 10"},
        "metrics": [
            {
                "metric_id": "stage 10|videoamme_results.json|speed.latency_mean_s",
                "json_path": "speed.latency_mean_s",
                "value": 160.685,
                "direction": "lower",
                "attempt_ordinal": 1,
            },
            {
                "metric_id": "stage 10|videoamme_results.json|speed.latency_mean_s",
                "json_path": "speed.latency_mean_s",
                "value": 185.539,
                "direction": "lower",
                "attempt_ordinal": 3,
            },
        ],
    }

    record = audit.history_record(event)

    assert len(record["metrics"]) == 1
    assert record["metrics"][0]["value"] == 185.539


def test_storage_path_includes_pr_run_and_stage():
    audit = load_audit_module()
    event = {
        "repository": "sgl-project/sglang-omni",
        "workflow": {"run_id": "28608687974", "run_attempt": "1", "job": "stage-10"},
        "pull_request": {"number": 911},
        "stage": {
            "label": "FP8 Thinker TP=2 (Video-AMME Talker accuracy + WER + speed)"
        },
    }

    path = audit.storage_path(event)

    assert path == (
        "events/sgl-project-sglang-omni/pr-911/run-28608687974/"
        "attempt-1/stage-10/"
        "fp8-thinker-tp-2-video-amme-talker-accuracy-wer-speed.json"
    )
