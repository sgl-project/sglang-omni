# SPDX-License-Identifier: Apache-2.0
"""FDB CI rejects quality regressions and incomplete scoring evidence."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import JsonValue

from tests.test_model import fdb_ci_config
from tests.utils import MetricCheckCollector, fdb_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_TEST = "tests/test_model/test_fdb_ci.py"
ENGINE = "cascade"
STOP_INTERVAL_S = 1.0
RESPONSE_INTERVAL_S = 2.0


@pytest.mark.parametrize(
    "marker_arguments,expected_exit_code,selected",
    [
        pytest.param(
            ["-m", "not benchmark and not accelerator"],
            pytest.ExitCode.NO_TESTS_COLLECTED,
            False,
            id="cpu-unit",
        ),
        pytest.param(
            ["-m", "accelerator and not benchmark"],
            pytest.ExitCode.NO_TESTS_COLLECTED,
            False,
            id="accelerator-unit",
        ),
        pytest.param([], pytest.ExitCode.OK, True, id="dedicated-fdb-stage"),
    ],
)
def test_the_ci_run_is_selected_only_by_its_dedicated_stage(
    marker_arguments: list[str], expected_exit_code: pytest.ExitCode, selected: bool
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            CI_TEST,
            *marker_arguments,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == expected_exit_code, result.stdout + result.stderr
    assert ("::test_full_duplex_bench" in result.stdout) == selected


def write_json(path: Path, document: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def read_json(path: Path) -> JsonValue:
    return json.loads(path.read_text())


def write_timing(
    sample: Path, stop: list[list[float]], response: list[list[float]]
) -> None:
    intervals = sample / "latency_intervals.json"
    write_json(intervals, {"latency_stop_list": stop, "latency_resp_list": response})
    write_json(
        sample / "receipts" / "timing-overlap.json",
        {
            "status": "ok",
            "intervals_sha256": hashlib.sha256(intervals.read_bytes()).hexdigest(),
        },
    )


def passing_variant() -> dict[str, JsonValue]:
    return {"status": "pass", "protocol_verdict": "pass", "errors": []}


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """A complete scored run whose every metric equals REFERENCES."""
    repeat = tmp_path / "repeat-1"
    pairs = fdb_ci.V15_PER_SUBSET * len(fdb_ci.V15_CATEGORIES)
    v10_samples = fdb_ci.V10_PER_SUBSET * len(fdb_ci.V10_SUBSETS)
    write_json(
        repeat / "recording" / "shard-0" / "run.json",
        {
            "samples": [
                {"variants": {"overlap": passing_variant(), "clean": passing_variant()}}
                for _ in range(pairs)
            ]
        },
    )
    write_json(
        repeat / "v10" / "recording" / "shard-0" / "run.json",
        {"samples": [{"variants": {"input": passing_variant()}}] * v10_samples},
    )
    behavior = {
        category: {
            "valid_n": fdb_ci.V15_PER_SUBSET,
            "valid_label_proportions": {
                "C_RESPOND": {"count": 3},
                "C_RESUME": {"count": 2},
            },
        }
        for category in fdb_ci.V15_CATEGORIES
    }
    write_json(repeat / "judge-qwen" / "summary.json", {"engines": {ENGINE: behavior}})
    write_json(
        repeat / "semantic-qwen" / "summary.json",
        {
            "categories": {
                "all": {
                    "interaction_handling": {"accepted": 10, "failed": 10},
                    "grounding": {"accepted": 15, "failed": 5},
                }
            }
        },
    )
    samples = repeat / "scores" / "engines" / ENGINE / "samples"
    for index in range(pairs):
        write_timing(
            samples / "category" / str(index),
            [[0.0, STOP_INTERVAL_S]],
            [[0.0, RESPONSE_INTERVAL_S]],
        )
    write_json(
        repeat / "v10" / "reference" / "unified.json",
        {
            "subsets": {
                subset: {
                    "counts": {"selected": fdb_ci.V10_PER_SUBSET},
                    "metrics": {
                        metric: {"count": 1, "n": 5, "value": 0.2}
                        for metric in ("pause_intrusion_rate", "response_rate_2s")
                    },
                }
                for subset in fdb_ci.V10_SUBSETS
            }
        },
    )
    write_json(
        repeat / "listening" / "summary.json",
        {
            "datasets": {
                "v1.5": {
                    "all": {
                        "omission_rate": 0.05,
                        "utmos_mean": 4.2,
                        "voice_similarity_mean": 70.0,
                    }
                }
            }
        },
    )
    (tmp_path / "RESULTS.md").write_text("Fixture results\n")
    return tmp_path


REFERENCES = {
    "v15_respond_on_interruption": 0.6,
    "v15_resume_on_backchannel": 0.4,
    "v15_resume_on_talking_to_other": 0.4,
    "v15_resume_on_background_speech": 0.4,
    "v15_overlap_handled": 0.5,
    "v15_grounded_reply": 0.75,
    "v15_stop_latency_mean_s": STOP_INTERVAL_S,
    "v15_response_latency_mean_s": RESPONSE_INTERVAL_S,
    "v10_synthetic_pause_intrusion_rate": 0.2,
    "v10_candor_pause_intrusion_rate": 0.2,
    "v10_turn_response_rate_2s": 0.2,
    "listening_omission_rate": 0.05,
    "listening_utmos_mean": 4.2,
    "listening_voice_similarity_mean": 70.0,
}


def gate(run_root: Path, should_run_semantic_judge: bool = True) -> None:
    metrics = fdb_ci.collect_fdb_metrics(run_root, ENGINE, should_run_semantic_judge)
    checks = MetricCheckCollector("FDB fixture")
    fdb_ci.check_no_regression(checks, metrics, REFERENCES, should_run_semantic_judge)
    checks.assert_all()


def test_the_reference_run_passes(run_root: Path) -> None:
    gate(run_root)


@pytest.mark.parametrize(
    "category,label,allowed_count,rejected_count",
    [
        pytest.param("user_interruption", "C_RESPOND", 2, 1, id="higher-is-better"),
        pytest.param("user_backchannel", "C_RESUME", 1, 0, id="resume"),
    ],
)
def test_a_behavior_rate_tolerates_one_sample(
    run_root: Path,
    category: str,
    label: str,
    allowed_count: int,
    rejected_count: int,
) -> None:
    path = run_root / "repeat-1" / "judge-qwen" / "summary.json"
    document = read_json(path)
    proportions = document["engines"][ENGINE][category]["valid_label_proportions"]
    proportions[label]["count"] = allowed_count
    write_json(path, document)
    gate(run_root)
    proportions[label]["count"] = rejected_count
    write_json(path, document)
    with pytest.raises(AssertionError, match="<"):
        gate(run_root)


def test_a_lower_is_better_rate_tolerates_one_sample(run_root: Path) -> None:
    path = run_root / "repeat-1" / "v10" / "reference" / "unified.json"
    document = read_json(path)
    rate = document["subsets"]["synthetic_pause_handling"]["metrics"][
        "pause_intrusion_rate"
    ]
    rate.update(count=2, value=0.4)
    write_json(path, document)
    gate(run_root)
    rate.update(count=3, value=0.6)
    write_json(path, document)
    with pytest.raises(AssertionError, match="v10_synthetic_pause_intrusion_rate.*>"):
        gate(run_root)


@pytest.mark.parametrize("metric", ["latency_stop_list", "latency_resp_list"])
def test_latency_gate_rejects_regression(run_root: Path, metric: str) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / ENGINE / "samples"
    for path in samples.glob("*/*/latency_intervals.json"):
        document = read_json(path)
        document[metric] = [[0.0, 10.0]]
        write_timing(
            path.parent, document["latency_stop_list"], document["latency_resp_list"]
        )
    with pytest.raises(AssertionError, match="latency_mean_s: .* >"):
        gate(run_root)


@pytest.mark.parametrize(
    "intervals",
    [[], [[0.0, float("nan")]], [[2.0, 1.0]]],
    ids=["empty", "nan", "negative"],
)
def test_invalid_timing_cannot_pass(
    run_root: Path, intervals: list[list[float]]
) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / ENGINE / "samples"
    for path in samples.glob("*/*/latency_intervals.json"):
        write_timing(path.parent, intervals, [[0.0, RESPONSE_INTERVAL_S]])
    with pytest.raises(AssertionError, match="latency_stop_list|v15_stop_latency"):
        gate(run_root)


def test_changed_timing_receipt_fails(run_root: Path) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / ENGINE / "samples"
    path = next(samples.glob("*/*/latency_intervals.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AssertionError, match="changed after scoring"):
        gate(run_root)


def test_one_client_pacing_failure_passes_and_a_protocol_failure_does_not(
    run_root: Path,
) -> None:
    path = run_root / "repeat-1" / "recording" / "shard-0" / "run.json"
    document = read_json(path)
    variants = document["samples"][0]["variants"]
    variants["overlap"] = {
        "status": "fail",
        "protocol_verdict": "pass",
        "errors": ["input pacing deviation 0.148s at append 24 exceeds 0.08s"],
    }
    write_json(path, document)
    gate(run_root)
    variants["clean"] = variants["overlap"]
    write_json(path, document)
    with pytest.raises(AssertionError, match="client pacing failures"):
        gate(run_root)
    variants["clean"] = {"status": "fail", "protocol_verdict": "fail", "errors": []}
    write_json(path, document)
    with pytest.raises(AssertionError, match="protocol failures"):
        gate(run_root)


def test_a_rate_without_scored_samples_fails(run_root: Path) -> None:
    path = run_root / "repeat-1" / "v10" / "reference" / "unified.json"
    document = read_json(path)
    document["subsets"]["candor_turn_taking"]["metrics"]["response_rate_2s"] = {
        "count": 0,
        "n": 0,
        "value": None,
    }
    write_json(path, document)
    with pytest.raises(AssertionError, match="v10_turn_response_rate_2s: no scored"):
        gate(run_root)


def test_the_semantic_metrics_follow_the_semantic_judge_switch(run_root: Path) -> None:
    """Gated and required when the judge runs; neither when it is skipped."""
    semantic = run_root / "repeat-1" / "semantic-qwen" / "summary.json"
    document = read_json(semantic)
    document["categories"]["all"]["grounding"] = {"accepted": 0, "failed": 20}
    write_json(semantic, document)
    with pytest.raises(AssertionError, match="v15_grounded_reply"):
        gate(run_root)
    with pytest.raises(AssertionError, match="semantic-qwen"):
        gate(run_root, should_run_semantic_judge=False)
    semantic.unlink()
    gate(run_root, should_run_semantic_judge=False)
    with pytest.raises(FileNotFoundError):
        gate(run_root)


@pytest.mark.parametrize("strategy_name", list(fdb_ci_config.FDB_CI_PRESETS))
def test_every_strategy_references_every_metric(strategy_name: str) -> None:
    preset = fdb_ci_config.FDB_CI_PRESETS[strategy_name]
    assert set(preset.references) == set(fdb_ci.METRIC_SPECS)


def test_every_duplex_ci_job_selects_a_known_strategy() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "test-duplex-ci.yaml").read_text()
    )
    selected = [
        step["env"]["FDB_CI_STRATEGY"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "FDB_CI_STRATEGY" in step.get("env", {})
    ]
    assert sorted(selected) == sorted(fdb_ci_config.FDB_CI_PRESETS)
