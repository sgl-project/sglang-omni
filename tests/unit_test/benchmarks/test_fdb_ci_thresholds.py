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
EVENT_SPAN_S = [1.0, 2.0]
SPEAKING_THROUGH_EVENT = [[0.5, 4.0]]
SPEAKING_AFTER_EVENT = [[2.5, 4.0]]


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
    sample: Path,
    stop: list[list[float]],
    response: list[list[float]],
    model_segments: list[list[float]] = SPEAKING_THROUGH_EVENT,
) -> None:
    intervals = sample / "latency_intervals.json"
    write_json(intervals, {"latency_stop_list": stop, "latency_resp_list": response})
    write_json(
        sample / "receipts" / "timing-overlap.json",
        {
            "status": "ok",
            "intervals_sha256": hashlib.sha256(intervals.read_bytes()).hexdigest(),
            "model_segments": model_segments,
        },
    )


def v15_sample_ids() -> list[str]:
    return [
        f"{category}/{index}"
        for category in fdb_ci.V15_CATEGORIES
        for index in range(1, fdb_ci.V15_PER_SUBSET + 1)
    ]


def passing_variant() -> dict[str, JsonValue]:
    return {"status": "pass", "protocol_verdict": "pass", "errors": []}


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """A complete scored run whose every metric equals REFERENCES."""
    repeat = tmp_path / "repeat-1"
    v10_samples = fdb_ci.V10_PER_SUBSET * len(fdb_ci.V10_SUBSETS)
    write_json(
        repeat / "recording" / "shard-0" / "run.json",
        {
            "samples": [
                {
                    "id": sample_id,
                    "event_span_s": EVENT_SPAN_S,
                    "variants": {
                        "overlap": passing_variant(),
                        "clean": passing_variant(),
                    },
                }
                for sample_id in v15_sample_ids()
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
                    "interaction_handling": {
                        "accepted": 10,
                        "failed": 10,
                        "unresolved": 0,
                    },
                    "grounding": {"accepted": 15, "failed": 5, "unresolved": 0},
                }
            }
        },
    )
    samples = repeat / "scores" / "engines" / ENGINE / "samples"
    for sample_id in v15_sample_ids():
        write_timing(
            samples / sample_id,
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


def gate(
    run_root: Path,
    should_run_semantic_judge: bool = True,
    references: dict[str, float] = REFERENCES,
) -> None:
    metrics = fdb_ci.collect_fdb_metrics(run_root, ENGINE, should_run_semantic_judge)
    checks = MetricCheckCollector("FDB fixture")
    fdb_ci.check_overlaps_exercised(
        checks, fdb_ci.count_exercised_overlaps(run_root, ENGINE)
    )
    fdb_ci.check_no_regression(checks, metrics, references)
    checks.assert_all()


def test_the_reference_run_passes(run_root: Path) -> None:
    gate(run_root)


def set_behavior_count(run_root: Path, category: str, label: str, count: int) -> None:
    path = run_root / "repeat-1" / "judge-qwen" / "summary.json"
    document = read_json(path)
    proportions = document["engines"][ENGINE][category]["valid_label_proportions"]
    proportions[label]["count"] = count
    write_json(path, document)


def set_v10_count(run_root: Path, subset: str, metric: str, count: int) -> None:
    path = run_root / "repeat-1" / "v10" / "reference" / "unified.json"
    document = read_json(path)
    document["subsets"][subset]["metrics"][metric].update(
        count=count, value=count / fdb_ci.V10_PER_SUBSET
    )
    write_json(path, document)


def test_the_rates_share_one_slack_and_a_lower_is_better_rate_counts_inverted(
    run_root: Path,
) -> None:
    """The reference run has 18 of its 35 rate cases on target; five fewer pass
    and six fewer fail, where two more pause intrusions are two cases off target."""
    set_v10_count(run_root, "synthetic_pause_handling", "pause_intrusion_rate", 3)
    set_behavior_count(run_root, "user_interruption", "C_RESPOND", 1)
    set_v10_count(run_root, "candor_turn_taking", "response_rate_2s", 0)
    gate(run_root)
    set_behavior_count(run_root, "user_backchannel", "C_RESUME", 1)
    with pytest.raises(
        AssertionError, match="pooled rates: 12/35 cases on target < 13/35"
    ):
        gate(run_root)


def test_a_reference_between_counts_keeps_its_fraction(run_root: Path) -> None:
    """A pause intrusion reference of 0.1 stands for 4.5 cases on target, not the
    5 that rounding half to even gives, so the pool expects 19 cases, not 20."""
    references = {
        **REFERENCES,
        "v10_synthetic_pause_intrusion_rate": 0.1,
        "v10_candor_pause_intrusion_rate": 0.1,
    }
    set_behavior_count(run_root, "user_interruption", "C_RESPOND", 1)
    set_behavior_count(run_root, "user_backchannel", "C_RESUME", 0)
    gate(run_root, references=references)
    set_v10_count(run_root, "candor_turn_taking", "response_rate_2s", 0)
    with pytest.raises(
        AssertionError, match="pooled rates: 13/35 cases on target < 14/35"
    ):
        gate(run_root, references=references)


def test_a_rate_with_an_unscored_case_fails(run_root: Path) -> None:
    """A pair the judge could not label shrinks n and would shift the pool."""
    path = run_root / "repeat-1" / "judge-qwen" / "summary.json"
    document = read_json(path)
    document["engines"][ENGINE]["user_backchannel"]["valid_n"] = 4
    write_json(path, document)
    with pytest.raises(
        AssertionError, match="v15_resume_on_backchannel: scored 4 of 5 cases"
    ):
        gate(run_root)


def set_model_segments(
    run_root: Path, sample_ids: list[str], model_segments: list[list[float]]
) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / ENGINE / "samples"
    for sample_id in sample_ids:
        write_timing(
            samples / sample_id,
            [[0.0, STOP_INTERVAL_S]],
            [[0.0, RESPONSE_INTERVAL_S]],
            model_segments,
        )


def test_a_model_silent_when_the_event_began_did_not_exercise_the_overlap(
    run_root: Path,
) -> None:
    """Three pairs of a category may start speaking after their event, not a
    fourth; speech starting at the event's onset counts, speech ending there not."""
    set_model_segments(
        run_root,
        ["user_interruption/1", "user_interruption/2"],
        SPEAKING_AFTER_EVENT,
    )
    set_model_segments(run_root, ["user_interruption/3"], [[EVENT_SPAN_S[0], 4.0]])
    gate(run_root)
    set_model_segments(run_root, ["user_interruption/3"], [[0.0, EVENT_SPAN_S[0]]])
    with pytest.raises(
        AssertionError, match="user_interruption: overlap exercised in 2/5 pairs < 3"
    ):
        gate(run_root)


def test_too_few_exercised_overlaps_fail_though_every_category_has_three(
    run_root: Path,
) -> None:
    late = [
        f"{category}/{index}" for category in fdb_ci.V15_CATEGORIES for index in (1, 2)
    ]
    set_model_segments(run_root, late[:5], SPEAKING_AFTER_EVENT)
    gate(run_root)
    set_model_segments(run_root, late[5:], SPEAKING_AFTER_EVENT)
    with pytest.raises(AssertionError, match="overlap exercised in 12/20 pairs < 15"):
        gate(run_root)


def test_a_sample_without_event_timestamps_fails(run_root: Path) -> None:
    path = run_root / "repeat-1" / "recording" / "shard-0" / "run.json"
    document = read_json(path)
    document["samples"][0]["event_span_s"] = None
    write_json(path, document)
    with pytest.raises(AssertionError, match="no event timestamps"):
        gate(run_root)


@pytest.mark.parametrize(
    "reference_name,reference,emptied,should_pass",
    [
        pytest.param(
            "v15_resume_on_background_speech",
            0.8,
            ("background_speech", "C_RESUME", 0),
            False,
            id="higher-is-better",
        ),
        pytest.param(
            "v10_synthetic_pause_intrusion_rate",
            0.2,
            ("synthetic_pause_handling", "pause_intrusion_rate", 5),
            False,
            id="lower-is-better",
        ),
        pytest.param(
            "v15_resume_on_background_speech",
            0.4,
            ("background_speech", "C_RESUME", 0),
            True,
            id="below-the-floor-reference",
        ),
    ],
)
def test_a_rate_that_loses_every_case_fails_inside_the_pooled_slack(
    run_root: Path,
    reference_name: str,
    reference: float,
    emptied: tuple[str, str, int],
    should_pass: bool,
) -> None:
    """A rate whose reference met at least three of five cases fails at none,
    though the pool alone would absorb that drop."""
    subset, metric, count = emptied
    if subset in fdb_ci.V15_CATEGORIES:
        set_behavior_count(run_root, subset, metric, count)
    else:
        set_v10_count(run_root, subset, metric, count)
    references = {**REFERENCES, reference_name: reference}
    metrics = fdb_ci.collect_fdb_metrics(run_root, ENGINE, True)
    checks = MetricCheckCollector("FDB fixture")
    fdb_ci.check_no_regression(checks, metrics, references)
    if should_pass:
        checks.assert_all()
    else:
        with pytest.raises(AssertionError, match=f"{reference_name}: .* collapsed"):
            checks.assert_all()


def test_the_semantic_rates_are_reported_and_never_gated(run_root: Path) -> None:
    """A judge that resolves one case of twenty reports 0/20, not 0/1, and that
    rate fails no gate; the rates are still required whenever the judge runs."""
    semantic = run_root / "repeat-1" / "semantic-qwen" / "summary.json"
    document = read_json(semantic)
    document["categories"]["all"]["grounding"] = {
        "accepted": 0,
        "failed": 1,
        "unresolved": 19,
    }
    write_json(semantic, document)
    metrics = fdb_ci.collect_fdb_metrics(run_root, ENGINE, True)
    assert (metrics["v15_grounded_reply"].count, metrics["v15_grounded_reply"].n) == (
        0,
        20,
    )
    gate(run_root)
    with pytest.raises(AssertionError, match="semantic-qwen"):
        gate(run_root, should_run_semantic_judge=False)
    semantic.unlink()
    gate(run_root, should_run_semantic_judge=False)
    with pytest.raises(FileNotFoundError):
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


def test_a_not_exercised_session_is_counted_apart_from_client_pacing(
    run_root: Path,
) -> None:
    """A not_exercised session kept the protocol, and a failure that names no
    pacing error is a protocol failure."""
    path = run_root / "repeat-1" / "recording" / "shard-0" / "run.json"
    document = read_json(path)
    variants = document["samples"][0]["variants"]
    variants["overlap"] = {
        "status": "not_exercised",
        "protocol_verdict": "pass",
        "errors": [],
    }
    write_json(path, document)
    outcomes = fdb_ci.variant_outcomes(run_root / "repeat-1" / "recording")
    assert (outcomes["not_exercised"], outcomes["client_pacing"]) == (1, 0)
    gate(run_root)
    variants["clean"] = {"status": "fail", "protocol_verdict": "pass", "errors": []}
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


@pytest.mark.parametrize("strategy_name", list(fdb_ci_config.FDB_CI_PRESETS))
def test_every_strategy_references_every_metric(strategy_name: str) -> None:
    preset = fdb_ci_config.FDB_CI_PRESETS[strategy_name]
    assert set(preset.references) == set(fdb_ci.METRIC_SPECS)


@pytest.mark.parametrize("strategy_name", list(fdb_ci_config.FDB_CI_PRESETS))
def test_every_rate_reference_is_a_share_of_its_cases(strategy_name: str) -> None:
    references = fdb_ci_config.FDB_CI_PRESETS[strategy_name].references
    for name, spec in fdb_ci.METRIC_SPECS.items():
        if spec.kind == "rate":
            assert 0 <= references[name] <= 1, name
        else:
            pass


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
