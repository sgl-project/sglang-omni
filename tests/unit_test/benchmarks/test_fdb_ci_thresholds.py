# SPDX-License-Identifier: Apache-2.0
"""FDB CI rejects quality regressions and incomplete scoring evidence."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from tests.test_model import test_minicpmo_fdb_ci as fdb_ci


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
def test_smoke_is_selected_only_by_dedicated_stage(
    marker_arguments: list[str], expected_exit_code: pytest.ExitCode, selected: bool
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            str(Path(fdb_ci.__file__).resolve()),
            *marker_arguments,
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected_exit_code, result.stdout + result.stderr
    assert ("::test_minicpmo_full_duplex_bench_smoke" in result.stdout) == selected


def write_json(path: Path, document: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


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


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    repeat = tmp_path / "repeat-1"
    for directory, count in (("recording", 16), ("v10/recording", 25)):
        write_json(
            repeat / directory / "shard-0" / "run.json",
            {
                "samples": [
                    {"variants": {"overlap": {"status": "pass"}}} for _ in range(count)
                ]
            },
        )
    write_json(
        repeat / "reference-audio" / "reference-manifest.json",
        {"counts": {"eligible_pairs": 8}},
    )
    for index in range(8):
        sample = (
            repeat
            / "scores"
            / "engines"
            / "minicpmo"
            / "samples"
            / "category"
            / str(index)
        )
        write_timing(sample, [[0.0, 1.0]], [[0.0, 2.0], [3.0, 5.0]])
    baseline_rates = (
        ("candor_turn_taking", "response_rate_5s", 5, 5),
        ("synthetic_user_interruption", "speaking_at_end_rate", 2, 4),
        ("synthetic_pause_handling", "pause_intrusion_rate", 1, 5),
        ("candor_pause_handling", "pause_intrusion_rate", 0, 5),
    )
    write_json(
        repeat / "v10" / "reference" / "unified.json",
        {
            "subsets": {
                subset: {
                    "counts": {"selected": 5, "eligible": 5},
                    "metrics": {metric: {"count": count, "n": n, "value": count / n}},
                }
                for subset, metric, count, n in baseline_rates
            }
        },
    )
    (tmp_path / "RESULTS.md").write_text("Fixture results\n")
    return tmp_path


def test_calibrated_results_pass(run_root: Path) -> None:
    fdb_ci.assert_fdb_results(run_root)


@pytest.mark.parametrize(
    "subset,metric,allowed_count,rejected_count,n",
    [
        ("candor_turn_taking", "response_rate_5s", 4, 3, 5),
        ("synthetic_user_interruption", "speaking_at_end_rate", 3, 4, 4),
        ("synthetic_pause_handling", "pause_intrusion_rate", 2, 3, 5),
        ("candor_pause_handling", "pause_intrusion_rate", 1, 2, 5),
    ],
)
def test_rate_gate_allows_one_additional_bad_outcome(
    run_root: Path,
    subset: str,
    metric: str,
    allowed_count: int,
    rejected_count: int,
    n: int,
) -> None:
    path = run_root / "repeat-1" / "v10" / "reference" / "unified.json"
    document = json.loads(path.read_text())
    document["subsets"][subset]["metrics"][metric] = {
        "count": allowed_count,
        "n": n,
        "value": allowed_count / n,
    }
    write_json(path, document)
    fdb_ci.assert_fdb_results(run_root)
    document["subsets"][subset]["metrics"][metric] = {
        "count": rejected_count,
        "n": n,
        "value": rejected_count / n,
    }
    write_json(path, document)
    with pytest.raises(AssertionError, match="outside"):
        fdb_ci.assert_fdb_results(run_root)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_interruption_requires_exercised_samples(run_root: Path, n: int) -> None:
    path = run_root / "repeat-1" / "v10" / "reference" / "unified.json"
    document = json.loads(path.read_text())
    document["subsets"]["synthetic_user_interruption"]["metrics"][
        "speaking_at_end_rate"
    ] = {"count": 0, "n": n, "value": 0.0 if n else None}
    write_json(path, document)
    with pytest.raises(AssertionError, match="insufficient metric coverage"):
        fdb_ci.assert_fdb_results(run_root)


@pytest.mark.parametrize("metric", ["latency_stop_list", "latency_resp_list"])
def test_latency_gate_rejects_regression(run_root: Path, metric: str) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / "minicpmo" / "samples"
    for path in samples.glob("*/*/latency_intervals.json"):
        document = json.loads(path.read_text())
        document[metric] = [[0.0, 4.0], [5.0, 9.0]]
        write_timing(
            path.parent, document["latency_stop_list"], document["latency_resp_list"]
        )
    with pytest.raises(AssertionError, match=f"{metric}:.* > .* s"):
        fdb_ci.assert_fdb_results(run_root)


@pytest.mark.parametrize("metric", ["latency_stop_list", "latency_resp_list"])
@pytest.mark.parametrize("intervals", [[], [[0.0, float("nan")]], [[2.0, 1.0]]])
def test_invalid_timing_cannot_pass(
    run_root: Path, metric: str, intervals: list[list[float]]
) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / "minicpmo" / "samples"
    for path in samples.glob("*/*/latency_intervals.json"):
        document = json.loads(path.read_text())
        document[metric] = intervals * 2
        write_timing(
            path.parent, document["latency_stop_list"], document["latency_resp_list"]
        )
    with pytest.raises(AssertionError, match=metric):
        fdb_ci.assert_fdb_results(run_root)


def test_changed_timing_receipt_fails(run_root: Path) -> None:
    samples = run_root / "repeat-1" / "scores" / "engines" / "minicpmo" / "samples"
    path = next(samples.glob("*/*/latency_intervals.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AssertionError, match="changed after scoring"):
        fdb_ci.assert_fdb_results(run_root)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0.0])
def test_missing_or_inconsistent_rate_fails(
    run_root: Path, value: float | None
) -> None:
    path = run_root / "repeat-1" / "v10" / "reference" / "unified.json"
    document = json.loads(path.read_text())
    document["subsets"]["candor_turn_taking"]["metrics"]["response_rate_5s"][
        "value"
    ] = value
    write_json(path, document)
    with pytest.raises(AssertionError, match="response_rate_5s"):
        fdb_ci.assert_fdb_results(run_root)


def test_protocol_failure_is_still_a_hard_gate(run_root: Path) -> None:
    path = run_root / "repeat-1" / "recording" / "shard-0" / "run.json"
    document = json.loads(path.read_text())
    document["samples"][0]["variants"]["overlap"]["status"] = "fail"
    write_json(path, document)
    with pytest.raises(AssertionError):
        fdb_ci.assert_fdb_results(run_root)
