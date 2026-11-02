# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o FDB protocol, coverage, and calibrated regression smoke on two H100s."""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download
from pydantic import JsonValue

from benchmarks.duplex.fdb.common import DEFAULT_MODEL_REVISION, MODEL_ID
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v10_dataset import SUBSET_TASKS
from benchmarks.duplex.v15_dataset import SUBSETS as V15_CATEGORIES

pytestmark = pytest.mark.benchmark

RUN_NAME = "ci"
V15_PER_SUBSET = 2
V10_PER_SUBSET = 5
GPUS = "0,1"

# Note (wenyao): PR #2864's five H100 DP2 runs fix the raw baseline independently of CI slack.
STOP_MEAN_S_REF = 1.432
RESPONSE_MEAN_S_REF = 2.7152727272727275
TURN_RESPONSE_RATE_REF = 1.0
INTERRUPTION_SPEAKING_AT_END_RATE_REF = 0.5
SYNTHETIC_PAUSE_INTRUSION_RATE_REF = 0.2
CANDOR_PAUSE_INTRUSION_RATE_REF = 0.0
LATENCY_TOLERANCE = 1.25
RATE_SAMPLE_TOLERANCE = 1

TIMING_LIMITS = {
    "latency_stop_list": (6, STOP_MEAN_S_REF * LATENCY_TOLERANCE),
    "latency_resp_list": (9, RESPONSE_MEAN_S_REF * LATENCY_TOLERANCE),
}
RATE_LIMITS = (
    ("candor_turn_taking", "response_rate_5s", 5, TURN_RESPONSE_RATE_REF, 1.0),
    (
        "synthetic_user_interruption",
        "speaking_at_end_rate",
        4,
        0.0,
        INTERRUPTION_SPEAKING_AT_END_RATE_REF,
    ),
    (
        "synthetic_pause_handling",
        "pause_intrusion_rate",
        5,
        0.0,
        SYNTHETIC_PAUSE_INTRUSION_RATE_REF,
    ),
    (
        "candor_pause_handling",
        "pause_intrusion_rate",
        5,
        0.0,
        CANDOR_PAUSE_INTRUSION_RATE_REF,
    ),
)


def read_json(path: Path) -> dict[str, JsonValue]:
    return json.loads(path.read_text())


def runbook(env: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "benchmarks.duplex.fdb", *arguments],
        env=env,
        check=True,
    )


def variant_statuses(recording: Path) -> Counter[str]:
    return Counter(
        variant["status"]
        for run in recording.glob("shard-*/run.json")
        for sample in read_json(run)["samples"]
        for variant in sample["variants"].values()
    )


def test_minicpmo_full_duplex_bench_smoke() -> None:
    fdb_work = Path(os.environ["OMNI_CI_HOME"]) / "fdb"
    env = {
        **os.environ,
        "FDB_WORK": str(fdb_work),
        "MODEL_PATH": snapshot_download(MODEL_ID, revision=DEFAULT_MODEL_REVISION),
        # note (luojiaxuan): with JUDGE=gpt, setup skips the 27B judge model; the
        # smoke never runs the judge step, so no API key is read.
        "JUDGE": "gpt",
    }
    run_root = fdb_work / "runs" / RUN_NAME
    # note (luojiaxuan): generate refuses a repeat that already holds a recording,
    # so a retried attempt starts from an empty run.
    if run_root.exists():
        shutil.rmtree(run_root)
    else:
        pass
    runbook(env, "setup")
    runbook(
        env,
        "generate",
        "--run-name",
        RUN_NAME,
        "--v15-per-subset",
        str(V15_PER_SUBSET),
        "--v10-per-subset",
        str(V10_PER_SUBSET),
        "--gpus",
        GPUS,
        "--num-shards",
        str(len(GPUS.split(","))),
    )
    runbook(env, "asr", "--run-name", RUN_NAME)
    runbook(env, "aggregate", "--run-name", RUN_NAME)
    assert_fdb_results(run_root)


def assert_fdb_results(run_root: Path) -> None:
    repeat = run_root / "repeat-1"
    pairs = V15_PER_SUBSET * len(V15_CATEGORIES)
    v10_samples = V10_PER_SUBSET * len(SUBSET_TASKS)
    assert variant_statuses(repeat / "recording") == {"pass": 2 * pairs}
    assert variant_statuses(repeat / "v10" / "recording") == {"pass": v10_samples}
    manifest = read_json(repeat / "reference-audio" / "reference-manifest.json")
    assert manifest["counts"]["eligible_pairs"] == pairs

    samples = repeat / "scores" / "engines" / "minicpmo" / "samples"
    receipts = list(samples.glob("*/*/receipts/timing-overlap.json"))
    assert Counter(read_json(path)["status"] for path in receipts) == {"ok": pairs}
    intervals: dict[str, list[float]] = {name: [] for name in TIMING_LIMITS}
    for path in receipts:
        interval_path = path.parent.parent / "latency_intervals.json"
        assert (
            file_sha256(interval_path) == read_json(path)["intervals_sha256"]
        ), f"Timing intervals changed after scoring: {interval_path}"
        document = read_json(interval_path)
        for name in intervals:
            intervals[name].extend(end - start for start, end in document[name])
    for name, (minimum_intervals, maximum_mean_s) in TIMING_LIMITS.items():
        values = intervals[name]
        assert (
            len(values) >= minimum_intervals
        ), f"{name}: insufficient timing intervals"
        assert all(math.isfinite(value) and value >= 0 for value in values), name
        mean_s = statistics.fmean(values)
        assert mean_s <= maximum_mean_s, f"{name}: {mean_s} > {maximum_mean_s} s"

    unified = read_json(repeat / "v10" / "reference" / "unified.json")
    # note (luojiaxuan): the unified scorer leaves backchannel to the official
    # evaluator, which needs the judge step.
    assert set(unified["subsets"]) == {
        subset for subset, task in SUBSET_TASKS.items() if task != "backchannel"
    }
    for subset, entry in unified["subsets"].items():
        assert entry["counts"]["selected"] == V10_PER_SUBSET, subset
        assert entry["counts"]["eligible"] == V10_PER_SUBSET, subset
    for subset, metric, minimum_n, reference_lower, reference_upper in RATE_LIMITS:
        rate = unified["subsets"][subset]["metrics"][metric]
        count, n, value = rate["count"], rate["n"], rate["value"]
        assert (
            isinstance(n, int) and minimum_n <= n <= V10_PER_SUBSET
        ), f"{subset}/{metric}: insufficient metric coverage, n={n}"
        assert (
            isinstance(count, int) and 0 <= count <= n
        ), f"{subset}/{metric}: {count}/{n}"
        assert isinstance(value, (int, float)) and math.isfinite(value), metric
        assert math.isclose(value, count / n), f"{subset}/{metric}: inconsistent rate"
        tolerance = RATE_SAMPLE_TOLERANCE / minimum_n
        lower = max(0.0, reference_lower - tolerance)
        upper = min(1.0, reference_upper + tolerance)
        assert (
            lower <= value <= upper
        ), f"{subset}/{metric}: {value} outside [{lower}, {upper}]"
    assert (run_root / "RESULTS.md").is_file()
