# SPDX-License-Identifier: Apache-2.0
"""JSON-only Higgs TTS stage-3 checks for GitHub-hosted runners."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.utils import (
    MetricCheckCollector,
    assert_per_request_fields,
    assert_streaming_consistency,
    assert_summary_metrics,
)

HIGGS_STAGE1_SPEED_RESULTS_DIR_ENV = "HIGGS_STAGE1_SPEED_RESULTS_DIR"
HIGGS_STAGE2_SPEED_RESULTS_DIR_ENV = "HIGGS_STAGE2_SPEED_RESULTS_DIR"
HIGGS_CONSISTENCY_CONCURRENCY_ENV = "HIGGS_CONSISTENCY_CONCURRENCY"
DEFAULT_CONCURRENCY = 16


def _load_speed_results(results_path: Path) -> dict:
    assert results_path.exists(), f"Speed results file not found: {results_path}"
    with results_path.open(encoding="utf-8") as f:
        speed_results = json.load(f)
    assert (
        "summary" in speed_results
    ), f"Missing 'summary' key in speed results. Keys: {list(speed_results.keys())}"
    assert "per_request" in speed_results, (
        "Missing 'per_request' key in speed results. "
        f"Keys: {list(speed_results.keys())}"
    )
    return speed_results


def _find_results(artifact_root: str, output_dir_name: str) -> dict:
    root = Path(artifact_root)
    matches = sorted(root.rglob(f"{output_dir_name}/speed_results.json"))
    assert (
        matches
    ), f"Downloaded speed results not found under {artifact_root}: {output_dir_name}"
    return _load_speed_results(matches[0])


def _concurrency() -> int:
    return int(os.environ.get(HIGGS_CONSISTENCY_CONCURRENCY_ENV, DEFAULT_CONCURRENCY))


@pytest.mark.benchmark
def test_higgs_streaming_consistency_from_artifacts() -> None:
    non_stream_root = os.environ.get(HIGGS_STAGE1_SPEED_RESULTS_DIR_ENV)
    stream_root = os.environ.get(HIGGS_STAGE2_SPEED_RESULTS_DIR_ENV)
    assert non_stream_root, f"{HIGGS_STAGE1_SPEED_RESULTS_DIR_ENV} is required"
    assert stream_root, f"{HIGGS_STAGE2_SPEED_RESULTS_DIR_ENV} is required"

    concurrency = _concurrency()
    non_stream_results = _find_results(
        non_stream_root,
        f"higgs_nonstream_c{concurrency}",
    )
    stream_results = _find_results(
        stream_root,
        f"higgs_stream_c{concurrency}",
    )

    checks = MetricCheckCollector("Higgs artifact streaming consistency")
    assert_summary_metrics(non_stream_results["summary"], collector=checks)
    assert_summary_metrics(stream_results["summary"], collector=checks)
    assert_per_request_fields(non_stream_results["per_request"], collector=checks)
    assert_per_request_fields(stream_results["per_request"], collector=checks)
    assert_streaming_consistency(
        non_stream_results["per_request"],
        stream_results["per_request"],
        collector=checks,
    )
    checks.assert_all()
