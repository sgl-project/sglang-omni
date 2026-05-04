# SPDX-License-Identifier: Apache-2.0
"""JSON-only S2-Pro stage-3 checks for GitHub-hosted runners.

Note (Chenyang):
To run locally, first run stage 1 and stage 2:

S2PRO_STAGE_OUTPUT_ROOT=$PWD/stage-results/nonstream \
pytest tests/test_model/test_s2pro_tts_ci.py -v -s -x --concurrency 8 \
  --s2pro-stage s2pro-stage-1-nonstream

S2PRO_STAGE_OUTPUT_ROOT=$PWD/stage-results/stream \
pytest tests/test_model/test_s2pro_tts_ci.py -v -s -x --concurrency 8 \
  --s2pro-stage s2pro-stage-2-stream

Then run this JSON-only stage 3 check:

S2PRO_STAGE1_SPEED_RESULTS_DIR=$PWD/stage-results/nonstream \
S2PRO_STAGE2_SPEED_RESULTS_DIR=$PWD/stage-results/stream \
pytest tests/test_model/test_s2pro_consistency_artifacts.py -v -s -x


Note (Chenyang):
    This stage does not compare generated token/content similarity or output
    quality. Those are covered by the earlier WER checks, and the CI DAG
    runs stage 3 only after stage 1 and stage 2 pass.

Author:
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.utils import assert_streaming_consistency

S2PRO_STAGE1_SPEED_RESULTS_DIR_ENV = "S2PRO_STAGE1_SPEED_RESULTS_DIR"
S2PRO_STAGE2_SPEED_RESULTS_DIR_ENV = "S2PRO_STAGE2_SPEED_RESULTS_DIR"
S2PRO_CONSISTENCY_CONCURRENCY_ENV = "S2PRO_CONSISTENCY_CONCURRENCY"
DEFAULT_CONSISTENCY_CONCURRENCY = 8
STREAMING_BENCHMARK_MAX_SAMPLES = 16


def _load_speed_results(results_root_env: str, output_dir_name: str) -> dict:
    results_root = os.environ.get(results_root_env)
    assert results_root, f"{results_root_env} must point to downloaded stage artifacts"

    matches = sorted(Path(results_root).rglob(f"{output_dir_name}/speed_results.json"))
    assert matches, f"Missing {output_dir_name}/speed_results.json under {results_root}"

    with open(matches[0]) as results_file:
        speed_results = json.load(results_file)
    assert "per_request" in speed_results, f"Missing 'per_request' key in {matches[0]}"
    return speed_results


def _selected_concurrency() -> int:
    option_value = os.environ.get(
        S2PRO_CONSISTENCY_CONCURRENCY_ENV,
        str(DEFAULT_CONSISTENCY_CONCURRENCY),
    )
    try:
        return int(option_value)
    except ValueError as exc:
        raise ValueError(
            f"{S2PRO_CONSISTENCY_CONCURRENCY_ENV} must be an integer"
        ) from exc


@pytest.mark.benchmark
def test_s2pro_streaming_consistency_from_artifacts() -> None:
    """Validate stage-1 (non-stream) vs stage-2 (stream) speed_results.json
    artifacts agree on structural invariants: prompt token counts, completion
    token counts, and audio duration within tolerance."""
    concurrency = _selected_concurrency()
    non_stream_results = _load_speed_results(
        S2PRO_STAGE1_SPEED_RESULTS_DIR_ENV,
        f"vc_nonstream_c{concurrency}",
    )
    stream_results = _load_speed_results(
        S2PRO_STAGE2_SPEED_RESULTS_DIR_ENV,
        f"vc_stream_c{concurrency}",
    )

    assert_streaming_consistency(
        non_stream_results["per_request"],
        stream_results["per_request"],
        expected_stream_count=STREAMING_BENCHMARK_MAX_SAMPLES,
    )
