# SPDX-License-Identifier: Apache-2.0
"""Full-Duplex-Bench CI for one interaction strategy, selected by FDB_CI_STRATEGY.

Usage:
    FDB_CI_STRATEGY=turn-based pytest tests/test_model/test_fdb_ci.py -s -x
"""

import sys

import pytest

from tests.test_model.fdb_ci_config import select_fdb_ci_preset
from tests.utils import MetricCheckCollector
from tests.utils.fdb_ci import (
    METRIC_SPECS,
    V15_PER_SUBSET,
    check_no_regression,
    check_overlaps_exercised,
    collect_fdb_metrics,
    count_exercised_overlaps,
    run_fdb_ci,
)

STRATEGY_NAME, PRESET = select_fdb_ci_preset()


@pytest.mark.benchmark
def test_full_duplex_bench() -> None:
    """Record, score and judge the strategy's selection; the overlaps must be
    exercised and no metric may regress."""
    strategy = PRESET.strategy
    run_root = run_fdb_ci(strategy)
    metrics = collect_fdb_metrics(
        run_root, strategy.engine_name, strategy.should_run_semantic_judge
    )
    print(f"FDB {STRATEGY_NAME} ({strategy.engine_name}) metrics:")
    for name, measurement in metrics.items():
        rate = (
            "" if measurement.n is None else f" ({measurement.count}/{measurement.n})"
        )
        gating = "" if METRIC_SPECS[name].is_gated else ", reported only"
        print(
            f"  {name:36} {measurement.value:.4f}{rate}"
            f"  reference {PRESET.references[name]:.4f}{gating}"
        )
    exercised = count_exercised_overlaps(run_root, strategy.engine_name)
    for category, count in exercised.items():
        print(f"  {category + ' overlaps exercised':36} {count}/{V15_PER_SUBSET}")
    checks = MetricCheckCollector(f"FDB {STRATEGY_NAME}")
    check_overlaps_exercised(checks, exercised)
    if PRESET.calibrated:
        check_no_regression(checks, metrics, PRESET.references)
    else:
        checks.fail(
            f"FDB {STRATEGY_NAME} references are uncalibrated; the metrics above "
            "were collected but this stage cannot qualify until calibration is "
            "complete"
        )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
