# SPDX-License-Identifier: Apache-2.0
"""Contracts for the CI threshold derivation in tests/utils."""

from __future__ import annotations

import pytest

from tests.utils import apply_slack, readable_upper_gate

SLACK_LOWER = 1.125


def test_a_sub_second_reference_keeps_its_slack() -> None:
    # note (luojiaxuan): the MMSU text reference is 0.201 s, and rounding
    # 0.201 * 1.125 to one decimal gives 0.2, a gate below the reference it
    # came from, which fails a run that matches its own calibration.
    thresholds = apply_slack(
        {
            16: {
                "throughput_qps": 78.8,
                "output_tok_per_req_s": 10.2,
                "latency_mean_s": 0.201,
            }
        }
    )[16]

    assert thresholds["latency_mean_s_max"] == pytest.approx(0.201 * SLACK_LOWER)
    assert thresholds["latency_mean_s_max"] > 0.201


@pytest.mark.parametrize(
    ("reference", "digits", "rounds_up"),
    [
        (7.964, 1, True),
        (11.249, 1, True),
        (0.851, 1, True),
        (0.5662, 2, True),
        (0.201, 1, False),
        (0.2941, 2, False),
    ],
)
def test_rounding_only_ever_loosens(
    reference: float, digits: int, rounds_up: bool
) -> None:
    gate = readable_upper_gate(reference, SLACK_LOWER, digits)
    slacked = reference * SLACK_LOWER

    assert gate >= slacked
    # note (luojiaxuan): the readable value stays the gate wherever it is the
    # looser of the two, so no calibrated suite gets a stricter gate.
    assert gate >= round(slacked, digits)
    if rounds_up:
        assert gate == pytest.approx(round(slacked, digits))
    else:
        assert gate == pytest.approx(slacked)


def test_higher_is_better_gates_are_untouched() -> None:
    thresholds = apply_slack(
        {
            8: {
                "throughput_qps": 78.8,
                "output_tok_per_req_s": 10.2,
                "latency_mean_s": 7.964,
            }
        }
    )[8]

    assert thresholds["throughput_qps_min"] == pytest.approx(68.95, abs=1e-9)
    assert thresholds["output_tok_per_req_s_min"] == pytest.approx(8.9, abs=1e-9)
