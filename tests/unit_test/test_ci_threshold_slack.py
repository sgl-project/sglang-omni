# SPDX-License-Identifier: Apache-2.0
"""Contracts for the CI threshold derivation in tests/utils."""

from __future__ import annotations

import pytest

from tests.utils import apply_slack

SLACK_HIGHER = 0.875
SLACK_LOWER = 1.125


def test_gates_are_the_slacked_references() -> None:
    reference = {
        "throughput_qps": 79.438,
        "output_tok_per_req_s": 10.3,
        "latency_mean_s": 0.201,
        "rtf_mean": 0.2939,
    }

    thresholds = apply_slack({16: reference})[16]

    assert thresholds == pytest.approx(
        {
            "throughput_qps_min": 79.438 * SLACK_HIGHER,
            "output_tok_per_req_s_min": 10.3 * SLACK_HIGHER,
            "latency_mean_s_max": 0.201 * SLACK_LOWER,
            "rtf_mean_max": 0.2939 * SLACK_LOWER,
        }
    )
    # note (luojiaxuan): 0.201 s is the MMSU text reference, small enough that
    # a gate readable to one decimal sits below the reference it came from.
    assert thresholds["latency_mean_s_max"] > reference["latency_mean_s"]


@pytest.mark.parametrize("latency", [0.201, 7.964])
def test_a_latency_gate_is_neither_rounded_up_nor_down(latency: float) -> None:
    thresholds = apply_slack({16: {"throughput_qps": 78.8, "latency_mean_s": latency}})

    assert thresholds[16]["latency_mean_s_max"] == pytest.approx(latency * SLACK_LOWER)
