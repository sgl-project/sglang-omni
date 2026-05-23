# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils import (
    MetricCheckCollector,
    assert_speed_thresholds,
    assert_summary_metrics,
)


def test_speed_thresholds_report_all_failed_metrics() -> None:
    summary = {
        "throughput_qps": 0.5,
        "tok_per_s_agg": 1.0,
        "latency_mean_s": 30.0,
        "rtf_mean": 4.0,
    }
    thresholds = {
        16: {
            "throughput_qps_min": 1.0,
            "tok_per_s_agg_min": 2.0,
            "latency_mean_s_max": 10.0,
            "rtf_mean_max": 2.0,
        }
    }

    with pytest.raises(AssertionError) as exc_info:
        assert_speed_thresholds(summary, thresholds, 16)

    message = str(exc_info.value)
    assert "speed thresholds failed 4 check(s)" in message
    assert "throughput_qps 0.5 < 1.0" in message
    assert "tok_per_s_agg 1.0 < 2.0" in message
    assert "latency_mean_s 30.0 > 10.0" in message
    assert "rtf_mean 4.0 > 2.0" in message


def test_shared_collector_combines_multiple_helpers() -> None:
    checks = MetricCheckCollector("combined metrics")
    assert_summary_metrics(
        {
            "failed_requests": 2,
            "audio_duration_mean_s": 0,
            "gen_tokens_mean": 0,
            "prompt_tokens_mean": 0,
        },
        collector=checks,
    )
    assert_speed_thresholds(
        {
            "throughput_qps": 0.5,
            "tok_per_s_agg": 1.0,
            "latency_mean_s": 30.0,
        },
        {
            16: {
                "throughput_qps_min": 1.0,
                "tok_per_s_agg_min": 2.0,
                "latency_mean_s_max": 10.0,
            }
        },
        16,
        collector=checks,
    )

    with pytest.raises(AssertionError) as exc_info:
        checks.assert_all()

    message = str(exc_info.value)
    assert "combined metrics failed 7 check(s)" in message
    assert "Expected 0 failed requests, got 2" in message
    assert "Expected positive audio duration, got 0" in message
    assert "throughput_qps 0.5 < 1.0" in message
