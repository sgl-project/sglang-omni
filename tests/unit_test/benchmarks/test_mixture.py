# SPDX-License-Identifier: Apache-2.0
"""Fixed window summaries and sweep rows for mixed streaming and non streaming load."""

from __future__ import annotations

import pytest

from benchmarks.benchmarker.data import RequestClass, RequestResult
from benchmarks.metrics.mixture import (
    MixtureLoad,
    MixtureRepeatResult,
    aggregate_mixture_sweep,
    compute_mixture_summaries,
    stream_finish_gap_max_s,
)


def finished_request(
    request_class: RequestClass,
    finished_at_s: float,
    *,
    sent_at_s: float | None = None,
    is_success: bool = True,
    audio_duration_s: float = 2.0,
) -> RequestResult:
    return RequestResult(
        request_id=f"{request_class}-{finished_at_s}",
        is_success=is_success,
        latency_s=1.0,
        audio_duration_s=audio_duration_s if is_success else 0.0,
        rtf=0.5,
        audio_ttfp_s=0.3 if request_class == "streaming" else None,
        request_class=request_class,
        sent_at_s=finished_at_s - 1.0 if sent_at_s is None else sent_at_s,
        finished_at_s=finished_at_s,
    )


@pytest.mark.parametrize(
    ("finish_times_s", "expected_gap_s"),
    [
        ([30.0, 35.0, 110.0], 75.0),
        ([5.0, 10.0], 110.0),
        ([100.0], 100.0),
        ([-3.0, 60.0, 125.0], 60.0),
        ([-3.0, 125.0], 120.0),
    ],
)
def test_stream_finish_gap_includes_both_window_edges(
    finish_times_s: list[float], expected_gap_s: float
) -> None:
    """With no stream finishing inside the window, the gap is the whole window."""
    assert stream_finish_gap_max_s(finish_times_s, 120.0) == expected_gap_s


def test_summaries_count_only_requests_that_finished_in_the_window() -> None:
    """Membership is by finish time on the closed window, whatever the send time."""
    results = [
        finished_request("non_streaming", -2.0),
        finished_request("non_streaming", 0.5, sent_at_s=-0.5),
        finished_request("non_streaming", 0.0),
        finished_request("non_streaming", 10.0),
        finished_request("non_streaming", 50.0),
        finished_request("non_streaming", 70.0, is_success=False),
        finished_request("non_streaming", 121.5, is_success=False),
        finished_request("streaming", 20.0),
        finished_request("streaming", 90.0),
        finished_request("streaming", 120.0),
        finished_request("streaming", 121.0, sent_at_s=119.5),
        finished_request("streaming", 125.0),
    ]

    summaries = compute_mixture_summaries(
        results,
        load=MixtureLoad(non_streaming_clients=2, streaming_clients=1),
        settle_s=10.0,
        duration_s=120.0,
        pool_size=4,
    )

    non_streaming = summaries["summary_by_class"]["non_streaming"]
    streaming = summaries["summary_by_class"]["streaming"]
    assert (non_streaming["total_requests"], non_streaming["failed_requests"]) == (5, 1)
    assert (streaming["total_requests"], streaming["failed_requests"]) == (3, 0)
    assert summaries["summary"]["total_requests"] == 8
    assert summaries["summary"]["wall_clock_s"] == 120.0
    assert streaming["stream_finish_gap_max_s"] == 70.0
    window = summaries["mixture"]
    assert window["issued_requests"] == {"non_streaming": 7, "streaming": 5}
    assert window["outside_window_requests"] == {"non_streaming": 2, "streaming": 2}
    assert window["outside_window_failed_requests"] == {
        "non_streaming": 1,
        "streaming": 0,
    }
    assert window["pool_cycles"] == {"non_streaming": 1.75, "streaming": 1.25}
    assert window["drain_s"] == 5.0


@pytest.mark.parametrize(
    "load",
    [
        MixtureLoad(non_streaming_clients=2, streaming_clients=1),
        MixtureLoad(non_streaming_clients=2, streaming_clients=0),
    ],
)
def test_class_audio_throughputs_add_up_to_the_combined_total(
    load: MixtureLoad,
) -> None:
    results = [
        finished_request("non_streaming", 10.0, audio_duration_s=3.0),
        finished_request("non_streaming", 50.0, audio_duration_s=6.0),
        finished_request("non_streaming", 130.0, audio_duration_s=9.0),
    ]
    if load.streaming_clients:
        results += [
            finished_request("streaming", 20.0, audio_duration_s=4.5),
            finished_request("streaming", 90.0, audio_duration_s=1.5),
        ]
    else:
        pass

    summaries = compute_mixture_summaries(
        results, load=load, settle_s=10.0, duration_s=120.0, pool_size=4
    )

    assert set(summaries["summary_by_class"]) == set(load.clients_by_class)
    class_totals_s_per_s = sum(
        class_summary["audio_throughput_s_per_s"]
        for class_summary in summaries["summary_by_class"].values()
    )
    assert summaries["summary"]["audio_throughput_s_per_s"] == pytest.approx(
        class_totals_s_per_s
    )
    assert summaries["summary_by_class"]["non_streaming"][
        "audio_throughput_s_per_s"
    ] == pytest.approx(9.0 / 120.0, abs=1e-3)


def repeat_result(
    audio_s_per_s_by_class: dict[RequestClass, float],
) -> MixtureRepeatResult:
    return {
        "repeat": 1,
        "output_dir": "point",
        "summary": {
            "failed_requests": 0,
            "audio_throughput_s_per_s": sum(audio_s_per_s_by_class.values()),
        },
        "summary_by_class": {
            request_class: {"audio_throughput_s_per_s": audio_s_per_s}
            for request_class, audio_s_per_s in audio_s_per_s_by_class.items()
        },
        "mixture": {
            "outside_window_failed_requests": {
                request_class: 0 for request_class in audio_s_per_s_by_class
            }
        },
    }


SINGLE_MODE_REPEATS = {
    MixtureLoad(non_streaming_clients=48, streaming_clients=0): [
        repeat_result({"non_streaming": 172.0})
    ],
    MixtureLoad(non_streaming_clients=96, streaming_clients=0): [
        repeat_result({"non_streaming": 168.0})
    ],
    MixtureLoad(non_streaming_clients=0, streaming_clients=16): [
        repeat_result({"streaming": 104.8})
    ],
    MixtureLoad(non_streaming_clients=0, streaming_clients=48): [
        repeat_result({"streaming": 130.8})
    ],
}


def test_normalized_work_holds_when_work_moves_to_streaming() -> None:
    """Total audio drops 3 percent when work moves to the streams, while normalized work holds (#3668)."""
    mixture_load = MixtureLoad(non_streaming_clients=48, streaming_clients=16)
    baseline_audio_s_per_s = {"non_streaming": 129.0, "streaming": 32.7}
    shifted_audio_s_per_s = {"non_streaming": 107.5, "streaming": 49.05}
    assert sum(shifted_audio_s_per_s.values()) < 0.97 * sum(
        baseline_audio_s_per_s.values()
    )

    for audio_s_per_s_by_class in (baseline_audio_s_per_s, shifted_audio_s_per_s):
        rows = aggregate_mixture_sweep(
            {
                **SINGLE_MODE_REPEATS,
                mixture_load: [repeat_result(audio_s_per_s_by_class)],
            }
        )
        mixture_row = rows[-1]
        assert mixture_row["category"] == "mixture"
        assert mixture_row["normalized_work"]["references_s_per_s"] == {
            "non_streaming": 172.0,
            "streaming": 130.8,
        }
        assert mixture_row["normalized_work"]["value"]["mean"] == pytest.approx(1.0)
        assert all(row["normalized_work"] is None for row in rows[:-1])

    without_streaming_reference = aggregate_mixture_sweep(
        {
            MixtureLoad(non_streaming_clients=48, streaming_clients=0): [
                repeat_result({"non_streaming": 172.0})
            ],
            MixtureLoad(non_streaming_clients=0, streaming_clients=16): [
                repeat_result({"streaming": 0.0})
            ],
            mixture_load: [repeat_result(baseline_audio_s_per_s)],
        }
    )
    assert without_streaming_reference[-1]["normalized_work"] is None


def test_starved_streams_read_as_the_worst_values_rather_than_missing_ones() -> None:
    """Streams that all finish after the window used to leave throughput and the gap empty."""
    mixture_load = MixtureLoad(non_streaming_clients=48, streaming_clients=16)
    stream_results = [
        finished_request("streaming", 121.0 + index, sent_at_s=1.0)
        for index in range(16)
    ]
    non_streaming_results = [
        finished_request("non_streaming", 10.0 + index, audio_duration_s=120.0)
        for index in range(43)
    ]

    summaries = compute_mixture_summaries(
        non_streaming_results + stream_results,
        load=mixture_load,
        settle_s=10.0,
        duration_s=120.0,
        pool_size=1088,
    )
    streaming = summaries["summary_by_class"]["streaming"]
    assert streaming["completed_requests"] == 0
    assert streaming["throughput_qps"] == streaming["audio_throughput_s_per_s"] == 0.0
    assert streaming["stream_finish_gap_max_s"] == 120.0
    rows = aggregate_mixture_sweep(
        {
            **SINGLE_MODE_REPEATS,
            mixture_load: [{"repeat": 1, "output_dir": "point", **summaries}],
        }
    )
    assert rows[-1]["normalized_work"]["value"]["mean"] == pytest.approx(43.0 / 172.0)

    only_streams = compute_mixture_summaries(
        stream_results,
        load=MixtureLoad(non_streaming_clients=0, streaming_clients=16),
        settle_s=10.0,
        duration_s=120.0,
        pool_size=1088,
    )
    assert only_streams["summary"]["audio_throughput_s_per_s"] == 0.0
