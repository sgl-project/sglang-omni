# SPDX-License-Identifier: Apache-2.0
"""The offload bench turns per-request timings into the metrics D1 compares."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.offload.run_offload_bench import (
    RequestTiming,
    percentile,
    read_prompts,
    summarize_requests,
)


def test_percentile_takes_the_nearest_rank_of_a_sample() -> None:
    sample = [float(value) for value in range(1, 11)]
    assert percentile(sample, 0.50) == 5.0
    assert percentile(sample, 0.95) == 10.0
    assert percentile(sample, 0.0) == 1.0
    assert percentile([0.4], 0.99) == 0.4
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_a_run_reports_time_to_first_audio_and_throughput() -> None:
    timings = [
        RequestTiming(
            first_chunk_s=0.2, chunk_gaps_s=[0.05, 0.06], audio_s=2.0, elapsed_s=2.5
        ),
        RequestTiming(
            first_chunk_s=0.4, chunk_gaps_s=[0.08], audio_s=1.0, elapsed_s=2.0
        ),
    ]
    latency = summarize_requests(timings, wall_s=4.0)
    assert latency.ttfa_p50_s == 0.2 and latency.ttfa_p95_s == 0.4
    assert latency.inter_chunk_p95_s > 0
    assert latency.throughput_qps == pytest.approx(0.5)
    assert latency.rtf_p95 == pytest.approx(2.0)


def test_a_run_without_audio_is_refused() -> None:
    timings = [RequestTiming(first_chunk_s=0.2, audio_s=0.0, elapsed_s=1.0)]
    with pytest.raises(ValueError):
        summarize_requests(timings, wall_s=1.0)
    with pytest.raises(ValueError):
        summarize_requests([], wall_s=1.0)


def test_prompts_come_from_lines_or_from_a_jsonl_field(tmp_path: Path) -> None:
    lines = tmp_path / "lines.txt"
    lines.write_text("first\n\n  second  \n", encoding="utf-8")
    assert read_prompts(lines) == ["first", "second"]
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"prompt": "third"}) + "\n", encoding="utf-8")
    assert read_prompts(rows) == ["third"]
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_prompts(empty)
