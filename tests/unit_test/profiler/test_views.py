# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_omni.profiler.views (timeline / stage / hop)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from sglang_omni.profiler.views import (
    build_report,
    hop_breakdown,
    reconstruct_timelines,
    stage_breakdown,
)


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for ev in events:
            fp.write(json.dumps(ev))
            fp.write("\n")


def _ev(request_id, stage, name, ts, **md):
    return {
        "request_id": request_id,
        "stage": stage,
        "event_name": name,
        "timestamp_ns": ts,
        "run_id": "run_test",
        "pid": os.getpid(),
        "metadata": md,
    }


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_reconstruct_timelines_sorts_per_request(tmp_path: Path) -> None:
    events = [
        _ev("r1", "coordinator", "request_admission", 1000),
        _ev("r2", "coordinator", "request_admission", 1100),
        _ev("r1", "encoder", "stage_input_received", 1500, from_stage="coordinator"),
        _ev("r1", "coordinator", "terminal_response", 5000, from_stage="thinker"),
    ]
    p = tmp_path / "events_test_1.jsonl"
    _write_events(p, events)

    tls = reconstruct_timelines(tmp_path)
    assert set(tls) == {"r1", "r2"}
    assert [e["event_name"] for e in tls["r1"].events] == [
        "request_admission",
        "stage_input_received",
        "terminal_response",
    ]
    rel = tls["r1"].to_relative()
    assert rel[0]["t_rel_ms"] == 0.0
    # 5000ns - 1000ns = 4000ns = 0.004ms
    assert rel[-1]["t_rel_ms"] == 0.004


def test_timeline_merges_multiple_files(tmp_path: Path) -> None:
    file_a = tmp_path / "events_coordinator_1.jsonl"
    file_b = tmp_path / "events_encoder_2.jsonl"
    _write_events(file_a, [_ev("r1", "coordinator", "request_admission", 100)])
    _write_events(
        file_b,
        [_ev("r1", "encoder", "stage_input_received", 200, from_stage="coordinator")],
    )

    tls = reconstruct_timelines(tmp_path)
    assert "r1" in tls
    names = [e["event_name"] for e in tls["r1"].events]
    assert names == ["request_admission", "stage_input_received"]


def test_iter_events_skips_malformed_lines(tmp_path: Path) -> None:
    """A garbage line must not break the loader."""
    p = tmp_path / "events_x_1.jsonl"
    with p.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(_ev("r1", "s", "a", 1)))
        fp.write("\n")
        fp.write("not-valid-json\n")
        fp.write(json.dumps(_ev("r1", "s", "b", 2)))
        fp.write("\n")
    tls = reconstruct_timelines(tmp_path)
    assert len(tls["r1"].events) == 2


# ---------------------------------------------------------------------------
# Stage breakdown
# ---------------------------------------------------------------------------


def test_stage_breakdown_pairs_open_close(tmp_path: Path) -> None:
    events = [
        _ev("r1", "encoder", "stage_input_received", 0, from_stage="coordinator"),
        _ev("r1", "encoder", "stage_complete", 2_000_000),  # 2ms
        _ev("r2", "encoder", "stage_input_received", 1, from_stage="coordinator"),
        _ev("r2", "encoder", "stage_complete", 4_000_001),  # 4ms
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = stage_breakdown(source=tmp_path)
    encoder_rows = [
        r
        for r in rows
        if r.stage == "encoder"
        and r.interval_name == "stage_input_received->stage_complete"
    ]
    assert len(encoder_rows) == 1
    row = encoder_rows[0]
    assert row.count == 2
    assert row.total_ms == 6.0
    assert row.avg_ms == 3.0
    assert row.max_ms == 4.0


def test_stage_breakdown_keeps_intervals_stage_local(tmp_path: Path) -> None:
    """An open on stage A must not pair with a close on stage B."""
    events = [
        _ev("r1", "encoder", "stage_input_received", 0, from_stage="coordinator"),
        _ev("r1", "thinker", "stage_complete", 1_000_000),
        # No matching close on encoder for r1 → no encoder interval emitted.
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = stage_breakdown(source=tmp_path)
    encoder_rows = [
        r
        for r in rows
        if r.stage == "encoder"
        and r.interval_name == "stage_input_received->stage_complete"
    ]
    assert encoder_rows == []


# ---------------------------------------------------------------------------
# Hop breakdown
# ---------------------------------------------------------------------------


def test_hop_breakdown_pairs_payload_send_recv(tmp_path: Path) -> None:
    events = [
        _ev("r1", "encoder", "stage_hop_sent", 0, to_stage="thinker"),
        _ev(
            "r1",
            "thinker",
            "stage_input_received",
            500_000,  # 0.5ms hop
            from_stage="encoder",
            kind="payload",
        ),
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = hop_breakdown(source=tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r.src_stage == "encoder"
    assert r.dst_stage == "thinker"
    assert r.kind == "payload"
    assert r.count == 1
    assert abs(r.total_ms - 0.5) < 1e-9


def test_hop_breakdown_pairs_stream_chunks_by_id(tmp_path: Path) -> None:
    events = [
        _ev(
            "r1",
            "thinker",
            "stage_stream_chunk_sent",
            0,
            to_stage="talker",
            chunk_id=0,
        ),
        _ev(
            "r1",
            "thinker",
            "stage_stream_chunk_sent",
            100_000,
            to_stage="talker",
            chunk_id=1,
        ),
        _ev(
            "r1",
            "talker",
            "stage_stream_chunk_received",
            1_000_000,
            from_stage="thinker",
            chunk_id=0,
        ),
        _ev(
            "r1",
            "talker",
            "stage_stream_chunk_received",
            1_500_000,
            from_stage="thinker",
            chunk_id=1,
        ),
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = hop_breakdown(source=tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r.src_stage == "thinker"
    assert r.dst_stage == "talker"
    assert r.kind == "stream_chunk"
    assert r.count == 2


def test_build_report_returns_all_three_views(tmp_path: Path) -> None:
    events = [
        _ev("r1", "coordinator", "request_admission", 0),
        _ev("r1", "encoder", "stage_input_received", 100, from_stage="coordinator"),
        _ev("r1", "encoder", "stage_complete", 2_000_000),
        _ev("r1", "encoder", "stage_hop_sent", 2_100_000, to_stage="thinker"),
        _ev(
            "r1",
            "thinker",
            "stage_input_received",
            3_000_000,
            from_stage="encoder",
        ),
        _ev("r1", "coordinator", "terminal_response", 10_000_000),
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rep = build_report(tmp_path)
    assert rep["request_count"] == 1
    assert "r1" in rep["timelines"]
    assert len(rep["timelines"]["r1"]) == 6
    assert any(r["stage"] == "encoder" for r in rep["stage_breakdown"])
    assert any(
        r["src"] == "encoder" and r["dst"] == "thinker" for r in rep["hop_breakdown"]
    )
