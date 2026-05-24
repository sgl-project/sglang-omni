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


def test_hop_breakdown_pairs_terminal_stream_chunks_to_coordinator(
    tmp_path: Path,
) -> None:
    events = [
        _ev(
            "r1",
            "decode",
            "stage_stream_chunk_sent",
            0,
            to_stage="coordinator",
            chunk_id=0,
        ),
        _ev(
            "r1",
            "decode",
            "stage_stream_chunk_sent",
            100_000,
            to_stage="coordinator",
            chunk_id=1,
        ),
        _ev(
            "r1",
            "coordinator",
            "stage_stream_chunk_received",
            1_000_000,
            from_stage="decode",
            chunk_id=0,
        ),
        _ev(
            "r1",
            "coordinator",
            "stage_stream_chunk_received",
            1_500_000,
            from_stage="decode",
            chunk_id=1,
        ),
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = hop_breakdown(source=tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r.src_stage == "decode"
    assert r.dst_stage == "coordinator"
    assert r.kind == "stream_chunk"
    assert r.count == 2


def test_stage_breakdown_covers_preprocess_encoder_and_prefill(
    tmp_path: Path,
) -> None:
    """The required intervals for #501 must be wired into the views layer."""
    events = [
        _ev("r1", "preprocessor", "preprocess_start", 0),
        _ev("r1", "preprocessor", "preprocess_end", 1_000_000),  # 1ms
        _ev("r1", "audio_encoder", "encoder_start", 1_100_000, modality="audio"),
        _ev("r1", "audio_encoder", "encoder_end", 6_100_000, modality="audio"),  # 5ms
        _ev("r1", "thinker", "scheduler_prefill_start", 6_200_000),
        _ev(
            "r1",
            "thinker",
            "stage_first_stream_chunk_sent",
            10_200_000,  # 4ms thinker TTFT
            to_stage="talker",
        ),
        _ev("r1", "talker", "scheduler_request_build_start", 10_300_000),
        _ev("r1", "talker", "scheduler_request_build_end", 10_700_000),  # 0.4ms
        _ev("r1", "talker", "scheduler_prefill_start", 10_800_000),
        _ev(
            "r1",
            "talker",
            "stage_first_stream_chunk_sent",
            14_800_000,  # 4ms first code chunk
            to_stage="code2wav",
        ),
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = stage_breakdown(source=tmp_path)
    by_key = {(r.stage, r.interval_name): r for r in rows}

    assert ("preprocessor", "preprocess_start->preprocess_end") in by_key
    assert by_key[("preprocessor", "preprocess_start->preprocess_end")].total_ms == 1.0

    assert ("audio_encoder", "encoder_start->encoder_end") in by_key
    assert by_key[("audio_encoder", "encoder_start->encoder_end")].total_ms == 5.0

    thinker_ttft_key = (
        "thinker",
        "scheduler_prefill_start->stage_first_stream_chunk_sent",
    )
    assert thinker_ttft_key in by_key
    assert by_key[thinker_ttft_key].total_ms == 4.0

    talker_build_key = (
        "talker",
        "scheduler_request_build_start->scheduler_request_build_end",
    )
    assert talker_build_key in by_key
    assert abs(by_key[talker_build_key].total_ms - 0.4) < 1e-9

    talker_ttfcc_key = (
        "talker",
        "scheduler_prefill_start->stage_first_stream_chunk_sent",
    )
    assert talker_ttfcc_key in by_key
    assert by_key[talker_ttfcc_key].total_ms == 4.0


def test_stage_breakdown_emits_both_intervals_sharing_opener(
    tmp_path: Path,
) -> None:
    """Two intervals sharing the same opener must both appear.

    Regression for review finding P1: ``scheduler_prefill_start`` participates
    in both ``-> scheduler_first_emit`` AND ``-> stage_first_stream_chunk_sent``.
    Before the fix, the earlier-arriving close (``scheduler_first_emit``) popped
    the opener, leaving nothing for the later close — so the issue #501
    "thinker first token" / TTFCC interval silently disappeared from the
    report.
    """
    events = [
        _ev("r1", "thinker", "scheduler_prefill_start", 0),
        _ev("r1", "thinker", "scheduler_first_emit", 3_000_000),  # 3 ms
        _ev("r1", "thinker", "stage_first_stream_chunk_sent", 7_000_000),  # 7 ms
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = stage_breakdown(source=tmp_path)
    by_key = {(r.stage, r.interval_name): r for r in rows}

    first_emit_key = ("thinker", "scheduler_prefill_start->scheduler_first_emit")
    first_chunk_key = (
        "thinker",
        "scheduler_prefill_start->stage_first_stream_chunk_sent",
    )
    assert first_emit_key in by_key, "scheduler_first_emit interval was dropped"
    assert first_chunk_key in by_key, (
        "stage_first_stream_chunk_sent interval was dropped — opener was "
        "consumed by the sibling pair"
    )
    assert by_key[first_emit_key].total_ms == 3.0
    assert by_key[first_chunk_key].total_ms == 7.0


def test_stage_breakdown_uses_prefill_start_not_queue_enter(
    tmp_path: Path,
) -> None:
    events = [
        _ev("r1", "thinker", "scheduler_queue_enter", 0),
        _ev("r1", "thinker", "scheduler_prefill_start", 5_000_000),
        _ev("r1", "thinker", "scheduler_first_emit", 7_000_000),
        _ev("r1", "thinker", "stage_first_stream_chunk_sent", 10_000_000),
    ]
    _write_events(tmp_path / "events_x.jsonl", events)
    rows = stage_breakdown(source=tmp_path)
    by_key = {(r.stage, r.interval_name): r for r in rows}

    assert (
        by_key[("thinker", "scheduler_prefill_start->scheduler_first_emit")].total_ms
        == 2.0
    )
    assert (
        by_key[
            ("thinker", "scheduler_prefill_start->stage_first_stream_chunk_sent")
        ].total_ms
        == 5.0
    )
    assert all("scheduler_queue_enter->" not in r.interval_name for r in rows)


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
