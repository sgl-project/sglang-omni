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
    serving_summary,
    stage_breakdown,
)


def write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for ev in events:
            fp.write(json.dumps(ev))
            fp.write("\n")


def make_ev(request_id, stage, name, ts, **md):
    return {
        "request_id": request_id,
        "stage": stage,
        "event_name": name,
        "timestamp_ns": ts,
        "run_id": "run_test",
        "pid": os.getpid(),
        "metadata": md,
    }


def test_serving_summary_intervals_and_batch_samples(tmp_path: Path) -> None:
    events = []
    for index, wait_ms in enumerate([1, 3, 5, 7, 9]):
        events.extend(
            [
                make_ev(str(index), "thinker", "scheduler_queue_enter", 0),
                make_ev(
                    str(index),
                    "thinker",
                    "scheduler_prefill_start",
                    wait_ms * 1_000_000,
                ),
                make_ev(
                    str(index),
                    "thinker",
                    "scheduler_prefill_end",
                    (wait_ms + 10) * 1_000_000,
                ),
            ]
        )
    for stage, batch_type, size in [
        ("thinker", "decode", 2),
        ("thinker", "decode", 6),
        ("talker", "prefill", 3),
        ("talker", "mixed", 4),
    ]:
        events.append(
            make_ev(
                "0",
                stage,
                "scheduler_batch_start",
                20_000_000,
                batch_type=batch_type,
                batch_size=size,
                waiting_requests=2,
                num_retracted_reqs=4,
                kv_usage=0.75,
                kv_used_tokens=75,
                kv_available_tokens=10,
                kv_evictable_tokens=15,
                request_build_pending=2,
                request_build_backlog=18,
            )
        )
    events.append(make_ev("0", "thinker", "scheduler_request_retracted", 21_000_000))
    events.append(make_ev("0", "thinker", "scheduler_batch_start", 22_000_000))
    write_events(tmp_path / "events_test.jsonl", events)
    report = build_report(tmp_path)
    summary = report["serving_summary"]
    assert report["request_count"] == 5
    assert summary["thinker"]["queue_wait_ms"] == {
        "count": 5,
        "avg": 5,
        "p50": 5,
        "p95": 8.6,
        "max": 9,
    }
    assert summary["thinker"]["prefill_ms"]["avg"] == 10
    assert summary["thinker"]["decode_batch_size"]["avg"] == 4
    assert summary["talker"]["prefill_batch_size"]["avg"] == 3
    assert summary["talker"]["mixed_batch_size"]["avg"] == 4
    assert "decode_batch_size" not in summary["talker"]
    assert summary["thinker"]["observed_retractions"] == 1
    assert "num_retracted_reqs" not in summary["thinker"]
    for metric, value in {
        "kv_usage": 0.75,
        "kv_used_tokens": 75,
        "kv_available_tokens": 10,
        "kv_evictable_tokens": 15,
        "request_build_pending": 2,
        "request_build_backlog": 18,
    }.items():
        assert summary["thinker"][metric] == {
            "count": 2,
            "avg": value,
            "p50": value,
            "p95": value,
            "max": value,
        }
    assert report["stage_breakdown"] == [
        row.to_dict() for row in stage_breakdown(source=tmp_path)
    ]


def test_serving_summary_code2wav_subbatches_and_optional_metadata(
    tmp_path: Path,
) -> None:
    executions = [
        {"batch_size": 4, "execution_mode": "cuda_graph"},
        {"batch_size": 2, "execution_mode": "eager", "fallback_reason": "ineligible"},
        {"batch_size": 1, "execution_mode": "eager", "fallback_reason": "ineligible"},
        {"batch_size": 1, "execution_mode": "eager", "fallback_reason": "key_miss"},
        {"batch_size": 1, "execution_mode": "eager", "fallback_reason": None},
    ]
    events = [
        make_ev(
            "r", "code2wav", "code2wav_batch_start", 0, batch_size=9, inbox_depth=3
        ),
        make_ev(
            "r",
            "code2wav",
            "code2wav_batch_end",
            1_000_000,
            batch_size=9,
            execution_mode="mixed",
            sub_batch_execution=executions,
        ),
        make_ev(
            "r",
            "code2wav",
            "code2wav_batch_end",
            2_000_000,
            execution_mode="eager",
            sub_batch_execution=[],
        ),
        make_ev("r", "code2wav", "code2wav_batch_end", 3_000_000),
    ]
    write_events(tmp_path / "events_test.jsonl", events)
    summary = build_report(tmp_path)["serving_summary"]["code2wav"]
    assert summary["execution_mode"] == {"cuda_graph": 1, "eager": 4}
    assert summary["graph_hit_count"] == 1
    assert summary["graph_fallback_count"] == 3
    assert summary["graph_attempt_success_rate"] == 0.25
    assert "graph_hit_rate" not in summary
    assert summary["fallback_reason"] == {"ineligible": 2, "key_miss": 1}
    assert summary["effective_batch_size"]["avg"] == 1.8
    assert summary["inbox_depth"]["count"] == 1
    assert summary["batch_ms"]["avg"] == 1
    assert serving_summary({}) == {}


def test_serving_summary_serial_code2wav(tmp_path: Path) -> None:
    events = []
    for index, (mode, reason) in enumerate(
        [
            ("cuda_graph", None),
            ("eager", "ineligible"),
            ("eager", None),
        ]
    ):
        events.extend(
            [
                make_ev(
                    "r",
                    "code2wav",
                    "code2wav_decode_start",
                    index * 2_000_000,
                    active_request_count=16,
                    inbox_depth=8,
                ),
                make_ev(
                    "r",
                    "code2wav",
                    "code2wav_decode_end",
                    (index * 2 + 1) * 1_000_000,
                    active_request_count=16,
                    inbox_depth=8,
                    execution_mode=mode,
                    fallback_reason=reason,
                ),
            ]
        )
    write_events(tmp_path / "events_test.jsonl", events)
    summary = build_report(tmp_path)["serving_summary"]["code2wav"]
    assert summary["effective_batch_size"] == {
        "count": 3,
        "avg": 1,
        "p50": 1,
        "p95": 1,
        "max": 1,
    }
    assert summary["execution_mode"] == {"cuda_graph": 1, "eager": 2}
    assert summary["graph_attempt_success_rate"] == 0.5
    assert summary["fallback_reason"] == {"ineligible": 1}
    assert summary["inbox_depth"]["count"] == 3
    assert summary["active_request_count"]["avg"] == 16
    assert summary["decode_ms"]["avg"] == 1


def test_serving_summary_intentional_eager_and_cli(tmp_path: Path, capsys) -> None:
    from sglang_omni.profiler.__main__ import main

    write_events(
        tmp_path / "events_test.jsonl",
        [
            make_ev("r", "code2wav", "code2wav_batch_end", 0, execution_mode="eager"),
        ],
    )
    summary = build_report(tmp_path)["serving_summary"]["code2wav"]
    assert summary["graph_fallback_count"] == 0
    assert summary["graph_attempt_success_rate"] is None
    assert main([str(tmp_path), "--format", "table"]) == 0
    assert "=== Serving Summary ===" in capsys.readouterr().out
    assert main([str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["serving_summary"]["code2wav"] == summary


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_reconstruct_timelines_sorts_per_request(tmp_path: Path) -> None:
    events = [
        make_ev("r1", "coordinator", "request_admission", 1000),
        make_ev("r2", "coordinator", "request_admission", 1100),
        make_ev(
            "r1", "encoder", "stage_input_received", 1500, from_stage="coordinator"
        ),
        make_ev("r1", "coordinator", "terminal_response", 5000, from_stage="thinker"),
    ]
    p = tmp_path / "events_test_1.jsonl"
    write_events(p, events)

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
    write_events(file_a, [make_ev("r1", "coordinator", "request_admission", 100)])
    write_events(
        file_b,
        [
            make_ev(
                "r1", "encoder", "stage_input_received", 200, from_stage="coordinator"
            )
        ],
    )

    tls = reconstruct_timelines(tmp_path)
    assert "r1" in tls
    names = [e["event_name"] for e in tls["r1"].events]
    assert names == ["request_admission", "stage_input_received"]


def test_iter_events_skips_malformed_lines(tmp_path: Path) -> None:
    """A garbage line must not break the loader."""
    p = tmp_path / "events_x_1.jsonl"
    with p.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(make_ev("r1", "s", "a", 1)))
        fp.write("\n")
        fp.write("not-valid-json\n")
        fp.write(json.dumps(make_ev("r1", "s", "b", 2)))
        fp.write("\n")
    tls = reconstruct_timelines(tmp_path)
    assert len(tls["r1"].events) == 2


# ---------------------------------------------------------------------------
# Stage breakdown
# ---------------------------------------------------------------------------


def test_stage_breakdown_pairs_open_close(tmp_path: Path) -> None:
    events = [
        make_ev("r1", "encoder", "stage_input_received", 0, from_stage="coordinator"),
        make_ev("r1", "encoder", "stage_complete", 2_000_000),  # 2ms
        make_ev("r2", "encoder", "stage_input_received", 1, from_stage="coordinator"),
        make_ev("r2", "encoder", "stage_complete", 4_000_001),  # 4ms
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev("r1", "encoder", "stage_input_received", 0, from_stage="coordinator"),
        make_ev("r1", "thinker", "stage_complete", 1_000_000),
        # No matching close on encoder for r1 → no encoder interval emitted.
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev("r1", "encoder", "stage_hop_sent", 0, to_stage="thinker"),
        make_ev(
            "r1",
            "thinker",
            "stage_input_received",
            500_000,  # 0.5ms hop
            from_stage="encoder",
            kind="payload",
        ),
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev(
            "r1",
            "thinker",
            "stage_stream_chunk_sent",
            0,
            to_stage="talker",
            chunk_id=0,
        ),
        make_ev(
            "r1",
            "thinker",
            "stage_stream_chunk_sent",
            100_000,
            to_stage="talker",
            chunk_id=1,
        ),
        make_ev(
            "r1",
            "talker",
            "stage_stream_chunk_received",
            1_000_000,
            from_stage="thinker",
            chunk_id=0,
        ),
        make_ev(
            "r1",
            "talker",
            "stage_stream_chunk_received",
            1_500_000,
            from_stage="thinker",
            chunk_id=1,
        ),
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev(
            "r1",
            "decode",
            "stage_stream_chunk_sent",
            0,
            to_stage="coordinator",
            chunk_id=0,
        ),
        make_ev(
            "r1",
            "decode",
            "stage_stream_chunk_sent",
            100_000,
            to_stage="coordinator",
            chunk_id=1,
        ),
        make_ev(
            "r1",
            "coordinator",
            "stage_stream_chunk_received",
            1_000_000,
            from_stage="decode",
            chunk_id=0,
        ),
        make_ev(
            "r1",
            "coordinator",
            "stage_stream_chunk_received",
            1_500_000,
            from_stage="decode",
            chunk_id=1,
        ),
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev("r1", "preprocessor", "preprocess_start", 0),
        make_ev("r1", "preprocessor", "preprocess_end", 1_000_000),  # 1ms
        make_ev("r1", "audio_encoder", "encoder_start", 1_100_000, modality="audio"),
        make_ev(
            "r1", "audio_encoder", "encoder_end", 6_100_000, modality="audio"
        ),  # 5ms
        make_ev("r1", "thinker", "scheduler_prefill_start", 6_200_000),
        make_ev(
            "r1",
            "thinker",
            "stage_first_stream_chunk_sent",
            10_200_000,  # 4ms thinker TTFT
            to_stage="talker",
        ),
        make_ev("r1", "talker", "scheduler_request_build_start", 10_300_000),
        make_ev("r1", "talker", "scheduler_request_build_end", 10_700_000),  # 0.4ms
        make_ev("r1", "talker", "scheduler_prefill_start", 10_800_000),
        make_ev(
            "r1",
            "talker",
            "stage_first_stream_chunk_sent",
            14_800_000,  # 4ms first code chunk
            to_stage="code2wav",
        ),
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev("r1", "thinker", "scheduler_prefill_start", 0),
        make_ev("r1", "thinker", "scheduler_first_emit", 3_000_000),  # 3 ms
        make_ev("r1", "thinker", "stage_first_stream_chunk_sent", 7_000_000),  # 7 ms
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
        make_ev("r1", "thinker", "scheduler_queue_enter", 0),
        make_ev("r1", "thinker", "scheduler_prefill_start", 5_000_000),
        make_ev("r1", "thinker", "scheduler_first_emit", 7_000_000),
        make_ev("r1", "thinker", "stage_first_stream_chunk_sent", 10_000_000),
    ]
    write_events(tmp_path / "events_x.jsonl", events)
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
    # note (luojiaxuan): queue wait is now its own interval (issue #1324
    # Q-PR2); TTFT-style intervals must still open at prefill_start only.
    assert (
        by_key[("thinker", "scheduler_queue_enter->scheduler_prefill_start")].total_ms
        == 5.0
    )
    assert all(
        not r.interval_name.startswith("scheduler_queue_enter->")
        or r.interval_name == "scheduler_queue_enter->scheduler_prefill_start"
        for r in rows
    )


def test_build_report_returns_all_three_views(tmp_path: Path) -> None:
    events = [
        make_ev("r1", "coordinator", "request_admission", 0),
        make_ev("r1", "encoder", "stage_input_received", 100, from_stage="coordinator"),
        make_ev("r1", "encoder", "stage_complete", 2_000_000),
        make_ev("r1", "encoder", "stage_hop_sent", 2_100_000, to_stage="thinker"),
        make_ev(
            "r1",
            "thinker",
            "stage_input_received",
            3_000_000,
            from_stage="encoder",
        ),
        make_ev("r1", "coordinator", "terminal_response", 10_000_000),
    ]
    write_events(tmp_path / "events_x.jsonl", events)
    rep = build_report(tmp_path)
    assert rep["request_count"] == 1
    assert "r1" in rep["timelines"]
    assert len(rep["timelines"]["r1"]) == 6
    assert any(r["stage"] == "encoder" for r in rep["stage_breakdown"])
    assert any(
        r["src"] == "encoder" and r["dst"] == "thinker" for r in rep["hop_breakdown"]
    )
