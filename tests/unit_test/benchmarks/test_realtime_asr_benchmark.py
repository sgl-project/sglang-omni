# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the realtime ASR benchmark client and metric definitions.

The client is exercised against an in-process fake /v1/realtime server that
speaks the transcription-intent protocol; the metric definitions are pinned
with hand-built traces.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
import websockets

from benchmarks.realtime_asr.client import (
    SAMPLE_RATE,
    ReceivedEvent,
    SentPacket,
    SessionTrace,
    run_session,
    split_packets,
)
from benchmarks.realtime_asr.metrics import (
    check_invariants,
    latency_metrics,
    percentile,
    summarize,
)

DECODE_INTERVAL_MS = 1000


def _pcm_seconds(seconds: float) -> bytes:
    return b"\x01\x00" * int(SAMPLE_RATE * seconds)


class _FakeTranscriptionServer:
    """Minimal stand-in for RealtimeTranscriptionSession.

    Manual mode: one partial per ``decode_interval`` of audio, final on commit.
    VAD mode: emits speech_started on the first packet, then behaves the same
    and finalizes on ``transcription.done``.
    """

    def __init__(self) -> None:
        self.received: list[dict] = []

    async def handler(self, websocket) -> None:
        index = 0
        turn_detection: dict | None = {"type": "server_vad"}

        async def send(event: dict) -> None:
            nonlocal index
            index += 1
            event.setdefault("event_index", index)
            await websocket.send(json.dumps(event))

        def session_payload() -> dict:
            return {
                "id": "sess_fake",
                "decode_interval_ms": DECODE_INTERVAL_MS,
                "turn_detection": turn_detection,
            }

        await send({"type": "session.created", "session": session_payload()})
        audio_bytes = 0
        next_refresh = SAMPLE_RATE * 2 * DECODE_INTERVAL_MS // 1000
        started = False
        async for raw in websocket:
            event = json.loads(raw)
            self.received.append(event)
            kind = event["type"]
            if kind == "session.update":
                turn_detection = event["session"].get("turn_detection")
                await send({"type": "session.updated", "session": session_payload()})
            elif kind == "input_audio_buffer.append":
                if turn_detection is not None and not started:
                    started = True
                    await send(
                        {
                            "type": "input_audio_buffer.speech_started",
                            "audio_start_ms": 0,
                            "segment_id": 0,
                        }
                    )
                audio_bytes += len(base64.b64decode(event["audio"]))
                if audio_bytes >= next_refresh:
                    next_refresh += SAMPLE_RATE * 2 * DECODE_INTERVAL_MS // 1000
                    await send(
                        {
                            "type": "transcription.segment",
                            "segment_id": 0,
                            "text": f"partial {audio_bytes}",
                            "is_final": False,
                        }
                    )
            elif kind in ("input_audio_buffer.commit", "transcription.done"):
                if audio_bytes:
                    await send(
                        {"type": "input_audio_buffer.committed", "segment_id": 0}
                    )
                    await send(
                        {
                            "type": "transcription.segment",
                            "segment_id": 0,
                            "text": "hello world",
                            "is_final": True,
                        }
                    )
                    audio_bytes = 0
                if kind == "transcription.done":
                    await send(
                        {"type": "transcription.completed", "text": "hello world"}
                    )
                    return


@pytest.fixture
def fake_server():
    server = _FakeTranscriptionServer()

    async def _run(coro):
        async with websockets.serve(server.handler, "127.0.0.1", 0) as ws_server:
            port = ws_server.sockets[0].getsockname()[1]
            return await coro(f"ws://127.0.0.1:{port}/v1/realtime?intent=transcription")

    return server, _run


def test_split_packets_covers_all_bytes_in_order():
    pcm = bytes(range(256)) * 100
    packets = split_packets(pcm, packet_ms=200)
    assert b"".join(packets) == pcm
    assert all(len(packet) == SAMPLE_RATE * 200 // 1000 * 2 for packet in packets[:-1])


def test_manual_session_records_packets_and_events(fake_server):
    server, run = fake_server
    pcm = _pcm_seconds(2.5)

    trace = asyncio.run(
        run(
            lambda url: run_session(
                url,
                pcm,
                packet_ms=500,
                paced=False,
                turn_detection=None,
                manual_commit=True,
            )
        )
    )

    assert trace.error is None
    assert trace.session["decode_interval_ms"] == DECODE_INTERVAL_MS
    assert trace.session["turn_detection"] is None
    assert [packet.audio_end_s for packet in trace.sent] == pytest.approx(
        [0.5, 1.0, 1.5, 2.0, 2.5]
    )
    assert trace.audio_duration_s == pytest.approx(2.5)
    assert [event["type"] for event in server.received] == (
        ["session.update"]
        + ["input_audio_buffer.append"] * 5
        + ["input_audio_buffer.commit", "transcription.done"]
    )
    assert len(trace.segments(is_final=False)) == 2
    assert len(trace.segments(is_final=True)) == 1
    assert trace.completed_text == "hello world"
    assert trace.commit_sent_s is not None and trace.done_sent_s >= trace.commit_sent_s
    assert check_invariants(trace) == []

    metrics = latency_metrics(trace)
    assert metrics["partial_count"] == 2
    assert metrics["segment_count"] == 1
    assert len(metrics["first_partial_latency_s"]) == 1
    assert len(metrics["partial_interval_s"]) == 1
    assert len(metrics["final_latency_s"]) == 1
    assert (
        metrics["done_to_completed_s"] is not None
        and metrics["done_to_completed_s"] >= 0
    )


def test_paced_send_spreads_packets_over_wall_clock(fake_server):
    _, run = fake_server
    pcm = _pcm_seconds(1.0)

    trace = asyncio.run(
        run(lambda url: run_session(url, pcm, packet_ms=100, paced=True))
    )

    assert trace.error is None
    # 10 packets at 100 ms: the last one leaves no earlier than ~0.9 s after t0.
    assert trace.sent[-1].send_s - trace.sent[0].send_s >= 0.85
    # VAD mode: no commit, trailing silence is not added by default.
    assert trace.commit_sent_s is None
    assert trace.events("input_audio_buffer.speech_started")


def test_trailing_silence_extends_sent_audio_but_not_duration(fake_server):
    _, run = fake_server
    pcm = _pcm_seconds(1.0)
    trace = asyncio.run(
        run(
            lambda url: run_session(
                url, pcm, packet_ms=200, paced=False, trailing_silence_ms=600
            )
        )
    )
    assert trace.audio_duration_s == pytest.approx(1.0)
    assert trace.sent[-1].audio_end_s == pytest.approx(1.6)


def test_timeout_is_reported_not_raised():
    async def _hang(websocket):
        await websocket.send(json.dumps({"type": "session.created", "session": {}}))
        await asyncio.sleep(5)

    async def _run():
        async with websockets.serve(_hang, "127.0.0.1", 0) as ws_server:
            port = ws_server.sockets[0].getsockname()[1]
            return await run_session(
                f"ws://127.0.0.1:{port}/v1/realtime", _pcm_seconds(0.2), timeout_s=0.3
            )

    trace = asyncio.run(_run())
    assert trace.error is not None and "timeout" in trace.error
    assert trace.end_s is not None
    assert "client error" in check_invariants(trace)[0]


# --- metric definitions on hand-built traces ---------------------------------


def _trace(
    events: list[tuple[float, dict]],
    *,
    sent_packet_s: float = 0.2,
    audio_s: float = 3.0,
    turn_detection: dict | None = {"type": "server_vad"},
) -> SessionTrace:
    trace = SessionTrace(
        url="ws://fake",
        session={
            "decode_interval_ms": DECODE_INTERVAL_MS,
            "turn_detection": turn_detection,
        },
    )
    trace.first_send_s = 100.0
    count = int(audio_s / sent_packet_s)
    for index in range(count):
        trace.sent.append(
            SentPacket(
                send_s=100.0 + index * sent_packet_s,
                audio_end_samples=round((index + 1) * sent_packet_s * SAMPLE_RATE),
                num_bytes=int(SAMPLE_RATE * sent_packet_s * 2),
            )
        )
    for offset, event in events:
        trace.received.append(ReceivedEvent(recv_s=100.0 + offset, event=event))
    trace.done_sent_s = 100.0 + audio_s
    trace.end_s = 100.0 + audio_s + 1.0
    return trace


def _seg(segment_id: int, text: str, *, final: bool, index: int) -> dict:
    return {
        "type": "transcription.segment",
        "segment_id": segment_id,
        "text": text,
        "is_final": final,
        "event_index": index,
    }


def test_first_partial_latency_counts_from_the_packet_crossing_the_refresh_point():
    # VAD says speech started at 400 ms; refresh point is 1400 ms of audio.
    # Packet 7 (audio_end 1.4 s) is sent at t0 + 1.2 s; partial arrives at t0 + 1.5 s.
    trace = _trace(
        [
            (
                0.25,
                {
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": 400,
                    "segment_id": 0,
                    "event_index": 1,
                },
            ),
            (1.5, _seg(0, "hel", final=False, index=2)),
            (2.6, _seg(0, "hello wor", final=False, index=3)),
            (
                3.1,
                {
                    "type": "input_audio_buffer.committed",
                    "segment_id": 0,
                    "event_index": 4,
                },
            ),
            (3.4, _seg(0, "hello world", final=True, index=5)),
            (
                3.5,
                {
                    "type": "transcription.completed",
                    "text": "hello world",
                    "event_index": 6,
                },
            ),
        ]
    )
    metrics = latency_metrics(trace)
    assert metrics["first_partial_latency_s"] == [pytest.approx(1.5 - 1.2)]
    assert metrics["partial_interval_s"] == [pytest.approx(1.1)]
    assert metrics["final_latency_s"] == [pytest.approx(0.3)]
    assert metrics["done_to_completed_s"] == pytest.approx(0.5)
    assert metrics["partial_count"] == 2 and metrics["segment_count"] == 1
    assert check_invariants(trace) == []


def test_manual_mode_first_segment_starts_at_zero():
    # No VAD events; refresh point is 1000 ms, reached by packet 5 sent at t0 + 0.8 s.
    trace = _trace(
        [
            (1.0, _seg(0, "h", final=False, index=1)),
            (
                3.1,
                {
                    "type": "input_audio_buffer.committed",
                    "segment_id": 0,
                    "event_index": 2,
                },
            ),
            (3.2, _seg(0, "hi", final=True, index=3)),
            (3.3, {"type": "transcription.completed", "text": "hi", "event_index": 4}),
        ],
        turn_detection=None,
    )
    assert latency_metrics(trace)["first_partial_latency_s"] == [pytest.approx(0.2)]


def test_refresh_point_lookup_is_exact_on_packet_boundaries():
    # 10 x 200 ms packets reach exactly 2000 ms; float accumulation of 0.2
    # would have pushed the trigger to packet 11 and made the latency negative.
    trace = _trace(
        [
            (1.9, _seg(0, "h", final=False, index=1)),
            (3.3, {"type": "transcription.completed", "text": "h", "event_index": 2}),
        ],
        turn_detection=None,
    )
    trace.session["decode_interval_ms"] = 2000
    # packet 10 (index 9) is sent at t0 + 1.8 s
    assert latency_metrics(trace)["first_partial_latency_s"] == [pytest.approx(0.1)]


def test_first_partial_skipped_when_segment_start_unknown():
    # Second segment without a speech_started event: no start, no latency.
    trace = _trace(
        [
            (1.0, _seg(1, "x", final=False, index=1)),
            (
                3.1,
                {
                    "type": "input_audio_buffer.committed",
                    "segment_id": 1,
                    "event_index": 2,
                },
            ),
            (3.2, _seg(1, "x", final=True, index=3)),
            (3.3, {"type": "transcription.completed", "text": "x", "event_index": 4}),
        ],
        turn_detection=None,
    )
    assert latency_metrics(trace)["first_partial_latency_s"] == []


def test_invariants_flag_each_protocol_violation():
    trace = _trace(
        [
            (1.0, _seg(0, "a", final=True, index=1)),
            (1.1, _seg(0, "b", final=False, index=3)),  # update after final
            (
                1.2,
                {"type": "error", "error": {"code": "boom"}, "event_index": 2},
            ),  # index goes back
            # no committed for segment 0, no completed
        ]
    )
    violations = check_invariants(trace)
    joined = "\n".join(violations)
    assert "server error event" in joined
    assert "event_index not strictly increasing" in joined
    assert "do not match final segments" in joined
    assert "updated after its final event" in joined
    assert "no transcription.completed" in joined


def test_percentile_and_summarize():
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    summary = summarize(values)
    assert summary == {
        "mean": 10.5,
        "p50": 10.0,
        "p95": 19.0,
        "min": 1.0,
        "max": 20.0,
        "n": 20,
    }
    assert summarize([]) is None


# --- paired WER delta ------------------------------------------------------


def _outputs(texts: dict[str, str | None]) -> list:
    """SampleOutputs with ref == hyp; ``None`` marks a failed sample."""
    from benchmarks.metrics.wer import SampleOutput
    from benchmarks.tasks.asr import apply_wer

    outputs = []
    for sample_id, text in texts.items():
        output = SampleOutput(sample_id=sample_id, target_text=text or "unused ref")
        if text is not None:
            output = apply_wer(output, text, "en")
        else:
            output.error = "HTTP 500"
        outputs.append(output)
    return outputs


def test_paired_wer_delta_ignores_samples_that_failed_on_one_side():
    from benchmarks.realtime_asr.metrics import paired_corpus_wer

    texts = {"a": "one two three", "b": "four five six", "c": "seven eight nine"}
    stream = _outputs(texts)
    http = _outputs({**texts, "c": None})  # identical transcripts, one HTTP failure

    paired = paired_corpus_wer(stream, http, lang="en")
    assert paired["common_evaluated"] == 2
    assert paired["corpus_wer_delta_vs_http"] == 0.0


def test_paired_wer_delta_is_none_when_baseline_has_no_successes():
    from benchmarks.realtime_asr.metrics import paired_corpus_wer

    stream = _outputs({"a": "one two three"})
    http = _outputs({"a": None})

    paired = paired_corpus_wer(stream, http, lang="en")
    assert paired["common_evaluated"] == 0
    assert paired["corpus_wer_delta_vs_http"] is None
    assert paired["http_corpus_wer_common"] is None
