# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingStreamingSegmenterScheduler."""

from __future__ import annotations

import threading
import time
from typing import Iterator

from sglang_omni.models.ming_omni.components.streaming_segmenter import (
    MingStreamingSegmenterScheduler,
)
from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


def _run_scheduler(
    scheduler: MingStreamingSegmenterScheduler,
) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return thread


def _drain_outbox(
    scheduler: MingStreamingSegmenterScheduler, *, until_request_id: str
) -> list[OutgoingMessage]:
    """Drain outbox until we see a final 'result' message for the request."""
    collected: list[OutgoingMessage] = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            msg = scheduler.outbox.get(timeout=0.2)
        except Exception:
            continue
        collected.append(msg)
        if msg.request_id == until_request_id and msg.type == "result":
            return collected
    raise AssertionError(
        f"timed out waiting for final result; got {[m.type for m in collected]}"
    )


def _push_text_chunks(
    scheduler: MingStreamingSegmenterScheduler,
    request_id: str,
    chunks: Iterator[str],
) -> None:
    for chunk in chunks:
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="stream_chunk",
                data=StreamItem(
                    chunk_id=0,
                    data=text_to_uint8_tensor(chunk),
                    from_stage="thinker",
                    metadata=None,
                ),
            )
        )


def test_segmenter_emits_sentence_then_finalizes_on_done():
    cfg = SegmenterConfig(
        segment_min_tokens=2,
        segment_max_tokens=20,
        first_segment_min_tokens=2,
        first_segment_max_wait_ms=10_000,
    )
    sched = MingStreamingSegmenterScheduler(config=cfg)
    thread = _run_scheduler(sched)
    try:
        rid = "req-1"
        # Main payload (handle) arrives first.
        payload = StagePayload(
            request_id=rid,
            request=None,
            data={"keep": "me"},
        )
        sched.inbox.put(
            IncomingMessage(request_id=rid, type="new_request", data=payload)
        )

        _push_text_chunks(
            sched,
            rid,
            iter(["Hello world.", " Tail piece"]),
        )
        sched.inbox.put(IncomingMessage(request_id=rid, type="stream_done"))

        messages = _drain_outbox(sched, until_request_id=rid)
    finally:
        sched.stop()
        thread.join(timeout=1.0)

    streams = [m for m in messages if m.type == "stream"]
    results = [m for m in messages if m.type == "result"]
    assert len(results) == 1
    assert results[0].data.data["segment_count"] == len(streams)
    # First non-empty segment must contain the first sentence.
    non_empty = [m for m in streams if m.metadata["text_len"] > 0]
    assert non_empty, f"no non-empty stream segments: {streams}"
    first_text = uint8_tensor_to_text(non_empty[0].data)
    assert "Hello world." in first_text
    # All stream messages must route to talker_stream.
    for m in streams:
        assert m.target == TALKER_STREAM_STAGE
    # Final segment is flagged.
    assert streams[-1].metadata["is_final_segment"] is True
    # Final-result payload must be a clean small dict (no tensor-laden
    # upstream state, which would break msgpack serialization to coord).
    assert "keep" not in results[0].data.data
    assert results[0].data.data["aborted"] is False


def test_segmenter_handles_stream_arriving_before_payload():
    cfg = SegmenterConfig(
        segment_min_tokens=2,
        segment_max_tokens=20,
        first_segment_min_tokens=2,
        first_segment_max_wait_ms=10_000,
    )
    sched = MingStreamingSegmenterScheduler(config=cfg)
    thread = _run_scheduler(sched)
    try:
        rid = "req-pre"
        # Stream arrives before the main payload.
        _push_text_chunks(sched, rid, iter(["Hi there. "]))
        sched.inbox.put(IncomingMessage(request_id=rid, type="stream_done"))
        # Give scheduler a chance to buffer.
        time.sleep(0.05)
        payload = StagePayload(request_id=rid, request=None, data={})
        sched.inbox.put(
            IncomingMessage(request_id=rid, type="new_request", data=payload)
        )
        messages = _drain_outbox(sched, until_request_id=rid)
    finally:
        sched.stop()
        thread.join(timeout=1.0)

    streams = [m for m in messages if m.type == "stream"]
    assert any(uint8_tensor_to_text(m.data).strip() == "Hi there." for m in streams)


def test_segmenter_first_segment_timeout_emits_before_punctuation():
    cfg = SegmenterConfig(
        segment_min_tokens=100,  # very large so punctuation alone won't fire
        segment_max_tokens=200,
        first_segment_min_tokens=2,
        first_segment_max_wait_ms=50,
    )
    sched = MingStreamingSegmenterScheduler(config=cfg)
    thread = _run_scheduler(sched)
    try:
        rid = "req-timeout"
        payload = StagePayload(request_id=rid, request=None, data={})
        sched.inbox.put(
            IncomingMessage(request_id=rid, type="new_request", data=payload)
        )
        # Push two short tokens that don't end in punctuation.
        _push_text_chunks(sched, rid, iter(["hello world"]))
        # Wait past first_segment_max_wait_ms; the inbox-empty tick should
        # flush the first segment.
        time.sleep(0.3)
        sched.inbox.put(IncomingMessage(request_id=rid, type="stream_done"))
        messages = _drain_outbox(sched, until_request_id=rid)
    finally:
        sched.stop()
        thread.join(timeout=1.0)

    streams = [m for m in messages if m.type == "stream"]
    non_empty = [m for m in streams if m.metadata["text_len"] > 0]
    # First non-empty stream message must arrive without trailing punctuation.
    assert non_empty, "first-segment timeout did not emit"
    first = uint8_tensor_to_text(non_empty[0].data)
    assert "hello" in first
