# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingStreamingTalkerScheduler with a fake talker."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch

from sglang_omni.models.ming_omni.components.streaming_talker import (
    MingStreamingTalkerScheduler,
)
from sglang_omni.models.ming_omni.components.streaming_text import text_to_uint8_tensor
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class _FakeTalker:
    """Stand-in for MingOmniTalker that yields one waveform per call."""

    def __init__(self, *, samples_per_call: int = 512, sample_rate: int = 44100):
        self.samples_per_call = samples_per_call
        self.sample_rate = sample_rate
        self.calls: list[str] = []

    def omni_audio_generation(
        self,
        *,
        tts_text: str,
        voice_name: str,
        audio_detokenizer: Any,
        stream: bool,
        abort_event: threading.Event | None = None,
    ):
        self.calls.append(tts_text)
        # Two chunks per segment so we can assert >1 emit.
        for _ in range(2):
            if abort_event is not None and abort_event.is_set():
                return
            wav = torch.zeros(self.samples_per_call, dtype=torch.float32)
            yield (wav, None, None, None)


def _make_scheduler(**kwargs) -> MingStreamingTalkerScheduler:
    talker = kwargs.pop("talker", None) or _FakeTalker()
    return MingStreamingTalkerScheduler(
        talker=talker,
        sample_rate=talker.sample_rate,
        **kwargs,
    )


def _run(scheduler: MingStreamingTalkerScheduler) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return thread


def _drain(
    scheduler: MingStreamingTalkerScheduler, *, until_request_id: str
) -> list[OutgoingMessage]:
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
        f"timed out waiting for result; got {[m.type for m in collected]}"
    )


def _segment(scheduler, rid: str, text: str, *, segment_id: int, final: bool = False):
    scheduler.inbox.put(
        IncomingMessage(
            request_id=rid,
            type="stream_chunk",
            data=StreamItem(
                chunk_id=0,
                data=text_to_uint8_tensor(text),
                from_stage="segmenter",
                metadata={
                    "segment_id": segment_id,
                    "is_final_segment": final,
                    "text_len": len(text),
                },
            ),
        )
    )


def test_streaming_talker_emits_audio_per_segment_then_finalizes():
    sched = _make_scheduler()
    thread = _run(sched)
    try:
        rid = "req-1"
        sched.inbox.put(
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=StagePayload(request_id=rid, request=None, data={"keep": "x"}),
            )
        )
        _segment(sched, rid, "Hello.", segment_id=0)
        _segment(sched, rid, "World.", segment_id=1, final=True)
        sched.inbox.put(IncomingMessage(request_id=rid, type="stream_done"))
        msgs = _drain(sched, until_request_id=rid)
    finally:
        sched.stop()
        thread.join(timeout=1.0)

    streams = [m for m in msgs if m.type == "stream"]
    results = [m for m in msgs if m.type == "result"]
    # 2 segments × 2 audio chunks each = 4 stream messages.
    assert len(streams) == 4, f"expected 4 audio chunks, got {len(streams)}"
    assert all(m.target is None for m in streams)
    for m in streams:
        assert m.data["modality"] == "audio"
        # waveform decodes to expected sample count.
        arr = np.frombuffer(m.data["audio_waveform"], dtype=np.float32)
        assert arr.size == 512
    assert len(results) == 1
    assert results[0].data.data["audio_chunk_count"] == 4
    assert results[0].data.data["segment_count"] == 2
    assert results[0].data.data["aborted"] is False
    # Final result must be a clean dict (no upstream-inherited fields)
    # so msgpack can serialize it to the coordinator.
    assert "keep" not in results[0].data.data
    assert results[0].data.data["modality"] == "audio"


def test_streaming_talker_stream_before_payload_buffers_correctly():
    sched = _make_scheduler()
    thread = _run(sched)
    try:
        rid = "req-pre"
        _segment(sched, rid, "Early bird.", segment_id=0, final=True)
        sched.inbox.put(IncomingMessage(request_id=rid, type="stream_done"))
        time.sleep(0.1)  # let scheduler buffer
        sched.inbox.put(
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=StagePayload(request_id=rid, request=None, data={}),
            )
        )
        msgs = _drain(sched, until_request_id=rid)
    finally:
        sched.stop()
        thread.join(timeout=1.0)

    streams = [m for m in msgs if m.type == "stream"]
    assert len(streams) == 2  # one segment × 2 chunks


def test_streaming_talker_abort_short_circuits_generation():
    class _SlowTalker(_FakeTalker):
        def omni_audio_generation(
            self, *, tts_text, voice_name, audio_detokenizer, stream, abort_event=None
        ):
            self.calls.append(tts_text)
            for _ in range(10):
                if abort_event is not None and abort_event.is_set():
                    return
                time.sleep(0.02)
                yield (torch.zeros(256, dtype=torch.float32), None, None, None)

    talker = _SlowTalker()
    sched = _make_scheduler(talker=talker)
    thread = _run(sched)
    try:
        rid = "req-abort"
        sched.inbox.put(
            IncomingMessage(
                request_id=rid,
                type="new_request",
                data=StagePayload(request_id=rid, request=None, data={}),
            )
        )
        _segment(sched, rid, "long segment", segment_id=0)
        # Let a couple chunks emit, then abort.
        time.sleep(0.05)
        sched.abort(rid)
        # Mark stream done so the scheduler finalizes the aborted request.
        sched.inbox.put(IncomingMessage(request_id=rid, type="stream_done"))
        # Also send the new_request first if not already — already sent.
        msgs = _drain(sched, until_request_id=rid)
    finally:
        sched.stop()
        thread.join(timeout=1.0)

    streams = [m for m in msgs if m.type == "stream"]
    # Aborted segment must produce fewer than the 10 chunks the slow talker
    # would otherwise generate.
    assert 0 < len(streams) < 10
    results = [m for m in msgs if m.type == "result"]
    assert len(results) == 1
    assert results[0].data.data["aborted"] is True
