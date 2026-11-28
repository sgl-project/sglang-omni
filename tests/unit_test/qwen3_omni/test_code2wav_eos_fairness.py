# SPDX-License-Identifier: Apache-2.0

"""Code2Wav tail finalization yields without losing audio or completion."""

import asyncio
import time

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    SessionIdentity,
    SessionOperation,
)
from sglang_omni.scheduling.message import IncomingMessage


class FakeCodec:
    total_upsample = 4

    def __init__(self):
        self.frames = []
        self.batches = []
        self.after_first_window = None

    def __call__(self, codes):
        self.batches.append(codes[:, 0].tolist())
        self.frames.extend(codes[0, 0].tolist())
        callback, self.after_first_window = self.after_first_window, None
        if callback is not None:
            callback()
        return codes[:, :1, :].float().repeat_interleave(self.total_upsample, dim=-1)


class StopAfterTail(Code2WavScheduler):
    def complete_stream_request(self, request_id, messages):
        super().complete_stream_request(request_id, messages)
        if request_id == "old":
            self.stop()


def make_scheduler(scheduler_type=Code2WavScheduler):
    codec = FakeCodec()
    return scheduler_type(codec, "cpu", stream_chunk_size=2, left_context_size=0), codec


def payload_for(request_id):
    return StagePayload(request_id, OmniRequest([], params={"stream": True}), {})


def enqueue_open(scheduler, index=0, *, priority=False):
    request_id = f"open-{index}"
    operation = SessionOperation(
        "open", SessionIdentity(f"new-{index}", 1), ("code2wav",)
    )
    request = OmniRequest(None, metadata={SESSION_METADATA_KEY: operation.to_dict()})
    message = IncomingMessage(
        request_id, "new_request", StagePayload(request_id, request, {})
    )
    if priority:
        scheduler.enqueue(message)
    else:
        scheduler.inbox.put(message)


def prepare_tail(scheduler, *, with_payload=True):
    frames = list(range(1, 12))
    if with_payload:
        scheduler.handle_streaming_new_request("old", payload_for("old"))
    state = scheduler.get_or_create_stream_state("old")
    if not with_payload:
        state.stream_enabled = True
    state.chunks.extend(torch.tensor([frame]) for frame in frames)
    return frames


def results(scheduler):
    return [message for message in scheduler.outbox.queue if message.type == "result"]


def test_new_session_open_completes_while_existing_tail_is_decoding():
    scheduler, codec = make_scheduler(StopAfterTail)
    frames = prepare_tail(scheduler)
    codec.after_first_window = lambda: enqueue_open(scheduler)
    scheduler.inbox.put(IncomingMessage("old", "stream_done", None))

    scheduler.start()

    assert [message.request_id for message in results(scheduler)] == ["open-0", "old"]
    assert results(scheduler)[0].data.data == {"opened": True}
    assert codec.frames == frames
    assert "old" not in scheduler.pending_finalizations


def test_early_eos_drains_then_completes_once_after_payload_arrives():
    scheduler, codec = make_scheduler()
    frames = prepare_tail(scheduler, with_payload=False)
    scheduler.handle_stream_done("old")
    while scheduler.pending_finalizations:
        scheduler.run_ready_step()
    assert codec.frames == frames
    assert not results(scheduler)
    assert any(message.type == "stream" for message in scheduler.outbox.queue)

    scheduler.handle_streaming_new_request("old", payload_for("old"))
    assert [message.request_id for message in results(scheduler)] == ["old"]
    assert codec.frames == frames
    assert "old" not in scheduler.pending_done
    assert "old" not in scheduler.stream_states


def test_abort_drops_pending_tail_without_decoding_more_frames():
    scheduler, codec = make_scheduler()
    frames = prepare_tail(scheduler)
    scheduler.handle_stream_done("old")
    scheduler.run_ready_step()
    assert codec.frames == frames[:2]

    scheduler.abort("old")
    scheduler.run_ready_step()
    assert codec.frames == frames[:2]
    assert "old" not in scheduler.pending_finalizations
    assert "old" not in scheduler.stream_states
    assert not results(scheduler)


def test_pending_tail_makes_progress_with_nonempty_inbox():
    scheduler, codec = make_scheduler(StopAfterTail)
    frames = prepare_tail(scheduler)
    scheduler.inbox.put(IncomingMessage("old", "stream_done", None))
    for index in range(256):
        enqueue_open(scheduler, index)

    scheduler.start()

    messages = results(scheduler)
    assert messages[0].request_id == "open-0"
    assert sum(message.request_id == "old" for message in messages) == 1
    assert scheduler.inbox.qsize() > 0
    assert codec.frames == frames


def test_session_open_lane_is_bounded_per_scheduler_iteration():
    scheduler, _ = make_scheduler()
    scheduler.inbox.put(IncomingMessage("old", "stream_done", None))
    for index in range(5):
        enqueue_open(scheduler, index, priority=True)
    loop = asyncio.new_event_loop()
    try:
        scheduler.service_deferred_work(loop)
    finally:
        loop.close()
    assert [message.request_id for message in results(scheduler)] == [
        "open-0",
        "open-1",
    ]
    assert len(scheduler.open_inbox) == 3
    assert scheduler.inbox.get_nowait().request_id == "old"


def session_message(operation, identity, request_id):
    op = SessionOperation(operation, identity, ("code2wav",))
    request = OmniRequest(None, metadata={SESSION_METADATA_KEY: op.to_dict()})
    return IncomingMessage(
        request_id, "new_request", StagePayload(request_id, request, {})
    )


def queue_early_close(scheduler, identity):
    for index in range(scheduler.OPEN_BATCH_BUDGET):
        enqueue_open(scheduler, index, priority=True)
    scheduler.enqueue(session_message("open", identity, "target-open"))
    scheduler.enqueue(session_message("close", identity, "target-close"))


def test_early_close_follows_its_open_behind_other_priority_messages():
    scheduler, _ = make_scheduler()
    identity = SessionIdentity("target", 7)
    queue_early_close(scheduler, identity)
    loop = asyncio.new_event_loop()
    try:
        scheduler.service_deferred_work(loop)
        if not scheduler.inbox.empty():
            scheduler.handle_message(scheduler.inbox.get_nowait(), loop)
        scheduler.service_deferred_work(loop)
    finally:
        loop.close()

    assert [msg.request_id for msg in results(scheduler)[-2:]] == [
        "target-open",
        "target-close",
    ]
    assert identity.id not in scheduler.session_units


def test_abort_before_dequeue_skips_open_and_still_completes_early_close():
    scheduler, _ = make_scheduler()
    identity = SessionIdentity("target", 7)
    queue_early_close(scheduler, identity)
    scheduler.abort("target-open")
    loop = asyncio.new_event_loop()
    try:
        scheduler.service_deferred_work(loop)
        scheduler.service_deferred_work(loop)
        assert [msg.request_id for msg in results(scheduler)[-1:]] == ["target-close"]
        assert identity.id not in scheduler.session_units

        replacement = SessionIdentity("target", 8)
        scheduler.enqueue(session_message("open", replacement, "replacement-open"))
        scheduler.service_deferred_work(loop)
        assert results(scheduler)[-1].request_id == "replacement-open"
        assert identity.id in scheduler.session_units
    finally:
        loop.close()


def test_close_after_open_ack_stays_behind_already_queued_read():
    scheduler, _ = make_scheduler()
    identity = SessionIdentity("target", 7)
    scheduler.enqueue(session_message("open", identity, "target-open"))
    loop = asyncio.new_event_loop()
    try:
        scheduler.service_deferred_work(loop)
        scheduler.enqueue(session_message("read", identity, "target-read"))
        scheduler.enqueue(session_message("close", identity, "target-close"))
        for _ in range(2):
            scheduler.handle_message(scheduler.inbox.get_nowait(), loop)
    finally:
        loop.close()

    assert [msg.request_id for msg in results(scheduler)] == [
        "target-open",
        "target-read",
        "target-close",
    ]
    assert identity.id not in scheduler.session_units
    assert identity.id not in scheduler.session_reads


def test_payload_arrives_during_deferred_eos_and_terminal_is_emitted_once():
    scheduler, codec = make_scheduler()
    frames = prepare_tail(scheduler, with_payload=False)
    scheduler.handle_stream_done("old")
    scheduler.run_ready_step()
    scheduler.handle_streaming_new_request("old", payload_for("old"))

    assert not results(scheduler)
    while scheduler.pending_finalizations:
        scheduler.run_ready_step()
    assert codec.frames == frames
    assert [msg.request_id for msg in results(scheduler)] == ["old"]
    assert "old" not in scheduler.stream_states
    assert "old" not in scheduler.pending_done


class SustainedOpenTraffic(StopAfterTail):
    scheduler_iterations = 0
    ordinary_frames_at_eos = 0

    def complete_stream_request(self, request_id, messages):
        if request_id == "old":
            state = self.stream_states.get("normal")
            if state is not None:
                self.ordinary_frames_at_eos = self.ready(state)
        super().complete_stream_request(request_id, messages)

    def service_deferred_work(self, loop):
        self.scheduler_iterations += 1
        if self.scheduler_iterations >= 80:
            self.stop()
            return False
        for n in range(3):
            enqueue_open(self, self.scheduler_iterations * 3 + n, priority=True)
        return super().service_deferred_work(loop)


def test_sustained_priority_opens_still_allow_inbox_chunks_and_eos_to_progress():
    scheduler, codec = make_scheduler(SustainedOpenTraffic)
    frames = prepare_tail(scheduler)
    scheduler.inbox.put(IncomingMessage("old", "stream_done", None))
    item = StreamItem(0, torch.tensor([99]), "talker", metadata={"stream": True})
    scheduler.inbox.put(IncomingMessage("normal", "stream_chunk", item))

    scheduler.start()

    assert scheduler.scheduler_iterations < 80
    assert len(scheduler.open_inbox) > 0
    assert scheduler.ordinary_frames_at_eos == 1
    assert codec.frames == frames
    assert [r.request_id for r in results(scheduler)].count("old") == 1


def test_full_tail_window_batches_with_compatible_active_stream():
    scheduler, codec = make_scheduler()
    scheduler.handle_streaming_new_request("normal", payload_for("normal"))
    normal = scheduler.get_or_create_stream_state("normal")
    normal.chunks.extend(torch.tensor([frame]) for frame in (51, 52))
    normal.stream_enabled = True
    prepare_tail(scheduler)
    scheduler.get_or_create_stream_state("old").stream_enabled = True
    scheduler.handle_stream_done("old")

    scheduler.run_ready_step()

    assert len(codec.batches) == 1
    assert sorted(codec.batches[0]) == [[1, 2], [51, 52]]
    assert {
        msg.request_id for msg in scheduler.outbox.queue if msg.type == "stream"
    } == {
        "normal",
        "old",
    }
    assert not results(scheduler)
    while scheduler.pending_finalizations:
        scheduler.run_ready_step()
    assert [msg.request_id for msg in results(scheduler)] == ["old"]
    assert "normal" in scheduler.stream_states


def test_overdue_open_and_audio_both_progress(monkeypatch):
    scheduler, codec = make_scheduler()
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)
    prepare_tail(scheduler)
    scheduler.handle_stream_done("old")
    enqueue_open(scheduler, priority=True)
    monkeypatch.setattr(time, "monotonic", lambda: 2.0)
    loop = asyncio.new_event_loop()
    try:
        assert not scheduler.service_deferred_work(loop)
        assert [msg.request_id for msg in results(scheduler)] == ["open-0"]
        enqueue_open(scheduler, index=1, priority=True)
        monkeypatch.setattr(time, "monotonic", lambda: 3.0)
        assert scheduler.service_deferred_work(loop)
    finally:
        loop.close()

    assert codec.frames == [1, 2]
    assert [msg.request_id for msg in results(scheduler)] == ["open-0"]
