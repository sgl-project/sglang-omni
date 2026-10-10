# SPDX-License-Identifier: Apache-2.0
"""Row-batched stream transport: one message carries a row per request."""

from __future__ import annotations

import asyncio
import queue
from types import MethodType, SimpleNamespace

import torch

from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from sglang_omni.pipeline.stage.stream_queue import (
    StreamItem,
    StreamItemBatch,
    StreamQueue,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler
from tests.unit_test.pipeline.helpers import make_stage


class BatchScheduler(FakeScheduler):
    accepts_stream_chunk_batch = True


def make_pair(receiver_scheduler, *, accept_early: bool = True):
    dispatcher = LocalStageDispatcher()
    receiver = make_stage(
        name="vocoder",
        scheduler=receiver_scheduler,
        can_accept_stream_before_payload=accept_early,
    )
    receiver.stream_queue = StreamQueue()
    sender = make_stage(
        name="talker",
        endpoints={"vocoder": "inproc://vocoder"},
        same_process_targets={"vocoder"},
        local_dispatcher=dispatcher,
    )
    dispatcher.register_many([sender, receiver])
    return sender, receiver


def drain(inbox: queue.Queue) -> list[IncomingMessage]:
    messages = []
    while not inbox.empty():
        messages.append(inbox.get_nowait())
    return messages


def batch_message(request_ids, *, target="vocoder"):
    data = torch.arange(len(request_ids) * 2).view(len(request_ids), 1, 2)
    return OutgoingMessage(
        request_id=request_ids[0],
        type="stream",
        data=data,
        target=target,
        metadata={"modality": "audio_codes"},
        request_ids=tuple(request_ids),
    )


def test_stream_item_batch_views_one_row_per_request() -> None:
    data = torch.arange(6).view(3, 1, 2)
    batch = StreamItemBatch(("a", "c"), (0, 2), (4, 7), data, "talker", {"k": 1})

    items = batch.items()

    assert [request_id for request_id, _ in items] == ["a", "c"]
    assert [item.chunk_id for _, item in items] == [4, 7]
    assert torch.equal(items[1][1].data, data[2])
    assert items[0][1].from_stage == "talker"


def test_a_colocated_batch_reaches_an_accepting_scheduler_as_one_message() -> None:
    scheduler = BatchScheduler()
    sender, _ = make_pair(scheduler)
    sender.active_requests.update({"a", "b"})

    async def run():
        await sender.route_stream_batch(batch_message(["a", "b"]))
        await sender.route_stream_batch(batch_message(["a", "b"]))

    asyncio.run(run())

    first, second = drain(scheduler.inbox)
    assert first.type == second.type == "stream_chunk_batch"
    assert first.data.request_ids == ("a", "b")
    assert first.data.chunk_ids == (0, 0)
    assert second.data.chunk_ids == (1, 1)
    assert first.data.from_stage == "talker"


def test_other_schedulers_still_receive_one_chunk_per_request() -> None:
    scheduler = FakeScheduler()
    sender, _ = make_pair(scheduler)
    sender.active_requests.update({"a", "b"})
    message = batch_message(["a", "b"])

    asyncio.run(sender.route_stream_batch(message))

    queued = drain(scheduler.inbox)
    assert [(m.type, m.request_id) for m in queued] == [
        ("stream_chunk", "a"),
        ("stream_chunk", "b"),
    ]
    assert torch.equal(queued[1].data.data, message.data[1])


def test_rows_of_finished_requests_are_dropped_before_sending() -> None:
    scheduler = BatchScheduler()
    sender, _ = make_pair(scheduler)
    sender.active_requests.update({"a", "c"})

    asyncio.run(sender.route_stream_batch(batch_message(["a", "b", "c"])))

    (queued,) = drain(scheduler.inbox)
    assert queued.data.request_ids == ("a", "c")
    assert queued.data.data.shape[0] == 2
    assert sender.stream_chunk_counters == {("a", "vocoder"): 1, ("c", "vocoder"): 1}


def test_the_receiver_drops_aborted_rows_and_rejects_early_chunks() -> None:
    scheduler = BatchScheduler()
    sender, receiver = make_pair(scheduler, accept_early=False)
    sender.active_requests.update({"a", "b", "c"})
    receiver.aborted.add("a")
    receiver.stream_queue.open("c")
    failures = []

    async def record_failure(request_id, error):
        failures.append(request_id)

    receiver.send_failure = record_failure

    asyncio.run(sender.route_stream_batch(batch_message(["a", "b", "c"])))

    (queued,) = drain(scheduler.inbox)
    assert queued.data.request_ids == ("c",)
    assert queued.data.rows == (2,)
    assert failures == ["b"]
    assert scheduler.aborted == ["b"]


def test_rows_bound_to_another_process_keep_the_per_request_path() -> None:
    sender = make_stage(
        name="talker",
        endpoints={"vocoder": "tcp://remote"},
        local_dispatcher=LocalStageDispatcher(),
    )
    sender.active_requests.update({"a", "b"})
    sent = []

    async def per_row(request_id, data, target, metadata=None):
        sent.append((request_id, data.shape, target))

    sender.send_stream_to_target = per_row

    asyncio.run(sender.route_stream_batch(batch_message(["a", "b"])))

    assert sent == [("a", (1, 2), "vocoder"), ("b", (1, 2), "vocoder")]


class RecordingStreamScheduler(StreamingSimpleScheduler):
    can_batch_stream_chunks = True
    accepts_stream_chunk_batch = True
    stream_chunk_batch_max = 3

    def __init__(self) -> None:
        super().__init__(None)
        self.batches: list[list[tuple[str, StreamItem]]] = []

    def on_stream_chunk_batch(self, items) -> None:
        self.batches.append(items)


def scheduler_batch(request_ids, first_chunk_id=0):
    return IncomingMessage(
        request_id=request_ids[0],
        type="stream_chunk_batch",
        data=StreamItemBatch(
            tuple(request_ids),
            tuple(range(len(request_ids))),
            tuple(first_chunk_id for _ in request_ids),
            torch.zeros(len(request_ids), 1, 2),
            "talker",
        ),
    )


def test_the_scheduler_expands_batches_and_filters_aborted_rows() -> None:
    scheduler = RecordingStreamScheduler()
    scheduler.record_aborted_request_id("a")
    scheduler.inbox.put(
        IncomingMessage("c", "stream_chunk", StreamItem(0, torch.zeros(1, 2), "talker"))
    )
    loop = asyncio.new_event_loop()
    try:
        assert not scheduler.message_aborted(scheduler_batch(["a", "b"]))
        scheduler.handle_message(scheduler_batch(["a", "b"]), loop)
    finally:
        loop.close()

    (items,) = scheduler.batches
    assert [request_id for request_id, _ in items] == ["b", "c"]


def test_a_batch_that_overflows_the_cap_waits_for_the_next_step() -> None:
    scheduler = RecordingStreamScheduler()
    overflow = scheduler_batch(["c", "d"], first_chunk_id=1)
    scheduler.inbox.put(overflow)

    batch = scheduler.collect_stream_chunk_batch(scheduler_batch(["a", "b"]))

    assert len(batch) == 1
    assert scheduler.pending_messages[0] is overflow


def fake_omni_scheduler(builder, *, session_rids=()):
    fake = SimpleNamespace(
        stream_output_builder=builder,
        session_bridge=None,
        aborted_request_ids={"gone"},
        first_emit_done=set(),
        outbox=queue.Queue(),
        active_session_unit=lambda rid: object() if rid in session_rids else None,
    )
    fake.emit_stream_batch = MethodType(OmniScheduler.emit_stream_batch, fake)
    fake.put_stream_messages = MethodType(OmniScheduler.put_stream_messages, fake)
    return fake


class BatchingBuilder:
    def __init__(self, result="batch") -> None:
        self.result = result
        self.calls = []
        self.batches = []

    def __call__(self, rid, data, req_output):
        self.calls.append(rid)
        return [OutgoingMessage(rid, "stream", data, target="vocoder")]

    def build_batch(self, entries):
        self.batches.append([rid for rid, _, _ in entries])
        if self.result is None:
            return None
        else:
            pass
        return [batch_message([rid for rid, _, _ in entries])]


def emit(fake, rids, skip=()):
    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id=rid, data=rid) for rid in rids]
    )
    mr_output = SimpleNamespace(outputs={rid: None for rid in rids})
    OmniScheduler.emit_stream_output(fake, sched_output, mr_output, skip)


def test_build_batch_emits_the_step_for_live_requests(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_event",
        lambda **kwargs: events.append(kwargs["request_id"]),
    )
    builder = BatchingBuilder()
    fake = fake_omni_scheduler(builder)

    emit(fake, ["a", "gone", "b", "done"], skip={"done"})

    assert builder.batches == [["a", "b"]]
    assert builder.calls == []
    assert fake.outbox.get_nowait().request_ids == ("a", "b")
    assert events == ["a", "b"]


def test_per_request_emission_covers_declined_and_session_steps() -> None:
    declining = BatchingBuilder(result=None)
    fake = fake_omni_scheduler(declining)
    emit(fake, ["a", "b"])
    assert declining.calls == ["a", "b"]

    builder = BatchingBuilder()
    fake = fake_omni_scheduler(builder, session_rids={"b"})
    fake.session_bridge = SimpleNamespace(
        stream_messages=lambda rid, data, out: [
            OutgoingMessage(rid, "stream", data, target="vocoder")
        ]
    )
    emit(fake, ["a", "b"])
    assert builder.batches == []
    assert builder.calls == ["a"]
