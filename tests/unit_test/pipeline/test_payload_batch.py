# SPDX-License-Identifier: Apache-2.0
"""A stage's small CPU results for one remote target travel in one message."""

from __future__ import annotations

import asyncio
import pickle

import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.stage.runtime import BATCHED_PAYLOAD_BYTES_LIMIT
from sglang_omni.proto.messages import DataReadyMessage, PayloadBatchMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    make_result_message,
)
from tests.unit_test.pipeline.helpers import make_stage


def drain(
    batch_payloads: bool, results
) -> tuple[RecordingStageControlPlane, FakeRelay]:
    relay = FakeRelay()
    control_plane = RecordingStageControlPlane()
    scheduler = FakeScheduler()
    stage_obj = make_stage(
        name="encode",
        get_next=lambda request_id, output: "lm",
        endpoints={"lm": "inproc://lm"},
        relay=relay,
        scheduler=scheduler,
        control_plane=control_plane,
        batch_payloads=batch_payloads,
    )

    async def run() -> None:
        for message in results:
            stage_obj.active_requests.add(message.request_id)
            scheduler.outbox.put(message)
        stage_obj.end_outbox_drain()
        await stage_obj.drain_outbox()

    asyncio.run(run())
    return control_plane, relay


def results() -> list:
    small = [
        make_result_message(f"req-{i}", data={"codes": torch.full((1, 8), i)})
        for i in range(3)
    ]
    large = make_result_message(
        "req-big",
        data={"embeddings": torch.zeros(BATCHED_PAYLOAD_BYTES_LIMIT // 4 + 1)},
    )
    return [*small, large]


def test_small_cpu_results_share_one_message_and_large_ones_keep_the_relay():
    control_plane, relay = drain(True, results())
    sent = [msg for _, _, msg in control_plane.sent_to_stage]
    batches = [msg for msg in sent if isinstance(msg, PayloadBatchMessage)]
    assert len(batches) == 1
    batch = deserialize_message(serialize_message(batches[0]))
    assert batch.request_ids == ["req-0", "req-1", "req-2"]
    payloads = pickle.loads(batch.pickled_payloads)
    assert [int(payload.data["codes"][0, 0]) for payload in payloads] == [0, 1, 2]
    relayed = [msg for msg in sent if isinstance(msg, DataReadyMessage)]
    assert [msg.request_id for msg in relayed] == ["req-big"]


def test_without_the_flag_every_result_takes_the_relay():
    control_plane, relay = drain(False, results())
    sent = [msg for _, _, msg in control_plane.sent_to_stage]
    assert all(isinstance(msg, DataReadyMessage) for msg in sent)
    assert [msg.request_id for msg in sent] == ["req-0", "req-1", "req-2", "req-big"]
    restored = asyncio.run(
        stage_io.read_payload(relay, "req-1", DataRef.from_dict(sent[1].data_ref))
    )
    assert int(restored.data["codes"][0, 0]) == 1


def test_a_received_batch_reaches_the_scheduler_as_one_request_each_in_order():
    scheduler = FakeScheduler()
    stage_obj = make_stage(name="lm", scheduler=scheduler)
    sent, _ = drain(True, results()[:3])
    batch = next(msg for _, _, msg in sent.sent_to_stage)

    async def run() -> None:
        await stage_obj.handle_message(batch)
        await asyncio.gather(*list(stage_obj.receive_tasks))

    asyncio.run(run())
    received = []
    while not scheduler.inbox.empty():
        received.append(scheduler.inbox.get_nowait())
    assert [message.request_id for message in received] == ["req-0", "req-1", "req-2"]
    assert all(message.type == "new_request" for message in received)
    assert int(received[2].data.data["codes"][0, 0]) == 2


class RecordingSocket:
    def __init__(self, received=()):
        self.sent = []
        self.received = list(received)

    async def send(self, msg, on_submitted=None):
        self.sent.append(msg)
        if on_submitted is not None:
            on_submitted()
        else:
            pass

    async def recv(self):
        return self.received.pop(0)


def test_held_completions_and_streams_reach_the_coordinator_in_one_message():
    from sglang_omni.pipeline.control_plane import (
        CoordinatorControlPlane,
        StageControlPlane,
    )
    from sglang_omni.proto.messages import (
        CompleteMessage,
        MessageBatchMessage,
        StreamMessage,
    )

    stage_plane = StageControlPlane(
        "code2wav", "inproc://a", "inproc://b", "inproc://c"
    )
    stage_plane.coordinator_socket = RecordingSocket()
    submitted = []
    events = [
        StreamMessage(request_id="u1", from_stage="code2wav", chunk={"pcm": b"\0\1"}),
        CompleteMessage(request_id="u1", from_stage="code2wav", success=True),
        CompleteMessage(request_id="u2", from_stage="code2wav", success=True),
    ]

    async def send() -> None:
        stage_plane.hold_coordinator_messages()
        await stage_plane.send_stream(events[0])
        await stage_plane.send_complete(
            events[1], on_submitted=lambda: submitted.append("u1")
        )
        await stage_plane.send_complete(events[2])
        assert stage_plane.coordinator_socket.sent == []
        await stage_plane.flush_coordinator_messages()
        await stage_plane.send_complete(events[2])

    asyncio.run(send())
    batch, alone = stage_plane.coordinator_socket.sent
    assert isinstance(batch, MessageBatchMessage) and len(batch.messages) == 3
    assert submitted == ["u1"] and alone is events[2]

    coordinator_plane = CoordinatorControlPlane("inproc://d", "inproc://e")
    coordinator_plane.completion_socket = RecordingSocket(
        [deserialize_message(serialize_message(batch)), alone]
    )

    async def receive() -> list:
        return [await coordinator_plane.recv_event() for _ in range(4)]

    received = asyncio.run(receive())
    assert [type(msg).__name__ for msg in received] == [
        "StreamMessage",
        "CompleteMessage",
        "CompleteMessage",
        "CompleteMessage",
    ]
    assert received[0].chunk == {"pcm": b"\0\1"}
    assert [msg.request_id for msg in received] == ["u1", "u1", "u2", "u2"]
