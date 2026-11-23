# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import pickle
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataKind, DataRef, TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher, LocalStreamChunk
from sglang_omni.pipeline.stage.runtime import BATCHED_STREAM_CHUNK_BYTES_LIMIT, Stage
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto import DataReadyMessage, OmniRequest, StagePayload
from sglang_omni.proto.messages import StreamChunkBatchMessage
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.scheduling.message import OutgoingMessage, put_messages
from tests.unit_test.fixtures.trace_capture import capture_comm_trace, events_named
from tests.unit_test.pipeline.helpers import make_stage


class FakeControlPlane:
    recv_endpoint = "inproc://stage"

    def __init__(self) -> None:
        self.streams = []
        self.stage_messages = []
        self.completions = []

    async def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def send_stream(self, msg) -> None:
        self.streams.append(msg)

    async def send_to_stage(self, target, endpoint, msg) -> None:
        self.stage_messages.append((target, endpoint, msg))

    async def send_complete(self, msg) -> None:
        self.completions.append(msg)


class FakeRelay:
    def __init__(self) -> None:
        self.device = "cpu"
        self.puts = []

    async def put_async(
        self,
        tensor,
        request_id=None,
        dst_rank=None,
        receiver_id=None,
    ):
        del dst_rank, receiver_id
        self.puts.append((request_id, tensor))
        return DoneOp(tensor.numel())

    def close(self) -> None:
        pass

    def cleanup(self, request_id: str) -> None:
        pass


class DoneOp:
    def __init__(self, size: int = 1) -> None:
        self.metadata = {"transfer_info": {"size": size}}

    async def wait_for_completion(self) -> None:
        pass

    def mark_receiver_done(self) -> None:
        pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        raise exc


class AbortOnReadRelay(FakeRelay):
    def __init__(self, on_wait) -> None:
        super().__init__()
        self.on_wait = on_wait
        self.gets = 0

    async def get_async(self, metadata, dest_tensor, request_id):
        del metadata, dest_tensor, request_id
        self.gets += 1
        return CallbackOp(self.on_wait)


class CallbackOp:
    def __init__(self, on_wait) -> None:
        self.on_wait = on_wait

    async def wait_for_completion(self) -> None:
        self.on_wait()

    def mark_receiver_done(self) -> None:
        pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        raise exc


# Stream chunks now always move through the transport router's relay (CUDA-IPC on
# GPU, shm on CPU). These helpers build real relay-backed messages over an
# in-process ShmRelay so the receive path is exercised end to end, matching what
# ``write_stream_chunk`` / ``write_payload`` produce on the wire.


async def make_relay_chunk(
    relay,
    *,
    request_id: str,
    from_stage: str,
    to_stage: str,
    chunk_id: int,
    data,
    metadata: dict | None = None,
) -> DataReadyMessage:
    object_id = f"{request_id}:stream:{from_stage}:{to_stage}:{chunk_id}"
    data_ref, op = await stage_io.write_tensor(
        relay,
        object_id,
        data,
        transport=TransportKind.SHM,
        kind=DataKind.STREAM_CHUNK,
        request_id=request_id,
        from_stage=from_stage,
        to_stage=to_stage,
    )
    pending_ops = [op]
    data_ref = await stage_io.with_stream_metadata(
        relay,
        data_ref,
        metadata,
        TransportKind.SHM,
        pending_ops,
    )
    return DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=to_stage,
        data_ref=data_ref.to_dict(),
        chunk_id=chunk_id,
    )


async def make_relay_payload(
    relay,
    payload: StagePayload,
    *,
    from_stage: str = "tts_engine",
    to_stage: str = "vocoder",
) -> DataReadyMessage:
    data_ref, op = await stage_io.write_payload(
        relay,
        payload.request_id,
        payload,
        transport=TransportKind.SHM,
    )
    return DataReadyMessage(
        request_id=payload.request_id,
        from_stage=from_stage,
        to_stage=to_stage,
        data_ref=data_ref.to_dict(),
    )


def test_terminal_scheduler_stream_routes_to_coordinator() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage.active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.1], "modality": "audio"},
            )
        )
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.2], "modality": "audio"},
            )
        )

        stage.end_outbox_drain()
        await stage.drain_outbox_external()

        assert len(control_plane.streams) == 2
        msg = control_plane.streams[0]
        assert msg.request_id == "req"
        assert msg.from_stage == "vocoder"
        assert msg.chunk == {"audio_data": [0.1], "modality": "audio"}
        assert msg.modality == "audio"
        assert [msg.chunk_id for msg in control_plane.streams] == [0, 1]

    asyncio.run(run())


def test_outbox_drain_reuses_one_executor_wakeup_for_ready_messages(
    monkeypatch,
) -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        run_in_executor = AsyncMock(side_effect=lambda _, get: get())
        monkeypatch.setattr(loop, "run_in_executor", run_in_executor)

        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage.active_requests.add("req-live")

        messages = [
            OutgoingMessage("req-live", "stream", {"sequence": 0}),
            OutgoingMessage("req-stale", "stream", {"sequence": 999}),
            OutgoingMessage("req-live", "stream", {"sequence": 1}),
            OutgoingMessage("req-live", "result", {"answer": "done"}),
        ]
        for message in messages:
            scheduler.outbox.put(message)

        stage.end_outbox_drain()
        await stage.drain_outbox_external()

        assert [
            (msg.chunk_id, msg.chunk["sequence"]) for msg in control_plane.streams
        ] == [(0, 0), (1, 1)]
        assert len(control_plane.completions) == 1
        completion = control_plane.completions[0]
        assert (completion.success, completion.result) == (True, {"answer": "done"})
        assert scheduler.outbox.empty()
        assert not stage.active_requests

        # Before this optimization each message required an executor round trip.
        run_in_executor.assert_awaited_once()

    asyncio.run(run())


def test_outbox_drain_yields_after_ready_message_batch(monkeypatch) -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        run_in_executor = AsyncMock(side_effect=lambda _, get: get())
        monkeypatch.setattr(loop, "run_in_executor", run_in_executor)

        execution_order: list[int | str] = []

        class RecordingControlPlane(FakeControlPlane):
            async def send_stream(self, msg) -> None:
                execution_order.append(msg.chunk["sequence"])
                await super().send_stream(msg)

        control_plane = RecordingControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage.active_requests.add("req-live")

        for sequence in range(1, 66):
            scheduler.outbox.put(
                OutgoingMessage("req-live", "stream", {"sequence": sequence})
            )

        async def competing_ready_coroutine() -> None:
            execution_order.append("competing-coroutine")

        competing_task = asyncio.create_task(competing_ready_coroutine())
        stage.end_outbox_drain()
        await stage.drain_outbox_external()
        await competing_task

        assert execution_order == [
            *range(1, 65),
            "competing-coroutine",
            65,
        ]
        assert [msg.chunk["sequence"] for msg in control_plane.streams] == list(
            range(1, 66)
        )
        assert scheduler.outbox.empty()
        assert run_in_executor.await_count == 2

    asyncio.run(run())


def test_explicit_scheduler_stream_target_keeps_stage_to_stage_routing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    async def run() -> None:
        control_plane = FakeControlPlane()
        relay = FakeRelay()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        codes = torch.empty(4096, 1, dtype=torch.long)
        stage = Stage(
            name="tts_engine",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"vocoder": "inproc://vocoder"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data=codes,
                target="vocoder",
                metadata={"modality": "audio_codes"},
            )
        )

        stage.end_outbox_drain()
        await stage.drain_outbox_external()

        assert control_plane.streams == []
        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        target, endpoint, msg = control_plane.stage_messages[0]
        assert target == "vocoder"
        assert endpoint == "inproc://vocoder"
        assert msg.request_id == "req"
        assert msg.from_stage == "tts_engine"
        assert msg.to_stage == "vocoder"
        assert msg.chunk_id == 0
        assert DataRef.from_dict(msg.data_ref).metadata == {"modality": "audio_codes"}

    asyncio.run(run())


def test_remote_stream_run_packs_each_target_around_a_relay_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    async def run() -> None:
        control_plane = FakeControlPlane()
        relay = FakeRelay()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="thinker",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={
                "decode": "inproc://decode",
                "talker_ar": "inproc://talker_ar",
            },
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.active_requests.update({"req-a", "req-b"})
        stage.replica_bindings["req-b"] = {"decode": 1}
        hidden = torch.zeros(
            BATCHED_STREAM_CHUNK_BYTES_LIMIT // 4 + 1, dtype=torch.float32
        )
        put_messages(
            scheduler.outbox,
            [
                OutgoingMessage(
                    "req-a", "stream", torch.tensor([1]), "decode", {"token_id": 1}
                ),
                OutgoingMessage(
                    "req-a", "stream", torch.tensor([1]), "talker_ar", {"token_id": 1}
                ),
                OutgoingMessage(
                    "req-b", "stream", torch.tensor([2]), "talker_ar", {"token_id": 2}
                ),
                OutgoingMessage(
                    "req-b", "stream", hidden, "talker_ar", {"token_id": 3}
                ),
                OutgoingMessage(
                    "req-b", "stream", torch.tensor([4]), "decode", {"token_id": 4}
                ),
                OutgoingMessage(
                    "req-a", "stream", torch.tensor([5]), "talker_ar", {"token_id": 5}
                ),
            ],
        )

        stage.end_outbox_drain()
        await stage.drain_outbox_external()

        def describe(message) -> list[tuple[str, int, int, int]]:
            if isinstance(message, StreamChunkBatchMessage):
                return [
                    (
                        chunk.request_id,
                        chunk.chunk_id,
                        chunk.metadata["token_id"],
                        int(chunk.data[0]),
                    )
                    for chunk in stage_io.unpack_stream_chunk_batch(message)
                ]
            else:
                metadata = DataRef.from_dict(message.data_ref).metadata
                return [(message.request_id, message.chunk_id, metadata["token_id"], 0)]

        assert [
            (target, describe(message))
            for target, _, message in control_plane.stage_messages
        ] == [
            ("decode", [("req-a", 0, 1, 1)]),
            ("talker_ar", [("req-a", 0, 1, 1), ("req-b", 0, 2, 2)]),
            ("talker_ar", [("req-b", 1, 3, 0)]),
            ("decode", [("req-b", 0, 4, 4)]),
            ("talker_ar", [("req-a", 1, 5, 5)]),
        ]
        assert len(relay.puts) == 1
        assert control_plane.stage_messages[1][2].replica_bindings == {
            "req-b": {"decode": 1}
        }

    with capture_comm_trace(monkeypatch) as events:
        asyncio.run(run())

    assert [
        (event["request_id"], event["to_stage"], event["chunk_id"], event["bytes"])
        for event in events_named(events, "comm_stream_send")
        if event["transport"] == "inline"
    ] == [
        ("req-a", "decode", 0, 8),
        ("req-a", "talker_ar", 0, 8),
        ("req-b", "talker_ar", 0, 8),
        ("req-b", "decode", 0, 8),
        ("req-a", "talker_ar", 1, 8),
    ]


@pytest.mark.parametrize(
    "metadata_device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires CUDA"
                ),
            ],
        ),
    ],
)
def test_stream_chunk_with_a_metadata_tensor_takes_the_relay(
    monkeypatch: pytest.MonkeyPatch, metadata_device: str
) -> None:
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="thinker",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"talker_ar": "inproc://talker_ar"},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
        )
        stage.active_requests.update({"req-a", "req-b"})
        put_messages(
            scheduler.outbox,
            [
                OutgoingMessage(
                    "req-a",
                    "stream",
                    torch.tensor([1]),
                    "talker_ar",
                    {"hidden": torch.zeros(2, device=metadata_device)},
                ),
                OutgoingMessage("req-b", "stream", torch.tensor([2]), "talker_ar"),
            ],
        )

        stage.end_outbox_drain()
        await stage.drain_outbox_external()

        assert [
            (
                (message.request_ids, message.dtypes)
                if isinstance(message, StreamChunkBatchMessage)
                else (message.request_id, DataRef.from_dict(message.data_ref).dtype)
            )
            for _, _, message in control_plane.stage_messages
        ] == [("req-a", "torch.int64"), (["req-b"], ["torch.int64"])]

    asyncio.run(run())


def test_stream_chunk_batch_round_trips_through_the_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = [
        torch.tensor([7], dtype=torch.long),
        torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        torch.tensor(5, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.arange(10, dtype=torch.long)[1::3],
        torch.arange(8, dtype=torch.long)[::3][:1],
        torch.tensor([1 + 2j, 3 + 4j]).conj(),
        torch.tensor(1 + 2j).conj().imag,
    ]
    chunks = [
        stage_io.BatchedStreamChunk(
            request_id=f"req-{index % 2}",
            chunk_id=index,
            data=tensor,
            metadata=None if index == 0 else {"token_id": index, "span": (index, 1)},
        )
        for index, tensor in enumerate(tensors)
    ]
    clone = torch.Tensor.clone
    cloned_shapes = []

    def counting_clone(tensor, *args, **kwargs):
        cloned_shapes.append(tuple(tensor.shape))
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)

    message = deserialize_message(
        serialize_message(
            stage_io.pack_stream_chunk_batch(
                chunks,
                from_stage="thinker",
                to_stage="talker_ar",
                replica_bindings={"req-1": {"talker_ar": 1}},
            )
        )
    )
    restored = stage_io.unpack_stream_chunk_batch(message)

    assert cloned_shapes == [(1,), (2,), (1,)]
    assert message.replica_bindings == {"req-1": {"talker_ar": 1}}
    assert [
        (chunk.request_id, chunk.chunk_id, chunk.metadata) for chunk in restored
    ] == [(chunk.request_id, chunk.chunk_id, chunk.metadata) for chunk in chunks]
    for original, received in zip(chunks, restored, strict=True):
        assert received.data.dtype == original.data.dtype
        assert torch.equal(received.data, original.data)
        assert received.data.untyped_storage().nbytes() == received.data.nbytes


class GatedRelay(FakeRelay):
    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__()
        self.gate = gate
        self.entered = asyncio.Event()

    async def get_async(self, metadata, dest_tensor, request_id):
        del metadata, dest_tensor, request_id
        self.entered.set()
        await self.gate.wait()
        return DoneOp()


def test_stream_chunk_batch_waits_only_behind_its_own_request_lane() -> None:
    async def run() -> None:
        gate = asyncio.Event()
        relay = GatedRelay(gate)
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            aborted=[],
            abort=lambda request_id: scheduler.aborted.append(request_id),
        )
        stage = Stage(
            name="talker_ar",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"thinker": "inproc://thinker"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.stream_queue = StreamQueue()
        stage.stream_queue.open("req-a")
        stage.stream_queue.open("req-b")
        stage.schedule_receive_task(
            await make_relay_chunk(
                relay,
                request_id="req-a",
                from_stage="thinker",
                to_stage="talker_ar",
                chunk_id=0,
                data=torch.zeros(4),
            )
        )
        stage.schedule_stream_chunk_batch(
            stage_io.pack_stream_chunk_batch(
                [
                    stage_io.BatchedStreamChunk(
                        request_id=request_id,
                        chunk_id=chunk_id,
                        data=torch.tensor([chunk_id]),
                        metadata={"token_id": chunk_id},
                    )
                    for request_id, chunk_id in (
                        ("req-a", 1),
                        ("req-b", 0),
                        ("req-c", 0),
                    )
                ],
                from_stage="thinker",
                to_stage="talker_ar",
                replica_bindings={},
            )
        )
        stage.schedule_receive_task(
            DataReadyMessage(
                request_id="req-a",
                from_stage="thinker",
                to_stage="talker_ar",
                data_ref=None,
                is_done=True,
            )
        )

        def delivered() -> list[tuple[str, str, int | None]]:
            messages = []
            while not scheduler.inbox.empty():
                message = scheduler.inbox.get_nowait()
                messages.append(
                    (
                        message.request_id,
                        message.type,
                        None if message.data is None else message.data.chunk_id,
                    )
                )
            return messages

        await relay.entered.wait()
        assert delivered() == [("req-b", "stream_chunk", 0)]
        gate.set()
        await asyncio.gather(*stage.receive_tasks)
        assert delivered() == [
            ("req-a", "stream_chunk", 0),
            ("req-a", "stream_chunk", 1),
            ("req-a", "stream_done", None),
        ]
        assert [
            (completion.request_id, completion.success)
            for completion in control_plane.completions
        ] == [("req-c", False)]
        assert scheduler.aborted == ["req-c"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption", ["unknown dtype", "tensor bytes not bytes", "metadata not a dict"]
)
def test_unreadable_stream_chunk_batch_fails_each_request_it_carries(
    corruption: str,
) -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            aborted=[],
            abort=lambda request_id: scheduler.aborted.append(request_id),
        )
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"thinker": "inproc://thinker"},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
        )
        stage.stream_queue = StreamQueue()
        stage.stream_queue.open("req-a")
        stage.stream_queue.open("req-b")
        message = stage_io.pack_stream_chunk_batch(
            [
                stage_io.BatchedStreamChunk(
                    request_id=request_id,
                    chunk_id=chunk_id,
                    data=torch.tensor([chunk_id]),
                    metadata=None,
                )
                for request_id, chunk_id in (("req-a", 0), ("req-b", 0), ("req-a", 1))
            ],
            from_stage="thinker",
            to_stage="decode",
            replica_bindings={},
        )
        if corruption == "unknown dtype":
            message.dtypes[1] = "torch.float8_e4m3fn"
        elif corruption == "tensor bytes not bytes":
            message.tensor_bytes[1] = 8
        else:
            message.pickled_metadata = pickle.dumps([None, ["invalid"], None])

        stage.schedule_stream_chunk_batch(message)
        await asyncio.gather(*stage.receive_tasks)

        assert scheduler.inbox.empty()
        assert [
            (completion.request_id, completion.success)
            for completion in control_plane.completions
        ] == [("req-a", False), ("req-b", False)]
        assert scheduler.aborted == ["req-a", "req-b"]

    asyncio.run(run())


def test_stream_queue_drops_chunk_after_request_cleanup() -> None:
    stream_queue = StreamQueue()
    stream_queue.open("req")
    stream_queue.close("req")

    stream_queue.put(
        "req",
        StreamItem(
            chunk_id=0,
            data=torch.tensor([1]),
            from_stage="thinker",
        ),
    )

    assert not stream_queue.has("req")


def test_stage_fails_pre_payload_stream_chunk_by_default() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.stream_queue = StreamQueue()
        codes = torch.arange(11, dtype=torch.float32)

        await stage.on_stream_chunk(
            await make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )

        assert scheduler.inbox.empty()
        assert len(control_plane.completions) == 1
        assert control_plane.completions[0].success is False
        assert "pre-payload stream data" in control_plane.completions[0].error

    asyncio.run(run())


def test_stage_routes_stream_chunk_after_payload_by_default() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.stream_queue = StreamQueue()
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage.on_data_ready(await make_relay_payload(relay, payload))
        codes = torch.arange(11, dtype=torch.float32)
        await stage.on_stream_chunk(
            await make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )
        payload_msg = scheduler.inbox.get_nowait()
        chunk_msg = scheduler.inbox.get_nowait()
        assert payload_msg.type == "new_request"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, codes)

    asyncio.run(run())


def test_stage_routes_pre_payload_stream_events_for_capable_receiver() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        stage.stream_queue = StreamQueue()
        codes = torch.arange(11, dtype=torch.float32)

        await stage.on_stream_chunk(
            await make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"modality": "audio_codes"},
            )
        )

        chunk_msg = scheduler.inbox.get_nowait()
        assert chunk_msg.request_id == "req"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, codes)
        assert chunk_msg.data.metadata == {"modality": "audio_codes"}

        await stage.on_stream_signal(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=None,
                is_done=True,
            )
        )
        early_done_msg = scheduler.inbox.get_nowait()
        assert early_done_msg.request_id == "req"
        assert early_done_msg.type == "stream_done"

        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage.on_data_ready(await make_relay_payload(relay, payload))
        payload_msg = scheduler.inbox.get_nowait()
        assert payload_msg.request_id == "req"
        assert payload_msg.type == "new_request"
        assert payload_msg.data.data == {"ready": True}

    asyncio.run(run())


def test_stage_stream_chunk_received_after_relay_materialization(monkeypatch) -> None:
    """Cross-GPU chunks emit receive only after relay data and metadata are restored."""
    order: list[str] = []

    async def fake_read_stream_chunk(self, *, relay, data_ref):
        del self, relay, data_ref
        order.append("stream_chunk_read")
        return "chunk", {"modality": "audio_codes"}

    async def fake_route(self, request_id, item):
        del self, request_id, item
        order.append("routed")

    monkeypatch.setattr(CommEngine, "read_stream_chunk", fake_read_stream_chunk)
    monkeypatch.setattr(Stage, "route_stream_item_or_fail", fake_route)
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: order.append(kwargs["event_name"]),
    )

    async def run() -> None:
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=FakeControlPlane(),
            relay=FakeRelay(),
            scheduler=SimpleNamespace(outbox=queue.Queue()),
            can_accept_stream_before_payload=True,
        )
        await stage.on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=(
                    await make_relay_chunk(
                        FakeRelay(),
                        request_id="req",
                        from_stage="tts_engine",
                        to_stage="vocoder",
                        chunk_id=0,
                        data=torch.empty(1),
                    )
                ).data_ref,
                chunk_id=0,
            )
        )

    asyncio.run(run())

    assert order == ["stream_chunk_read", "stage_stream_chunk_received", "routed"]


def test_stage_stream_error_fails_request_even_with_stream_queue() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            aborted=[],
            abort=lambda request_id: scheduler.aborted.append(request_id),
        )
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage.stream_queue = StreamQueue()
        stage.stream_queue.open("req")

        await stage.queue_stream_error(
            "req",
            from_stage="thinker",
            error=RuntimeError("stream failed"),
        )

        assert scheduler.aborted == ["req"]
        assert len(control_plane.completions) == 1
        assert control_plane.completions[0].success is False
        assert control_plane.completions[0].error == "stream failed"
        assert not stage.stream_queue.has("req")
        assert "req" in stage.aborted

    asyncio.run(run())


def test_terminal_request_can_be_readmitted_after_cleanup() -> None:
    async def run() -> None:
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={},
            control_plane=FakeControlPlane(),
            relay=FakeRelay(),
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        stage.stream_queue = StreamQueue()
        stage.stream_queue.open("req")
        stage.active_requests.add("req")

        stage.clear_request_state("req")
        await stage.receive_local_payload(
            "req",
            "tts_engine",
            StagePayload(request_id="req", request=OmniRequest(inputs=None), data={}),
        )
        incoming = scheduler.inbox.get_nowait()
        assert incoming.request_id == "req"
        assert incoming.type == "new_request"
        assert "req" in stage.active_requests
        assert stage.stream_queue.has("req")
        assert stage.control_plane.completions == []

    asyncio.run(run())


def test_write_stream_chunk_uses_relay() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        relay = FakeRelay()
        codes = torch.empty(11, 1, dtype=torch.long)

        data_ref, ops = await stage_io.write_stream_chunk(
            relay,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            transport=TransportKind.SHM,
        )
        await control_plane.send_to_stage(
            "vocoder",
            "inproc://vocoder",
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=data_ref.to_dict(),
                chunk_id=0,
            ),
        )
        for op in ops:
            op.mark_receiver_done()
            await op.wait_for_completion()

        assert len(relay.puts) == 1
        assert relay.puts[0][0] == "req:stream:tts_engine:vocoder:0"
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        expected_size = codes.contiguous().view(torch.uint8).numel()
        data_ref = DataRef.from_dict(msg.data_ref)
        assert data_ref.buffer.info == {"transfer_info": {"size": expected_size}}

    asyncio.run(run())


def test_stage_drops_stream_chunk_after_abort_during_relay_read() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        codes = torch.empty(11, 1, dtype=torch.long)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = AbortOnReadRelay(lambda: stage.on_abort("req"))
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.stream_queue = None

        await stage.on_stream_chunk(
            await make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(run())


def test_stage_drains_relay_stream_chunk_for_already_aborted_request() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        codes = torch.empty(11, 1, dtype=torch.long)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = AbortOnReadRelay(lambda: None)
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.aborted.add("req")
        await stage.on_stream_chunk(
            await make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"latency": torch.zeros(1)},
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 2

    asyncio.run(run())


def test_stage_drains_relay_payload_for_already_aborted_request() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = AbortOnReadRelay(lambda: None)
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage.aborted.add("req")
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage.on_data_ready(await make_relay_payload(relay, payload))

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(run())


def test_stage_drops_payload_after_abort_during_relay_read() -> None:
    async def run() -> None:
        control_plane = FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = AbortOnReadRelay(lambda: stage.on_abort("req"))
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage.on_data_ready(await make_relay_payload(relay, payload))

        assert scheduler.inbox.empty()

    asyncio.run(run())


class RecordingLocalDispatcher(LocalStageDispatcher):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_batches: list[tuple[str, list[tuple[str, int]]]] = []

    async def send_stream_chunks(
        self, *, from_stage: str, to_stage: str, chunks: list[LocalStreamChunk]
    ) -> None:
        self.chunk_batches.append(
            (to_stage, [(chunk.request_id, chunk.chunk_id) for chunk in chunks])
        )
        await super().send_stream_chunks(
            from_stage=from_stage, to_stage=to_stage, chunks=chunks
        )


def make_local_stream_pair(
    *, targets: tuple[str, ...] = ("vocoder",), accept_before_payload: bool = True
) -> tuple[RecordingLocalDispatcher, Stage, dict[str, Stage]]:
    dispatcher = RecordingLocalDispatcher()
    sender = make_stage(
        name="tts_engine",
        endpoints={target: f"inproc://{target}" for target in targets},
        stream_targets=list(targets),
        same_process_targets=set(targets),
        local_dispatcher=dispatcher,
    )
    receivers = {
        target: make_stage(
            name=target, can_accept_stream_before_payload=accept_before_payload
        )
        for target in targets
    }
    for receiver in receivers.values():
        receiver.stream_queue = StreamQueue()
    dispatcher.register_many([sender, *receivers.values()])
    return dispatcher, sender, receivers


def frame_message(
    request_id: str, frame: int, target: str = "vocoder"
) -> OutgoingMessage:
    return OutgoingMessage(
        request_id=request_id, type="stream", data=frame, target=target
    )


def test_same_process_stream_run_reaches_the_target_in_one_dispatch() -> None:
    async def run() -> None:
        dispatcher, sender, receivers = make_local_stream_pair()
        sender.active_requests.update({"req-a", "req-b"})
        put_messages(
            sender.scheduler.outbox,
            [
                frame_message("req-a", 0),
                frame_message("req-b", 0),
                frame_message("req-a", 1),
                frame_message("req-b", 1),
                OutgoingMessage(request_id="req-a", type="result", data={"ok": 1}),
            ],
        )

        sender.end_outbox_drain()
        await sender.drain_outbox_external()

        assert dispatcher.chunk_batches == [
            ("vocoder", [("req-a", 0), ("req-b", 0), ("req-a", 1), ("req-b", 1)])
        ]
        inbox = receivers["vocoder"].scheduler.inbox
        delivered = [inbox.get_nowait() for _ in range(5)]
        assert inbox.empty()
        assert [(message.request_id, message.type) for message in delivered] == [
            ("req-a", "stream_chunk"),
            ("req-b", "stream_chunk"),
            ("req-a", "stream_chunk"),
            ("req-b", "stream_chunk"),
            ("req-a", "stream_done"),
        ]
        assert [message.data.data for message in delivered[:4]] == [0, 0, 1, 1]
        assert {message.data.from_stage for message in delivered[:4]} == {"tts_engine"}

    asyncio.run(run())


def test_stream_run_drops_inactive_and_receiver_aborted_requests() -> None:
    async def run() -> None:
        dispatcher, sender, receivers = make_local_stream_pair()
        sender.active_requests.update({"req-a", "req-b"})
        receivers["vocoder"].aborted.add("req-b")
        put_messages(
            sender.scheduler.outbox,
            [
                frame_message("req-stale", 0),
                frame_message("req-a", 0),
                frame_message("req-b", 0),
                frame_message("req-a", 1),
            ],
        )

        sender.end_outbox_drain()
        await sender.drain_outbox_external()

        assert dispatcher.chunk_batches == [
            ("vocoder", [("req-a", 0), ("req-b", 0), ("req-a", 1)])
        ]
        assert ("req-stale", "vocoder") not in sender.stream_chunk_counters
        inbox = receivers["vocoder"].scheduler.inbox
        delivered = [inbox.get_nowait() for _ in range(2)]
        assert inbox.empty()
        assert [(message.request_id, message.data.data) for message in delivered] == [
            ("req-a", 0),
            ("req-a", 1),
        ]

    asyncio.run(run())


def test_stream_run_rejects_only_the_request_without_its_payload() -> None:
    async def run() -> None:
        _, sender, receivers = make_local_stream_pair(accept_before_payload=False)
        receiver = receivers["vocoder"]
        receiver.stream_queue.open("req-a")
        sender.active_requests.update({"req-a", "req-b"})
        put_messages(
            sender.scheduler.outbox,
            [
                frame_message("req-a", 0),
                frame_message("req-b", 0),
                frame_message("req-a", 1),
            ],
        )

        sender.end_outbox_drain()
        await sender.drain_outbox_external()

        inbox = receiver.scheduler.inbox
        delivered = [inbox.get_nowait() for _ in range(2)]
        assert inbox.empty()
        assert [(message.request_id, message.data.data) for message in delivered] == [
            ("req-a", 0),
            ("req-a", 1),
        ]
        assert [
            (completion.request_id, completion.success)
            for completion in receiver.control_plane.completions
        ] == [("req-b", False)]
        assert receiver.scheduler.aborted == ["req-b"]

    asyncio.run(run())


def test_stream_run_dispatches_the_same_process_run_before_an_awaited_remote_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )

    async def run() -> None:
        sends: list[str] = []

        class OrderedDispatcher(LocalStageDispatcher):
            async def send_stream_chunks(
                self, *, from_stage: str, to_stage: str, chunks: list[LocalStreamChunk]
            ) -> None:
                sends.append(to_stage)
                await super().send_stream_chunks(
                    from_stage=from_stage, to_stage=to_stage, chunks=chunks
                )

        class OrderedControlPlane(FakeControlPlane):
            async def send_to_stage(self, target, endpoint, msg) -> None:
                sends.append(target)
                await super().send_to_stage(target, endpoint, msg)

        dispatcher = OrderedDispatcher()
        relay = FakeRelay()
        sender = make_stage(
            name="talker_ar",
            endpoints={"code2wav": "inproc://code2wav", "monitor": "inproc://monitor"},
            control_plane=OrderedControlPlane(),
            relay=relay,
            same_process_targets={"code2wav"},
            local_dispatcher=dispatcher,
        )
        receiver = make_stage(name="code2wav", can_accept_stream_before_payload=True)
        receiver.stream_queue = StreamQueue()
        dispatcher.register_many([sender, receiver])
        sender.active_requests.add("req-a")
        hidden = torch.zeros(
            BATCHED_STREAM_CHUNK_BYTES_LIMIT // 4 + 1, dtype=torch.float32
        )
        put_messages(
            sender.scheduler.outbox,
            [
                frame_message("req-a", 0, "code2wav"),
                OutgoingMessage("req-a", "stream", hidden, "monitor"),
            ],
        )

        sender.end_outbox_drain()
        await sender.drain_outbox_external()

        assert sends == ["code2wav", "monitor"]
        assert len(relay.puts) == 1
        assert receiver.scheduler.inbox.qsize() == 1

    asyncio.run(run())


def test_stream_run_keeps_the_order_across_alternating_targets() -> None:
    async def run() -> None:
        dispatcher, sender, receivers = make_local_stream_pair(
            targets=("vocoder", "monitor")
        )
        sender.active_requests.add("req-a")
        put_messages(
            sender.scheduler.outbox,
            [
                frame_message("req-a", 0, "vocoder"),
                frame_message("req-a", 0, "monitor"),
                frame_message("req-a", 1, "vocoder"),
            ],
        )

        sender.end_outbox_drain()
        await sender.drain_outbox_external()

        assert dispatcher.chunk_batches == [
            ("vocoder", [("req-a", 0)]),
            ("monitor", [("req-a", 0)]),
            ("vocoder", [("req-a", 1)]),
        ]
        assert receivers["monitor"].scheduler.inbox.qsize() == 1
        assert receivers["vocoder"].scheduler.inbox.qsize() == 2

    asyncio.run(run())
