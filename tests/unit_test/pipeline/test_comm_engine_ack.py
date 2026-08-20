# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.proto import DataAckMessage, DataReadyMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingStageControlPlane,
    make_stage_payload,
)


class _AckedOp:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
        self.acked = False
        self.waited = False
        self.failed: BaseException | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def mark_receiver_done(self) -> None:
        self.acked = True

    def mark_receiver_failed(self, exc: BaseException) -> None:
        self.failed = exc

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        del timeout
        self.waited = True
        if self.failed is not None:
            raise self.failed
        if not self.acked:
            raise RuntimeError("waited before receiver ack")


class _AckedRelay:
    device = "cpu"

    def __init__(self) -> None:
        self.storage: dict[str, torch.Tensor] = {}
        self.ops: list[_AckedOp] = []
        self.receiver_ids: list[str | None] = []

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> _AckedOp:
        del dst_rank
        self.receiver_ids.append(receiver_id)
        key = str(request_id)
        self.storage[key] = tensor.detach().clone()
        op = _AckedOp({"transfer_info": {"size": int(tensor.numel())}, "key": key})
        self.ops.append(op)
        return op

    async def get_async(
        self,
        metadata: dict[str, Any],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> _AckedOp:
        key = str(metadata.get("key", request_id))
        stored = self.storage[key]
        dest_tensor.reshape(-1)[: stored.numel()].copy_(stored.reshape(-1))
        return _AckedOp(metadata)

    def cleanup(self, request_id: str) -> None:
        del request_id

    def close(self) -> None:
        pass


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition was not met")
        await asyncio.sleep(0)


def _consume_task_exception(task: asyncio.Task, _label: str) -> None:
    if not task.cancelled():
        task.exception()


def _stream_ack(
    *,
    request_id: str,
    object_id: str,
) -> DataAckMessage:
    return DataAckMessage(
        request_id=request_id,
        from_stage="consumer",
        to_stage="producer",
        object_id=object_id,
    )


async def _send_stream_for_ack_test(
    engine: CommEngine,
    relay: _AckedRelay,
    control_plane: RecordingStageControlPlane,
    *,
    request_id: str,
    metadata: dict[str, Any] | None = None,
) -> DataRef:
    await engine.send_stream_chunk(
        relay=relay,
        control_plane=control_plane,
        request_id=request_id,
        data=torch.ones(2),
        target_stage="consumer",
        target_endpoint="inproc://consumer",
        from_stage="producer",
        chunk_id=0,
        metadata=metadata,
        transport=TransportKind.SHM,
    )
    return DataRef.from_dict(control_plane.sent_to_stage[-1][2].data_ref)


def test_comm_engine_releases_sender_op_after_data_ack() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})

        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )

        op = relay.ops[0]
        assert not op.waited
        target, _, msg = control_plane.sent_to_stage[0]
        assert target == "receiver"
        assert relay.receiver_ids == ["receiver"]
        assert DataRef.from_dict(msg.data_ref).object_id == data_ref.object_id

        engine.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="receiver",
                to_stage="sender",
                object_id=data_ref.object_id,
            )
        )
        await _wait_until(lambda: op.waited)
        assert op.acked

    asyncio.run(_run())


def test_comm_engine_stream_sends_with_reused_semantics_coexist() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="producer",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            ),
            task_done_callback=_consume_task_exception,
        )

        metadata = {"embedding": torch.ones(1)}
        data_ref_a = await _send_stream_for_ack_test(
            engine,
            relay,
            control_plane,
            request_id="req-reused",
            metadata=metadata,
        )
        data_ref_b = await _send_stream_for_ack_test(
            engine,
            relay,
            control_plane,
            request_id="req-reused",
            metadata=metadata,
        )

        assert data_ref_a.object_id != data_ref_b.object_id
        metadata_a = data_ref_a.metadata_tensors[0].ref.object_id
        metadata_b = data_ref_b.metadata_tensors[0].ref.object_id
        assert metadata_a != metadata_b
        assert metadata_a.startswith(f"{data_ref_a.object_id}:meta:0")
        assert metadata_b.startswith(f"{data_ref_b.object_id}:meta:0")
        assert set(engine._pending) == {
            data_ref_a.object_id,
            data_ref_b.object_id,
        }

        ops_a = relay.ops[:2]
        ops_b = relay.ops[2:]
        engine.ack_transfer(
            _stream_ack(request_id="req-reused", object_id=data_ref_a.object_id)
        )
        await _wait_until(
            lambda: data_ref_a.object_id not in engine._pending
            and all(op.waited for op in ops_a)
        )
        assert all(op.acked for op in ops_a)
        assert data_ref_b.object_id in engine._pending
        assert not any(op.acked or op.waited for op in ops_b)

        engine.ack_transfer(
            _stream_ack(request_id="req-reused", object_id=data_ref_b.object_id)
        )
        await _wait_until(
            lambda: data_ref_b.object_id not in engine._pending
            and all(op.waited for op in ops_b)
        )
        assert all(op.acked for op in ops_b)

    asyncio.run(_run())


def test_stream_stale_ack_does_not_complete_reused_send() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="producer",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 0.05},
                injected_relay=relay,
            ),
            task_done_callback=_consume_task_exception,
        )

        data_ref_a = await _send_stream_for_ack_test(
            engine,
            relay,
            control_plane,
            request_id="req-reused",
        )
        pending_a = engine._pending[data_ref_a.object_id]
        assert pending_a.task is not None

        await _wait_until(
            lambda: data_ref_a.object_id not in engine._pending
            and pending_a.task.done(),
            timeout=5.0,
        )
        with pytest.raises(asyncio.TimeoutError):
            await pending_a.task
        assert relay.ops[0].failed is not None

        engine._ack_timeout_s = 1.0
        data_ref_b = await _send_stream_for_ack_test(
            engine,
            relay,
            control_plane,
            request_id="req-reused",
        )
        assert data_ref_b.object_id in engine._pending
        pending_b = engine._pending[data_ref_b.object_id]
        op_b = relay.ops[1]

        engine.ack_transfer(
            _stream_ack(request_id="req-reused", object_id=data_ref_a.object_id)
        )
        assert data_ref_b.object_id in engine._pending
        assert not pending_b.ack.done()
        assert not op_b.acked
        assert not op_b.waited
        assert data_ref_a.object_id != data_ref_b.object_id

        engine.ack_transfer(
            _stream_ack(request_id="req-reused", object_id=data_ref_b.object_id)
        )
        await _wait_until(
            lambda: data_ref_b.object_id not in engine._pending and op_b.waited
        )
        assert op_b.acked

    asyncio.run(_run())


def test_comm_engine_ignores_unknown_data_ack() -> None:
    engine = CommEngine(
        CommRouter(
            stage_name="sender",
            gpu_id=None,
            same_process_targets=set(),
            gpu_stage_names=set(),
        )
    )

    engine.ack_transfer(
        DataAckMessage(
            request_id="req-1",
            from_stage="receiver",
            to_stage="sender",
            object_id="missing",
        )
    )


def test_comm_engine_ignores_duplicate_data_ack() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})
        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )
        ack = DataAckMessage(
            request_id="req-1",
            from_stage="receiver",
            to_stage="sender",
            object_id=data_ref.object_id,
        )

        engine.ack_transfer(ack)
        await _wait_until(lambda: relay.ops[0].waited)

        engine.ack_transfer(ack)

    asyncio.run(_run())


def test_data_messages_reject_missing_data_ref() -> None:
    with pytest.raises(TypeError, match="data_ref"):
        DataReadyMessage(
            request_id="req-1",
            from_stage="a",
            to_stage="b",
            data_ref=None,
        ).to_dict()

    with pytest.raises(TypeError, match="success"):
        DataAckMessage.from_dict(
            {
                "type": "data_ack",
                "request_id": "req-1",
                "from_stage": "b",
                "to_stage": "a",
                "object_id": "obj",
            }
        )

    with pytest.raises(TypeError, match="is_done"):
        DataReadyMessage.from_dict(
            {
                "type": "data_ready",
                "request_id": "req-1",
                "from_stage": "a",
                "to_stage": "b",
                "is_done": "false",
            }
        )

    with pytest.raises(TypeError, match="chunk_id"):
        DataReadyMessage.from_dict(
            {
                "type": "data_ready",
                "request_id": "req-1",
                "from_stage": "a",
                "to_stage": "b",
                "data_ref": {"version": 1},
                "chunk_id": True,
            }
        )

    with pytest.raises(ValueError, match="both done and error"):
        DataReadyMessage(
            request_id="req-1",
            from_stage="a",
            to_stage="b",
            data_ref=None,
            is_done=True,
            error="boom",
        ).to_dict()


def test_data_ref_rejects_bool_int_fields() -> None:
    data_ref = {
        "_type": "DataRef",
        "version": 1,
        "kind": "stream_chunk",
        "object_id": "obj",
        "transport": "shm",
        "layout": "raw_tensor",
        "buffer": {"transport": "shm", "info": {}, "length": 1},
        "tensors": [],
        "metadata_tensors": [],
        "shape": [1],
        "dtype": "torch.uint8",
        "offset": 0,
    }

    data_ref["version"] = True
    with pytest.raises(TypeError, match="version must be int"):
        DataRef.from_dict(data_ref)

    data_ref["version"] = 1
    data_ref["shape"] = [True]
    with pytest.raises(TypeError, match="shape must be list\\[int\\]"):
        DataRef.from_dict(data_ref)
