# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.kv_transfer import KVBufferRegion, KVPageDestination, KVPool
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import (
    DataAckMessage,
    DataReadyMessage,
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    KVTransferReadyMessage,
)
from tests.unit_test.fixtures.pipeline_fakes import FakeOp, FakeRelay
from tests.unit_test.pipeline.helpers import make_stage


class _PagedRelay(FakeRelay):
    def __init__(self) -> None:
        super().__init__()
        self.put_ops: list[FakeOp] = []
        self.get_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def register_kv_pool(self, pool: KVPool) -> None:
        del pool

    def prepare_kv_destination(self, pool_id: str) -> dict[str, Any]:
        return {"fake_kv": {"pool_id": pool_id}}

    async def put_kv_pages(
        self,
        *,
        source_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_ref: dict[str, Any],
    ) -> FakeOp:
        del source_pool_id, destination_ref
        op = FakeOp(
            {
                "transfer_info": {"size": len(source_page_indices)},
                "fake_kv": True,
                "key": "kv-put",
            },
            self.log,
        )
        self.put_ops.append(op)
        return op

    async def get_kv_pages(
        self,
        metadata: dict[str, Any],
        *,
        destination_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_page_indices: tuple[int, ...],
        request_id: str,
    ) -> FakeOp:
        assert metadata["fake_kv"] is True
        self.get_calls.append(
            (destination_pool_id, source_page_indices, destination_page_indices)
        )
        return FakeOp(
            {"transfer_info": {"size": 0}, "key": f"{request_id}:get"},
            self.log,
        )

    async def put_async(self, *args: Any, **kwargs: Any) -> FakeOp:
        del args, kwargs
        raise AssertionError("paged KV transfer must not use staging put_async")


class _Receiver:
    def __init__(self, page_indices: tuple[int, ...]) -> None:
        self.page_indices = page_indices
        self.committed: list[str] = []
        self.aborted: list[str] = []

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        return KVPageDestination(request.target_pool_id, self.page_indices)

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        del destination
        self.committed.append(request.request_id)

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        del destination, error
        self.aborted.append(request.request_id)


class _LinkedControlPlane:
    def __init__(self) -> None:
        self.source: CommEngine | None = None
        self.destination: Any = None
        self.messages: list[Any] = []

    def connect(self, source: CommEngine, destination: Any) -> None:
        self.source = source
        self.destination = destination

    async def send_to_stage(
        self,
        next_stage: str,
        next_stage_endpoint: str,
        message: Any,
    ) -> None:
        del next_stage, next_stage_endpoint
        self.messages.append(message)
        assert self.source is not None and self.destination is not None
        if isinstance(message, KVTransferPrepareMessage):
            await self.destination._on_kv_transfer_prepare(message)
        elif isinstance(message, KVTransferReadyMessage):
            self.source.kv_transfer_ready(message)
        elif isinstance(message, DataReadyMessage):
            await self.destination._on_data_ready(message)
        elif isinstance(message, DataAckMessage):
            self.source.ack_transfer(message)
        else:
            raise AssertionError(f"unexpected message {type(message).__name__}")


class _DropDataReadyControlPlane(_LinkedControlPlane):
    async def send_to_stage(
        self,
        next_stage: str,
        next_stage_endpoint: str,
        message: Any,
    ) -> None:
        if isinstance(message, DataReadyMessage):
            self.messages.append(message)
            return
        await super().send_to_stage(next_stage, next_stage_endpoint, message)


def _engine(stage_name: str, relay: _PagedRelay) -> CommEngine:
    return CommEngine(
        CommRouter(
            stage_name=stage_name,
            gpu_id=0,
            same_process_targets=set(),
            gpu_stage_names={"source", "destination"},
            injected_relay=relay,
            comm_config={"ack_timeout_s": 1.0},
        )
    )


def _linked_pair(
    control_plane: _LinkedControlPlane | None = None,
) -> tuple[_PagedRelay, CommEngine, Any, _LinkedControlPlane]:
    relay = _PagedRelay()
    control_plane = control_plane or _LinkedControlPlane()
    source = _engine("source", relay)
    destination = make_stage(
        name="destination",
        endpoints={"source": "unused"},
        gpu_id=0,
        gpu_stage_names={"source", "destination"},
        relay=relay,
        control_plane=control_plane,
    )
    control_plane.connect(source, destination)
    return relay, source, destination, control_plane


def _pool(pool_id: str, *, buffer_name: str = "layer.0.kv") -> KVPool:
    tensor = torch.zeros((6, 4), dtype=torch.uint8)
    return KVPool(
        pool_id=pool_id,
        layout_id="NHD",
        page_size=1,
        buffers=(KVBufferRegion(buffer_name, tensor, bytes_per_page=4),),
    )


def test_kv_control_messages_round_trip() -> None:
    layout = KVPoolLayout(
        layout_id="NHD",
        page_size=1,
        buffers=(KVBufferSpec("layer.0.kv", bytes_per_page=8),),
    )
    messages = (
        KVTransferPrepareMessage(
            request_id="request",
            transfer_id="transfer",
            from_stage="prefill",
            to_stage="decode",
            source_pool_id="source",
            target_pool_id="destination",
            source_page_indices=(1, 4),
            source_layout=layout,
            metadata={"sequence_length": 2},
        ),
        KVTransferReadyMessage(
            request_id="request",
            transfer_id="transfer",
            from_stage="decode",
            to_stage="prefill",
            success=True,
            destination_pool_id="destination",
            destination_page_indices=(3, 5),
            destination_ref={
                "transport": "shm",
                "info": {"transfer_info": {"size": 16}},
                "length": 16,
            },
        ),
    )

    for message in messages:
        assert deserialize_message(serialize_message(message)) == message


def test_kv_transfer_requires_cuda_ipc_topology() -> None:
    async def _run() -> None:
        relay = _PagedRelay()
        source = CommEngine(
            CommRouter(
                stage_name="source",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                injected_relay=relay,
            )
        )
        source.register_kv_pool(_pool("source_pool"))
        lease = Mock()

        with pytest.raises(NotImplementedError, match="only cuda_ipc"):
            await source.send_kv_pages(
                control_plane=_LinkedControlPlane(),
                request_id="request",
                source_pool_id="source_pool",
                source_page_indices=(0,),
                target_pool_id="destination_pool",
                from_stage="source",
                to_stage="destination",
                target_endpoint="unused",
                lease=lease,
            )
        lease.release.assert_called_once_with()

    asyncio.run(_run())


def test_kv_transfer_uses_common_data_ready_lifecycle() -> None:
    async def _run() -> None:
        relay, source, destination, control_plane = _linked_pair()
        source.register_kv_pool(_pool("source_pool"))
        destination._comm.register_kv_pool(_pool("destination_pool"))
        receiver = _Receiver((0, 3))
        destination._comm.register_kv_receiver("destination_pool", receiver)
        lease = Mock()

        data_ref = await source.send_kv_pages(
            control_plane=control_plane,
            request_id="request",
            transfer_id="transfer",
            source_pool_id="source_pool",
            source_page_indices=(1, 4),
            target_pool_id="destination_pool",
            from_stage="source",
            to_stage="destination",
            target_endpoint="unused",
            lease=lease,
        )

        assert data_ref.object_id == "transfer"
        assert relay.get_calls == [("destination_pool", (1, 4), (0, 3))]
        assert receiver.committed == ["request"]
        assert receiver.aborted == []
        assert destination.scheduler.inbox.empty()
        lease.release.assert_called_once_with()
        assert relay.put_ops[0].waited
        assert ("op_ack", "kv-put") in relay.log.events
        assert [type(message) for message in control_plane.messages] == [
            KVTransferPrepareMessage,
            KVTransferReadyMessage,
            DataReadyMessage,
            DataAckMessage,
        ]

    asyncio.run(_run())


def test_kv_transfer_rejects_layout_mismatch() -> None:
    async def _run() -> None:
        relay, source, destination, control_plane = _linked_pair()
        source.register_kv_pool(_pool("source_pool"))
        destination._comm.register_kv_pool(
            _pool("destination_pool", buffer_name="different")
        )
        receiver = _Receiver((0,))
        destination._comm.register_kv_receiver("destination_pool", receiver)
        lease = Mock()

        with pytest.raises(RuntimeError, match="layouts do not match"):
            await source.send_kv_pages(
                control_plane=control_plane,
                request_id="request",
                transfer_id="transfer",
                source_pool_id="source_pool",
                source_page_indices=(1,),
                target_pool_id="destination_pool",
                from_stage="source",
                to_stage="destination",
                target_endpoint="unused",
                lease=lease,
            )

        assert receiver.aborted == ["request"]
        lease.release.assert_called_once_with()
        assert not relay.put_ops
        assert [type(message) for message in control_plane.messages] == [
            KVTransferPrepareMessage,
            KVTransferReadyMessage,
        ]

    asyncio.run(_run())


def test_kv_ack_timeout_retains_pending_sender_resources() -> None:
    async def _run() -> None:
        relay, source, destination, control_plane = _linked_pair(
            _DropDataReadyControlPlane()
        )
        source._ack_timeout_s = 0.0
        source.register_kv_pool(_pool("source_pool"))
        destination._comm.register_kv_pool(_pool("destination_pool"))
        destination._comm.register_kv_receiver("destination_pool", _Receiver((0,)))
        lease = Mock()

        with pytest.raises(TimeoutError):
            await source.send_kv_pages(
                control_plane=control_plane,
                request_id="request",
                transfer_id="transfer",
                source_pool_id="source_pool",
                source_page_indices=(1,),
                target_pool_id="destination_pool",
                from_stage="source",
                to_stage="destination",
                target_endpoint="unused",
                lease=lease,
            )

        lease.release.assert_not_called()
        assert not relay.put_ops[0].waited
        assert relay.put_ops[0].failed is None
        assert "transfer" not in source._pending
        assert len(source._retained_pending_kv_transfers) == 1

    asyncio.run(_run())


def test_kv_cleanup_aborts_reserved_destination() -> None:
    relay = _PagedRelay()
    destination = _engine("destination", relay)
    destination_pool = _pool("destination_pool")
    destination.register_kv_pool(destination_pool)
    receiver = _Receiver((2,))
    destination.register_kv_receiver("destination_pool", receiver)
    prepare = KVTransferPrepareMessage(
        request_id="request",
        transfer_id="transfer",
        from_stage="source",
        to_stage="destination",
        source_pool_id="source_pool",
        target_pool_id="destination_pool",
        source_page_indices=(0,),
        source_layout=destination_pool.layout,
    )

    assert destination.prepare_kv_receive(prepare).success
    destination.cleanup("request")

    assert receiver.aborted == ["request"]
