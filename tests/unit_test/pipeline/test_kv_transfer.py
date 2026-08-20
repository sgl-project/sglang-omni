# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch

from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.kv_transfer import KVBufferRegion, KVPageDestination, KVPool
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline.control_plane import (
    deserialize_message,
    send_to_endpoint,
    serialize_message,
)
from sglang_omni.proto import (
    DataReadyMessage,
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    KVTransferReadyMessage,
)
from tests.unit_test.fixtures.pipeline_fakes import FakeOp, FakeRelay


@pytest.fixture(autouse=True)
def _cuda_ipc_capable_platform(monkeypatch):
    """Paged KV transfer is cuda_ipc-only, so pin the transport policy that provides
    it; a platform without cuda_ipc rejects these edges instead.
    """
    import sglang_omni.platforms as platforms
    from sglang_omni.comm.data_ref import TransportKind

    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
        raising=False,
    )


class _PagedRelay(FakeRelay):
    def __init__(self, *, tp_rank: int = 0) -> None:
        super().__init__()
        self.tp_rank = tp_rank
        self.put_ops: list[FakeOp] = []
        self.get_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        self.received_source_tp_ranks: list[int] = []

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
                "source_tp_rank": self.tp_rank,
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
        self.received_source_tp_ranks.append(metadata["source_tp_rank"])
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


class _FailingReceiver(_Receiver):
    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        del request
        raise RuntimeError("rank-local reserve failed")


class _BlockingOp(FakeOp):
    def __init__(self, request_id: str, started: asyncio.Queue[str]) -> None:
        super().__init__({"transfer_info": {"size": 0}, "key": "blocking-get"})
        self.request_id = request_id
        self.started = started

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        del timeout
        self.started.put_nowait(self.request_id)
        await asyncio.Future()


class _BlockingPagedRelay(_PagedRelay):
    def __init__(self) -> None:
        super().__init__()
        self.copy_started: asyncio.Queue[str] = asyncio.Queue()

    async def get_kv_pages(
        self,
        metadata: dict[str, Any],
        *,
        destination_pool_id: str,
        source_page_indices: tuple[int, ...],
        destination_page_indices: tuple[int, ...],
        request_id: str,
    ) -> FakeOp:
        del metadata
        self.get_calls.append(
            (destination_pool_id, source_page_indices, destination_page_indices)
        )
        return _BlockingOp(request_id, self.copy_started)


def _pool(pool_id: str, *, buffer_name: str = "layer.0.kv") -> KVPool:
    tensor = torch.zeros((6, 4), dtype=torch.uint8)
    return KVPool(
        pool_id=pool_id,
        layout_id="NHD",
        page_size=1,
        buffers=(KVBufferRegion(buffer_name, tensor, bytes_per_page=4),),
    )


def _kv_endpoints(tp_size: int) -> dict[str, tuple[str, ...]]:
    namespace = uuid4().hex
    return {
        stage_name: tuple(
            f"inproc://{namespace}-{stage_name}-rank{tp_rank}"
            for tp_rank in range(tp_size)
        )
        for stage_name in ("source", "destination")
    }


def _engine(
    stage_name: str,
    relay: _PagedRelay,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    rank_endpoints: dict[str, tuple[str, ...]] | None = None,
) -> CommEngine:
    return CommEngine(
        CommRouter(
            stage_name=stage_name,
            gpu_id=0,
            same_process_targets=set(),
            gpu_stage_names={"source", "destination"},
            injected_relay=relay,
            comm_config={"ack_timeout_s": 1.0},
        ),
        tp_rank=tp_rank,
        tp_size=tp_size,
        rank_endpoints=rank_endpoints,
    )


async def _start_pair(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    relay: _PagedRelay | None = None,
    endpoints: dict[str, tuple[str, ...]] | None = None,
) -> tuple[_PagedRelay, CommEngine, CommEngine]:
    relay = relay or _PagedRelay(tp_rank=tp_rank)
    endpoints = endpoints or _kv_endpoints(tp_size)
    source = _engine(
        "source",
        relay,
        tp_rank=tp_rank,
        tp_size=tp_size,
        rank_endpoints=endpoints,
    )
    destination = _engine(
        "destination",
        relay,
        tp_rank=tp_rank,
        tp_size=tp_size,
        rank_endpoints=endpoints,
    )
    await source.start()
    await destination.start()
    return relay, source, destination


def test_kv_control_messages_round_trip_without_rank_envelope() -> None:
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
        encoded = message.to_dict()
        assert "tp_rank" not in encoded
        assert "tp_size" not in encoded
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
                request_id="request",
                source_pool_id="source_pool",
                source_page_indices=(0,),
                target_pool_id="destination_pool",
                to_stage="destination",
                lease=lease,
            )
        lease.release.assert_called_once_with()

    asyncio.run(_run())


def test_kv_transfer_requires_matching_tp_endpoint_counts() -> None:
    async def _run() -> None:
        relay = _PagedRelay()
        endpoints = _kv_endpoints(2)
        endpoints["destination"] = tuple(
            f"inproc://destination-rank{rank}-{uuid4().hex}" for rank in range(4)
        )
        source = _engine(
            "source",
            relay,
            tp_size=2,
            rank_endpoints=endpoints,
        )
        source.register_kv_pool(_pool("source_pool"))

        with pytest.raises(
            NotImplementedError,
            match="source has tp_size=2, destination has tp_size=4",
        ):
            await source.send_kv_pages(
                request_id="request",
                source_pool_id="source_pool",
                source_page_indices=(0,),
                target_pool_id="destination_pool",
                to_stage="destination",
            )

    asyncio.run(_run())


def test_kv_transfer_uses_rank_endpoint_for_full_lifecycle() -> None:
    async def _run() -> None:
        relay, source, destination = await _start_pair()
        try:
            source.register_kv_pool(_pool("source_pool"))
            destination.register_kv_pool(_pool("destination_pool"))
            receiver = _Receiver((0, 3))
            destination.register_kv_receiver("destination_pool", receiver)
            lease = Mock()

            data_ref = await source.send_kv_pages(
                request_id="request",
                transfer_id="transfer",
                source_pool_id="source_pool",
                source_page_indices=(1, 4),
                target_pool_id="destination_pool",
                to_stage="destination",
                lease=lease,
            )

            assert data_ref.object_id == "transfer"
            assert relay.get_calls == [("destination_pool", (1, 4), (0, 3))]
            assert receiver.committed == ["request"]
            assert receiver.aborted == []
            lease.release.assert_called_once_with()
            assert relay.put_ops[0].waited
            assert ("op_ack", "kv-put") in relay.log.events
        finally:
            await source.close()
            await destination.close()

    asyncio.run(_run())


def test_equal_tp_transfer_copies_each_shard_over_its_peer_endpoint() -> None:
    async def _run() -> None:
        endpoints = _kv_endpoints(2)
        rank_state: list[
            tuple[_PagedRelay, CommEngine, CommEngine, _Receiver, Mock]
        ] = []

        try:
            for tp_rank in range(2):
                relay, source, destination = await _start_pair(
                    tp_rank=tp_rank,
                    tp_size=2,
                    endpoints=endpoints,
                )
                source.register_kv_pool(_pool("source_pool"))
                destination.register_kv_pool(_pool("destination_pool"))
                receiver = _Receiver((tp_rank, tp_rank + 2))
                destination.register_kv_receiver("destination_pool", receiver)
                lease = Mock()
                rank_state.append((relay, source, destination, receiver, lease))

            await asyncio.gather(
                *(
                    source.send_kv_pages(
                        request_id="request",
                        transfer_id="transfer",
                        source_pool_id="source_pool",
                        source_page_indices=(1, 4),
                        target_pool_id="destination_pool",
                        to_stage="destination",
                        lease=lease,
                    )
                    for _, source, _, _, lease in rank_state
                )
            )

            for tp_rank, (relay, source, destination, receiver, lease) in enumerate(
                rank_state
            ):
                assert source.rank_endpoints["destination"][source.tp_rank] == (
                    endpoints["destination"][tp_rank]
                )
                assert destination.rank_endpoints["source"][destination.tp_rank] == (
                    endpoints["source"][tp_rank]
                )
                assert relay.get_calls == [
                    ("destination_pool", (1, 4), (tp_rank, tp_rank + 2))
                ]
                assert relay.received_source_tp_ranks == [tp_rank]
                assert receiver.committed == ["request"]
                lease.release.assert_called_once_with()
        finally:
            for _, source, destination, _, _ in rank_state:
                await source.close()
                await destination.close()

    asyncio.run(_run())


def test_rank_local_prepare_failure_returns_to_same_source_rank() -> None:
    async def _run() -> None:
        _, source, destination = await _start_pair(tp_rank=1, tp_size=2)
        try:
            source.register_kv_pool(_pool("source_pool"))
            destination.register_kv_pool(_pool("destination_pool"))
            destination.register_kv_receiver("destination_pool", _FailingReceiver(()))
            lease = Mock()

            with pytest.raises(RuntimeError, match="rank-local reserve failed"):
                await source.send_kv_pages(
                    request_id="request",
                    transfer_id="transfer",
                    source_pool_id="source_pool",
                    source_page_indices=(1, 4),
                    target_pool_id="destination_pool",
                    to_stage="destination",
                    lease=lease,
                )

            lease.release.assert_called_once_with()
        finally:
            await source.close()
            await destination.close()

    asyncio.run(_run())


def test_kv_transfer_rejects_layout_mismatch() -> None:
    async def _run() -> None:
        relay, source, destination = await _start_pair()
        try:
            source.register_kv_pool(_pool("source_pool"))
            destination.register_kv_pool(
                _pool("destination_pool", buffer_name="different")
            )
            receiver = _Receiver((0,))
            destination.register_kv_receiver("destination_pool", receiver)
            lease = Mock()

            with pytest.raises(RuntimeError, match="layouts do not match"):
                await source.send_kv_pages(
                    request_id="request",
                    transfer_id="transfer",
                    source_pool_id="source_pool",
                    source_page_indices=(1,),
                    target_pool_id="destination_pool",
                    to_stage="destination",
                    lease=lease,
                )

            assert receiver.aborted == ["request"]
            lease.release.assert_called_once_with()
            assert not relay.put_ops
        finally:
            await source.close()
            await destination.close()

    asyncio.run(_run())


def test_kv_ack_timeout_retains_pending_sender_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        relay, source, destination = await _start_pair()

        async def drop_data_ready(
            sockets: dict[str, Any], target_endpoint: str, message: Any
        ) -> None:
            if isinstance(message, DataReadyMessage):
                return
            await send_to_endpoint(sockets, target_endpoint, message)

        monkeypatch.setattr(
            "sglang_omni.comm.engine.send_to_endpoint",
            drop_data_ready,
        )
        source._ack_timeout_s = 0.1
        source.register_kv_pool(_pool("source_pool"))
        destination.register_kv_pool(_pool("destination_pool"))
        destination.register_kv_receiver("destination_pool", _Receiver((0,)))
        lease = Mock()

        try:
            with pytest.raises(TimeoutError):
                await source.send_kv_pages(
                    request_id="request",
                    transfer_id="transfer",
                    source_pool_id="source_pool",
                    source_page_indices=(1,),
                    target_pool_id="destination_pool",
                    to_stage="destination",
                    lease=lease,
                )

            lease.release.assert_not_called()
            assert not relay.put_ops[0].waited
            assert relay.put_ops[0].failed is None
            assert "transfer" not in source._pending
            assert len(source._retained_pending_kv_transfers) == 1
        finally:
            await source.close()
            await destination.close()

    asyncio.run(_run())


def test_rank_endpoint_dispatches_concurrent_kv_copies_and_aborts_once() -> None:
    async def _run() -> None:
        relay = _BlockingPagedRelay()
        _, source, destination = await _start_pair(relay=relay)
        source.register_kv_pool(_pool("source_pool"))
        destination.register_kv_pool(_pool("destination_pool"))
        receiver = _Receiver((0,))
        destination.register_kv_receiver("destination_pool", receiver)

        def start_transfer(index: int) -> asyncio.Task:
            return asyncio.create_task(
                source.send_kv_pages(
                    request_id=f"request-{index}",
                    transfer_id=f"transfer-{index}",
                    source_pool_id="source_pool",
                    source_page_indices=(index,),
                    target_pool_id="destination_pool",
                    to_stage="destination",
                    lease=Mock(),
                )
            )

        transfers: list[asyncio.Task] = []
        try:
            transfers.append(start_transfer(1))
            assert await asyncio.wait_for(relay.copy_started.get(), 5.0) == "request-1"

            transfers.append(start_transfer(2))
            assert await asyncio.wait_for(relay.copy_started.get(), 5.0) == "request-2"

            await destination.close()
            assert sorted(receiver.aborted) == ["request-1", "request-2"]
        finally:
            await destination.close()
            await source.close()
            await asyncio.gather(*transfers, return_exceptions=True)

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
