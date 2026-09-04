# SPDX-License-Identifier: Apache-2.0
"""Omni communication engine facade used by pipeline stages."""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable
from uuid import uuid4

import msgspec
import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import (
    BackendRef,
    DataKind,
    DataLayout,
    DataRef,
    TransportKind,
)
from sglang_omni.comm.kv_transfer import (
    KVPageDestination,
    KVPageLease,
    KVPool,
    KVReceiver,
)
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline.control_plane import PullSocket, PushSocket, send_to_endpoint
from sglang_omni.platforms import current_platform
from sglang_omni.profiler.comm_trace import elapsed_ms as _comm_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import now_ns as _comm_now_ns
from sglang_omni.proto import (
    DataAckMessage,
    DataReadyMessage,
    KVTransferPrepareMessage,
    KVTransferReadyMessage,
    StagePayload,
)
from sglang_omni.relay.base import Relay

logger = logging.getLogger(__name__)


@dataclass
class _InboundKVTransfer:
    request: KVTransferPrepareMessage
    receiver: KVReceiver
    destination: KVPageDestination
    copy_started: bool = False
    abort_error: BaseException | None = None


class _PendingTransfer(msgspec.Struct):
    ops: list[Any]
    ack: asyncio.Future[None]
    task: asyncio.Task | None = None
    lease: KVPageLease | None = None
    retain_pending_on_failure: bool = False
    receiver_terminal: bool = False


class _PayloadSendJob(msgspec.Struct, frozen=True):
    relay: Relay
    control_plane: Any
    request_id: str
    payload: StagePayload
    transport: TransportKind
    from_stage: str
    to_stage: str
    target_endpoint: str
    ready: asyncio.Future[DataRef]
    enqueued_ns: int
    replica_bindings: dict[str, int] | None = None


class _StreamSendJob(msgspec.Struct, frozen=True):
    relay: Relay
    control_plane: Any
    request_id: str
    data: torch.Tensor
    target_stage: str
    target_endpoint: str
    from_stage: str
    chunk_id: int
    metadata: dict[str, Any] | None
    transport: TransportKind
    ready: asyncio.Future[DataRef]
    enqueued_ns: int
    replica_bindings: dict[str, int] | None = None


class CommEngine:
    """Stage-owned communication engine.

    It owns locality classification and data_ref-based relay IO. Stages keep
    routing semantics; the engine owns byte movement mechanics.
    """

    def __init__(
        self,
        router: CommRouter,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        rank_endpoints: dict[str, tuple[str, ...]] | None = None,
        task_done_callback: Callable[[asyncio.Task, str], None] | None = None,
    ) -> None:
        self.router = router
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.rank_endpoints = rank_endpoints or {}
        cfg = router.comm_config
        queue_size = int(cfg["send_queue_size"]) if "send_queue_size" in cfg else 1024
        self._ack_timeout_s = (
            float(cfg["ack_timeout_s"]) if "ack_timeout_s" in cfg else 30.0
        )
        self._send_queue_size = queue_size
        self._send_queues: dict[
            str, asyncio.Queue[_PayloadSendJob | _StreamSendJob]
        ] = {}
        self._send_workers: dict[str, asyncio.Task] = {}
        self._pending: dict[str, _PendingTransfer] = {}
        self._stream_send_sequence = count()
        # Failed pending KV transfers stay pinned until this dying process exits.
        self._retained_pending_kv_transfers: list[_PendingTransfer] = []
        self._kv_pools: dict[str, KVPool] = {}
        self._kv_receivers: dict[str, KVReceiver] = {}
        self._kv_ready: dict[str, asyncio.Future[KVTransferReadyMessage]] = {}
        self._outbound_kv_requests: dict[str, str] = {}
        self._inbound_kv: dict[str, _InboundKVTransfer] = {}
        self._aborted_kv_requests: set[str] = set()
        self._rank_recv_socket: PullSocket | None = None
        self._rank_send_sockets: dict[str, PushSocket] = {}
        self._rank_control_task: asyncio.Task | None = None
        self._rank_receive_tasks: set[asyncio.Task[None]] = set()
        self._task_done_callback = task_done_callback
        self._closed = False

    async def start(self) -> None:
        """Start this process's rank-local communication endpoint."""

        if not self.rank_endpoints or self._rank_recv_socket is not None:
            return
        if self._closed:
            raise RuntimeError("comm engine is closed")
        recv_socket = PullSocket(
            self.rank_endpoints[self.router.stage_name][self.tp_rank], bind=True
        )
        await recv_socket.start()
        self._rank_recv_socket = recv_socket
        task = asyncio.create_task(self._run_rank_control(recv_socket))
        self._rank_control_task = task
        self._track_task(
            task,
            f"rank endpoint {self.router.stage_name}_rank{self.tp_rank}",
        )

    def outbound(self, target: str) -> TransportKind:
        return self.router.outbound(target)

    def outbound_stream(self, target: str, data: torch.Tensor) -> TransportKind:
        return self.router.outbound_stream(target, data)

    def relay(self, kind: TransportKind) -> Relay:
        return self.router.relay(kind)

    def inbound_relay(self, from_stage: str) -> Relay:
        return self.router.inbound_relay(from_stage)

    async def write_payload(
        self,
        *,
        relay: Relay,
        request_id: str,
        payload: StagePayload,
        transport: TransportKind,
        from_stage: str,
        to_stage: str,
    ) -> tuple[DataRef, Any]:
        return await stage_io.write_payload(
            relay,
            request_id,
            payload,
            transport=transport,
            from_stage=from_stage,
            to_stage=to_stage,
        )

    async def send_payload(
        self,
        *,
        relay: Relay,
        control_plane: Any,
        request_id: str,
        payload: StagePayload,
        transport: TransportKind,
        from_stage: str,
        to_stage: str,
        target_endpoint: str,
        replica_bindings: dict[str, int] | None = None,
    ) -> DataRef:
        if not isinstance(payload, StagePayload):
            raise TypeError(
                f"send_payload expects StagePayload, got {type(payload).__name__}"
            )
        queue = self._send_queue_for(to_stage)
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[DataRef] = loop.create_future()
        enqueue_start = _comm_now_ns()
        await queue.put(
            _PayloadSendJob(
                relay=relay,
                control_plane=control_plane,
                request_id=request_id,
                payload=payload,
                transport=transport,
                from_stage=from_stage,
                to_stage=to_stage,
                target_endpoint=target_endpoint,
                ready=ready,
                enqueued_ns=enqueue_start,
                replica_bindings=replica_bindings,
            )
        )
        _comm_trace(
            "comm_send_enqueue",
            kind="payload",
            request_id=request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            transport=transport.value,
            queue_key=to_stage,
            elapsed_ms=round(_comm_elapsed_ms(enqueue_start), 6),
        )
        return await ready

    @property
    def local_payload_device(self) -> str | None:
        """This stage's own accelerator, or None for a host-only stage."""
        if self.router.gpu_id is None:
            return None
        return f"{current_platform.device_type}:{self.router.gpu_id}"

    async def read_payload(
        self,
        *,
        relay: Relay,
        request_id: str,
        data_ref: DataRef,
    ) -> StagePayload:
        return await stage_io.read_payload(
            relay, request_id, data_ref, self.local_payload_device
        )

    async def read_data(
        self,
        *,
        relay: Relay,
        request_id: str,
        data_ref: DataRef,
    ) -> StagePayload | None:
        """Read one non-stream DataReady object.

        Payload data is returned to the Stage input handler. Paged KV data is
        installed into its pre-reserved destination and therefore has no value
        to route through the input handler.
        """

        if data_ref.kind is DataKind.STAGE_PAYLOAD:
            return await self.read_payload(
                relay=relay,
                request_id=request_id,
                data_ref=data_ref,
            )
        if data_ref.kind is DataKind.KV_PAGES:
            await self._read_kv_pages(
                relay=relay,
                request_id=request_id,
                data_ref=data_ref,
            )
            return None
        raise NotImplementedError(
            f"unsupported non-stream data kind {data_ref.kind.value!r}"
        )

    async def send_stream_chunk(
        self,
        *,
        relay: Relay,
        control_plane: Any,
        request_id: str,
        data: torch.Tensor,
        target_stage: str,
        target_endpoint: str,
        from_stage: str,
        chunk_id: int,
        metadata: dict[str, Any] | None,
        transport: TransportKind,
        replica_bindings: dict[str, int] | None = None,
    ) -> None:
        queue = self._send_queue_for(target_stage)
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[DataRef] = loop.create_future()
        enqueue_start = _comm_now_ns()
        await queue.put(
            _StreamSendJob(
                relay=relay,
                control_plane=control_plane,
                request_id=request_id,
                data=data,
                target_stage=target_stage,
                target_endpoint=target_endpoint,
                from_stage=from_stage,
                chunk_id=chunk_id,
                metadata=metadata,
                transport=transport,
                ready=ready,
                enqueued_ns=enqueue_start,
                replica_bindings=replica_bindings,
            )
        )
        _comm_trace(
            "comm_send_enqueue",
            kind="stream_chunk",
            request_id=request_id,
            from_stage=from_stage,
            to_stage=target_stage,
            chunk_id=chunk_id,
            transport=transport.value,
            queue_key=target_stage,
            elapsed_ms=round(_comm_elapsed_ms(enqueue_start), 6),
        )
        _ = await ready

    async def read_stream_chunk(
        self,
        *,
        relay: Relay,
        data_ref: DataRef,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        read_start = _comm_now_ns()
        data, metadata = await stage_io.read_stream_chunk(
            relay, data_ref, self.local_payload_device
        )
        _comm_trace(
            "comm_stream_read",
            object_id=data_ref.object_id,
            transport=data_ref.transport.value,
            bytes=data_ref.buffer.length,
            elapsed_ms=round(_comm_elapsed_ms(read_start), 6),
        )
        return data, metadata

    def register_kv_pool(self, pool: KVPool) -> None:
        self._kv_pools[pool.pool_id] = pool

    def register_kv_receiver(self, pool_id: str, receiver: KVReceiver) -> None:
        self._kv_receivers[pool_id] = receiver

    async def send_kv_pages(
        self,
        *,
        request_id: str,
        source_pool_id: str,
        source_page_indices: tuple[int, ...],
        target_pool_id: str,
        to_stage: str,
        metadata: dict[str, Any] | None = None,
        transfer_id: str | None = None,
        lease: KVPageLease | None = None,
    ) -> DataRef:
        """Transfer this rank's pages to the same rank of ``to_stage``."""
        from_stage = self.router.stage_name
        transfer_id = transfer_id or (
            f"{request_id}:kv_pages:{from_stage}:{to_stage}:{uuid4().hex}"
        )
        num_pages = len(source_page_indices)
        send_start = _comm_now_ns()
        _comm_trace(
            "comm_kv_send_start",
            request_id=request_id,
            transfer_id=transfer_id,
            from_stage=from_stage,
            to_stage=to_stage,
            source_pool_id=source_pool_id,
            target_pool_id=target_pool_id,
            num_pages=num_pages,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        try:
            pool = self._kv_pools.get(source_pool_id)
            if pool is None:
                raise KeyError(f"unknown source KV pool {source_pool_id!r}")
            pool.validate_page_indices(source_page_indices)
            if transfer_id in self._kv_ready or transfer_id in self._pending:
                raise RuntimeError(f"duplicate KV transfer {transfer_id!r}")

            transport = self.router.outbound(to_stage)
            if transport is not TransportKind.CUDA_IPC:
                raise NotImplementedError(
                    "paged KV transfer currently supports only cuda_ipc; topology "
                    f"selected {transport.value}"
                )
            target_tp_size = len(self.rank_endpoints[to_stage])
            if target_tp_size != self.tp_size:
                raise NotImplementedError(
                    "paged KV transfer requires matching tensor parallel sizes: "
                    f"{from_stage} has tp_size={self.tp_size}, "
                    f"{to_stage} has tp_size={target_tp_size}"
                )
            relay = self.relay(transport)
            relay.register_kv_pool(pool)

            ready_future = asyncio.get_running_loop().create_future()
            prepare_start = _comm_now_ns()
            self._kv_ready[transfer_id] = ready_future
            self._outbound_kv_requests[transfer_id] = request_id
            await send_to_endpoint(
                self._rank_send_sockets,
                self.rank_endpoints[to_stage][self.tp_rank],
                KVTransferPrepareMessage(
                    request_id=request_id,
                    transfer_id=transfer_id,
                    from_stage=from_stage,
                    to_stage=to_stage,
                    source_pool_id=source_pool_id,
                    target_pool_id=target_pool_id,
                    source_page_indices=source_page_indices,
                    source_layout=pool.layout,
                    metadata=dict(metadata or {}),
                ),
            )
            ready = await asyncio.wait_for(
                ready_future,
                timeout=self._ack_timeout_s,
            )
            _comm_trace(
                "comm_kv_ready",
                transfer_id=transfer_id,
                success=ready.success,
                error=ready.error,
                wait_ms=round(_comm_elapsed_ms(prepare_start), 6),
            )
            if not ready.success:
                raise RuntimeError(ready.error)
            op = await relay.put_kv_pages(
                source_pool_id=source_pool_id,
                source_page_indices=source_page_indices,
                destination_ref=ready.destination_ref,
                transfer_id=transfer_id,
            )
            data_ref = DataRef(
                version=1,
                object_id=transfer_id,
                kind=DataKind.KV_PAGES,
                transport=transport,
                layout=DataLayout.PAGED,
                buffer=BackendRef.from_relay_info(
                    transport=transport,
                    relay_info=op.metadata,
                ),
            )
            self._register_pending(
                data_ref.object_id,
                [op],
                lease=lease,
                retain_pending_on_failure=True,
            )
            # DataReady may expose the source from this point onward. Pending
            # now owns the lease even if publication or its caller is cancelled.
            lease = None
            try:
                await send_to_endpoint(
                    self._rank_send_sockets,
                    self.rank_endpoints[to_stage][self.tp_rank],
                    DataReadyMessage(
                        request_id=request_id,
                        from_stage=from_stage,
                        to_stage=to_stage,
                        data_ref=data_ref.to_dict(),
                    ),
                )
            except BaseException as exc:
                self._fail_pending(data_ref.object_id, exc)
                raise
            pending_task = self._arm_pending(data_ref.object_id)
            await asyncio.shield(pending_task)
            _comm_trace(
                "comm_kv_transfer_complete",
                transfer_id=transfer_id,
                num_pages=num_pages,
                bytes=op.metadata.get("transfer_info", {}).get("size", -1),
                elapsed_ms=round(_comm_elapsed_ms(send_start), 6),
            )
            return data_ref
        except BaseException as exc:
            _comm_trace(
                "comm_kv_transfer_failed",
                transfer_id=transfer_id,
                num_pages=num_pages,
                error=type(exc).__name__,
                detail=exc,
                elapsed_ms=round(_comm_elapsed_ms(send_start), 6),
            )
            raise
        finally:
            self._kv_ready.pop(transfer_id, None)
            self._outbound_kv_requests.pop(transfer_id, None)
            if lease is not None:
                lease.release()

    async def _run_rank_control(self, recv_socket: PullSocket) -> None:
        try:
            while not self._closed:
                message = await recv_socket.recv()
                if isinstance(message, KVTransferPrepareMessage):
                    if message.request_id in self._aborted_kv_requests:
                        ready = self._kv_ready_failure(
                            message,
                            f"request {message.request_id!r} was aborted",
                        )
                    else:
                        ready = self.prepare_kv_receive(message)
                    await send_to_endpoint(
                        self._rank_send_sockets,
                        self.rank_endpoints[message.from_stage][self.tp_rank],
                        ready,
                    )
                    continue

                if isinstance(message, KVTransferReadyMessage):
                    future = self._kv_ready.get(message.transfer_id)
                    if future is None:
                        logger.debug(
                            "Ignoring stale KV transfer ready for %s",
                            message.transfer_id,
                        )
                    elif not future.done():
                        future.set_result(message)
                    continue

                if isinstance(message, DataReadyMessage):
                    task = asyncio.create_task(self._receive_rank_kv(message))
                    self._rank_receive_tasks.add(task)
                    task.add_done_callback(self._rank_receive_tasks.discard)
                    self._track_task(
                        task,
                        f"KV receive {message.request_id}:{message.from_stage}",
                    )
                    continue

                self.ack_transfer(message)
        except asyncio.CancelledError:
            pass

    async def _receive_rank_kv(self, message: DataReadyMessage) -> None:
        data_ref = DataRef.from_dict(message.data_ref)
        error: Exception | None = None
        if message.request_id in self._aborted_kv_requests:
            error = RuntimeError(f"request {message.request_id!r} was aborted")
        else:
            try:
                await self.read_data(
                    relay=self.relay(data_ref.transport),
                    request_id=message.request_id,
                    data_ref=data_ref,
                )
            except Exception as exc:
                logger.exception(
                    "KV relay read failed for %s on %s rank %d",
                    message.request_id,
                    self.router.stage_name,
                    self.tp_rank,
                )
                error = exc
                self.cleanup(message.request_id)

        await send_to_endpoint(
            self._rank_send_sockets,
            self.rank_endpoints[message.from_stage][self.tp_rank],
            DataAckMessage(
                request_id=message.request_id,
                from_stage=self.router.stage_name,
                to_stage=message.from_stage,
                object_id=data_ref.object_id,
                success=error is None,
                error=None if error is None else (str(error) or type(error).__name__),
            ),
        )

    def prepare_kv_receive(
        self,
        message: KVTransferPrepareMessage,
    ) -> KVTransferReadyMessage:
        if message.transfer_id in self._inbound_kv:
            raise RuntimeError(f"duplicate inbound KV transfer {message.transfer_id!r}")
        transport = self.router.inbound(message.from_stage)
        if transport is not TransportKind.CUDA_IPC:
            return self._kv_ready_failure(
                message,
                "paged KV transfer currently supports only cuda_ipc; topology "
                f"selected {transport.value}",
            )
        relay = self.relay(transport)
        receiver = self._kv_receivers.get(message.target_pool_id)
        if receiver is None:
            return self._kv_ready_failure(
                message,
                f"unknown target KV pool {message.target_pool_id!r}",
            )

        destination: KVPageDestination | None = None
        try:
            destination = receiver.reserve(message)
            if len(destination.page_indices) != len(message.source_page_indices):
                raise ValueError(
                    "KV receiver must reserve one destination page per source page"
                )
            pool = self._kv_pools.get(destination.pool_id)
            if pool is None:
                raise KeyError(
                    f"destination KV pool {destination.pool_id!r} is not registered"
                )
            pool.validate_page_indices(destination.page_indices)
            if not message.source_layout.compatible_with(pool.layout):
                raise ValueError("source and destination KV pool layouts do not match")
            relay.register_kv_pool(pool)
            relay_info = relay.prepare_kv_destination(destination.pool_id)
            self._inbound_kv[message.transfer_id] = _InboundKVTransfer(
                request=message,
                receiver=receiver,
                destination=destination,
            )
            _comm_trace(
                "comm_kv_prepare_ready",
                request_id=message.request_id,
                transfer_id=message.transfer_id,
                from_stage=message.from_stage,
                to_stage=message.to_stage,
                destination_pool_id=destination.pool_id,
                num_pages=len(destination.page_indices),
            )
            return KVTransferReadyMessage(
                request_id=message.request_id,
                transfer_id=message.transfer_id,
                from_stage=message.to_stage,
                to_stage=message.from_stage,
                success=True,
                destination_pool_id=destination.pool_id,
                destination_page_indices=destination.page_indices,
                destination_ref=relay_info,
            )
        except Exception as exc:
            with suppress(Exception):
                receiver.abort(message, destination, exc)
            return self._kv_ready_failure(message, str(exc) or type(exc).__name__)

    async def _read_kv_pages(
        self,
        *,
        relay: Relay,
        request_id: str,
        data_ref: DataRef,
    ) -> None:
        if data_ref.transport is not TransportKind.CUDA_IPC:
            raise NotImplementedError(
                "paged KV transfer currently supports only cuda_ipc; data_ref "
                f"uses {data_ref.transport.value}"
            )
        state = self._inbound_kv.get(data_ref.object_id)
        if state is None:
            raise KeyError(f"unknown inbound KV transfer {data_ref.object_id!r}")

        read_start = _comm_now_ns()
        try:
            state.copy_started = True
            op = await relay.get_kv_pages(
                data_ref.buffer.info,
                destination_pool_id=state.destination.pool_id,
                source_page_indices=state.request.source_page_indices,
                destination_page_indices=state.destination.page_indices,
                request_id=request_id,
                transfer_id=data_ref.object_id,
            )
            await op.wait_for_completion(timeout=self._ack_timeout_s)
            if state.abort_error is not None:
                raise state.abort_error
            state.receiver.commit(state.request, state.destination)
            _comm_trace(
                "comm_kv_read_complete",
                transfer_id=data_ref.object_id,
                request_id=request_id,
                num_pages=len(state.destination.page_indices),
                elapsed_ms=round(_comm_elapsed_ms(read_start), 6),
            )
        except (asyncio.CancelledError, Exception) as exc:
            _comm_trace(
                "comm_kv_read_failed",
                transfer_id=data_ref.object_id,
                request_id=request_id,
                error=type(exc).__name__,
                detail=exc,
                elapsed_ms=round(_comm_elapsed_ms(read_start), 6),
            )
            with suppress(Exception):
                state.receiver.abort(state.request, state.destination, exc)
            raise
        finally:
            self._inbound_kv.pop(data_ref.object_id, None)

    @staticmethod
    def _kv_ready_failure(
        message: KVTransferPrepareMessage,
        error: str,
    ) -> KVTransferReadyMessage:
        _comm_trace(
            "comm_kv_prepare_rejected",
            request_id=message.request_id,
            transfer_id=message.transfer_id,
            from_stage=message.from_stage,
            to_stage=message.to_stage,
            target_pool_id=message.target_pool_id,
            num_pages=len(message.source_page_indices),
            error=error,
        )
        return KVTransferReadyMessage(
            request_id=message.request_id,
            transfer_id=message.transfer_id,
            from_stage=message.to_stage,
            to_stage=message.from_stage,
            success=False,
            error=error,
        )

    def cleanup(self, request_id: str) -> None:
        self._aborted_kv_requests.add(request_id)
        if len(self._aborted_kv_requests) > 10000:
            excess = len(self._aborted_kv_requests) - 5000
            for stale_request_id in list(self._aborted_kv_requests)[:excess]:
                self._aborted_kv_requests.discard(stale_request_id)
        error = RuntimeError(f"KV transfer request {request_id!r} was cleaned up")
        for transfer_id, state in list(self._inbound_kv.items()):
            if state.request.request_id != request_id:
                continue
            if state.copy_started:
                state.abort_error = error
                continue
            with suppress(Exception):
                state.receiver.abort(state.request, state.destination, error)
            self._inbound_kv.pop(transfer_id, None)
        for transfer_id, outbound_request_id in list(
            self._outbound_kv_requests.items()
        ):
            if outbound_request_id != request_id:
                continue
            future = self._kv_ready.get(transfer_id)
            if future is not None and not future.done():
                future.set_exception(error)
            self._fail_pending(transfer_id, error)
        self.router.cleanup(request_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        rank_control_task = self._rank_control_task
        self._rank_control_task = None
        if rank_control_task is not None:
            rank_control_task.cancel()
            await asyncio.gather(rank_control_task, return_exceptions=True)
        rank_receive_tasks = tuple(self._rank_receive_tasks)
        for task in rank_receive_tasks:
            task.cancel()
        await asyncio.gather(*rank_receive_tasks, return_exceptions=True)
        self._rank_receive_tasks.clear()
        if self._rank_recv_socket is not None:
            self._rank_recv_socket.close()
            self._rank_recv_socket = None
        for socket in self._rank_send_sockets.values():
            socket.close()
        self._rank_send_sockets.clear()
        for task in self._send_workers.values():
            task.cancel()
        self._send_workers.clear()
        self._send_queues.clear()
        for object_id in list(self._pending):
            self._fail_pending(object_id, RuntimeError("comm engine closed"))
        close_error = RuntimeError("comm engine closed")
        for state in self._inbound_kv.values():
            with suppress(Exception):
                state.receiver.abort(state.request, state.destination, close_error)
        self._inbound_kv.clear()
        for future in self._kv_ready.values():
            if not future.done():
                future.set_exception(close_error)
        self._kv_ready.clear()
        self._outbound_kv_requests.clear()
        self._aborted_kv_requests.clear()
        self.router.close()

    def ack_transfer(self, ack: DataAckMessage) -> None:
        if ack.to_stage != self.router.stage_name:
            raise ValueError(
                f"data_ack for {ack.to_stage!r} delivered to {self.router.stage_name!r}"
            )
        pending = self._pending.get(ack.object_id)
        if pending is None:
            logger.debug(
                "Ignoring stale data_ack for %s from %s to %s",
                ack.object_id,
                ack.from_stage,
                ack.to_stage,
            )
            return
        if ack.success:
            pending.receiver_terminal = True
            if not pending.ack.done():
                pending.ack.set_result(None)
            return
        error = ack.error
        if error is None:
            raise ValueError("failed data_ack is missing error")
        pending.receiver_terminal = True
        if not pending.ack.done():
            pending.ack.set_exception(RuntimeError(error))

    def _send_queue_for(
        self, queue_key: str
    ) -> asyncio.Queue[_PayloadSendJob | _StreamSendJob]:
        if self._closed:
            raise RuntimeError("comm engine is closed")
        queue = self._send_queues.get(queue_key)
        if queue is None:
            queue = asyncio.Queue(maxsize=self._send_queue_size)
            self._send_queues[queue_key] = queue
        task = self._send_workers.get(queue_key)
        if task is None or task.done():
            task = asyncio.create_task(self._run_send_worker(queue_key, queue))
            self._send_workers[queue_key] = task
            self._track_task(task, f"comm sender {queue_key}")
        return queue

    async def _run_send_worker(
        self,
        queue_key: str,
        queue: asyncio.Queue[_PayloadSendJob | _StreamSendJob],
    ) -> None:
        while not self._closed:
            job = await queue.get()
            try:
                if isinstance(job, _PayloadSendJob):
                    await self._run_payload_send(job, queue_key)
                else:
                    await self._run_stream_send(job, queue_key)
            finally:
                queue.task_done()
                del job

    async def _run_payload_send(self, job: _PayloadSendJob, queue_key: str) -> None:
        object_id: str | None = None
        send_start = _comm_now_ns()
        write_ms = -1.0
        control_ms = -1.0
        try:
            write_start = _comm_now_ns()
            data_ref, op = await stage_io.write_payload(
                job.relay,
                job.request_id,
                job.payload,
                transport=job.transport,
                from_stage=job.from_stage,
                to_stage=job.to_stage,
            )
            write_ms = _comm_elapsed_ms(write_start)
            object_id = data_ref.object_id
            control_start = _comm_now_ns()
            await self._publish_data_ready(
                control_plane=job.control_plane,
                request_id=job.request_id,
                from_stage=job.from_stage,
                to_stage=job.to_stage,
                target_endpoint=job.target_endpoint,
                data_ref=data_ref,
                ops=[op],
                replica_bindings=job.replica_bindings,
            )
            control_ms = _comm_elapsed_ms(control_start)
            _comm_trace(
                "comm_payload_send",
                request_id=job.request_id,
                from_stage=job.from_stage,
                to_stage=job.to_stage,
                transport=job.transport.value,
                queue_key=queue_key,
                queue_wait_ms=round((send_start - job.enqueued_ns) / 1_000_000.0, 6),
                write_ms=round(write_ms, 6),
                control_send_ms=round(control_ms, 6),
                elapsed_ms=round(_comm_elapsed_ms(send_start), 6),
            )
            job.ready.set_result(data_ref)
        except Exception as exc:
            if object_id is not None:
                self._fail_pending(object_id, exc)
            if not job.ready.done():
                job.ready.set_exception(exc)

    async def _run_stream_send(self, job: _StreamSendJob, queue_key: str) -> None:
        object_id: str | None = None
        stream_object_id = (
            f"{job.request_id}:stream:{job.from_stage}:{job.target_stage}:"
            f"{job.chunk_id}:{next(self._stream_send_sequence)}"
        )
        send_start = _comm_now_ns()
        write_ms = -1.0
        control_ms = -1.0
        try:
            write_start = _comm_now_ns()
            data_ref, ops = await stage_io.write_stream_chunk(
                job.relay,
                request_id=job.request_id,
                data=job.data,
                target_stage=job.target_stage,
                from_stage=job.from_stage,
                chunk_id=job.chunk_id,
                object_id=stream_object_id,
                metadata=job.metadata,
                transport=job.transport,
            )
            write_ms = _comm_elapsed_ms(write_start)
            object_id = data_ref.object_id
            control_start = _comm_now_ns()
            await self._publish_data_ready(
                control_plane=job.control_plane,
                request_id=job.request_id,
                from_stage=job.from_stage,
                to_stage=job.target_stage,
                target_endpoint=job.target_endpoint,
                data_ref=data_ref,
                ops=ops,
                chunk_id=job.chunk_id,
                replica_bindings=job.replica_bindings,
            )
            control_ms = _comm_elapsed_ms(control_start)
            _comm_trace(
                "comm_stream_send",
                request_id=job.request_id,
                from_stage=job.from_stage,
                to_stage=job.target_stage,
                chunk_id=job.chunk_id,
                transport=job.transport.value,
                bytes=job.data.nbytes,
                queue_key=queue_key,
                queue_wait_ms=round((send_start - job.enqueued_ns) / 1_000_000.0, 6),
                write_ms=round(write_ms, 6),
                control_send_ms=round(control_ms, 6),
                elapsed_ms=round(_comm_elapsed_ms(send_start), 6),
            )
            job.ready.set_result(data_ref)
        except Exception as exc:
            if object_id is not None:
                self._fail_pending(object_id, exc)
            if not job.ready.done():
                job.ready.set_exception(exc)

    async def _publish_data_ready(
        self,
        *,
        control_plane: Any,
        request_id: str,
        from_stage: str,
        to_stage: str,
        target_endpoint: str,
        data_ref: DataRef,
        ops: list[Any],
        chunk_id: int | None = None,
        replica_bindings: dict[str, int] | None = None,
    ) -> asyncio.Task:
        """Publish a relay object and arm its existing ACK lifecycle."""

        object_id = data_ref.object_id
        self._register_pending(object_id, ops)
        return await self._publish_registered_data_ready(
            control_plane=control_plane,
            request_id=request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            target_endpoint=target_endpoint,
            data_ref=data_ref,
            chunk_id=chunk_id,
            replica_bindings=replica_bindings,
        )

    async def _publish_registered_data_ready(
        self,
        *,
        control_plane: Any,
        request_id: str,
        from_stage: str,
        to_stage: str,
        target_endpoint: str,
        data_ref: DataRef,
        chunk_id: int | None = None,
        replica_bindings: dict[str, int] | None = None,
    ) -> asyncio.Task:
        object_id = data_ref.object_id
        try:
            await control_plane.send_to_stage(
                to_stage,
                target_endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=from_stage,
                    to_stage=to_stage,
                    data_ref=data_ref.to_dict(),
                    chunk_id=chunk_id,
                    replica_bindings=replica_bindings,
                ),
            )
        except BaseException as exc:
            self._fail_pending(object_id, exc)
            raise
        return self._arm_pending(object_id)

    def _register_pending(
        self,
        object_id: str,
        ops: list[Any],
        *,
        lease: KVPageLease | None = None,
        retain_pending_on_failure: bool = False,
    ) -> None:
        if object_id in self._pending:
            raise RuntimeError(f"duplicate pending transfer {object_id!r}")
        self._pending[object_id] = _PendingTransfer(
            ops=ops,
            ack=asyncio.get_running_loop().create_future(),
            lease=lease,
            retain_pending_on_failure=retain_pending_on_failure,
        )

    def _arm_pending(self, object_id: str) -> asyncio.Task:
        pending = self._pending[object_id]
        assert pending.task is None
        pending.task = asyncio.create_task(self._watch_pending(object_id, pending))
        self._track_task(pending.task, f"comm ack {object_id}")
        return pending.task

    async def _watch_pending(self, object_id: str, pending: _PendingTransfer) -> None:
        retained = False
        try:
            ack = (
                asyncio.shield(pending.ack)
                if pending.retain_pending_on_failure
                else pending.ack
            )
            await asyncio.wait_for(ack, timeout=self._ack_timeout_s)
            for op in pending.ops:
                op.mark_receiver_done()
            for op in pending.ops:
                await op.wait_for_completion(timeout=self._ack_timeout_s)
        except asyncio.CancelledError as exc:
            if pending.retain_pending_on_failure and not pending.receiver_terminal:
                # A local failure is not proof that the peer stopped reading.
                self._retain_pending_kv_transfer(object_id, pending, exc)
                retained = True
            raise
        except Exception as exc:
            if pending.retain_pending_on_failure and not pending.receiver_terminal:
                self._retain_pending_kv_transfer(object_id, pending, exc)
                retained = True
                raise
            for op in pending.ops:
                with suppress(Exception):
                    op.mark_receiver_failed(exc)
            for op in pending.ops:
                with suppress(Exception):
                    await op.wait_for_completion(timeout=self._ack_timeout_s)
            raise
        finally:
            if not retained:
                self._pending.pop(object_id, None)
                if pending.lease is not None:
                    pending.lease.release()

    def _retain_pending_kv_transfer(
        self,
        object_id: str,
        pending: _PendingTransfer,
        error: BaseException,
    ) -> None:
        self._pending.pop(object_id, None)
        self._retained_pending_kv_transfers.append(pending)
        _comm_trace(
            "comm_kv_pending_retained",
            object_id=object_id,
            retained_count=len(self._retained_pending_kv_transfers),
            num_ops=len(pending.ops),
            error=type(error).__name__,
        )
        logger.error(
            "Retaining pending KV transfer %s after sender failure: %s",
            object_id,
            error,
        )

    def _fail_pending(self, object_id: str, exc: BaseException) -> None:
        pending = self._pending.get(object_id)
        if pending is None:
            return
        if not pending.ack.done():
            pending.ack.set_exception(exc)
        if pending.task is None:
            self._arm_pending(object_id)

    def _track_task(self, task: asyncio.Task, label: str) -> None:
        if self._task_done_callback is not None:
            task.add_done_callback(lambda done: self._task_done_callback(done, label))
            return

        def _log_failure(done: asyncio.Task) -> None:
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                logger.exception(
                    "%s failed",
                    label,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(_log_failure)
