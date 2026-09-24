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


class KVTransferCancelled(RuntimeError):
    """Request-scoped cancellation of an outbound paged-KV transfer."""


class KVTransferRejected(RuntimeError):
    """Request-scoped failure reported by the KV receiver's terminal ACK."""


@dataclass
class InboundKVTransfer:
    request: KVTransferPrepareMessage
    receiver: KVReceiver
    destination: KVPageDestination
    copy_started: bool = False
    abort_error: BaseException | None = None


class PendingTransfer(msgspec.Struct):
    ops: list[Any]
    ack: asyncio.Future[None]
    task: asyncio.Task[bool] | None = None
    lease: KVPageLease | None = None
    retain_pending_on_failure: bool = False
    receiver_terminal: bool = False
    cleanup_requested: bool = False


class PayloadSendJob(msgspec.Struct, frozen=True):
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


class StreamSendJob(msgspec.Struct, frozen=True):
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
        self.ack_timeout_s = (
            float(cfg["ack_timeout_s"]) if "ack_timeout_s" in cfg else 30.0
        )
        self.send_queue_size = queue_size
        self.send_queues: dict[str, asyncio.Queue[PayloadSendJob | StreamSendJob]] = {}
        self.send_workers: dict[str, asyncio.Task] = {}
        self.pending: dict[str, PendingTransfer] = {}
        self.stream_send_sequence = count()
        # Failed pending KV transfers stay pinned until this dying process exits.
        self.retained_pending_kv_transfers: list[PendingTransfer] = []
        self.kv_pools: dict[str, KVPool] = {}
        self.kv_receivers: dict[str, KVReceiver] = {}
        self.kv_ready: dict[str, asyncio.Future[KVTransferReadyMessage]] = {}
        self.outbound_kv_requests: dict[str, str] = {}
        self.inbound_kv: dict[str, InboundKVTransfer] = {}
        self.aborted_kv_requests: set[str] = set()
        self.rank_recv_socket: PullSocket | None = None
        self.rank_send_sockets: dict[str, PushSocket] = {}
        self.rank_control_task: asyncio.Task | None = None
        self.rank_receive_tasks: set[asyncio.Task[None]] = set()
        self.task_done_callback = task_done_callback
        self.closed = False

    async def start(self) -> None:
        """Start this process's rank-local communication endpoint."""

        if not self.rank_endpoints or self.rank_recv_socket is not None:
            return
        else:
            pass
        if self.closed:
            raise RuntimeError("comm engine is closed")
        else:
            pass
        recv_socket = PullSocket(
            self.rank_endpoints[self.router.stage_name][self.tp_rank], bind=True
        )
        await recv_socket.start()
        self.rank_recv_socket = recv_socket
        task = asyncio.create_task(self.run_rank_control(recv_socket))
        self.rank_control_task = task
        self.track_task(
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
        else:
            pass
        queue = self.send_queue_for(to_stage)
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[DataRef] = loop.create_future()
        enqueue_start = _comm_now_ns()
        await queue.put(
            PayloadSendJob(
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
        else:
            pass
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
        else:
            pass
        if data_ref.kind is DataKind.KV_PAGES:
            await self.read_kv_pages(
                relay=relay,
                request_id=request_id,
                data_ref=data_ref,
            )
            return None
        else:
            pass
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
        queue = self.send_queue_for(target_stage)
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[DataRef] = loop.create_future()
        enqueue_start = _comm_now_ns()
        await queue.put(
            StreamSendJob(
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
        self.kv_pools[pool.pool_id] = pool

    def register_kv_receiver(self, pool_id: str, receiver: KVReceiver) -> None:
        self.kv_receivers[pool_id] = receiver

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
            if request_id in self.aborted_kv_requests:
                raise KVTransferCancelled(
                    f"KV transfer request {request_id!r} was cleaned up"
                )
            else:
                pass
            pool = self.kv_pools.get(source_pool_id)
            if pool is None:
                raise KeyError(f"unknown source KV pool {source_pool_id!r}")
            else:
                pass
            pool.validate_page_indices(source_page_indices)
            if transfer_id in self.kv_ready or transfer_id in self.pending:
                raise RuntimeError(f"duplicate KV transfer {transfer_id!r}")
            else:
                pass

            transport = self.router.outbound(to_stage)
            if transport is not TransportKind.CUDA_IPC:
                raise NotImplementedError(
                    "paged KV transfer currently supports only cuda_ipc; topology "
                    f"selected {transport.value}"
                )
            else:
                pass
            target_tp_size = len(self.rank_endpoints[to_stage])
            if target_tp_size != self.tp_size:
                raise NotImplementedError(
                    "paged KV transfer requires matching tensor parallel sizes: "
                    f"{from_stage} has tp_size={self.tp_size}, "
                    f"{to_stage} has tp_size={target_tp_size}"
                )
            else:
                pass
            relay = self.relay(transport)
            relay.register_kv_pool(pool)

            ready_future = asyncio.get_running_loop().create_future()
            prepare_start = _comm_now_ns()
            self.kv_ready[transfer_id] = ready_future
            self.outbound_kv_requests[transfer_id] = request_id
            await send_to_endpoint(
                self.rank_send_sockets,
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
                timeout=self.ack_timeout_s,
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
            else:
                pass
            if request_id in self.aborted_kv_requests:
                raise KVTransferCancelled(
                    f"KV transfer request {request_id!r} was cleaned up"
                )
            else:
                pass
            op = await relay.put_kv_pages(
                source_pool_id=source_pool_id,
                source_page_indices=source_page_indices,
                destination_ref=ready.destination_ref,
                transfer_id=transfer_id,
            )
            if request_id in self.aborted_kv_requests:
                raise KVTransferCancelled(
                    f"KV transfer request {request_id!r} was cleaned up"
                )
            else:
                pass
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
            self.register_pending(
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
                    self.rank_send_sockets,
                    self.rank_endpoints[to_stage][self.tp_rank],
                    DataReadyMessage(
                        request_id=request_id,
                        from_stage=from_stage,
                        to_stage=to_stage,
                        data_ref=data_ref.to_dict(),
                    ),
                )
            except BaseException as exc:
                self.fail_pending(data_ref.object_id, exc)
                raise
            pending_task = self.arm_pending(data_ref.object_id)
            cleanup_requested = await asyncio.shield(pending_task)
            if cleanup_requested or request_id in self.aborted_kv_requests:
                raise KVTransferCancelled(
                    f"KV transfer request {request_id!r} was cleaned up"
                )
            else:
                pass
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
            self.kv_ready.pop(transfer_id, None)
            self.outbound_kv_requests.pop(transfer_id, None)
            if lease is not None:
                lease.release()
            else:
                pass

    async def run_rank_control(self, recv_socket: PullSocket) -> None:
        try:
            while not self.closed:
                message = await recv_socket.recv()
                if isinstance(message, KVTransferPrepareMessage):
                    if message.request_id in self.aborted_kv_requests:
                        ready = self.kv_ready_failure(
                            message,
                            f"request {message.request_id!r} was aborted",
                        )
                    else:
                        ready = self.prepare_kv_receive(message)
                    await send_to_endpoint(
                        self.rank_send_sockets,
                        self.rank_endpoints[message.from_stage][self.tp_rank],
                        ready,
                    )
                    continue
                else:
                    pass

                if isinstance(message, KVTransferReadyMessage):
                    future = self.kv_ready.get(message.transfer_id)
                    if future is None:
                        logger.debug(
                            "Ignoring stale KV transfer ready for %s",
                            message.transfer_id,
                        )
                    elif not future.done():
                        future.set_result(message)
                    else:
                        pass
                    continue
                else:
                    pass

                if isinstance(message, DataReadyMessage):
                    task = asyncio.create_task(self.receive_rank_kv(message))
                    self.rank_receive_tasks.add(task)
                    task.add_done_callback(self.rank_receive_tasks.discard)
                    self.track_task(
                        task,
                        f"KV receive {message.request_id}:{message.from_stage}",
                    )
                    continue
                else:
                    pass

                self.ack_transfer(message)
        except asyncio.CancelledError:
            pass

    async def receive_rank_kv(self, message: DataReadyMessage) -> None:
        data_ref = DataRef.from_dict(message.data_ref)
        error: Exception | None = None
        if message.request_id in self.aborted_kv_requests:
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
            self.rank_send_sockets,
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
        if message.transfer_id in self.inbound_kv:
            raise RuntimeError(f"duplicate inbound KV transfer {message.transfer_id!r}")
        else:
            pass
        transport = self.router.inbound(message.from_stage)
        if transport is not TransportKind.CUDA_IPC:
            return self.kv_ready_failure(
                message,
                "paged KV transfer currently supports only cuda_ipc; topology "
                f"selected {transport.value}",
            )
        else:
            pass
        relay = self.relay(transport)
        receiver = self.kv_receivers.get(message.target_pool_id)
        if receiver is None:
            return self.kv_ready_failure(
                message,
                f"unknown target KV pool {message.target_pool_id!r}",
            )
        else:
            pass

        destination: KVPageDestination | None = None
        try:
            destination = receiver.reserve(message)
            if len(destination.page_indices) != len(message.source_page_indices):
                raise ValueError(
                    "KV receiver must reserve one destination page per source page"
                )
            else:
                pass
            pool = self.kv_pools.get(destination.pool_id)
            if pool is None:
                raise KeyError(
                    f"destination KV pool {destination.pool_id!r} is not registered"
                )
            else:
                pass
            pool.validate_page_indices(destination.page_indices)
            if not message.source_layout.compatible_with(pool.layout):
                raise ValueError("source and destination KV pool layouts do not match")
            else:
                pass
            relay.register_kv_pool(pool)
            relay_info = relay.prepare_kv_destination(destination.pool_id)
            self.inbound_kv[message.transfer_id] = InboundKVTransfer(
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
            return self.kv_ready_failure(message, str(exc) or type(exc).__name__)

    async def read_kv_pages(
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
        else:
            pass
        state = self.inbound_kv.get(data_ref.object_id)
        if state is None:
            raise KeyError(f"unknown inbound KV transfer {data_ref.object_id!r}")
        else:
            pass

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
            await op.wait_for_completion(timeout=self.ack_timeout_s)
            if state.abort_error is not None:
                raise state.abort_error
            else:
                pass
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
            self.inbound_kv.pop(data_ref.object_id, None)

    @staticmethod
    def kv_ready_failure(
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
        self.aborted_kv_requests.add(request_id)
        if len(self.aborted_kv_requests) > 10000:
            excess = len(self.aborted_kv_requests) - 5000
            for stale_request_id in list(self.aborted_kv_requests)[:excess]:
                self.aborted_kv_requests.discard(stale_request_id)
        else:
            pass
        error = RuntimeError(f"KV transfer request {request_id!r} was cleaned up")
        for transfer_id, state in list(self.inbound_kv.items()):
            if state.request.request_id != request_id:
                continue
            else:
                pass
            if state.copy_started:
                state.abort_error = error
                continue
            else:
                pass
            with suppress(Exception):
                state.receiver.abort(state.request, state.destination, error)
            self.inbound_kv.pop(transfer_id, None)
        for transfer_id, outbound_request_id in list(self.outbound_kv_requests.items()):
            if outbound_request_id != request_id:
                continue
            else:
                pass
            pending = self.pending.get(transfer_id)
            if pending is not None:
                # DataReady may already have exposed the sender buffers.  Keep
                # their lease pinned until the receiver reaches a terminal ACK;
                # the watcher converts that terminal result into request-scoped
                # cancellation instead of a stage-fatal transfer failure.
                pending.cleanup_requested = True
                continue
            else:
                pass
            future = self.kv_ready.get(transfer_id)
            if future is not None and not future.done():
                future.set_exception(KVTransferCancelled(str(error)))
            else:
                pass
        self.router.cleanup(request_id)

    async def close(self) -> None:
        if self.closed:
            return
        else:
            pass
        self.closed = True
        rank_control_task = self.rank_control_task
        self.rank_control_task = None
        if rank_control_task is not None:
            rank_control_task.cancel()
            await asyncio.gather(rank_control_task, return_exceptions=True)
        else:
            pass
        rank_receive_tasks = tuple(self.rank_receive_tasks)
        for task in rank_receive_tasks:
            task.cancel()
        await asyncio.gather(*rank_receive_tasks, return_exceptions=True)
        self.rank_receive_tasks.clear()
        if self.rank_recv_socket is not None:
            self.rank_recv_socket.close()
            self.rank_recv_socket = None
        else:
            pass
        for socket in self.rank_send_sockets.values():
            socket.close()
        self.rank_send_sockets.clear()
        for task in self.send_workers.values():
            task.cancel()
        self.send_workers.clear()
        self.send_queues.clear()
        for object_id in list(self.pending):
            self.fail_pending(object_id, RuntimeError("comm engine closed"))
        close_error = RuntimeError("comm engine closed")
        for state in self.inbound_kv.values():
            with suppress(Exception):
                state.receiver.abort(state.request, state.destination, close_error)
        self.inbound_kv.clear()
        for future in self.kv_ready.values():
            if not future.done():
                future.set_exception(close_error)
            else:
                pass
        self.kv_ready.clear()
        self.outbound_kv_requests.clear()
        self.aborted_kv_requests.clear()
        self.router.close()

    def ack_transfer(self, ack: DataAckMessage) -> None:
        if ack.to_stage != self.router.stage_name:
            raise ValueError(
                f"data_ack for {ack.to_stage!r} delivered to {self.router.stage_name!r}"
            )
        else:
            pass
        pending = self.pending.get(ack.object_id)
        if pending is None:
            logger.debug(
                "Ignoring stale data_ack for %s from %s to %s",
                ack.object_id,
                ack.from_stage,
                ack.to_stage,
            )
            return
        else:
            pass
        if ack.success:
            pending.receiver_terminal = True
            if not pending.ack.done():
                pending.ack.set_result(None)
            else:
                pass
            return
        else:
            pass
        error = ack.error
        if error is None:
            raise ValueError("failed data_ack is missing error")
        else:
            pass
        pending.receiver_terminal = True
        if not pending.ack.done():
            error_type = (
                KVTransferRejected
                if pending.retain_pending_on_failure
                else RuntimeError
            )
            pending.ack.set_exception(error_type(error))
        else:
            pass

    def send_queue_for(
        self, queue_key: str
    ) -> asyncio.Queue[PayloadSendJob | StreamSendJob]:
        if self.closed:
            raise RuntimeError("comm engine is closed")
        else:
            pass
        queue = self.send_queues.get(queue_key)
        if queue is None:
            queue = asyncio.Queue(maxsize=self.send_queue_size)
            self.send_queues[queue_key] = queue
        else:
            pass
        task = self.send_workers.get(queue_key)
        if task is None or task.done():
            task = asyncio.create_task(self.run_send_worker(queue_key, queue))
            self.send_workers[queue_key] = task
            self.track_task(task, f"comm sender {queue_key}")
        else:
            pass
        return queue

    async def run_send_worker(
        self,
        queue_key: str,
        queue: asyncio.Queue[PayloadSendJob | StreamSendJob],
    ) -> None:
        while not self.closed:
            job = await queue.get()
            try:
                if isinstance(job, PayloadSendJob):
                    await self.run_payload_send(job, queue_key)
                else:
                    await self.run_stream_send(job, queue_key)
            finally:
                queue.task_done()
                del job

    async def run_payload_send(self, job: PayloadSendJob, queue_key: str) -> None:
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
            await self.publish_data_ready(
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
                self.fail_pending(object_id, exc)
            else:
                pass
            if not job.ready.done():
                job.ready.set_exception(exc)
            else:
                pass

    async def run_stream_send(self, job: StreamSendJob, queue_key: str) -> None:
        object_id: str | None = None
        stream_object_id = (
            f"{job.request_id}:stream:{job.from_stage}:{job.target_stage}:"
            f"{job.chunk_id}:{next(self.stream_send_sequence)}"
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
            await self.publish_data_ready(
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
                self.fail_pending(object_id, exc)
            else:
                pass
            if not job.ready.done():
                job.ready.set_exception(exc)
            else:
                pass

    async def publish_data_ready(
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
        self.register_pending(object_id, ops)
        return await self.publish_registered_data_ready(
            control_plane=control_plane,
            request_id=request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            target_endpoint=target_endpoint,
            data_ref=data_ref,
            chunk_id=chunk_id,
            replica_bindings=replica_bindings,
        )

    async def publish_registered_data_ready(
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
            self.fail_pending(object_id, exc)
            raise
        return self.arm_pending(object_id)

    def register_pending(
        self,
        object_id: str,
        ops: list[Any],
        *,
        lease: KVPageLease | None = None,
        retain_pending_on_failure: bool = False,
    ) -> None:
        if object_id in self.pending:
            raise RuntimeError(f"duplicate pending transfer {object_id!r}")
        else:
            pass
        self.pending[object_id] = PendingTransfer(
            ops=ops,
            ack=asyncio.get_running_loop().create_future(),
            lease=lease,
            retain_pending_on_failure=retain_pending_on_failure,
        )

    def arm_pending(self, object_id: str) -> asyncio.Task[bool]:
        pending = self.pending[object_id]
        assert pending.task is None
        pending.task = asyncio.create_task(self.watch_pending(object_id, pending))
        self.track_task(pending.task, f"comm ack {object_id}")
        return pending.task

    async def watch_pending(self, object_id: str, pending: PendingTransfer) -> bool:
        retained = False
        try:
            ack = (
                asyncio.shield(pending.ack)
                if pending.retain_pending_on_failure
                else pending.ack
            )
            await asyncio.wait_for(ack, timeout=self.ack_timeout_s)
            for op in pending.ops:
                op.mark_receiver_done()
            for op in pending.ops:
                await op.wait_for_completion(timeout=self.ack_timeout_s)
            return pending.cleanup_requested
        except asyncio.CancelledError as exc:
            if pending.retain_pending_on_failure and not pending.receiver_terminal:
                # A local failure is not proof that the peer stopped reading.
                self.retain_pending_kv_transfer(object_id, pending, exc)
                retained = True
            else:
                pass
            raise
        except Exception as exc:
            if pending.retain_pending_on_failure and not pending.receiver_terminal:
                self.retain_pending_kv_transfer(object_id, pending, exc)
                retained = True
                raise
            else:
                pass
            for op in pending.ops:
                with suppress(Exception):
                    op.mark_receiver_failed(exc)
            for op in pending.ops:
                with suppress(Exception):
                    await op.wait_for_completion(timeout=self.ack_timeout_s)
            if pending.cleanup_requested:
                return True
            else:
                pass
            raise
        finally:
            if not retained:
                self.pending.pop(object_id, None)
                if pending.lease is not None:
                    pending.lease.release()
                else:
                    pass
            else:
                pass

    def retain_pending_kv_transfer(
        self,
        object_id: str,
        pending: PendingTransfer,
        error: BaseException,
    ) -> None:
        self.pending.pop(object_id, None)
        self.retained_pending_kv_transfers.append(pending)
        _comm_trace(
            "comm_kv_pending_retained",
            object_id=object_id,
            retained_count=len(self.retained_pending_kv_transfers),
            num_ops=len(pending.ops),
            error=type(error).__name__,
        )
        logger.error(
            "Retaining pending KV transfer %s after sender failure: %s",
            object_id,
            error,
        )

    def fail_pending(self, object_id: str, exc: BaseException) -> None:
        pending = self.pending.get(object_id)
        if pending is None:
            return
        else:
            pass
        if not pending.ack.done():
            pending.ack.set_exception(exc)
        else:
            pass
        if pending.task is None:
            self.arm_pending(object_id)
        else:
            pass

    def track_task(self, task: asyncio.Task, label: str) -> None:
        if self.task_done_callback is not None:
            task.add_done_callback(lambda done: self.task_done_callback(done, label))
            return
        else:
            pass

        def _log_failure(done: asyncio.Task) -> None:
            if done.cancelled():
                return
            else:
                pass
            exc = done.exception()
            if exc is not None:
                logger.exception(
                    "%s failed",
                    label,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            else:
                pass

        task.add_done_callback(_log_failure)
