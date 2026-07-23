# SPDX-License-Identifier: Apache-2.0
"""Omni communication engine facade used by pipeline stages."""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Callable

import msgspec
import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.comm.router import CommRouter
from sglang_omni.profiler.comm_trace import elapsed_ms as _comm_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import now_ns as _comm_now_ns
from sglang_omni.proto import DataAckMessage, DataReadyMessage, StagePayload
from sglang_omni.relay.base import Relay

logger = logging.getLogger(__name__)


class _PendingTransfer(msgspec.Struct):
    ops: list[Any]
    ack: asyncio.Future[None]
    task: asyncio.Task | None = None


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


class CommEngine:
    """Stage-owned communication engine.

    It owns locality classification and data_ref-based relay IO. Stages keep
    routing semantics; the engine owns byte movement mechanics.
    """

    def __init__(
        self,
        router: CommRouter,
        *,
        task_done_callback: Callable[[asyncio.Task, str], None] | None = None,
    ) -> None:
        self.router = router
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
        self._task_done_callback = task_done_callback
        self._closed = False

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

    async def read_payload(
        self,
        *,
        relay: Relay,
        request_id: str,
        data_ref: DataRef,
    ) -> StagePayload:
        return await stage_io.read_payload(relay, request_id, data_ref)

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
        return await stage_io.read_stream_chunk(relay, data_ref)

    def cleanup(self, request_id: str) -> None:
        self.router.cleanup(request_id)

    def close(self) -> None:
        self._closed = True
        for task in self._send_workers.values():
            task.cancel()
        self._send_workers.clear()
        self._send_queues.clear()
        for object_id in list(self._pending):
            self._fail_pending(object_id, RuntimeError("comm engine closed"))
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
            if not pending.ack.done():
                pending.ack.set_result(None)
            return
        error = ack.error
        if error is None:
            raise ValueError("failed data_ack is missing error")
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
            self._register_pending(object_id, [op])
            control_start = _comm_now_ns()
            await job.control_plane.send_to_stage(
                job.to_stage,
                job.target_endpoint,
                DataReadyMessage(
                    request_id=job.request_id,
                    from_stage=job.from_stage,
                    to_stage=job.to_stage,
                    data_ref=data_ref.to_dict(),
                ),
            )
            control_ms = _comm_elapsed_ms(control_start)
            self._arm_pending(object_id)
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
                metadata=job.metadata,
                transport=job.transport,
            )
            write_ms = _comm_elapsed_ms(write_start)
            object_id = data_ref.object_id
            self._register_pending(object_id, ops)
            control_start = _comm_now_ns()
            await job.control_plane.send_to_stage(
                job.target_stage,
                job.target_endpoint,
                DataReadyMessage(
                    request_id=job.request_id,
                    from_stage=job.from_stage,
                    to_stage=job.target_stage,
                    data_ref=data_ref.to_dict(),
                    chunk_id=job.chunk_id,
                ),
            )
            control_ms = _comm_elapsed_ms(control_start)
            self._arm_pending(object_id)
            _comm_trace(
                "comm_stream_send",
                request_id=job.request_id,
                from_stage=job.from_stage,
                to_stage=job.target_stage,
                chunk_id=job.chunk_id,
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

    def _register_pending(self, object_id: str, ops: list[Any]) -> None:
        if object_id in self._pending:
            raise RuntimeError(f"duplicate pending transfer {object_id!r}")
        self._pending[object_id] = _PendingTransfer(
            ops=ops,
            ack=asyncio.get_running_loop().create_future(),
        )

    def _arm_pending(self, object_id: str) -> None:
        pending = self._pending[object_id]
        assert pending.task is None
        pending.task = asyncio.create_task(self._watch_pending(object_id, pending))
        self._track_task(pending.task, f"comm ack {object_id}")

    async def _watch_pending(self, object_id: str, pending: _PendingTransfer) -> None:
        try:
            await asyncio.wait_for(pending.ack, timeout=self._ack_timeout_s)
            for op in pending.ops:
                op.mark_receiver_done()
            for op in pending.ops:
                await op.wait_for_completion(timeout=self._ack_timeout_s)
        except Exception as exc:
            for op in pending.ops:
                with suppress(Exception):
                    op.mark_receiver_failed(exc)
            for op in pending.ops:
                with suppress(Exception):
                    await op.wait_for_completion(timeout=self._ack_timeout_s)
            raise
        finally:
            self._pending.pop(object_id, None)

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
