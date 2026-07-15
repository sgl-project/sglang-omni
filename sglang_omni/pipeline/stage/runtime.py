# SPDX-License-Identifier: Apache-2.0
"""Stage — IO shell for pipeline processing.

Handles: control plane messaging, data plane (relay) IO, input aggregation,
stream chunk routing, abort tracking, profiling.

Dispatches all compute to scheduler (OmniScheduler or SimpleScheduler).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import queue as _queue_mod
import threading
from contextlib import suppress
from typing import Any, Awaitable, Callable, Literal

import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline.stage.input import DirectInput, InputHandler
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.pipeline.tp_control import TPLeaderFanout, TPWorkMessage
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_recorder
from sglang_omni.profiler.event_recorder import set_active_stage as _set_active_stage
from sglang_omni.profiler.torch_profiler import TorchProfiler
from sglang_omni.proto import (
    AdminMessage,
    AdminResult,
    AdminResultMessage,
    CompleteMessage,
    DataAckMessage,
    DataReadyMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
)
from sglang_omni.relay.base import Relay
from sglang_omni.scheduling.messages import IncomingMessage

logger = logging.getLogger(__name__)

GetNextFn = Callable[[str, Any], str | list[str] | None]
GetStreamDoneTargetsFn = Callable[[str, Any], str | list[str] | None]


def _error_text(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


class Stage:
    """IO shell for one pipeline stage.

    All stage compute is dispatched through the scheduler inbox/outbox
    contract, independent of scheduler implementation.

    Note on ``role``: ``role="single"`` means this stage owns its own ZMQ
    control plane and relay reader (i.e. it is NOT a TP follower). It does
    **not** imply this stage has its OS process to itself — since the
    declarative topology PR, multiple ``role="single"`` stages can share
    one OS process (and one asyncio event loop). When they do, they share
    a failure domain: see ``_run_process`` in ``stage_workers.py``.
    ``role="leader"`` / ``role="follower"`` continue to denote TP rank 0
    vs rank > 0 within a multi-rank TP stage; TP stages must own their OS
    process exclusively.
    """

    def __init__(
        self,
        name: str,
        role: Literal["single", "leader", "follower"],
        get_next: GetNextFn,
        gpu_id: int | None,
        endpoints: dict[str, str],
        control_plane: Any,
        placement_gpu_id: int | None = None,
        input_handler: InputHandler | None = None,
        relay: Relay | None = None,
        comm_config: dict[str, Any] | None = None,
        scheduler: Any = None,
        project_payload: dict[str, Callable[[Any], Any]] | None = None,
        stream_targets: list[str] | None = None,
        get_stream_done_targets: GetStreamDoneTargetsFn | None = None,
        gpu_stage_names: set[str] | None = None,
        stage_gpu_ids: dict[str, tuple[int, ...]] | None = None,
        remote_stage_names: set[str] | None = None,
        same_process_targets: set[str] | None = None,
        local_dispatcher: Any | None = None,
        can_accept_stream_before_payload: bool = False,
        disable_direct_cuda_ipc_payload: bool = False,
        tp_fanout: TPLeaderFanout | None = None,
        is_terminal: bool = False,
    ):
        self.name = name
        self.role = role
        self.get_next = get_next
        self.gpu_id = gpu_id
        self.endpoints = endpoints
        self.control_plane = control_plane
        self.input_handler = input_handler or DirectInput()
        self.scheduler = scheduler
        self._project_payload = project_payload or {}
        self._stream_targets = stream_targets or []
        self.get_stream_done_targets = get_stream_done_targets
        self._same_process_targets = same_process_targets or set()
        self._local_dispatcher = local_dispatcher
        self._can_accept_stream_before_payload = can_accept_stream_before_payload
        self._disable_direct_cuda_ipc_payload = disable_direct_cuda_ipc_payload
        self._tp_fanout = tp_fanout
        self._is_terminal = is_terminal
        self._owns_external_io = role in {"single", "leader"}

        self._comm = CommEngine(
            CommRouter(
                stage_name=name,
                gpu_id=self.gpu_id,
                placement_gpu_id=placement_gpu_id,
                same_process_targets=self._same_process_targets,
                gpu_stage_names=gpu_stage_names or set(),
                stage_gpu_ids=stage_gpu_ids,
                remote_stage_names=remote_stage_names or set(),
                comm_config=comm_config or {},
                injected_relay=relay,
            ),
            task_done_callback=self._on_background_task_done,
        )

        self._running = False
        self._aborted: set[str] = set()
        self._active_requests: set[str] = set()
        self._stream_queue: StreamQueue | None = None
        self._stream_chunk_counters: dict[tuple[str, str], int] = {}
        self._first_stream_chunk_seen: set[str] = set()
        self._local_stream_targets: dict[str, set[str]] = {}
        self._nonlocal_stream_targets: dict[str, set[str]] = {}
        self._receive_tasks: set[asyncio.Task] = set()
        self._receive_lane_tails: dict[tuple[str, str], asyncio.Future[None]] = {}
        self._scheduler_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._scheduler_crash_error: BaseException | None = None
        self._background_task_error: BaseException | None = None

    async def start(self) -> None:
        if self._running:
            return
        await self.control_plane.start()
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Start scheduler in dedicated thread
        if self.scheduler is not None:

            def _run_scheduler():
                # Active-stage binding so ``emit(stage=None)`` from
                # scheduler-thread descendants resolves to this stage.
                _set_active_stage(self.name)
                try:
                    if self.gpu_id is not None:
                        import torch

                        torch.cuda.set_device(int(self.gpu_id))
                        logger.info(
                            "Scheduler thread for stage %s set CUDA device to %s",
                            self.name,
                            self.gpu_id,
                        )
                    self.scheduler.start()
                except Exception as exc:
                    logger.exception("Scheduler thread for stage %s crashed", self.name)
                    self._running = False
                    loop = self._loop
                    if loop is not None and not loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._handle_scheduler_crash(exc),
                            loop,
                        )

            self._scheduler_thread = threading.Thread(
                target=_run_scheduler,
                name=f"scheduler-{self.name}",
                daemon=True,
            )
            self._scheduler_thread.start()

        logger.info("Stage %s started", self.name)

    async def stop(self) -> None:
        self._running = False
        cleanup_error: Exception | None = None

        def _record_cleanup_error(component: str, exc: Exception) -> None:
            nonlocal cleanup_error
            logger.warning(
                "Stage %s %s cleanup failed: %s",
                self.name,
                component,
                exc,
                exc_info=True,
            )
            if cleanup_error is None:
                cleanup_error = exc

        receive_tasks = list(self._receive_tasks)
        for task in receive_tasks:
            task.cancel()
        await asyncio.gather(*receive_tasks, return_exceptions=True)
        self._receive_tasks.clear()
        self._receive_lane_tails.clear()
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
            except Exception as exc:
                _record_cleanup_error("scheduler", exc)
        try:
            self.control_plane.close()
        except Exception as exc:
            _record_cleanup_error("control plane", exc)
        if self._tp_fanout is not None:
            try:
                self._tp_fanout.close()
            except Exception as exc:
                _record_cleanup_error("TP fanout", exc)
        try:
            self._comm.close()
        except Exception as exc:
            _record_cleanup_error("comm", exc)
        logger.info("Stage %s stopped", self.name)
        if cleanup_error is not None:
            raise RuntimeError(f"Stage {self.name} cleanup failed") from cleanup_error

    async def run(self) -> None:
        await self.start()

        abort_task = asyncio.create_task(self._abort_listener())
        outbox_task = asyncio.create_task(self._drain_outbox())
        abort_task.add_done_callback(
            lambda task: self._on_background_task_done(task, "abort listener")
        )
        outbox_task.add_done_callback(
            lambda task: self._on_background_task_done(task, "outbox drain")
        )

        try:
            while self._running:
                msg = await self.control_plane.recv()
                if (
                    self.role == "leader"
                    and self._tp_fanout is not None
                    and isinstance(
                        msg,
                        (
                            ShutdownMessage,
                            ProfilerStartMessage,
                            ProfilerStopMessage,
                            AdminMessage,
                        ),
                    )
                ):
                    await self._tp_fanout.fanout_control(msg)
                if isinstance(msg, ShutdownMessage):
                    break
                if isinstance(msg, TPWorkMessage):
                    await self._execute(msg.data)
                    continue
                await self._handle_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            if self._scheduler_crash_error is None:
                raise
        finally:
            await self.stop()
            abort_task.cancel()
            outbox_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task
            with suppress(asyncio.CancelledError):
                await outbox_task
            if self._background_task_error is not None:
                raise self._background_task_error
            if self._scheduler_crash_error is not None:
                raise RuntimeError(
                    f"Scheduler thread for stage {self.name} crashed"
                ) from self._scheduler_crash_error

    async def _handle_message(self, msg: Any) -> None:
        if isinstance(msg, SubmitMessage):
            await self._on_submit(msg)
        elif isinstance(msg, DataAckMessage):
            self._comm.ack_transfer(msg)
        elif isinstance(msg, DataReadyMessage):
            self._schedule_receive_task(msg)
        elif isinstance(msg, ProfilerStartMessage):
            self._on_profiler_start(msg)
        elif isinstance(msg, ProfilerStopMessage):
            self._on_profiler_stop(msg)
        elif isinstance(msg, AdminMessage):
            await self._on_admin(msg)

    def _schedule_receive_task(
        self,
        msg: DataReadyMessage,
    ) -> None:
        if msg.is_done or msg.error is not None:
            handler = self._on_stream_signal
            label = f"stream signal {msg.request_id}:{msg.from_stage}"
        elif msg.chunk_id is not None:
            handler = self._on_stream_chunk
            label = f"stream chunk {msg.request_id}:{msg.from_stage}:{msg.chunk_id}"
        else:
            handler = self._on_data_ready
            label = f"payload {msg.request_id}:{msg.from_stage}"

        lane = (msg.request_id, msg.from_stage)
        predecessor = self._receive_lane_tails.get(lane)
        completion = asyncio.get_running_loop().create_future()
        self._receive_lane_tails[lane] = completion
        task = asyncio.create_task(
            self._run_receive_task(
                handler(msg, predecessor),
                lane,
                predecessor,
                completion,
            )
        )
        self._receive_tasks.add(task)
        task.add_done_callback(self._receive_tasks.discard)
        task.add_done_callback(lambda done: self._on_background_task_done(done, label))

    async def _run_receive_task(
        self,
        coro: Awaitable[None],
        lane: tuple[str, str],
        predecessor: asyncio.Future[None] | None,
        completion: asyncio.Future[None],
    ) -> None:
        try:
            await coro
        finally:
            if predecessor is not None:
                await predecessor
            if not completion.done():
                completion.set_result(None)
            if self._receive_lane_tails.get(lane) is completion:
                self._receive_lane_tails.pop(lane, None)

    @staticmethod
    async def _wait_for_receive_predecessor(
        predecessor: asyncio.Future[None] | None,
    ) -> None:
        if predecessor is not None:
            await predecessor

    async def _on_submit(self, msg: SubmitMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_input_received",
            metadata={"from_stage": "coordinator", "kind": "submit"},
        )

        payload = msg.data  # StagePayload from coordinator
        await self._execute(payload)

    async def _on_data_ready(
        self,
        msg: DataReadyMessage,
        predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            await self._discard_payload_data(msg)
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        if stage_io.is_direct_cuda_ipc_payload_ref(msg.data_ref):
            try:
                payload = stage_io.deserialize_direct_cuda_ipc_payload(msg.data_ref)
            except Exception as exc:
                logger.exception(
                    "Stage %s: direct IPC payload deserialize failed for %s",
                    self.name,
                    request_id,
                )
                await self._wait_for_receive_predecessor(predecessor)
                await self._send_failure(
                    request_id, f"direct IPC payload deserialize failed: {exc}"
                )
                return
            await self._wait_for_receive_predecessor(predecessor)
            await self._receive_payload_from_stage(request_id, msg.from_stage, payload)
            return

        data_ref = self._data_ref_from_message(msg)
        relay = self._comm.relay(data_ref.transport)
        try:
            payload = await self._comm.read_payload(
                relay=relay,
                request_id=request_id,
                data_ref=data_ref,
            )
        except Exception as exc:
            logger.exception(
                "Stage %s: relay read failed for %s", self.name, request_id
            )
            await self._send_data_ack(
                msg, data_ref, success=False, error=_error_text(exc)
            )
            relay.cleanup(request_id)
            await self._wait_for_receive_predecessor(predecessor)
            await self._send_failure(request_id, f"relay read failed: {exc}")
            return
        await self._send_data_ack(msg, data_ref, success=True)

        await self._wait_for_receive_predecessor(predecessor)
        await self._receive_payload_from_stage(request_id, msg.from_stage, payload)

    async def receive_local_payload(
        self,
        request_id: str,
        from_stage: str,
        payload: Any,
    ) -> None:
        await self._receive_payload_from_stage(request_id, from_stage, payload)

    async def receive_local_stream_chunk(
        self,
        request_id: str,
        from_stage: str,
        chunk_id: int,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        item = StreamItem(
            chunk_id=chunk_id,
            data=data,
            from_stage=from_stage,
            metadata=metadata,
        )
        self._emit_stream_chunk_received(
            request_id=request_id,
            from_stage=from_stage,
            chunk_id=chunk_id,
        )
        await self._route_stream_item_or_fail(request_id, item)

    async def receive_local_stream_signal(
        self,
        request_id: str,
        from_stage: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        await self._receive_stream_signal(
            request_id,
            from_stage,
            is_done=is_done,
            error=error,
        )

    async def _receive_payload_from_stage(
        self,
        request_id: str,
        from_stage: str,
        payload: Any,
    ) -> None:
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        if request_id in self._aborted:
            return

        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_input_received",
            metadata={"from_stage": from_stage, "kind": "payload"},
        )
        merged = self.input_handler.receive(request_id, from_stage, payload)
        if merged is not None:
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_aggregate_ready",
                metadata={"from_stage": from_stage},
            )
            await self._execute(merged)

    async def _on_stream_chunk(
        self,
        msg: DataReadyMessage,
        predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            await self._discard_stream_chunk_data(msg)
            return
        self._active_requests.add(request_id)

        if stage_io.is_direct_cuda_ipc_stream_chunk_ref(msg.data_ref):
            try:
                data, metadata = stage_io.deserialize_direct_cuda_ipc_stream_chunk(
                    msg.data_ref
                )
            except Exception as exc:
                logger.error(
                    "Stage %s: direct IPC deserialize failed for %s: %s",
                    self.name,
                    request_id,
                    exc,
                )
                await self._wait_for_receive_predecessor(predecessor)
                await self._queue_stream_error(request_id, msg.from_stage, exc)
                return
            await self._wait_for_receive_predecessor(predecessor)
            if request_id in self._aborted:
                return
            item = StreamItem(
                chunk_id=msg.chunk_id,
                data=data,
                from_stage=msg.from_stage,
                metadata=metadata,
            )
            self._emit_stream_chunk_received(
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                chunk_id=msg.chunk_id,
            )
            await self._route_stream_item_or_fail(request_id, item)
            return

        data_ref = self._data_ref_from_message(msg)
        relay = self._comm.relay(data_ref.transport)
        try:
            data, metadata = await self._comm.read_stream_chunk(
                relay=relay,
                data_ref=data_ref,
            )
        except Exception as exc:
            logger.error(
                "Stage %s: stream chunk read failed for %s: %s",
                self.name,
                request_id,
                exc,
            )
            await self._send_data_ack(
                msg, data_ref, success=False, error=_error_text(exc)
            )
            await self._wait_for_receive_predecessor(predecessor)
            await self._queue_stream_error(request_id, msg.from_stage, exc)
            return
        await self._send_data_ack(msg, data_ref, success=True)

        await self._wait_for_receive_predecessor(predecessor)
        if request_id in self._aborted:
            return

        item = StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata,
        )
        self._emit_stream_chunk_received(
            request_id=msg.request_id,
            from_stage=msg.from_stage,
            chunk_id=msg.chunk_id,
        )
        await self._route_stream_item_or_fail(request_id, item)

    def _emit_stream_chunk_received(
        self,
        *,
        request_id: str,
        from_stage: str,
        chunk_id: int | None,
    ) -> None:
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_stream_chunk_received",
            metadata={"from_stage": from_stage, "chunk_id": chunk_id},
        )

    async def _route_stream_item_or_fail(
        self, request_id: str, item: StreamItem
    ) -> None:
        if self._open_pre_payload_stream_if_allowed(request_id):
            self._route_stream_item(request_id, item)
            return
        with suppress(Exception):
            self.scheduler.abort(request_id)
        await self._send_failure(
            request_id,
            (
                f"Stage {self.name}: stream chunk from {item.from_stage!r} arrived "
                "before the request payload, but this stage is not configured to "
                "accept pre-payload stream data"
            ),
        )

    async def _queue_stream_error(
        self,
        request_id: str,
        from_stage: str | None,
        error: BaseException,
    ) -> None:
        if request_id in self._aborted:
            return
        logger.error(
            "Stage %s: stream error from %s for %s: %s",
            self.name,
            from_stage,
            request_id,
            error,
        )
        with suppress(Exception):
            self.scheduler.abort(request_id)
        await self._send_failure(request_id, str(error))

    def _data_ref_from_message(self, msg: DataReadyMessage) -> DataRef:
        if msg.data_ref is None:
            raise ValueError("data_ready message is missing transfer data_ref")
        return DataRef.from_dict(msg.data_ref)

    async def _send_data_ack(
        self,
        msg: DataReadyMessage,
        data_ref: DataRef,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        endpoint = self.endpoints.get(msg.from_stage)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for ack target "
                f"{msg.from_stage!r}"
            )
        await self.control_plane.send_to_stage(
            msg.from_stage,
            endpoint,
            DataAckMessage(
                request_id=msg.request_id,
                from_stage=self.name,
                to_stage=msg.from_stage,
                object_id=data_ref.object_id,
                success=success,
                error=error,
            ),
        )

    async def _discard_payload_data(self, msg: DataReadyMessage) -> None:
        if stage_io.is_direct_cuda_ipc_payload_ref(msg.data_ref):
            imported = stage_io.deserialize_direct_cuda_ipc_payload(msg.data_ref)
            del imported
            return
        request_id = msg.request_id
        data_ref = self._data_ref_from_message(msg)
        relay = self._comm.relay(data_ref.transport)
        try:
            await self._comm.read_payload(
                relay=relay,
                request_id=request_id,
                data_ref=data_ref,
            )
        except Exception as exc:
            logger.debug(
                "Stage %s: failed to drain aborted payload for %s",
                self.name,
                request_id,
                exc_info=True,
            )
            await self._send_data_ack(
                msg, data_ref, success=False, error=_error_text(exc)
            )
            self._comm.cleanup(request_id)
            return
        await self._send_data_ack(msg, data_ref, success=True)

    async def _discard_stream_chunk_data(self, msg: DataReadyMessage) -> None:
        if stage_io.is_direct_cuda_ipc_stream_chunk_ref(msg.data_ref):
            imported = stage_io.deserialize_direct_cuda_ipc_stream_chunk(msg.data_ref)
            del imported
            return
        if msg.chunk_id is None:
            raise ValueError("stream chunk discard requires chunk_id")
        data_ref = self._data_ref_from_message(msg)
        relay = self._comm.relay(data_ref.transport)
        try:
            await self._comm.read_stream_chunk(relay=relay, data_ref=data_ref)
        except Exception as exc:
            logger.debug(
                "Stage %s: failed to drain aborted stream chunk for %s",
                self.name,
                msg.request_id,
                exc_info=True,
            )
            await self._send_data_ack(
                msg, data_ref, success=False, error=_error_text(exc)
            )
            return
        await self._send_data_ack(msg, data_ref, success=True)

    async def _on_stream_signal(
        self,
        msg: DataReadyMessage,
        predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        await self._wait_for_receive_predecessor(predecessor)
        await self._receive_stream_signal(
            msg.request_id,
            msg.from_stage,
            is_done=msg.is_done,
            error=msg.error,
        )

    async def _receive_stream_signal(
        self,
        request_id: str,
        from_stage: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if error:
            await self._queue_stream_error(
                request_id,
                from_stage,
                RuntimeError(error),
            )
            return

        if is_done:
            if not self._open_pre_payload_stream_if_allowed(request_id):
                with suppress(Exception):
                    self.scheduler.abort(request_id)
                await self._send_failure(
                    request_id,
                    (
                        f"Stage {self.name}: stream_done from {from_stage!r} "
                        "arrived before the request payload, but this stage is not "
                        "configured to accept pre-payload stream data"
                    ),
                )
                return
            self._stream_queue.put_done(request_id, from_stage=from_stage)
            self.scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_done",
                )
            )

    def _open_pre_payload_stream_if_allowed(self, request_id: str) -> bool:
        if self._stream_queue is None:
            return False
        if self._stream_queue.has(request_id):
            return True
        if not self._can_accept_stream_before_payload:
            return False
        self._active_requests.add(request_id)
        self._stream_queue.open(request_id)
        return True

    def _route_stream_item(self, request_id: str, item: StreamItem) -> None:
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="stream_chunk", data=item)
        )

    async def _execute(self, payload: Any) -> None:
        request_id = payload.request_id
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_dispatch",
        )
        if (
            self.role == "leader"
            and self._tp_fanout is not None
            and getattr(self.scheduler, "requires_tp_work_fanout", False)
        ):
            self._tp_fanout.fanout_work(payload)
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )

    async def _on_admin(self, msg: AdminMessage) -> None:
        operation = msg.operation
        if self.role == "leader" and self._tp_fanout is not None:
            local = await self._run_admin_operation(operation)
            try:
                follower_msgs = await self._tp_fanout.collect_admin_results(
                    operation.op_id,
                    timeout_s=float(
                        60.0 if operation.timeout_s is None else operation.timeout_s
                    ),
                )
            except Exception as exc:
                local.success = False
                local.error = str(exc)
                local.message = "failed to collect TP follower admin results"
                follower_msgs = []

            rank_results = [local] + [item.result for item in follower_msgs]
            success = all(item.success for item in rank_results)
            errors = [item.error for item in rank_results if item.error]
            data = dict(local.data)
            data["tp_size"] = len(rank_results)
            data["rank_results"] = [item.to_dict() for item in rank_results]
            result = AdminResult(
                op_id=operation.op_id,
                stage=self.name,
                action=operation.action,
                success=success,
                message=(
                    local.message if success else "; ".join(errors) or local.message
                ),
                data=data,
                error=None if success else "; ".join(errors) or local.error,
                rank=0,
                role=self.role,
            )
            await self.control_plane.send_admin_result(AdminResultMessage(result))
            return

        result = await self._run_admin_operation(operation)
        await self.control_plane.send_admin_result(AdminResultMessage(result))

    async def _run_admin_operation(self, operation: Any) -> AdminResult:
        try:
            handler = getattr(self.scheduler, "admin", None)
            if handler is None:
                return self._admin_result(
                    operation,
                    success=True,
                    message="stage does not support admin operations",
                    data={"skipped": True, "unsupported": True},
                )
            action = operation.action
            payload = dict(operation.payload)
            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(None, lambda: handler(action, payload))
            if inspect.isawaitable(outcome):
                outcome = await outcome
            return self._admin_result_from_outcome(operation, outcome)
        except Exception as exc:
            logger.exception(
                "Stage %s admin operation failed: action=%s",
                self.name,
                getattr(operation, "action", None),
            )
            return self._admin_result(
                operation,
                success=False,
                message=str(exc),
                error=str(exc),
            )

    def _admin_result_from_outcome(self, operation: Any, outcome: Any) -> AdminResult:
        if isinstance(outcome, AdminResult):
            return outcome
        if isinstance(outcome, dict):
            data = dict(outcome.get("data") or {})
            for key, value in outcome.items():
                if key not in {"success", "message", "data", "error"}:
                    data.setdefault(key, value)
            return self._admin_result(
                operation,
                success=bool(outcome.get("success", True)),
                message=str(outcome.get("message") or "ok"),
                data=data,
                error=outcome.get("error"),
            )
        return self._admin_result(
            operation,
            success=True,
            message="ok",
            data={"result": outcome},
        )

    def _admin_result(
        self,
        operation: Any,
        *,
        success: bool,
        message: str = "",
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AdminResult:
        return AdminResult(
            op_id=operation.op_id,
            stage=self.name,
            action=operation.action,
            success=success,
            message=message,
            data=dict(data or {}),
            error=error,
            rank=getattr(self.scheduler, "tp_rank", None),
            role=self.role,
        )

    # ------------------------------------------------------------------
    # Outbox drain: scheduler results → route downstream
    # ------------------------------------------------------------------

    async def _drain_outbox(self) -> None:
        if self._owns_external_io:
            await self._drain_outbox_external()
        else:
            await self._drain_outbox_follower()

    async def _drain_outbox_external(self) -> None:
        """Drain scheduler outbox and route results downstream."""
        loop = asyncio.get_running_loop()
        while self._running or not self.scheduler.outbox.empty():
            try:
                out = await loop.run_in_executor(
                    None, lambda: self.scheduler.outbox.get(timeout=0.1)
                )
            except _queue_mod.Empty:
                continue

            if out.request_id not in self._active_requests:
                continue

            if out.type == "result":
                await self._route_result(out.request_id, out.data)
            elif out.type == "stream":
                if out.target is None:
                    if self._stream_targets:
                        await asyncio.gather(
                            *(
                                self._send_stream_to_target(
                                    out.request_id,
                                    out.data,
                                    target,
                                    out.metadata,
                                )
                                for target in self._stream_targets
                            )
                        )
                    else:
                        await self._send_stream_to_coordinator(
                            out.request_id,
                            out.data,
                            out.metadata,
                        )
                else:
                    await self._send_stream_to_target(
                        out.request_id,
                        out.data,
                        out.target,
                        out.metadata,
                    )
            elif out.type == "error":
                await self._send_failure(out.request_id, str(out.data))

    async def _drain_outbox_follower(self) -> None:
        """Drain follower outbox without emitting external stage traffic."""
        loop = asyncio.get_running_loop()
        while self._running or not self.scheduler.outbox.empty():
            try:
                out = await loop.run_in_executor(
                    None, lambda: self.scheduler.outbox.get(timeout=0.1)
                )
            except _queue_mod.Empty:
                continue

            if out.type == "result":
                self._clear_request_state(out.request_id)
            elif out.type == "stream":
                continue
            elif out.type == "error":
                raise RuntimeError(
                    f"TP follower stage {self.name} received scheduler error: {out.data}"
                )

    async def _route_result(self, request_id: str, result: Any) -> None:
        """Route a completed result to next stage(s) or complete at coordinator."""
        if not self._owns_external_io:
            self._clear_request_state(request_id)
            return
        # Send stream done to the active stream targets for this request.
        stream_targets = self._stream_targets
        if self.get_stream_done_targets is not None:
            resolved = self.get_stream_done_targets(request_id, result)
            if isinstance(resolved, str):
                stream_targets = [resolved]
            elif isinstance(resolved, list):
                stream_targets = resolved
            elif resolved is None:
                stream_targets = []
        stream_targets_for_request = set(stream_targets)
        for target in stream_targets:
            await self._send_stream_signal_to_target(
                request_id,
                target,
                is_done=True,
            )

        next_stages = self.get_next(request_id, result)
        if next_stages is None:
            # Terminal: notify coordinator
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": True},
            )
            await self.control_plane.send_complete(
                CompleteMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    success=True,
                    result=result.data if isinstance(result, StagePayload) else result,
                )
            )
        else:
            if isinstance(next_stages, str):
                next_stages = [next_stages]
            is_single_target = len(next_stages) == 1
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": False, "next": list(next_stages)},
            )
            for target in next_stages:
                await self._send_to_stage(
                    request_id,
                    target,
                    result,
                    allow_local_object=is_single_target,
                    allow_projected_local_object=not is_single_target,
                    stream_targets_for_request=stream_targets_for_request,
                )

        self._clear_request_state(request_id)

    async def _send_to_stage(
        self,
        request_id: str,
        target: str,
        payload: Any,
        *,
        allow_local_object: bool = False,
        allow_projected_local_object: bool = False,
        stream_targets_for_request: set[str] | None = None,
    ) -> None:
        if not self._owns_external_io:
            raise RuntimeError(
                f"Follower stage {self.name} cannot send downstream data"
            )
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for target {target!r}"
            )
        projector = self._project_payload.get(target)
        projected_payload = projector(payload) if projector is not None else payload
        use_local_object = allow_local_object or (
            allow_projected_local_object
            and self._is_isolated_projected_payload(
                payload,
                projected_payload,
                projector_present=projector is not None,
            )
        )

        if (
            use_local_object
            and target in self._same_process_targets
            and self._can_send_full_payload_locally(
                request_id,
                target,
                (
                    set(self._stream_targets)
                    if stream_targets_for_request is None
                    else stream_targets_for_request
                ),
            )
        ):
            if self._local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process target {target!r} requires "
                    "a local dispatcher"
                )

            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_hop_sent",
                metadata={"to_stage": target, "transport": "local_object"},
            )
            await self._local_dispatcher.send_payload(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                payload=projected_payload,
            )
            return

        can_use_direct_cuda_ipc = self._comm.router.can_use_direct_cuda_ipc(target)
        if (
            not self._disable_direct_cuda_ipc_payload
            and can_use_direct_cuda_ipc
            and stage_io.payload_has_cuda_tensor(projected_payload)
        ):
            try:
                direct_ref = stage_io.serialize_direct_cuda_ipc_payload(
                    projected_payload
                )
            except RuntimeError as exc:
                if "received from another process" not in str(exc):
                    raise
            else:
                await self.control_plane.send_to_stage(
                    target,
                    endpoint,
                    DataReadyMessage(
                        request_id=request_id,
                        from_stage=self.name,
                        to_stage=target,
                        data_ref=direct_ref,
                    ),
                )
                _emit_event(
                    request_id=request_id,
                    stage=self.name,
                    event_name="stage_hop_sent",
                    metadata={"to_stage": target, "transport": "torch_cuda_ipc"},
                )
                return

        transport_kind, relay = self._comm.router.relay_for_payload(
            target, projected_payload
        )
        await self._comm.send_payload(
            relay=relay,
            control_plane=self.control_plane,
            request_id=request_id,
            payload=projected_payload,
            transport=transport_kind,
            from_stage=self.name,
            to_stage=target,
            target_endpoint=endpoint,
        )
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_hop_sent",
            metadata={"to_stage": target, "transport": transport_kind.value},
        )

    @staticmethod
    def _is_isolated_projected_payload(
        original_payload: Any,
        projected_payload: Any,
        *,
        projector_present: bool,
    ) -> bool:
        if not projector_present or projected_payload is original_payload:
            return False
        if not isinstance(original_payload, StagePayload):
            raise TypeError(
                "projected local-object dispatch requires the original payload "
                f"to be StagePayload, got {type(original_payload).__name__}"
            )
        if not isinstance(projected_payload, StagePayload):
            raise TypeError(
                "projected local-object dispatch requires projectors to return "
                f"StagePayload, got {type(projected_payload).__name__}"
            )
        if projected_payload.data is original_payload.data:
            return False
        return not Stage._shares_mutable_container(
            original_payload.data, projected_payload.data
        )

    @staticmethod
    def _shares_mutable_container(original: Any, projected: Any) -> bool:
        original_ids = Stage._collect_mutable_container_ids(original)
        if not original_ids:
            return False
        return Stage._contains_mutable_container_id(projected, original_ids)

    @staticmethod
    def _collect_mutable_container_ids(
        obj: Any, seen: set[int] | None = None
    ) -> set[int]:
        seen = set() if seen is None else seen
        obj_id = id(obj)
        if obj_id in seen:
            return set()
        seen.add(obj_id)

        ids: set[int] = set()
        if isinstance(obj, (dict, list, set, bytearray)):
            ids.add(obj_id)

        for child in Stage._iter_container_children(obj):
            ids.update(Stage._collect_mutable_container_ids(child, seen))
        return ids

    @staticmethod
    def _contains_mutable_container_id(
        obj: Any, original_ids: set[int], seen: set[int] | None = None
    ) -> bool:
        seen = set() if seen is None else seen
        obj_id = id(obj)
        if obj_id in seen:
            return False
        seen.add(obj_id)

        if isinstance(obj, (dict, list, set, bytearray)) and obj_id in original_ids:
            return True
        return any(
            Stage._contains_mutable_container_id(child, original_ids, seen)
            for child in Stage._iter_container_children(obj)
        )

    @staticmethod
    def _iter_container_children(obj: Any):
        if isinstance(obj, dict):
            return obj.values()
        if isinstance(obj, (list, tuple, set, frozenset)):
            return obj
        return ()

    def _can_send_full_payload_locally(
        self,
        request_id: str,
        target: str,
        stream_targets_for_request: set[str],
    ) -> bool:
        if target in self._nonlocal_stream_targets.get(request_id, set()):
            return False
        if target not in stream_targets_for_request:
            return True
        return target in self._local_stream_targets.get(request_id, set())

    def _record_local_stream_target(self, request_id: str, target: str) -> None:
        self._local_stream_targets.setdefault(request_id, set()).add(target)

    def _record_nonlocal_stream_target(self, request_id: str, target: str) -> None:
        self._nonlocal_stream_targets.setdefault(request_id, set()).add(target)

    async def _send_stream_to_target(
        self,
        request_id: str,
        data: Any,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._owns_external_io:
            return
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for stream target "
                f"{target!r}"
            )
        key = (request_id, target)
        chunk_id = self._stream_chunk_counters.get(key, 0)
        self._stream_chunk_counters[key] = chunk_id + 1
        chunk_modality = (
            metadata.get("modality") if isinstance(metadata, dict) else None
        )
        if request_id not in self._first_stream_chunk_seen:
            self._first_stream_chunk_seen.add(request_id)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_first_stream_chunk_sent",
                metadata={"to_stage": target, "modality": chunk_modality},
            )
        if target in self._same_process_targets:
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_stream_chunk_sent",
                metadata={
                    "to_stage": target,
                    "chunk_id": chunk_id,
                    "modality": chunk_modality,
                    "transport": "local_object",
                },
            )
            if self._local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process stream target {target!r} "
                    "requires a local dispatcher"
                )
            self._record_local_stream_target(request_id, target)
            await self._local_dispatcher.send_stream_chunk(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                chunk_id=chunk_id,
                data=data,
                metadata=metadata,
            )
            return
        self._record_nonlocal_stream_target(request_id, target)
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "relay-backed stream chunks must be torch.Tensor, got "
                f"{type(data).__name__}"
            )
        if data.is_cuda and self._comm.router.can_use_direct_cuda_ipc(target):
            direct_ref = stage_io.serialize_direct_cuda_ipc_stream_chunk(data, metadata)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_stream_chunk_sent",
                metadata={
                    "to_stage": target,
                    "chunk_id": chunk_id,
                    "modality": chunk_modality,
                    "transport": "torch_cuda_ipc",
                },
            )
            await self.control_plane.send_to_stage(
                target,
                endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    to_stage=target,
                    data_ref=direct_ref,
                    chunk_id=chunk_id,
                ),
            )
            return
        transport_kind, relay = self._comm.router.relay_for_stream(target, data)
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_stream_chunk_sent",
            metadata={
                "to_stage": target,
                "chunk_id": chunk_id,
                "modality": chunk_modality,
                "transport": transport_kind.value,
            },
        )
        await self._comm.send_stream_chunk(
            relay=relay,
            control_plane=self.control_plane,
            request_id=request_id,
            data=data,
            target_stage=target,
            target_endpoint=endpoint,
            from_stage=self.name,
            chunk_id=chunk_id,
            metadata=metadata,
            transport=transport_kind,
        )

    async def _send_stream_signal_to_target(
        self,
        request_id: str,
        target: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        if not self._owns_external_io:
            return
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for stream target "
                f"{target!r}"
            )
        if target in self._same_process_targets:
            if self._local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process stream target {target!r} "
                    "requires a local dispatcher"
                )
            self._record_local_stream_target(request_id, target)
            await self._local_dispatcher.send_stream_signal(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                is_done=is_done,
                error=error,
            )
            return
        self._record_nonlocal_stream_target(request_id, target)
        await stage_io.send_stream_signal(
            self.control_plane,
            request_id=request_id,
            target_stage=target,
            target_endpoint=endpoint,
            from_stage=self.name,
            is_done=is_done,
            error=error,
        )

    async def _send_stream_to_coordinator(
        self,
        request_id: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Forward a terminal stage's stream chunk to the Coordinator."""
        if not self._is_terminal:
            raise RuntimeError(
                f"Stage {self.name!r} emitted untargeted stream chunk but isn't "
                "terminal. Set ``terminal=True``, or use ``target=...`` / "
                "``stream_to=[...]``."
            )
        if not self._owns_external_io:
            return
        if request_id in self._aborted:
            return
        modality = metadata.get("modality") if isinstance(metadata, dict) else None
        if modality is None and isinstance(data, dict):
            modality = data.get("modality")
        key = (request_id, "coordinator")
        chunk_id = self._stream_chunk_counters.get(key, 0)
        self._stream_chunk_counters[key] = chunk_id + 1
        msg = StreamMessage(
            request_id=request_id,
            from_stage=self.name,
            chunk=data,
            stage_name=self.name,
            modality=modality,
            chunk_id=chunk_id,
        )
        if request_id not in self._first_stream_chunk_seen:
            self._first_stream_chunk_seen.add(request_id)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_first_stream_chunk_sent",
                metadata={
                    "to_stage": "coordinator",
                    "chunk_id": chunk_id,
                    "modality": modality,
                },
            )
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_stream_chunk_sent",
            metadata={
                "to_stage": "coordinator",
                "chunk_id": chunk_id,
                "modality": modality,
            },
        )
        await self.control_plane.send_stream(msg)

    async def _send_failure(self, request_id: str, error: str) -> None:
        self._record_aborted_request_id(request_id)
        if not self._owns_external_io:
            self._clear_request_state(request_id)
            raise RuntimeError(f"Follower stage {self.name} failed: {error}")
        await self.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.name,
                success=False,
                error=error,
            )
        )
        self._clear_request_state(request_id)

    def _clear_request_state(self, request_id: str) -> None:
        self._active_requests.discard(request_id)
        self.input_handler.cancel(request_id)
        if self._stream_queue is not None:
            self._stream_queue.close(request_id)
        stale_keys = [
            key for key in self._stream_chunk_counters if key[0] == request_id
        ]
        for key in stale_keys:
            self._stream_chunk_counters.pop(key, None)
        self._first_stream_chunk_seen.discard(request_id)
        self._local_stream_targets.pop(request_id, None)
        self._nonlocal_stream_targets.pop(request_id, None)

    async def _handle_scheduler_crash(self, exc: BaseException) -> None:
        if self._scheduler_crash_error is not None:
            return
        self._scheduler_crash_error = exc
        if not self._owns_external_io:
            self.control_plane.close()
            return
        error = f"scheduler crashed: {exc}"
        active_request_ids = [
            request_id
            for request_id in list(self._active_requests)
            if request_id not in self._aborted
        ]
        for request_id in active_request_ids:
            with suppress(Exception):
                self.scheduler.abort(request_id)
            await self._send_failure(request_id, error)
            with suppress(Exception):
                self._comm.cleanup(request_id)
        self.control_plane.close()

    async def _abort_listener(self) -> None:
        try:
            while self._running:
                abort_msg = await self.control_plane.recv_abort()
                if self.role == "leader" and self._tp_fanout is not None:
                    await self._tp_fanout.fanout_abort(abort_msg)
                self._on_abort(abort_msg.request_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            if self._scheduler_crash_error is None and self._running:
                logger.exception("Stage %s abort listener crashed", self.name)

    def _record_aborted_request_id(self, request_id: str) -> None:
        self._aborted.add(request_id)
        if len(self._aborted) > 10000:
            excess = len(self._aborted) - 5000
            it = iter(self._aborted)
            to_remove = [next(it) for _ in range(excess)]
            self._aborted -= set(to_remove)

    def _on_abort(self, request_id: str) -> None:
        self._record_aborted_request_id(request_id)
        self._comm.cleanup(request_id)
        self._clear_request_state(request_id)
        self.scheduler.abort(request_id)

    def _on_profiler_start(self, msg: ProfilerStartMessage) -> None:
        run_id = msg.run_id
        if msg.enable_torch and not TorchProfiler.is_active():
            base_tpl = msg.trace_path_template.format(run_id=run_id, stage=self.name)
            template = f"{base_tpl}_pid{os.getpid()}"
            prof_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
            if prof_dir and not os.path.isabs(template):
                template = os.path.join(prof_dir, template)
            TorchProfiler.start(template, run_id=run_id)
        if msg.event_dir is not None:
            try:
                _get_recorder().start(
                    run_id=run_id, event_dir=msg.event_dir, stage=self.name
                )
            except Exception:
                logger.warning(
                    "Stage %s failed to start request event recorder",
                    self.name,
                    exc_info=True,
                )

    def _on_profiler_stop(self, msg: ProfilerStopMessage) -> None:
        # run_id=None is a wildcard (stop whatever's active).
        if TorchProfiler.is_active() and (
            msg.run_id is None or TorchProfiler.get_active_run_id() == msg.run_id
        ):
            TorchProfiler.stop(run_id=msg.run_id)
        recorder = _get_recorder()
        if recorder.is_active() and (
            msg.run_id is None or recorder.active_run_id() == msg.run_id
        ):
            recorder.stop(run_id=msg.run_id)

    def _on_background_task_done(self, task: asyncio.Task, label: str) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        logger.exception(
            "Stage %s %s task crashed",
            self.name,
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        if self._background_task_error is None:
            self._background_task_error = exc
        self._running = False
        self.control_plane.close()

    def info(self) -> StageInfo:
        return StageInfo(
            name=self.name,
            control_endpoint=self.control_plane.recv_endpoint,
        )
