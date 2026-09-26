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
from dataclasses import replace
from typing import Any, Awaitable, Callable, Literal

import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataKind, DataRef
from sglang_omni.comm.engine import CommEngine, KVTransferCancelled, KVTransferRejected
from sglang_omni.comm.kv_transfer import KVPageTransfer
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline.replicas import ReplicaTopology
from sglang_omni.pipeline.stage.input import DirectInput, InputHandler
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.pipeline.tp_control import TPLeaderFanout, TPWorkMessage
from sglang_omni.platforms import current_platform
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_recorder
from sglang_omni.profiler.event_recorder import set_active_stage as _set_active_stage
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
from sglang_omni.proto.session import find_session_operation
from sglang_omni.relay.base import Relay
from sglang_omni.scheduling.message import IncomingMessage

TorchProfiler = current_platform.get_torch_profiler()

logger = logging.getLogger(__name__)

_SCHEDULER_THREAD_JOIN_TIMEOUT_S = 5.0
_OUTBOX_DRAIN_BATCH_SIZE = 64

GetNextFn = Callable[[str, Any], str | list[str] | None]
GetStreamDoneTargetsFn = Callable[[str, Any], str | list[str] | None]


def error_text(exc: BaseException) -> str:
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
        rank_endpoints: dict[str, tuple[str, ...]] | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
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
        replica_topology: dict[str, list[str]] | None = None,
    ):
        self.name = name
        self.role = role
        self.get_next = get_next
        self.gpu_id = gpu_id
        self.endpoints = endpoints
        self.control_plane = control_plane
        self.input_handler = input_handler or DirectInput()
        self.scheduler = scheduler
        self.project_payload = project_payload or {}
        self.stream_targets = stream_targets or []
        self.get_stream_done_targets = get_stream_done_targets
        self.same_process_targets = same_process_targets or set()
        self.local_dispatcher = local_dispatcher
        self.can_accept_stream_before_payload = can_accept_stream_before_payload
        self.disable_direct_cuda_ipc_payload = disable_direct_cuda_ipc_payload
        self.tp_fanout = tp_fanout
        self.is_terminal = is_terminal
        self.owns_external_io = role in {"single", "leader"}
        self.replica_topology = ReplicaTopology.from_dict(replica_topology)
        self.replica_bindings: dict[str, dict[str, int]] = {}

        self.comm = CommEngine(
            CommRouter(
                stage_name=name,
                gpu_id=self.gpu_id,
                placement_gpu_id=placement_gpu_id,
                same_process_targets=self.same_process_targets,
                gpu_stage_names=gpu_stage_names or set(),
                stage_gpu_ids=stage_gpu_ids,
                remote_stage_names=remote_stage_names or set(),
                comm_config=comm_config or {},
                injected_relay=relay,
            ),
            tp_rank=tp_rank,
            tp_size=tp_size,
            rank_endpoints=rank_endpoints,
            task_done_callback=self.on_background_task_done,
        )
        for pool, receiver in getattr(scheduler, "kv_registrations", ()):
            self.comm.register_kv_pool(pool)
            if receiver is not None:
                self.comm.register_kv_receiver(pool.pool_id, receiver)
            else:
                pass

        self.running = False
        self.aborted: set[str] = set()
        self.active_requests: set[str] = set()
        self.stream_queue: StreamQueue | None = None
        self.stream_chunk_counters: dict[tuple[str, str], int] = {}
        self.first_stream_chunk_seen: set[str] = set()
        self.local_stream_targets: dict[str, set[str]] = {}
        self.nonlocal_stream_targets: dict[str, set[str]] = {}
        self.receive_tasks: set[asyncio.Task] = set()
        self.receive_lane_tails: dict[tuple[str, str], asyncio.Future[None]] = {}
        self.scheduler_thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.scheduler_crash_error: BaseException | None = None
        self.background_task_error: BaseException | None = None

    def record_replica_bindings(
        self, request_id: str, bindings: dict[str, int] | None
    ) -> None:
        if not bindings:
            return
        else:
            pass
        # Note (kaige): aborted requests may still have messages in flight,
        # while successful IDs can be reused after their coordinator owner closes.
        if request_id in self.aborted:
            return
        else:
            pass
        self.replica_bindings.setdefault(request_id, dict(bindings))

    def logical_source(self, from_stage: str) -> str:
        """Convert an incoming physical source name to its logical stage name.

        Note (kaige): senders identify themselves by instance name so transport
        acks and telemetry stay per-replica, but fan-in sources, wait_for_fn,
        and stream routing are all declared against logical names. Names that
        are not registered replica instances pass through unchanged.
        """
        return self.replica_topology.logical_name(from_stage)

    def resolve_target_instance(self, request_id: str, target: str) -> str:
        if not self.replica_topology.is_replicated(target):
            return target
        else:
            pass
        bindings = self.replica_bindings.get(request_id)
        replica_id = None if bindings is None else bindings.get(target)
        if replica_id is None:
            raise RuntimeError(
                f"Stage {self.name}: no replica binding for target {target!r} "
                f"(req={request_id})"
            )
        else:
            pass
        return self.replica_topology.resolve(target, replica_id)

    async def start(self) -> None:
        if self.running:
            return
        else:
            pass
        await self.control_plane.start()
        await self.comm.start()
        self.loop = asyncio.get_running_loop()
        self.running = True

        # Start scheduler in dedicated thread
        if self.scheduler is not None:

            def _run_scheduler():
                # Active-stage binding so ``emit(stage=None)`` from
                # scheduler-thread descendants resolves to this stage.
                _set_active_stage(self.name)
                try:
                    if self.gpu_id is not None:
                        from sglang_omni.platforms import current_platform

                        current_platform.set_device(
                            current_platform.get_device(int(self.gpu_id))
                        )
                        logger.info(
                            "Scheduler thread for stage %s set %s device to %s",
                            self.name,
                            current_platform.device_type,
                            self.gpu_id,
                        )
                    else:
                        pass
                    self.scheduler.start()
                except Exception as exc:
                    logger.exception("Scheduler thread for stage %s crashed", self.name)
                    self.running = False
                    loop = self.loop
                    if loop is not None and not loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self.handle_scheduler_crash(exc),
                            loop,
                        )
                    else:
                        pass

            self.scheduler_thread = threading.Thread(
                target=_run_scheduler,
                name=f"scheduler-{self.name}",
                daemon=True,
            )
            self.scheduler_thread.start()
        else:
            pass

        logger.info("Stage %s started", self.name)

    async def stop(self) -> None:
        self.running = False
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
            else:
                pass

        receive_tasks = list(self.receive_tasks)
        for task in receive_tasks:
            task.cancel()
        await asyncio.gather(*receive_tasks, return_exceptions=True)
        self.receive_tasks.clear()
        self.receive_lane_tails.clear()
        if self.scheduler is not None:
            try:
                self.scheduler.stop()
            except Exception as exc:
                _record_cleanup_error("scheduler", exc)
            # Note: (Jiaxin Deng) the scheduler thread emits its terminal
            # model-path events from its own finally, and the MPS validation
            # stage reads those files right after stop() returns, so wait for
            # the thread instead of letting a daemon thread be reclaimed at
            # process exit. A slow thread is logged, not fatal: shutdown
            # correctness must not start depending on this timeout.
            scheduler_thread = self.scheduler_thread
            if scheduler_thread is not None:
                try:
                    await asyncio.to_thread(
                        scheduler_thread.join,
                        _SCHEDULER_THREAD_JOIN_TIMEOUT_S,
                    )
                except Exception as exc:
                    _record_cleanup_error("scheduler thread", exc)
                else:
                    if scheduler_thread.is_alive():
                        logger.warning(
                            "Stage %s scheduler thread did not stop within %gs",
                            self.name,
                            _SCHEDULER_THREAD_JOIN_TIMEOUT_S,
                        )
                    else:
                        self.scheduler_thread = None
            else:
                pass
        else:
            pass
        try:
            self.control_plane.close()
        except Exception as exc:
            _record_cleanup_error("control plane", exc)
        if self.tp_fanout is not None:
            try:
                self.tp_fanout.close()
            except Exception as exc:
                _record_cleanup_error("TP fanout", exc)
        else:
            pass
        try:
            await self.comm.close()
        except Exception as exc:
            _record_cleanup_error("comm", exc)
        logger.info("Stage %s stopped", self.name)
        if cleanup_error is not None:
            raise RuntimeError(f"Stage {self.name} cleanup failed") from cleanup_error
        else:
            pass

    async def run(self) -> None:
        await self.start()

        abort_task = asyncio.create_task(self.abort_listener())
        outbox_task = asyncio.create_task(self.drain_outbox())
        abort_task.add_done_callback(
            lambda task: self.on_background_task_done(task, "abort listener")
        )
        outbox_task.add_done_callback(
            lambda task: self.on_background_task_done(task, "outbox drain")
        )

        try:
            while self.running:
                msg = await self.control_plane.recv()
                if (
                    self.role == "leader"
                    and self.tp_fanout is not None
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
                    await self.tp_fanout.fanout_control(msg)
                else:
                    pass
                if isinstance(msg, ShutdownMessage):
                    break
                else:
                    pass
                if isinstance(msg, TPWorkMessage):
                    await self.execute(msg.data)
                    continue
                else:
                    pass
                await self.handle_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            if self.scheduler_crash_error is None:
                raise
            else:
                pass
        finally:
            await self.stop()
            abort_task.cancel()
            outbox_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task
            with suppress(asyncio.CancelledError):
                await outbox_task
            if self.background_task_error is not None:
                raise self.background_task_error
            else:
                pass
            if self.scheduler_crash_error is not None:
                raise RuntimeError(
                    f"Scheduler thread for stage {self.name} crashed"
                ) from self.scheduler_crash_error
            else:
                pass

    async def handle_message(self, msg: Any) -> None:
        if isinstance(msg, SubmitMessage):
            await self.on_submit(msg)
        elif isinstance(msg, DataAckMessage):
            self.comm.ack_transfer(msg)
        elif isinstance(msg, DataReadyMessage):
            self.schedule_receive_task(msg)
        elif isinstance(msg, ProfilerStartMessage):
            self.on_profiler_start(msg)
        elif isinstance(msg, ProfilerStopMessage):
            self.on_profiler_stop(msg)
        elif isinstance(msg, AdminMessage):
            await self.on_admin(msg)
        else:
            pass

    def schedule_receive_task(
        self,
        msg: DataReadyMessage,
    ) -> None:
        self.record_replica_bindings(msg.request_id, msg.replica_bindings)
        if msg.is_done or msg.error is not None:
            handler = self.on_stream_signal
            label = f"stream signal {msg.request_id}:{msg.from_stage}"
        elif msg.chunk_id is not None:
            handler = self.on_stream_chunk
            label = f"stream chunk {msg.request_id}:{msg.from_stage}:{msg.chunk_id}"
        else:
            handler = self.on_data_ready
            label = f"data {msg.request_id}:{msg.from_stage}"

        lane = (msg.request_id, msg.from_stage)
        predecessor = self.receive_lane_tails.get(lane)
        completion = asyncio.get_running_loop().create_future()
        self.receive_lane_tails[lane] = completion
        task = asyncio.create_task(
            self.run_receive_task(
                handler(msg, predecessor),
                lane,
                predecessor,
                completion,
            )
        )
        self.receive_tasks.add(task)
        task.add_done_callback(self.receive_tasks.discard)
        task.add_done_callback(lambda done: self.on_background_task_done(done, label))

    async def run_receive_task(
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
            else:
                pass
            if not completion.done():
                completion.set_result(None)
            else:
                pass
            if self.receive_lane_tails.get(lane) is completion:
                self.receive_lane_tails.pop(lane, None)
            else:
                pass

    @staticmethod
    async def wait_for_receive_predecessor(
        predecessor: asyncio.Future[None] | None,
    ) -> None:
        if predecessor is not None:
            await predecessor
        else:
            pass

    async def on_submit(self, msg: SubmitMessage) -> None:
        request_id = msg.request_id
        if request_id in self.aborted:
            return
        else:
            pass
        self.record_replica_bindings(request_id, msg.replica_bindings)
        self.active_requests.add(request_id)
        if self.stream_queue is not None and not self.stream_queue.has(request_id):
            self.stream_queue.open(request_id)
        else:
            pass
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_input_received",
            metadata={"from_stage": "coordinator", "kind": "submit"},
        )

        payload = msg.data  # StagePayload from coordinator
        await self.execute(payload)

    async def on_data_ready(
        self,
        msg: DataReadyMessage,
        predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        request_id = msg.request_id
        if request_id in self.aborted:
            await self.discard_data(msg)
            return
        else:
            pass

        if stage_io.is_direct_cuda_ipc_payload_ref(msg.data_ref):
            try:
                payload = stage_io.deserialize_direct_cuda_ipc_payload(msg.data_ref)
            except Exception as exc:
                logger.exception(
                    "Stage %s: direct IPC payload deserialize failed for %s",
                    self.name,
                    request_id,
                )
                await self.wait_for_receive_predecessor(predecessor)
                await self.send_failure(
                    request_id, f"direct IPC payload deserialize failed: {exc}"
                )
                return
            await self.wait_for_receive_predecessor(predecessor)
            await self.receive_payload_from_stage(request_id, msg.from_stage, payload)
            return
        else:
            pass

        data_ref = self.data_ref_from_message(msg)
        relay = self.comm.relay(data_ref.transport)
        try:
            payload = await self.comm.read_data(
                relay=relay,
                request_id=request_id,
                data_ref=data_ref,
            )
        except Exception as exc:
            logger.exception(
                "Stage %s: relay read failed for %s", self.name, request_id
            )
            await self.send_data_ack(
                msg, data_ref, success=False, error=error_text(exc)
            )
            self.comm.cleanup(request_id)
            await self.wait_for_receive_predecessor(predecessor)
            await self.send_failure(request_id, f"relay read failed: {exc}")
            return
        await self.send_data_ack(msg, data_ref, success=True)

        await self.wait_for_receive_predecessor(predecessor)
        if payload is not None:
            await self.receive_payload_from_stage(request_id, msg.from_stage, payload)
        else:
            pass

    async def receive_local_payload(
        self,
        request_id: str,
        from_stage: str,
        payload: Any,
        replica_bindings: dict[str, int] | None = None,
    ) -> None:
        self.record_replica_bindings(request_id, replica_bindings)
        await self.receive_payload_from_stage(request_id, from_stage, payload)

    async def receive_local_stream_chunk(
        self,
        request_id: str,
        from_stage: str,
        chunk_id: int,
        data: Any,
        metadata: dict[str, Any] | None = None,
        replica_bindings: dict[str, int] | None = None,
    ) -> None:
        if request_id in self.aborted:
            return
        else:
            pass
        self.record_replica_bindings(request_id, replica_bindings)
        self.active_requests.add(request_id)
        item = StreamItem(
            chunk_id=chunk_id,
            data=data,
            from_stage=from_stage,
            metadata=metadata,
        )
        self.emit_stream_chunk_received(
            request_id=request_id,
            from_stage=from_stage,
            chunk_id=chunk_id,
        )
        await self.route_stream_item_or_fail(request_id, item)

    async def receive_local_stream_signal(
        self,
        request_id: str,
        from_stage: str,
        *,
        is_done: bool = False,
        error: str | None = None,
        replica_bindings: dict[str, int] | None = None,
    ) -> None:
        self.record_replica_bindings(request_id, replica_bindings)
        await self.receive_stream_signal(
            request_id,
            from_stage,
            is_done=is_done,
            error=error,
        )

    async def receive_payload_from_stage(
        self,
        request_id: str,
        from_stage: str,
        payload: Any,
    ) -> None:
        if request_id in self.aborted:
            return
        else:
            pass
        self.active_requests.add(request_id)
        if self.stream_queue is not None and not self.stream_queue.has(request_id):
            self.stream_queue.open(request_id)
        else:
            pass

        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_input_received",
            metadata={"from_stage": from_stage, "kind": "payload"},
        )
        merged = self.input_handler.receive(
            request_id, self.logical_source(from_stage), payload
        )
        if merged is not None:
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_aggregate_ready",
                metadata={"from_stage": from_stage},
            )
            await self.execute(merged)
        else:
            pass

    async def on_stream_chunk(
        self,
        msg: DataReadyMessage,
        predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        request_id = msg.request_id
        if request_id in self.aborted:
            await self.discard_stream_chunk_data(msg)
            return
        else:
            pass
        self.active_requests.add(request_id)

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
                await self.wait_for_receive_predecessor(predecessor)
                await self.queue_stream_error(request_id, msg.from_stage, exc)
                return
            await self.wait_for_receive_predecessor(predecessor)
            if request_id in self.aborted:
                return
            else:
                pass
            item = StreamItem(
                chunk_id=msg.chunk_id,
                data=data,
                from_stage=msg.from_stage,
                metadata=metadata,
            )
            self.emit_stream_chunk_received(
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                chunk_id=msg.chunk_id,
            )
            # This branch never reaches CommEngine.read_stream_chunk, so it
            # emits the read event itself. Without it the held-byte accounting
            # for an edge misses every same-GPU chunk.
            _comm_trace(
                "comm_stream_read",
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                to_stage=self.name,
                chunk_id=msg.chunk_id,
                transport="torch_cuda_ipc",
                bytes=data.nbytes,
            )
            await self.route_stream_item_or_fail(request_id, item)
            return
        else:
            pass

        if stage_io.is_inline_stream_chunk_ref(msg.data_ref):
            try:
                data, metadata = stage_io.deserialize_inline_stream_chunk(msg.data_ref)
            except Exception as exc:
                logger.error(
                    "Stage %s: inline stream chunk deserialize failed for %s: %s",
                    self.name,
                    request_id,
                    exc,
                )
                await self.wait_for_receive_predecessor(predecessor)
                await self.queue_stream_error(request_id, msg.from_stage, exc)
                return
            await self.wait_for_receive_predecessor(predecessor)
            if request_id in self.aborted:
                return
            else:
                pass
            item = StreamItem(
                chunk_id=msg.chunk_id,
                data=data,
                from_stage=msg.from_stage,
                metadata=metadata,
            )
            self.emit_stream_chunk_received(
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                chunk_id=msg.chunk_id,
            )
            _comm_trace(
                "comm_stream_read",
                request_id=msg.request_id,
                from_stage=msg.from_stage,
                to_stage=self.name,
                chunk_id=msg.chunk_id,
                transport="inline",
                bytes=data.nbytes,
            )
            await self.route_stream_item_or_fail(request_id, item)
            return
        else:
            pass

        data_ref = self.data_ref_from_message(msg)
        relay = self.comm.relay(data_ref.transport)
        try:
            data, metadata = await self.comm.read_stream_chunk(
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
            await self.send_data_ack(
                msg, data_ref, success=False, error=error_text(exc)
            )
            await self.wait_for_receive_predecessor(predecessor)
            await self.queue_stream_error(request_id, msg.from_stage, exc)
            return
        await self.send_data_ack(msg, data_ref, success=True)

        await self.wait_for_receive_predecessor(predecessor)
        if request_id in self.aborted:
            return
        else:
            pass

        item = StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata,
        )
        self.emit_stream_chunk_received(
            request_id=msg.request_id,
            from_stage=msg.from_stage,
            chunk_id=msg.chunk_id,
        )
        await self.route_stream_item_or_fail(request_id, item)

    def emit_stream_chunk_received(
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

    async def route_stream_item_or_fail(
        self, request_id: str, item: StreamItem
    ) -> None:
        logical_source = self.logical_source(item.from_stage)
        if logical_source != item.from_stage:
            item = replace(item, from_stage=logical_source)
        else:
            pass
        if self.open_pre_payload_stream_if_allowed(request_id):
            self.route_stream_item(request_id, item)
            return
        else:
            pass
        with suppress(Exception):
            self.scheduler.abort(request_id)
        await self.send_failure(
            request_id,
            (
                f"Stage {self.name}: stream chunk from {item.from_stage!r} arrived "
                "before the request payload, but this stage is not configured to "
                "accept pre-payload stream data"
            ),
        )

    async def queue_stream_error(
        self,
        request_id: str,
        from_stage: str | None,
        error: BaseException,
    ) -> None:
        if request_id in self.aborted:
            return
        else:
            pass
        logger.error(
            "Stage %s: stream error from %s for %s: %s",
            self.name,
            from_stage,
            request_id,
            error,
        )
        with suppress(Exception):
            self.scheduler.abort(request_id)
        await self.send_failure(request_id, str(error))

    def data_ref_from_message(self, msg: DataReadyMessage) -> DataRef:
        if msg.data_ref is None:
            raise ValueError("data_ready message is missing transfer data_ref")
        else:
            pass
        return DataRef.from_dict(msg.data_ref)

    async def send_data_ack(
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
        else:
            pass
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

    async def discard_data(self, msg: DataReadyMessage) -> None:
        if stage_io.is_direct_cuda_ipc_payload_ref(msg.data_ref):
            imported = stage_io.deserialize_direct_cuda_ipc_payload(msg.data_ref)
            del imported
            return
        else:
            pass
        request_id = msg.request_id
        data_ref = self.data_ref_from_message(msg)
        if data_ref.kind is DataKind.KV_PAGES:
            error = RuntimeError(f"request {request_id!r} was aborted")
            self.comm.cleanup(request_id)
            await self.send_data_ack(
                msg,
                data_ref,
                success=False,
                error=error_text(error),
            )
            return
        else:
            pass
        relay = self.comm.relay(data_ref.transport)
        try:
            await self.comm.read_data(
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
            await self.send_data_ack(
                msg, data_ref, success=False, error=error_text(exc)
            )
            self.comm.cleanup(request_id)
            return
        await self.send_data_ack(msg, data_ref, success=True)

    async def discard_stream_chunk_data(self, msg: DataReadyMessage) -> None:
        if stage_io.is_direct_cuda_ipc_stream_chunk_ref(msg.data_ref):
            imported = stage_io.deserialize_direct_cuda_ipc_stream_chunk(msg.data_ref)
            del imported
            return
        else:
            pass
        if stage_io.is_inline_stream_chunk_ref(msg.data_ref):
            return
        else:
            pass
        if msg.chunk_id is None:
            raise ValueError("stream chunk discard requires chunk_id")
        else:
            pass
        data_ref = self.data_ref_from_message(msg)
        relay = self.comm.relay(data_ref.transport)
        try:
            await self.comm.read_stream_chunk(relay=relay, data_ref=data_ref)
        except Exception as exc:
            logger.debug(
                "Stage %s: failed to drain aborted stream chunk for %s",
                self.name,
                msg.request_id,
                exc_info=True,
            )
            await self.send_data_ack(
                msg, data_ref, success=False, error=error_text(exc)
            )
            return
        await self.send_data_ack(msg, data_ref, success=True)

    async def on_stream_signal(
        self,
        msg: DataReadyMessage,
        predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        await self.wait_for_receive_predecessor(predecessor)
        await self.receive_stream_signal(
            msg.request_id,
            msg.from_stage,
            is_done=msg.is_done,
            error=msg.error,
        )

    async def receive_stream_signal(
        self,
        request_id: str,
        from_stage: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        if request_id in self.aborted:
            return
        else:
            pass
        self.active_requests.add(request_id)
        if error:
            await self.queue_stream_error(
                request_id,
                from_stage,
                RuntimeError(error),
            )
            return
        else:
            pass

        if is_done:
            if not self.open_pre_payload_stream_if_allowed(request_id):
                with suppress(Exception):
                    self.scheduler.abort(request_id)
                await self.send_failure(
                    request_id,
                    (
                        f"Stage {self.name}: stream_done from {from_stage!r} "
                        "arrived before the request payload, but this stage is not "
                        "configured to accept pre-payload stream data"
                    ),
                )
                return
            else:
                pass
            self.stream_queue.put_done(
                request_id, from_stage=self.logical_source(from_stage)
            )
            self.scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_done",
                )
            )
        else:
            pass

    def open_pre_payload_stream_if_allowed(self, request_id: str) -> bool:
        if self.stream_queue is None:
            return False
        else:
            pass
        if self.stream_queue.has(request_id):
            return True
        else:
            pass
        if not self.can_accept_stream_before_payload:
            return False
        else:
            pass
        self.active_requests.add(request_id)
        self.stream_queue.open(request_id)
        return True

    def route_stream_item(self, request_id: str, item: StreamItem) -> None:
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="stream_chunk", data=item)
        )

    async def execute(self, payload: Any) -> None:
        request_id = payload.request_id
        if request_id in self.aborted:
            return
        else:
            pass
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_dispatch",
        )
        if (
            self.role == "leader"
            and self.tp_fanout is not None
            and getattr(self.scheduler, "requires_tp_work_fanout", False)
        ):
            self.tp_fanout.fanout_work(payload)
        else:
            pass
        msg = IncomingMessage(request_id=request_id, type="new_request", data=payload)
        enqueue = getattr(self.scheduler, "enqueue", None)
        if enqueue is not None:
            enqueue(msg)
        else:
            self.scheduler.inbox.put(msg)

    async def on_admin(self, msg: AdminMessage) -> None:
        operation = msg.operation
        if self.role == "leader" and self.tp_fanout is not None:
            local = await self.run_admin_operation(operation)
            try:
                follower_msgs = await self.tp_fanout.collect_admin_results(
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
        else:
            pass

        result = await self.run_admin_operation(operation)
        await self.control_plane.send_admin_result(AdminResultMessage(result))

    async def run_admin_operation(self, operation: Any) -> AdminResult:
        try:
            handler = getattr(self.scheduler, "admin", None)
            if handler is None:
                return self.admin_result(
                    operation,
                    success=True,
                    message="stage does not support admin operations",
                    data={"skipped": True, "unsupported": True},
                )
            else:
                pass
            action = operation.action
            payload = dict(operation.payload)
            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(None, lambda: handler(action, payload))
            if inspect.isawaitable(outcome):
                outcome = await outcome
            else:
                pass
            return self.admin_result_from_outcome(operation, outcome)
        except Exception as exc:
            logger.exception(
                "Stage %s admin operation failed: action=%s",
                self.name,
                getattr(operation, "action", None),
            )
            return self.admin_result(
                operation,
                success=False,
                message=str(exc),
                error=str(exc),
            )

    def admin_result_from_outcome(self, operation: Any, outcome: Any) -> AdminResult:
        if isinstance(outcome, AdminResult):
            return outcome
        else:
            pass
        if isinstance(outcome, dict):
            data = dict(outcome.get("data") or {})
            for key, value in outcome.items():
                if key not in {"success", "message", "data", "error"}:
                    data.setdefault(key, value)
                else:
                    pass
            return self.admin_result(
                operation,
                success=bool(outcome.get("success", True)),
                message=str(outcome.get("message") or "ok"),
                data=data,
                error=outcome.get("error"),
            )
        else:
            pass
        return self.admin_result(
            operation,
            success=True,
            message="ok",
            data={"result": outcome},
        )

    def admin_result(
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

    async def drain_outbox(self) -> None:
        if self.owns_external_io:
            await self.drain_outbox_external()
        else:
            await self.drain_outbox_follower()

    async def drain_outbox_external(self) -> None:
        """Drain scheduler outbox and route results downstream."""
        loop = asyncio.get_running_loop()
        outbox = self.scheduler.outbox
        while self.running or not outbox.empty():
            try:
                out = await loop.run_in_executor(None, lambda: outbox.get(timeout=0.1))
            except _queue_mod.Empty:
                continue

            for batch_index in range(_OUTBOX_DRAIN_BATCH_SIZE):
                if out.type == "admitted":
                    if out.request_id not in self.aborted:
                        self.record_replica_bindings(
                            out.request_id, (out.metadata or {}).get("replica_bindings")
                        )
                        self.active_requests.add(out.request_id)
                    else:
                        pass
                elif out.type == "kv_transfer":
                    if out.request_id in self.active_requests:
                        self.launch_kv_transfer(out.data)
                    else:
                        self.discard_kv_transfer(out.data)
                elif out.request_id in self.active_requests:
                    if out.type == "result":
                        await self.route_result(out.request_id, out.data)
                    elif out.type == "stream":
                        if out.target is None:
                            if self.stream_targets:
                                await asyncio.gather(
                                    *(
                                        self.send_stream_to_target(
                                            out.request_id,
                                            out.data,
                                            target,
                                            out.metadata,
                                        )
                                        for target in self.stream_targets
                                    )
                                )
                            else:
                                await self.send_stream_to_coordinator(
                                    out.request_id,
                                    out.data,
                                    out.metadata,
                                )
                        else:
                            await self.send_stream_to_target(
                                out.request_id,
                                out.data,
                                out.target,
                                out.metadata,
                            )
                    elif out.type == "error":
                        await self.send_failure(out.request_id, str(out.data))
                    else:
                        pass
                else:
                    pass

                if batch_index + 1 >= _OUTBOX_DRAIN_BATCH_SIZE:
                    await asyncio.sleep(0)
                    break
                else:
                    pass

                try:
                    out = outbox.get_nowait()
                except _queue_mod.Empty:
                    break

    async def drain_outbox_follower(self) -> None:
        """Drain follower outbox without emitting external stage traffic."""
        loop = asyncio.get_running_loop()
        while self.running or not self.scheduler.outbox.empty():
            try:
                out = await loop.run_in_executor(
                    None, lambda: self.scheduler.outbox.get(timeout=0.1)
                )
            except _queue_mod.Empty:
                continue

            if out.type == "result":
                self.clear_request_state(out.request_id)
            elif out.type == "stream":
                continue
            elif out.type == "admitted":
                self.active_requests.add(out.request_id)
            elif out.type == "kv_transfer":
                raise RuntimeError(
                    f"TP follower stage {self.name} cannot publish a KV transfer"
                )
            elif out.type == "error":
                raise RuntimeError(
                    f"TP follower stage {self.name} received scheduler error: {out.data}"
                )
            else:
                pass

    def launch_kv_transfer(self, transfer: KVPageTransfer) -> None:
        started = False

        async def send():
            nonlocal started
            started = True
            await self.send_kv_transfer(transfer)

        def done(task):
            if not started:
                self.discard_kv_transfer(transfer)
            else:
                pass
            self.receive_tasks.discard(task)
            self.on_background_task_done(task, f"KV transfer {transfer.request_id}")

        task = asyncio.create_task(send())
        self.receive_tasks.add(task)
        task.add_done_callback(done)

    async def send_kv_transfer(self, transfer: KVPageTransfer) -> None:
        if not isinstance(transfer, KVPageTransfer):
            raise TypeError(
                "kv_transfer outbox messages require KVPageTransfer data, got "
                f"{type(transfer).__name__}"
            )
        else:
            pass
        lease = transfer.lease
        try:
            if transfer.request_id in self.aborted:
                return
            else:
                pass
            to_stage = self.resolve_target_instance(
                transfer.request_id, transfer.to_stage
            )
            target_pool_id = (
                transfer.target_pool_id
                if to_stage == transfer.to_stage
                else f"{to_stage}:kv"
            )
            metadata = {
                **transfer.metadata,
                "replica_bindings": self.replica_bindings.get(transfer.request_id),
            }
            # From here CommEngine owns the lease, including cancellation and
            # copies retained while their remote completion is uncertain.
            lease = None
            await self.comm.send_kv_pages(
                request_id=transfer.request_id,
                source_pool_id=transfer.source_pool_id,
                source_page_indices=transfer.source_page_indices,
                target_pool_id=target_pool_id,
                to_stage=to_stage,
                metadata=metadata,
                transfer_id=transfer.transfer_id,
                lease=transfer.lease,
            )
        except KVTransferCancelled:
            # Request cleanup is terminal for this transfer, but not for the
            # stage's long-lived outbox drain.
            return
        except Exception as exc:
            logger.exception(
                "Stage %s KV transfer failed for %s",
                self.name,
                transfer.request_id,
            )
            await self.send_failure(transfer.request_id, error_text(exc))
            return
        finally:
            if lease is not None:
                lease.release()
            else:
                pass
            self.clear_request_state(transfer.request_id)

    @staticmethod
    def discard_kv_transfer(transfer: Any) -> None:
        if isinstance(transfer, KVPageTransfer) and transfer.lease is not None:
            transfer.lease.release()
        else:
            pass

    async def route_result(self, request_id: str, result: Any) -> None:
        """Route a completed result to next stage(s) or complete at coordinator."""
        if not self.owns_external_io:
            self.clear_request_state(request_id)
            return
        else:
            pass
        session_operation = (
            find_session_operation(result.request.metadata)
            if isinstance(result, StagePayload)
            else None
        )
        if session_operation is not None and session_operation.operation != "append":
            await self.control_plane.send_complete(
                CompleteMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    success=True,
                    result=result.data,
                )
            )
            self.clear_request_state(request_id)
            return
        else:
            pass
        if session_operation is not None:
            owners = session_operation.stages
            if self.name not in owners or self.stream_targets:
                await self.send_failure(
                    request_id, "session route must use fixed, linear payload edges"
                )
                return
            else:
                pass
            index = owners.index(self.name)
            if index + 1 < len(owners):
                expected = self.logical_source(owners[index + 1])
            else:
                expected = None
            actual = self.get_next(request_id, result)
            if isinstance(actual, list) and len(actual) == 1:
                actual_target = actual[0]
            else:
                actual_target = actual
            if actual_target != expected:
                await self.send_failure(
                    request_id, "session route differs from the stage payload route"
                )
                return
            else:
                pass
        else:
            pass
        # Send stream done to the active stream targets for this request.
        stream_targets = self.stream_targets
        if self.get_stream_done_targets is not None:
            resolved = self.get_stream_done_targets(request_id, result)
            if isinstance(resolved, str):
                stream_targets = [resolved]
            elif isinstance(resolved, list):
                stream_targets = resolved
            elif resolved is None:
                stream_targets = []
            else:
                pass
        else:
            pass
        stream_targets_for_request = set(stream_targets)
        for target in stream_targets:
            await self.send_stream_signal_to_target(
                request_id,
                target,
                is_done=True,
            )

        next_stages = (
            self.get_next(request_id, result) if session_operation is None else actual
        )
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
            else:
                pass
            is_single_target = len(next_stages) == 1
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": False, "next": list(next_stages)},
            )
            for target in next_stages:
                await self.send_to_stage(
                    request_id,
                    target,
                    result,
                    allow_local_object=is_single_target,
                    allow_projected_local_object=not is_single_target,
                    stream_targets_for_request=stream_targets_for_request,
                )

        self.clear_request_state(request_id)

    async def send_to_stage(
        self,
        request_id: str,
        target: str,
        payload: Any,
        *,
        allow_local_object: bool = False,
        allow_projected_local_object: bool = False,
        stream_targets_for_request: set[str] | None = None,
    ) -> None:
        if not self.owns_external_io:
            raise RuntimeError(
                f"Follower stage {self.name} cannot send downstream data"
            )
        else:
            pass
        target = self.resolve_target_instance(request_id, target)
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for target {target!r}"
            )
        else:
            pass
        projector = self.project_payload.get(target)
        projected_payload = projector(payload) if projector is not None else payload
        use_local_object = allow_local_object or (
            allow_projected_local_object
            and self.is_isolated_projected_payload(
                payload,
                projected_payload,
                projector_present=projector is not None,
            )
        )

        if (
            use_local_object
            and target in self.same_process_targets
            and self.can_send_full_payload_locally(
                request_id,
                target,
                (
                    set(self.stream_targets)
                    if stream_targets_for_request is None
                    else stream_targets_for_request
                ),
            )
        ):
            if self.local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process target {target!r} requires "
                    "a local dispatcher"
                )
            else:
                pass

            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_hop_sent",
                metadata={"to_stage": target, "transport": "local_object"},
            )
            await self.local_dispatcher.send_payload(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                payload=projected_payload,
                replica_bindings=self.replica_bindings.get(request_id),
            )
            return
        else:
            pass

        can_use_direct_cuda_ipc = self.comm.router.can_use_direct_cuda_ipc(target)
        if (
            not self.disable_direct_cuda_ipc_payload
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
                    pass
            else:
                await self.control_plane.send_to_stage(
                    target,
                    endpoint,
                    DataReadyMessage(
                        request_id=request_id,
                        from_stage=self.name,
                        to_stage=target,
                        data_ref=direct_ref,
                        replica_bindings=self.replica_bindings.get(request_id),
                    ),
                )
                _emit_event(
                    request_id=request_id,
                    stage=self.name,
                    event_name="stage_hop_sent",
                    metadata={"to_stage": target, "transport": "torch_cuda_ipc"},
                )
                return
        else:
            pass

        transport_kind, relay = self.comm.router.relay_for_payload(
            target, projected_payload
        )
        await self.comm.send_payload(
            relay=relay,
            control_plane=self.control_plane,
            request_id=request_id,
            payload=projected_payload,
            transport=transport_kind,
            from_stage=self.name,
            to_stage=target,
            target_endpoint=endpoint,
            replica_bindings=self.replica_bindings.get(request_id),
        )
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_hop_sent",
            metadata={"to_stage": target, "transport": transport_kind.value},
        )

    @staticmethod
    def is_isolated_projected_payload(
        original_payload: Any,
        projected_payload: Any,
        *,
        projector_present: bool,
    ) -> bool:
        if not projector_present or projected_payload is original_payload:
            return False
        else:
            pass
        if not isinstance(original_payload, StagePayload):
            raise TypeError(
                "projected local-object dispatch requires the original payload "
                f"to be StagePayload, got {type(original_payload).__name__}"
            )
        else:
            pass
        if not isinstance(projected_payload, StagePayload):
            raise TypeError(
                "projected local-object dispatch requires projectors to return "
                f"StagePayload, got {type(projected_payload).__name__}"
            )
        else:
            pass
        if projected_payload.data is original_payload.data:
            return False
        else:
            pass
        return not Stage.shares_mutable_container(
            original_payload.data, projected_payload.data
        )

    @staticmethod
    def shares_mutable_container(original: Any, projected: Any) -> bool:
        original_ids = Stage.collect_mutable_container_ids(original)
        if not original_ids:
            return False
        else:
            pass
        return Stage.contains_mutable_container_id(projected, original_ids)

    @staticmethod
    def collect_mutable_container_ids(
        obj: Any, seen: set[int] | None = None
    ) -> set[int]:
        seen = set() if seen is None else seen
        obj_id = id(obj)
        if obj_id in seen:
            return set()
        else:
            pass
        seen.add(obj_id)

        ids: set[int] = set()
        if isinstance(obj, (dict, list, set, bytearray)):
            ids.add(obj_id)
        else:
            pass

        for child in Stage.iter_container_children(obj):
            ids.update(Stage.collect_mutable_container_ids(child, seen))
        return ids

    @staticmethod
    def contains_mutable_container_id(
        obj: Any, original_ids: set[int], seen: set[int] | None = None
    ) -> bool:
        seen = set() if seen is None else seen
        obj_id = id(obj)
        if obj_id in seen:
            return False
        else:
            pass
        seen.add(obj_id)

        if isinstance(obj, (dict, list, set, bytearray)) and obj_id in original_ids:
            return True
        else:
            pass
        return any(
            Stage.contains_mutable_container_id(child, original_ids, seen)
            for child in Stage.iter_container_children(obj)
        )

    @staticmethod
    def iter_container_children(obj: Any):
        if isinstance(obj, dict):
            return obj.values()
        else:
            pass
        if isinstance(obj, (list, tuple, set, frozenset)):
            return obj
        else:
            pass
        return ()

    def can_send_full_payload_locally(
        self,
        request_id: str,
        target: str,
        stream_targets_for_request: set[str],
    ) -> bool:
        if target in self.nonlocal_stream_targets.get(request_id, set()):
            return False
        else:
            pass
        if target not in stream_targets_for_request:
            return True
        else:
            pass
        return target in self.local_stream_targets.get(request_id, set())

    def record_local_stream_target(self, request_id: str, target: str) -> None:
        self.local_stream_targets.setdefault(request_id, set()).add(target)

    def record_nonlocal_stream_target(self, request_id: str, target: str) -> None:
        self.nonlocal_stream_targets.setdefault(request_id, set()).add(target)

    async def send_stream_to_target(
        self,
        request_id: str,
        data: Any,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.owns_external_io:
            return
        else:
            pass
        target = self.resolve_target_instance(request_id, target)
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for stream target "
                f"{target!r}"
            )
        else:
            pass
        key = (request_id, target)
        chunk_id = self.stream_chunk_counters.get(key, 0)
        self.stream_chunk_counters[key] = chunk_id + 1
        chunk_modality = (
            metadata.get("modality") if isinstance(metadata, dict) else None
        )
        if request_id not in self.first_stream_chunk_seen:
            self.first_stream_chunk_seen.add(request_id)
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_first_stream_chunk_sent",
                metadata={"to_stage": target, "modality": chunk_modality},
            )
        else:
            pass
        if target in self.same_process_targets:
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
            if self.local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process stream target {target!r} "
                    "requires a local dispatcher"
                )
            else:
                pass
            self.record_local_stream_target(request_id, target)
            await self.local_dispatcher.send_stream_chunk(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                chunk_id=chunk_id,
                data=data,
                metadata=metadata,
                replica_bindings=self.replica_bindings.get(request_id),
            )
            return
        else:
            pass
        self.record_nonlocal_stream_target(request_id, target)
        metadata = stage_io.strip_process_local_metadata(metadata)
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "relay-backed stream chunks must be torch.Tensor, got "
                f"{type(data).__name__}"
            )
        else:
            pass
        if data.is_cuda and self.comm.router.can_use_direct_cuda_ipc(target):
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
            # This branch skips the stream send worker and router.outbound_stream,
            # so it records both the transport it chose and the bytes it sent.
            self.comm.router.note_transport_choice("stream", target, "torch_cuda_ipc")
            _comm_trace(
                "comm_stream_send",
                request_id=request_id,
                from_stage=self.name,
                to_stage=target,
                chunk_id=chunk_id,
                transport="torch_cuda_ipc",
                bytes=data.nbytes,
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
                    replica_bindings=self.replica_bindings.get(request_id),
                ),
            )
            return
        else:
            pass
        inline_ref = stage_io.serialize_inline_stream_chunk(data, metadata)
        if inline_ref is not None:
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_stream_chunk_sent",
                metadata={
                    "to_stage": target,
                    "chunk_id": chunk_id,
                    "modality": chunk_modality,
                    "transport": "inline",
                },
            )
            self.comm.router.note_transport_choice("stream", target, "inline")
            _comm_trace(
                "comm_stream_send",
                request_id=request_id,
                from_stage=self.name,
                to_stage=target,
                chunk_id=chunk_id,
                transport="inline",
                bytes=data.nbytes,
            )
            await self.control_plane.send_to_stage(
                target,
                endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    to_stage=target,
                    data_ref=inline_ref,
                    chunk_id=chunk_id,
                    replica_bindings=self.replica_bindings.get(request_id),
                ),
            )
            return
        else:
            pass
        transport_kind, relay = self.comm.router.relay_for_stream(target, data)
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
        await self.comm.send_stream_chunk(
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
            replica_bindings=self.replica_bindings.get(request_id),
        )

    async def send_stream_signal_to_target(
        self,
        request_id: str,
        target: str,
        *,
        is_done: bool = False,
        error: str | None = None,
    ) -> None:
        if not self.owns_external_io:
            return
        else:
            pass
        target = self.resolve_target_instance(request_id, target)
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            raise RuntimeError(
                f"Stage {self.name}: no endpoint configured for stream target "
                f"{target!r}"
            )
        else:
            pass
        if target in self.same_process_targets:
            if self.local_dispatcher is None:
                raise RuntimeError(
                    f"Stage {self.name}: same-process stream target {target!r} "
                    "requires a local dispatcher"
                )
            else:
                pass
            self.record_local_stream_target(request_id, target)
            await self.local_dispatcher.send_stream_signal(
                from_stage=self.name,
                to_stage=target,
                request_id=request_id,
                is_done=is_done,
                error=error,
                replica_bindings=self.replica_bindings.get(request_id),
            )
            return
        else:
            pass
        self.record_nonlocal_stream_target(request_id, target)
        await stage_io.send_stream_signal(
            self.control_plane,
            request_id=request_id,
            target_stage=target,
            target_endpoint=endpoint,
            from_stage=self.name,
            is_done=is_done,
            error=error,
            replica_bindings=self.replica_bindings.get(request_id),
        )

    async def send_stream_to_coordinator(
        self,
        request_id: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Forward a terminal stage's stream chunk to the Coordinator."""
        if not self.is_terminal:
            raise RuntimeError(
                f"Stage {self.name!r} emitted untargeted stream chunk but isn't "
                "terminal. Set ``terminal=True``, or use ``target=...`` / "
                "``stream_to=[...]``."
            )
        else:
            pass
        if not self.owns_external_io:
            return
        else:
            pass
        if request_id in self.aborted:
            return
        else:
            pass
        modality = metadata.get("modality") if isinstance(metadata, dict) else None
        if modality is None and isinstance(data, dict):
            modality = data.get("modality")
        else:
            pass
        key = (request_id, "coordinator")
        chunk_id = self.stream_chunk_counters.get(key, 0)
        self.stream_chunk_counters[key] = chunk_id + 1
        msg = StreamMessage(
            request_id=request_id,
            from_stage=self.name,
            chunk=data,
            stage_name=self.name,
            modality=modality,
            chunk_id=chunk_id,
        )
        if request_id not in self.first_stream_chunk_seen:
            self.first_stream_chunk_seen.add(request_id)
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
        else:
            pass
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

    async def send_failure(self, request_id: str, error: str) -> None:
        self.record_aborted_request_id(request_id)
        if not self.owns_external_io:
            self.clear_request_state(request_id)
            raise RuntimeError(f"Follower stage {self.name} failed: {error}")
        else:
            pass
        await self.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.name,
                success=False,
                error=error,
            )
        )
        self.clear_request_state(request_id)

    def clear_request_state(self, request_id: str) -> None:
        self.active_requests.discard(request_id)
        self.input_handler.cancel(request_id)
        if self.stream_queue is not None:
            self.stream_queue.close(request_id)
        else:
            pass
        stale_keys = [key for key in self.stream_chunk_counters if key[0] == request_id]
        for key in stale_keys:
            self.stream_chunk_counters.pop(key, None)
        self.first_stream_chunk_seen.discard(request_id)
        self.local_stream_targets.pop(request_id, None)
        self.nonlocal_stream_targets.pop(request_id, None)
        self.replica_bindings.pop(request_id, None)

    async def handle_scheduler_crash(self, exc: BaseException) -> None:
        if self.scheduler_crash_error is not None:
            return
        else:
            pass
        self.scheduler_crash_error = exc
        if not self.owns_external_io:
            self.control_plane.close()
            return
        else:
            pass
        error = f"scheduler crashed: {exc}"
        active_request_ids = [
            request_id
            for request_id in list(self.active_requests)
            if request_id not in self.aborted
        ]
        for request_id in active_request_ids:
            with suppress(Exception):
                self.scheduler.abort(request_id)
            await self.send_failure(request_id, error)
            with suppress(Exception):
                self.comm.cleanup(request_id)
        self.control_plane.close()

    async def abort_listener(self) -> None:
        try:
            while self.running:
                abort_msg = await self.control_plane.recv_abort()
                if self.role == "leader" and self.tp_fanout is not None:
                    await self.tp_fanout.fanout_abort(abort_msg)
                else:
                    pass
                self.on_abort(abort_msg.request_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            if self.scheduler_crash_error is None and self.running:
                logger.exception("Stage %s abort listener crashed", self.name)
            else:
                pass

    def record_aborted_request_id(self, request_id: str) -> None:
        self.record_bounded_request_id(self.aborted, request_id)

    @staticmethod
    def record_bounded_request_id(ids: set[str], request_id: str) -> None:
        ids.add(request_id)
        if len(ids) > 10000:
            excess = len(ids) - 5000
            it = iter(ids)
            to_remove = [next(it) for _ in range(excess)]
            ids -= set(to_remove)
        else:
            pass

    def on_abort(self, request_id: str) -> None:
        self.record_aborted_request_id(request_id)
        self.comm.cleanup(request_id)
        self.clear_request_state(request_id)
        self.scheduler.abort(request_id)

    def on_profiler_start(self, msg: ProfilerStartMessage) -> None:
        run_id = msg.run_id
        if msg.enable_torch and not TorchProfiler.is_active():
            base_tpl = msg.trace_path_template.format(run_id=run_id, stage=self.name)
            template = f"{base_tpl}_pid{os.getpid()}"
            prof_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
            if prof_dir and not os.path.isabs(template):
                template = os.path.join(prof_dir, template)
            else:
                pass
            TorchProfiler.start(template, run_id=run_id)
        else:
            pass
        if msg.event_dir is not None and self.owns_external_io:
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
        else:
            pass

    def on_profiler_stop(self, msg: ProfilerStopMessage) -> None:
        # run_id=None is a wildcard (stop whatever's active).
        if TorchProfiler.is_active() and (
            msg.run_id is None or TorchProfiler.get_active_run_id() == msg.run_id
        ):
            TorchProfiler.stop(run_id=msg.run_id)
        else:
            pass
        recorder = _get_recorder()
        if recorder.is_active() and (
            msg.run_id is None or recorder.active_run_id() == msg.run_id
        ):
            recorder.stop(run_id=msg.run_id)
        else:
            pass

    def on_background_task_done(self, task: asyncio.Task, label: str) -> None:
        if task.cancelled():
            return
        else:
            pass
        exc = task.exception()
        if exc is None:
            return
        else:
            pass
        if isinstance(exc, KVTransferRejected):
            # The ACK watcher also propagates this to _send_kv_transfer(), which
            # reports the request failure even if the local abort arrives later.
            return
        else:
            pass
        logger.exception(
            "Stage %s %s task crashed",
            self.name,
            label,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        if self.background_task_error is None:
            self.background_task_error = exc
        else:
            pass
        self.running = False
        self.control_plane.close()

    def info(self) -> StageInfo:
        return StageInfo(
            name=self.name,
            control_endpoint=self.control_plane.recv_endpoint,
        )
