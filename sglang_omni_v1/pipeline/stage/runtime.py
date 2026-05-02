# SPDX-License-Identifier: Apache-2.0
"""Stage — IO shell for pipeline processing.

Handles: control plane messaging, data plane (relay) IO, input aggregation,
stream chunk routing, abort tracking, profiling.

Dispatches all compute to scheduler (OmniScheduler or SimpleScheduler).
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue as _queue_mod
import threading
from contextlib import suppress
from typing import Any, Callable, Literal

from sglang_omni_v1.pipeline import relay_io
from sglang_omni_v1.pipeline.stage.input import DirectInput, InputHandler
from sglang_omni_v1.pipeline.stage.stream_queue import (
    StreamItem,
    StreamQueue,
    StreamSignal,
)
from sglang_omni_v1.pipeline.tp_control import TPLeaderFanout
from sglang_omni_v1.profiler.torch_profiler import TorchProfiler
from sglang_omni_v1.proto import (
    CompleteMessage,
    DataReadyMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
    StageInfo,
    SubmitMessage,
)
from sglang_omni_v1.relay.base import Relay, create_relay
from sglang_omni_v1.scheduling.messages import IncomingMessage

logger = logging.getLogger(__name__)

GetNextFn = Callable[[str, Any], str | list[str] | None]


class Stage:
    """IO shell for one pipeline stage.

    For AR stages: owns a Scheduler (runs in dedicated thread),
    communicates via inbox/outbox queues.

    For simple stages: calls compute_fn directly.
    """

    def __init__(
        self,
        name: str,
        role: Literal["single", "leader", "follower"],
        get_next: GetNextFn,
        gpu_id: int | None,
        endpoints: dict[str, str],
        control_plane: Any,
        input_handler: InputHandler | None = None,
        relay: Relay | None = None,
        relay_config: dict[str, Any] | None = None,
        scheduler: Any = None,
        project_payload: dict[str, Callable[[Any], Any]] | None = None,
        stream_targets: list[str] | None = None,
        same_gpu_targets: set[str] | None = None,
        tp_fanout: TPLeaderFanout | None = None,
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
        self._same_gpu_targets = same_gpu_targets or set()
        self._tp_fanout = tp_fanout
        self._owns_external_io = role in {"single", "leader"}

        # --- Relay ---
        if relay is not None:
            self.relay = relay
        else:
            config = relay_config or {}
            engine_id = config.get("worker_id", f"{name}_relay")
            relay_type = config.get("relay_type", "nixl").lower()
            gpu_id = config.get("gpu_id")
            if gpu_id is not None:
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
                if relay_type == "nccl":
                    device = "cuda"
            self.relay = create_relay(
                relay_type,
                engine_id=engine_id,
                slot_size_mb=config.get("slot_size_mb", 64),
                credits=config.get("credits", 2),
                device=device,
                rank=config.get("rank"),
                world_size=config.get("world_size"),
                send_to_ranks=config.get("send_to_ranks", []),
                recv_from_ranks=config.get("recv_from_ranks", []),
            )

        # --- State ---
        self._running = False
        self._aborted: set[str] = set()
        self._active_requests: set[str] = set()
        self._stream_queue: StreamQueue | None = None
        self._pending_stream_data: dict[str, list[StreamItem | StreamSignal]] = {}
        self._stream_chunk_counters: dict[tuple[str, str], int] = {}
        self._scheduler_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._scheduler_crash_error: BaseException | None = None
        self._background_task_error: BaseException | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        await self.control_plane.start()
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Start scheduler in dedicated thread
        if self.scheduler is not None:

            def _run_scheduler():
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
        if self.scheduler is not None:
            self.scheduler.stop()
        self.control_plane.close()
        if self._tp_fanout is not None:
            self._tp_fanout.close()
        self.relay.close()
        logger.info("Stage %s stopped", self.name)

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
                        (ShutdownMessage, ProfilerStartMessage, ProfilerStopMessage),
                    )
                ):
                    await self._tp_fanout.fanout_control(msg)
                if isinstance(msg, ShutdownMessage):
                    break
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

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: Any) -> None:
        if isinstance(msg, SubmitMessage):
            await self._on_submit(msg)
        elif isinstance(msg, DataReadyMessage):
            if msg.is_done or msg.error:
                self._on_stream_signal(msg)
            elif msg.chunk_id is not None:
                await self._on_stream_chunk(msg)
            else:
                await self._on_data_ready(msg)
        elif isinstance(msg, ProfilerStartMessage):
            self._on_profiler_start(msg)
        elif isinstance(msg, ProfilerStopMessage):
            self._on_profiler_stop(msg)

    async def _on_submit(self, msg: SubmitMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        payload = msg.data  # StagePayload from coordinator
        await self._execute(payload)

    async def _on_data_ready(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            self.relay.cleanup(request_id)
            return
        self._active_requests.add(request_id)
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        # Read payload from relay
        try:
            payload = await relay_io.read_payload(
                self.relay, request_id, msg.shm_metadata
            )
        except Exception as exc:
            logger.exception(
                "Stage %s: relay read failed for %s", self.name, request_id
            )
            self.relay.cleanup(request_id)
            await self._send_failure(request_id, f"relay read failed: {exc}")
            return

        # Input aggregation
        merged = self.input_handler.receive(request_id, msg.from_stage, payload)
        if merged is not None:
            await self._execute(merged)
            # Flush any stream data that arrived before this request was ready
            for pending in self._pending_stream_data.pop(request_id, []):
                if self._stream_queue is not None:
                    if isinstance(pending, StreamItem):
                        self._stream_queue.put(request_id, pending)
                        self._route_stream_item(request_id, pending)
                    elif isinstance(pending, StreamSignal):
                        if pending.error:
                            self._stream_queue.put_error(request_id, pending.error)
                        elif pending.is_done:
                            self._stream_queue.put_done(
                                request_id, from_stage=pending.from_stage
                            )
                            self.scheduler.inbox.put(
                                IncomingMessage(
                                    request_id=request_id,
                                    type="stream_done",
                                )
                            )

    async def _on_stream_chunk(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)

        # Same-GPU CUDA IPC
        if isinstance(msg.shm_metadata, dict) and msg.shm_metadata.get("_ipc"):
            try:
                item = self._deserialize_ipc_chunk(msg)
            except Exception as exc:
                logger.error(
                    "Stage %s: IPC deserialize failed for %s: %s",
                    self.name,
                    request_id,
                    exc,
                )
                self._queue_stream_error(request_id, msg.from_stage, exc)
                return
            self._route_stream_item(request_id, item)
            return

        # Cross-GPU: relay
        blob_key = f"{request_id}:stream:{msg.from_stage}:{msg.to_stage}:{msg.chunk_id}"
        try:
            data = await relay_io.read_blob(self.relay, blob_key, msg.shm_metadata)
            metadata = await self._read_chunk_metadata(msg.shm_metadata, blob_key)
        except Exception as exc:
            logger.error(
                "Stage %s: stream chunk read failed for %s: %s",
                self.name,
                request_id,
                exc,
            )
            self._queue_stream_error(request_id, msg.from_stage, exc)
            return

        item = StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata,
        )
        if self._stream_queue is None or not self._stream_queue.has(request_id):
            self._pending_stream_data.setdefault(request_id, []).append(item)
            return
        self._route_stream_item(request_id, item)

    def _queue_stream_error(
        self,
        request_id: str,
        from_stage: str | None,
        error: BaseException,
    ) -> None:
        if self._stream_queue is not None and self._stream_queue.has(request_id):
            self._stream_queue.put_error(
                request_id,
                error,
                from_stage=from_stage,
            )
            return
        self._pending_stream_data.setdefault(request_id, []).append(
            StreamSignal(from_stage=from_stage, error=error)
        )

    async def _read_chunk_metadata(
        self, shm_metadata: dict, blob_key: str
    ) -> dict | None:
        metadata = {}
        chunk_meta = (
            shm_metadata.get("chunk_metadata")
            if isinstance(shm_metadata, dict)
            else None
        )
        if isinstance(chunk_meta, dict):
            metadata.update(chunk_meta)
        tensor_blobs = (
            shm_metadata.get("chunk_metadata_tensors", {})
            if isinstance(shm_metadata, dict)
            else {}
        )
        if isinstance(tensor_blobs, dict):
            tensor_dict = {}
            for path, info in tensor_blobs.items():
                if not isinstance(info, dict):
                    continue
                meta_blob_key = info.get("blob_key")
                meta_metadata = info.get("relay_metadata")
                if isinstance(meta_blob_key, str) and isinstance(meta_metadata, dict):
                    tensor_dict[path] = await relay_io.read_blob(
                        self.relay, meta_blob_key, meta_metadata
                    )
            if tensor_dict:
                metadata = relay_io.restore_tensors(metadata, tensor_dict)
        return metadata or None

    def _on_stream_signal(self, msg: DataReadyMessage) -> None:
        request_id = msg.request_id
        if request_id in self._aborted:
            return
        self._active_requests.add(request_id)
        if self._stream_queue is None or not self._stream_queue.has(request_id):
            if msg.error:
                self._pending_stream_data.setdefault(request_id, []).append(
                    StreamSignal(
                        from_stage=msg.from_stage, error=RuntimeError(msg.error)
                    )
                )
            elif msg.is_done:
                self._pending_stream_data.setdefault(request_id, []).append(
                    StreamSignal(from_stage=msg.from_stage, is_done=True)
                )
            return
        if msg.error:
            self._stream_queue.put_error(request_id, RuntimeError(msg.error))
        elif msg.is_done:
            self._stream_queue.put_done(request_id, from_stage=msg.from_stage)
            self.scheduler.inbox.put(
                IncomingMessage(
                    request_id=request_id,
                    type="stream_done",
                )
            )

    @staticmethod
    def _deserialize_ipc_chunk(msg: DataReadyMessage) -> StreamItem:
        import pickle as _pickle

        ipc_meta = msg.shm_metadata
        data = _pickle.loads(ipc_meta["tensor_bytes"])
        metadata = {}
        raw_meta = ipc_meta.get("metadata", {})
        if isinstance(raw_meta, dict):
            for key, value in raw_meta.items():
                if isinstance(value, dict) and "_ipc_tensor" in value:
                    metadata[key] = _pickle.loads(value["_ipc_tensor"])
                else:
                    metadata[key] = value
        return StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata or None,
        )

    def _route_stream_item(self, request_id: str, item: StreamItem) -> None:
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="stream_chunk", data=item)
        )

    # ------------------------------------------------------------------
    # Dispatch: always through scheduler
    # ------------------------------------------------------------------

    async def _execute(self, payload: Any) -> None:
        request_id = payload.request_id
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
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
                    for target in self._stream_targets:
                        await self._send_stream_to_target(
                            out.request_id,
                            out.data,
                            target,
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

    # ------------------------------------------------------------------
    # Routing: send results to next stage(s) or coordinator
    # ------------------------------------------------------------------

    async def _route_result(self, request_id: str, result: Any) -> None:
        """Route a completed result to next stage(s) or complete at coordinator."""
        if not self._owns_external_io:
            self._clear_request_state(request_id)
            return
        # Send stream done to all stream targets
        for target in self._stream_targets:
            endpoint = self.endpoints.get(target)
            if endpoint:
                await relay_io.send_stream_signal(
                    self.control_plane,
                    request_id=request_id,
                    target_stage=target,
                    target_endpoint=endpoint,
                    from_stage=self.name,
                    is_done=True,
                )

        next_stages = self.get_next(request_id, result)
        if next_stages is None:
            # Terminal: notify coordinator
            await self.control_plane.send_complete(
                CompleteMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    success=True,
                    result=result.data,
                )
            )
        else:
            if isinstance(next_stages, str):
                next_stages = [next_stages]
            for target in next_stages:
                await self._send_to_stage(request_id, target, result)

        self._clear_request_state(request_id)

    async def _send_to_stage(self, request_id: str, target: str, payload: Any) -> None:
        if not self._owns_external_io:
            raise RuntimeError(
                f"Follower stage {self.name} cannot send downstream data"
            )
        endpoint = self.endpoints.get(target)
        if endpoint is None:
            logger.warning("Stage %s: no endpoint for %s", self.name, target)
            return
        projector = self._project_payload.get(target)
        projected_payload = projector(payload) if projector is not None else payload
        metadata, op = await relay_io.write_payload(
            self.relay, request_id, projected_payload
        )
        msg = DataReadyMessage(
            request_id=request_id,
            from_stage=self.name,
            to_stage=target,
            shm_metadata=metadata,
        )
        await self.control_plane.send_to_stage(target, endpoint, msg)
        await op.wait_for_completion()

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
            return
        key = (request_id, target)
        chunk_id = self._stream_chunk_counters.get(key, 0)
        self._stream_chunk_counters[key] = chunk_id + 1
        await relay_io.send_stream_chunk(
            self.relay,
            self.control_plane,
            request_id=request_id,
            data=data,
            target_stage=target,
            target_endpoint=endpoint,
            from_stage=self.name,
            chunk_id=chunk_id,
            metadata=metadata,
            same_gpu_targets=self._same_gpu_targets,
        )

    async def _send_failure(self, request_id: str, error: str) -> None:
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
        self._pending_stream_data.pop(request_id, None)
        stale_keys = [
            key for key in self._stream_chunk_counters if key[0] == request_id
        ]
        for key in stale_keys:
            self._stream_chunk_counters.pop(key, None)

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
                self.relay.cleanup(request_id)
        self.control_plane.close()

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

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

    def _on_abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        if len(self._aborted) > 10000:
            excess = len(self._aborted) - 5000
            it = iter(self._aborted)
            to_remove = [next(it) for _ in range(excess)]
            self._aborted -= set(to_remove)
        self._active_requests.discard(request_id)
        self.input_handler.cancel(request_id)
        self.relay.cleanup(request_id)
        if self._stream_queue is not None:
            self._stream_queue.close(request_id)
        self._pending_stream_data.pop(request_id, None)
        self.scheduler.abort(request_id)

    # ------------------------------------------------------------------
    # Profiler (kept from original)
    # ------------------------------------------------------------------

    def _on_profiler_start(self, msg: ProfilerStartMessage) -> None:
        if TorchProfiler.is_active():
            return
        run_id = msg.run_id
        base_tpl = msg.trace_path_template.format(run_id=run_id, stage=self.name)
        template = f"{base_tpl}_pid{os.getpid()}"
        prof_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
        if prof_dir and not os.path.isabs(template):
            template = os.path.join(prof_dir, template)
        TorchProfiler.start(template, run_id=run_id)

    def _on_profiler_stop(self, msg: ProfilerStopMessage) -> None:
        if (
            TorchProfiler.is_active()
            and TorchProfiler.get_active_run_id() == msg.run_id
        ):
            TorchProfiler.stop(run_id=msg.run_id)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

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
