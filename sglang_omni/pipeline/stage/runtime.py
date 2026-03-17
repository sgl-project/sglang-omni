# SPDX-License-Identifier: Apache-2.0
"""Stage abstraction for pipeline processing."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from operator import methodcaller
from typing import Any, Callable

from sglang_omni.pipeline.control_plane import StageControlPlane
from sglang_omni.pipeline.stage.input import DirectInput, InputHandler
from sglang_omni.pipeline.stage.router import WorkerRouter
from sglang_omni.pipeline.stage.stream_queue import (
    StreamItem,
    StreamQueue,
    StreamSignal,
)
from sglang_omni.pipeline.stage.work import InputRef
from sglang_omni.pipeline.worker.data_plane import DataPlaneAdapter, _restore_tensors
from sglang_omni.pipeline.worker.runtime import Worker
from sglang_omni.profiler.torch_profiler import TorchProfiler
from sglang_omni.proto import (
    DataReadyMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
    StageInfo,
    SubmitMessage,
)
from sglang_omni.relay.base import Relay, create_relay

logger = logging.getLogger(__name__)


# Type alias for get_next function.
# Returns: next stage name, list of next stages (fan-out), or None for END.
GetNextFn = Callable[[str, Any], str | list[str] | None]


class Stage:
    """A processing stage in the pipeline.

    Responsibilities:
    - Receive work (via control plane)
    - Handle input aggregation (via input_handler)
    - Queue work for workers
    - Workers process and route output
    """

    def __init__(
        self,
        name: str,
        get_next: GetNextFn,
        recv_endpoint: str,
        coordinator_endpoint: str,
        abort_endpoint: str,
        endpoints: dict[str, str],
        input_handler: InputHandler | None = None,
        relay: Relay | None = None,
        relay_config: dict[str, Any] | None = None,
    ):
        """Initialize a stage.

        Args:
            name: Stage name (unique identifier)
            get_next: Function to determine next stage
                      (request_id, output) -> stage_name, list of stage names, or None
            recv_endpoint: ZMQ endpoint to receive work
            coordinator_endpoint: ZMQ endpoint to send completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            endpoints: Dict of stage_name -> endpoint for routing
            input_handler: Input handler for aggregation (default: DirectInput)
            relay: Relay instance for data transfer (default: constructed from relay_config)
            relay_config: Configuration dict for Relay (if relay is None)
        """
        self.name = name
        self.get_next = get_next
        self.endpoints = endpoints
        self.input_handler = input_handler or DirectInput()

        self.control_plane = StageControlPlane(
            stage_name=name,
            recv_endpoint=recv_endpoint,
            coordinator_endpoint=coordinator_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # --- Relay Initialization ---
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
                    logger.info("NcclRelay using default CUDA device selection.")
                    device = "cuda"

            logger.info(
                "Initializing %s for stage %s (device=%s)", relay_type, name, device
            )

            relay_kwargs = {
                "engine_id": engine_id,
                "slot_size_mb": config.get("slot_size_mb", 64),
                "credits": config.get("credits", 2),
                "device": device,
                # NCCL-specific parameters (ignored by Shm/Nixl)
                "rank": config.get("rank"),
                "world_size": config.get("world_size"),
                "send_to_ranks": config.get("send_to_ranks", []),
                "recv_from_ranks": config.get("recv_from_ranks", []),
            }

            self.relay = create_relay(relay_type, **relay_kwargs)

        self.router = WorkerRouter()
        self._data_plane = DataPlaneAdapter(self.relay)

        # Workers
        self.workers: list[Worker] = []

        # State
        self._running = False
        self._aborted_requests: set[str] = set()
        self._stream_queue: StreamQueue | None = (
            None  # Set by compiler for streaming-receiving stages
        )
        self._pending_stream_data: dict[str, list[StreamItem | StreamSignal]] = {}

        # Profiler
        self._profiler_run_id: str | None = None
        self._profiler_trace_template: str | None = None

    async def _on_profiler_start(self, msg: ProfilerStartMessage) -> None:
        if TorchProfiler.is_active():
            active = TorchProfiler.get_active_run_id()
            if active == msg.run_id:
                logger.info(
                    "Stage %s profiler already active for run_id=%s",
                    self.name,
                    msg.run_id,
                )
                return
            logger.warning(
                "Stage %s profiler already active (run_id=%s), ignoring new profiler (run_id=%s)",
                self.name,
                active,
                msg.run_id,
            )
            return

        run_id = msg.run_id
        base_tpl = msg.trace_path_template.format(run_id=run_id, stage=self.name)
        template = f"{base_tpl}_pid{os.getpid()}"

        prof_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
        if prof_dir and not os.path.isabs(template):
            template = os.path.join(prof_dir, template)

        logger.info(
            "Stage %s starting torch profiler run_id=%s template=%s",
            self.name,
            run_id,
            template,
        )
        trace_path = TorchProfiler.start(template, run_id=msg.run_id)

        self._profiler_run_id = run_id
        self._profiler_trace_template = template
        logger.info(
            "Stage %s starting torch profiler run_id=%s template=%s expected_trace=%s",
            self.name,
            msg.run_id,
            template,
            trace_path,
        )

    async def _on_profiler_stop(self, msg: ProfilerStopMessage) -> None:
        if not TorchProfiler.is_active():
            logger.info(
                "Stage %s profiler not active; ignore stop run_id=%s",
                self.name,
                msg.run_id,
            )
            self._profiler_run_id = None
            self._profiler_trace_template = None
            return

        active = TorchProfiler.get_active_run_id()
        if active != msg.run_id:
            logger.warning(
                "Stage %s profiler active_run_id=%s; ignoring stop for run_id=%s",
                self.name,
                active,
                msg.run_id,
            )
            return

        logger.info("Stage %s stopping torch profiler run_id=%s", self.name, msg.run_id)
        result = TorchProfiler.stop(run_id=msg.run_id)
        logger.info(
            "Stage %s stopping torch profiler run_id=%s result=%s",
            self.name,
            msg.run_id,
            result,
        )

        self._profiler_run_id = None
        self._profiler_trace_template = None

    def add_worker(self, worker: Worker) -> None:
        """Add a worker to this stage."""
        queue = self.router.add_worker()
        worker.bind(self, queue)
        self.workers.append(worker)

    async def start(self) -> None:
        """Start the stage (idempotent — safe to call more than once)."""
        if self._running:
            return
        await self.control_plane.start()
        self._running = True
        logger.info("Stage %s started", self.name)

    async def stop(self) -> None:
        """Stop the stage."""
        self._running = False

        # Signal workers to stop
        for worker in self.workers:
            await worker.queue.put(None)

        self.control_plane.close()

        if hasattr(self.relay, "close"):
            self.relay.close()

        logger.info("Stage %s stopped", self.name)

    async def run(self) -> None:
        """Main loop: receive work, handle input, queue for workers."""
        await self.start()

        # Start workers
        worker_tasks = [asyncio.create_task(w.run()) for w in self.workers]

        # Start abort listener
        abort_task = asyncio.create_task(self._abort_listener())

        try:
            while self._running:
                # Receive work
                msg = await self.control_plane.recv()
                logger.info("Stage %s received msg: %s", self.name, type(msg).__name__)

                if isinstance(msg, ShutdownMessage):
                    logger.info("Stage %s received shutdown", self.name)
                    break

                await self._handle_message(msg)

        except asyncio.CancelledError:
            logger.info("Stage %s cancelled", self.name)
        except Exception as e:
            logger.error("Stage %s error: %s", self.name, e)
            raise
        finally:
            # Stop
            await self.stop()

            # Cancel abort listener
            abort_task.cancel()
            try:
                await abort_task
            except asyncio.CancelledError:
                pass

            # Wait for workers
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    async def _abort_listener(self) -> None:
        """Background task to listen for abort broadcasts."""
        try:
            while self._running:
                abort_msg = await self.control_plane.recv_abort()
                self._on_abort(abort_msg.request_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Stage %s abort listener error: %s", self.name, e)

    async def _handle_message(
        self,
        msg: (
            DataReadyMessage
            | SubmitMessage
            | ProfilerStartMessage
            | ProfilerStopMessage
        ),
    ) -> None:
        """Handle an incoming message."""
        if isinstance(msg, SubmitMessage):
            await self._process_submit(msg)
        elif isinstance(msg, DataReadyMessage):
            if msg.is_done or msg.error:
                self._handle_stream_signal(msg)
            elif msg.chunk_id is not None:
                await self._handle_stream_chunk(msg)
            else:
                await self._process_data_ready(msg)
        elif isinstance(msg, ProfilerStartMessage):
            await self._on_profiler_start(msg)
        elif isinstance(msg, ProfilerStopMessage):
            await self._on_profiler_stop(msg)
        else:
            logger.warning(
                "Stage %s received unexpected message: %s", self.name, type(msg)
            )

    async def _process_submit(self, msg: SubmitMessage) -> None:
        """Process initial submission from coordinator."""
        request_id = msg.request_id
        logger.debug("Stage %s received submit: req=%s", self.name, request_id)

        if request_id in self._aborted_requests:
            logger.debug("Stage %s skipping aborted req=%s", self.name, request_id)
            return

        # Open stream queue for this request if stage has one
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        input_ref = InputRef.from_payload("coordinator", msg.data)
        work = self.input_handler.receive(request_id, "coordinator", input_ref)
        if work is not None:
            self.router.enqueue(work)

    async def _process_data_ready(self, msg: DataReadyMessage) -> None:
        """Process data ready notification from previous stage.

        Eagerly reads relay data so the sender's credit is released immediately.
        This prevents timeouts when an AggregatedInput handler defers processing
        until all sources arrive.
        """
        request_id = msg.request_id
        logger.debug(
            "Stage %s received data_ready: req=%s from %s",
            self.name,
            request_id,
            msg.from_stage,
        )

        if request_id in self._aborted_requests:
            logger.debug("Stage %s skipping aborted req=%s", self.name, request_id)
            self.relay.cleanup(request_id)
            return

        # Open stream queue for this request if stage has one
        if self._stream_queue is not None and not self._stream_queue.has(request_id):
            self._stream_queue.open(request_id)

        # Eagerly read from relay to release sender's credit/notification.
        if msg.shm_metadata:
            try:
                payload = await self._data_plane.read_payload(
                    request_id, msg.shm_metadata
                )
                input_ref = InputRef.from_payload(msg.from_stage, payload)
            except Exception:
                logger.exception(
                    "Stage %s: eager relay read failed for req=%s from %s",
                    self.name,
                    request_id,
                    msg.from_stage,
                )
                input_ref = InputRef.from_metadata(msg.from_stage, msg.shm_metadata)
        else:
            input_ref = InputRef.from_metadata(msg.from_stage, msg.shm_metadata)

        work = self.input_handler.receive(request_id, msg.from_stage, input_ref)
        if work is not None:
            self.router.enqueue(work)
            # Flush pending stream data that arrived before this request was assigned
            pending_stream = self._pending_stream_data.pop(request_id, [])
            for pending in pending_stream:
                if self._stream_queue is not None:
                    if isinstance(pending, StreamItem):
                        self._stream_queue.put(request_id, pending)
                    elif isinstance(pending, StreamSignal):
                        if pending.error:
                            self._stream_queue.put_error(request_id, pending.error)
                        elif pending.is_done:
                            self._stream_queue.put_done(
                                request_id, from_stage=pending.from_stage
                            )

    async def _handle_stream_chunk(self, msg: DataReadyMessage) -> None:
        """Handle a streaming data chunk from an upstream stage."""
        request_id = msg.request_id
        if request_id in self._aborted_requests:
            return

        # ── Same-GPU CUDA IPC path: deserialize tensor directly (zero copy) ──
        if isinstance(msg.shm_metadata, dict) and msg.shm_metadata.get("_ipc"):
            try:
                item = self._deserialize_ipc_chunk(msg)
            except Exception as exc:
                logger.error(
                    "Stage %s: IPC stream chunk deserialize failed for %s: %s",
                    self.name,
                    request_id,
                    exc,
                )
                if self._stream_queue is not None and self._stream_queue.has(
                    request_id
                ):
                    self._stream_queue.put_error(request_id, exc)
                return
            self._route_stream_item(request_id, item)
            return

        # ── Cross-GPU: read from relay (existing path) ──
        # blob_key must match the sender format in worker/runtime.py _do_stream_send
        blob_key = f"{request_id}:stream:{msg.from_stage}:{msg.to_stage}:{msg.chunk_id}"
        try:
            data = await self._data_plane.read_blob(blob_key, msg.shm_metadata)

            # Restore metadata tensors from relay
            metadata: dict[str, Any] = {}
            chunk_metadata = (
                msg.shm_metadata.get("chunk_metadata")
                if isinstance(msg.shm_metadata, dict)
                else None
            )
            if isinstance(chunk_metadata, dict):
                metadata.update(chunk_metadata)
            metadata_tensor_blobs = (
                msg.shm_metadata.get("chunk_metadata_tensors", {})
                if isinstance(msg.shm_metadata, dict)
                else {}
            )
            if isinstance(metadata_tensor_blobs, dict):
                tensor_dict: dict[str, Any] = {}
                for path, info in metadata_tensor_blobs.items():
                    if not isinstance(path, str) or not isinstance(info, dict):
                        continue
                    meta_blob_key = info.get("blob_key")
                    meta_metadata = info.get("relay_metadata")
                    if not isinstance(meta_blob_key, str) or not isinstance(
                        meta_metadata, dict
                    ):
                        continue
                    tensor_dict[path] = await self._data_plane.read_blob(
                        meta_blob_key, meta_metadata
                    )
                if tensor_dict:
                    metadata = _restore_tensors(metadata, tensor_dict)
        except Exception as exc:
            logger.error(
                "Stage %s: stream chunk read failed for %s: %s",
                self.name,
                request_id,
                exc,
            )
            if self._stream_queue is not None and self._stream_queue.has(request_id):
                self._stream_queue.put_error(request_id, exc)
            return

        item = StreamItem(
            chunk_id=msg.chunk_id,
            data=data,
            from_stage=msg.from_stage,
            metadata=metadata or None,
        )
        self._route_stream_item(request_id, item)

    @staticmethod
    def _deserialize_ipc_chunk(msg: DataReadyMessage) -> StreamItem:
        """Deserialize a same-GPU CUDA IPC stream chunk from control-plane metadata.

        Uses ``pickle.loads`` which reconstructs CUDA tensors via
        ``cudaIpcOpenMemHandle`` — zero data copy, the returned tensor
        points directly to the sender's GPU memory.
        """
        import pickle as _pickle

        ipc_meta = msg.shm_metadata
        data = _pickle.loads(ipc_meta["tensor_bytes"])

        metadata: dict[str, Any] = {}
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
        """Route a stream item to the queue or buffer it if the worker is not assigned yet."""
        worker_idx = self.router.get_worker_index(request_id)
        if worker_idx is None:
            self._pending_stream_data.setdefault(request_id, []).append(item)
            logger.debug(
                "Stage %s: buffered early stream chunk for %s", self.name, request_id
            )
            return

        if self._stream_queue is not None:
            self._stream_queue.put(request_id, item)

    def _handle_stream_signal(self, msg: DataReadyMessage) -> None:
        """Handle a streaming EOS or error signal."""
        request_id = msg.request_id
        if request_id in self._aborted_requests:
            return
        if self._stream_queue is None or not self._stream_queue.has(request_id):
            # Buffer signal if queue not open yet
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

    def _on_abort(self, request_id: str) -> None:
        """Handle abort for a request."""
        logger.debug("Stage %s: aborting req=%s", self.name, request_id)
        self._aborted_requests.add(request_id)
        self.router.clear_request(request_id)
        self.input_handler.cancel(request_id)
        self.relay.cleanup(request_id)

        # Close stream queue and discard buffered stream data
        if self._stream_queue is not None:
            self._stream_queue.close(request_id)
        self._pending_stream_data.pop(request_id, None)

        # Notify workers' engines
        for worker in self.workers:
            asyncio.create_task(worker.executor.abort(request_id))

    def info(self) -> StageInfo:
        """Return stage info."""
        return StageInfo(
            name=self.name,
            control_endpoint=self.control_plane.recv_endpoint,
        )

    def health(self) -> dict[str, Any]:
        """Return health status."""
        relay_health: dict[str, Any] = {"status": "error"}
        with suppress(Exception):
            relay_health = methodcaller("health")(self.relay)

        return {
            "name": self.name,
            "running": self._running,
            "queue_size": self.router.queue_size(),
            "num_workers": self.router.num_workers(),
            "relay": relay_health,
        }
