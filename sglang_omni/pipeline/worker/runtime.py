# SPDX-License-Identifier: Apache-2.0
"""Worker that runs the processing loop."""

from __future__ import annotations

import asyncio
import io
import logging
import queue as _queue_mod
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.executors.engine_executor import is_stream_request_params
from sglang_omni.executors.interface import Executor
from sglang_omni.pipeline.stage.work import WorkDescriptor
from sglang_omni.pipeline.worker.data_plane import DataPlaneAdapter
from sglang_omni.proto import (
    CompleteMessage,
    DataReadyMessage,
    StagePayload,
    StreamMessage,
)

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage.runtime import Stage

logger = logging.getLogger(__name__)


@dataclass
class _PendingStreamItem:
    request_id: str
    data: Any
    target_stage: str
    metadata: dict | None = None
    is_done: bool = False
    error: str | None = None


class Worker:
    """Worker that runs the processing loop.

    Requests are processed concurrently: each dequeued work item becomes an
    independent task.  A dedicated dispatcher task consumes results from the
    executor (in completion order) and routes them to the corresponding request
    task via per-request :class:`asyncio.Future` objects.

    Back-pressure is provided naturally by the work queue (upstream) and by
    the executor's ``add_request`` (downstream).
    """

    def __init__(self, executor: Executor, role: str | None = None):
        self.executor = executor
        self.engine = executor  # Backward-compatible alias.
        self.role = role
        self.stage: Stage | None = None
        self.data_plane: DataPlaneAdapter | None = None
        self.queue: asyncio.Queue[WorkDescriptor | None] | None = None
        self._running = False
        self._result_waiters: dict[str, asyncio.Future[StagePayload]] = {}
        # Streaming send state (thread-safe queue for cross-thread enqueue)
        self._stream_send_queue: _queue_mod.Queue[_PendingStreamItem] = (
            _queue_mod.Queue()
        )
        self._stream_send_task: asyncio.Task | None = None
        self._stream_chunk_counters: dict[tuple[str, str], int] = {}
        self._stream_targets: list[str] = []  # Set by compiler
        self._bootstrap_targets: set[str] = set()  # Set by compiler
        self._same_gpu_targets: set[str] = (
            set()
        )  # Set by compiler for CUDA IPC zero-copy
        self._stream_running: bool = False

    def bind(self, stage: Stage, queue: asyncio.Queue[WorkDescriptor | None]) -> None:
        """Bind this worker to a stage."""
        self.stage = stage
        self.data_plane = DataPlaneAdapter(stage.relay)
        self.queue = queue

    async def run(self) -> None:
        """Main processing loop."""
        if self.stage is None or self.queue is None or self.data_plane is None:
            raise RuntimeError("Worker not bound to a stage")

        try:
            await self.executor.start()
            self._running = True

            # Start streaming send loop
            self._stream_running = True
            self._stream_send_task = asyncio.create_task(self._stream_send_loop())

            logger.info("Worker started for stage %s", self.stage.name)

            inflight: set[asyncio.Task[None]] = set()
            dispatcher = asyncio.create_task(self._dispatch_results())

            try:
                while self._running:
                    work = await self.queue.get()
                    if work is None:
                        break

                    task = asyncio.create_task(self._process_request(work))
                    inflight.add(task)
                    task.add_done_callback(inflight.discard)

                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)
            finally:
                dispatcher.cancel()
                try:
                    await dispatcher
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            logger.info("Worker cancelled for stage %s", self.stage.name)
        finally:
            self._running = False
            # Stop streaming send loop
            self._stream_running = False
            if self._stream_send_task is not None:
                try:
                    await self._stream_send_task
                except asyncio.CancelledError:
                    pass
                self._stream_send_task = None
            await self.executor.stop()

    async def _dispatch_results(self) -> None:
        """Single consumer that routes executor results to per-request futures."""
        while True:
            try:
                result = await self.executor.get_result()
            except asyncio.CancelledError:
                break
            except Exception as e:
                request_id = getattr(e, "request_id", None)
                if request_id is not None:
                    fut = self._result_waiters.pop(request_id, None)
                    if fut is not None and not fut.done():
                        fut.set_exception(e)
                        continue
                logger.exception("Worker dispatcher: get_result error")
                continue

            fut = self._result_waiters.pop(result.request_id, None)
            if fut is not None and not fut.done():
                fut.set_result(result)
            else:
                logger.warning(
                    "Worker dispatcher: no waiter for request %s",
                    result.request_id,
                )

    async def _process_request(self, work: WorkDescriptor) -> None:
        """Process a single request."""
        request_id = work.request_id
        logger.debug("Worker %s: processing request %s", self.stage.name, request_id)
        try:
            if self.data_plane is None:
                raise RuntimeError("Worker not bound to a data plane")
            payloads = await self._load_inputs(work)
            logger.debug("Worker %s: loaded inputs for %s", self.stage.name, request_id)
            merged = self._merge_payloads(work, payloads)
            if not isinstance(merged, StagePayload):
                raise TypeError(f"Expected StagePayload, got {type(merged)}")
            if merged.request_id != request_id:
                raise ValueError(
                    "Merged payload request_id mismatch "
                    f"(expected={request_id} got={merged.request_id})"
                )

            bootstrap_targets = self._get_stream_bootstrap_targets()
            for stage_name in bootstrap_targets:
                sent = await self._send_to_next(request_id, stage_name, merged)
                if not sent:
                    return

            # Register future BEFORE add_request so the dispatcher can
            # route the result even if the executor completes synchronously.
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[StagePayload] = loop.create_future()
            self._result_waiters[request_id] = fut

            logger.debug(
                "Worker %s: adding request %s to executor", self.stage.name, request_id
            )
            await self.executor.add_request(merged)
            logger.debug(
                "Worker %s: request %s added, waiting for result",
                self.stage.name,
                request_id,
            )

            stream_task: asyncio.Task[None] | None = None
            stream_fn = getattr(self.executor, "stream", None)
            request_params = (
                merged.request.params if merged.request is not None else None
            )
            is_stream_request = is_stream_request_params(request_params)
            if is_stream_request and callable(stream_fn):
                stream_iter = stream_fn(request_id)
                if stream_iter is not None:
                    stream_task = asyncio.create_task(
                        self._forward_stream(request_id, stream_iter)
                    )

            output_payload = await fut
            logger.debug("Worker %s: got result for %s", self.stage.name, request_id)

            # Signal stream done to all downstream streaming targets
            for target in self._stream_targets:
                try:
                    self._enqueue_stream_done(request_id, target)
                except Exception:
                    logger.debug(
                        "Worker: failed to send stream done for %s to %s",
                        request_id,
                        target,
                    )

            if not isinstance(output_payload, StagePayload):
                raise TypeError(
                    "Executor must return StagePayload, " f"got {type(output_payload)}"
                )
            if output_payload.request_id != request_id:
                raise ValueError(
                    "Output payload request_id mismatch "
                    f"(expected={request_id} got={output_payload.request_id})"
                )

            # Route
            next_stage = self.stage.get_next(request_id, output_payload)

            logger.debug(
                "Worker %s: next_stage=%s for %s",
                self.stage.name,
                next_stage,
                request_id,
            )
            if next_stage is None:
                if stream_task is not None:
                    await self._finish_stream_task(stream_task)
                await self._send_complete(request_id, output_payload.data)
                logger.debug(
                    "Worker %s: sent complete for %s", self.stage.name, request_id
                )
            else:
                if stream_task is not None:
                    await self._finish_stream_task(stream_task)
                for stage_name in self._normalize_next_stages(next_stage):
                    if stage_name in bootstrap_targets:
                        continue
                    sent = await self._send_to_next(
                        request_id, stage_name, output_payload
                    )
                    if not sent:
                        return
                    logger.debug(
                        "Worker %s: routed %s to %s",
                        self.stage.name,
                        request_id,
                        stage_name,
                    )

        except asyncio.CancelledError:
            logger.debug("Worker: request %s cancelled", request_id)
        except Exception as e:
            logger.exception("Worker: request %s failed", request_id)
            self._notify_stream_error(request_id, str(e))
            await self._send_failure(request_id, str(e))
        finally:
            self._result_waiters.pop(request_id, None)
            if self.stage is not None:
                self.stage.router.clear_request(request_id)
                # Close the stream queue entry to prevent per-request leaks
                if self.stage._stream_queue is not None:
                    self.stage._stream_queue.close(request_id)
                self.stage._pending_stream_data.pop(request_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_inputs(self, work: WorkDescriptor) -> dict[str, StagePayload]:
        payloads: dict[str, StagePayload] = {}
        for ref in work.inputs:
            if ref.payload is not None:
                payloads[ref.source] = ref.payload
                continue
            if ref.metadata is None:
                raise ValueError(f"Missing metadata for source={ref.source}")
            payloads[ref.source] = await self.data_plane.read_payload(
                work.request_id, ref.metadata
            )
        return payloads

    @staticmethod
    def _merge_payloads(
        work: WorkDescriptor,
        payloads: dict[str, StagePayload],
    ) -> StagePayload:
        if work.merge is not None:
            return work.merge(payloads)
        if len(payloads) != 1:
            raise ValueError("Multiple inputs require a merge function")
        return next(iter(payloads.values()))

    @staticmethod
    def _normalize_next_stages(next_stage: str | list[str]) -> list[str]:
        match next_stage:
            case str() as stage:
                return [stage]
            case list() as stages if stages:
                return stages
            case list():
                raise ValueError("get_next returned an empty stage list")
            case _:
                raise TypeError(
                    "get_next must return a stage name, list of stage names, or None"
                )

    async def _send_complete(self, request_id: str, result: Any) -> None:
        """Send completion to coordinator."""
        logger.debug("Worker: %s completed (END)", request_id)
        await self.stage.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.stage.name,
                success=True,
                result=result,
            )
        )

    async def _send_to_next(
        self, request_id: str, next_stage: str, payload: StagePayload
    ) -> bool:
        """Send data to next stage."""
        logger.debug("Worker: routing %s to %s", request_id, next_stage)

        try:
            endpoint = self.stage.endpoints.get(next_stage)
            if endpoint is None:
                await self._send_failure(request_id, f"Unknown stage: {next_stage}")
                return False
            metadata, op = await self.data_plane.write_payload(request_id, payload)

            await self.stage.control_plane.send_to_stage(
                next_stage,
                endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=self.stage.name,
                    to_stage=next_stage,
                    shm_metadata=metadata,
                ),
            )

            await op.wait_for_completion()
            self.data_plane.cleanup(request_id)
            return True

        except Exception as e:
            logger.exception("Worker: failed to write data for req=%s", request_id)
            await self._send_failure(request_id, f"Failed to write data: {e}")
            return False

    async def _send_failure(self, request_id: str, error: str) -> None:
        """Send failure to coordinator."""
        await self.stage.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.stage.name,
                success=False,
                error=error,
            )
        )

    async def _forward_stream(self, request_id: str, stream_iter: Any) -> None:
        """Forward streaming chunks to the coordinator."""
        if self.stage is None:
            return
        try:
            async for chunk in stream_iter:
                if chunk is None:
                    continue
                await self.stage.control_plane.send_stream(
                    StreamMessage(
                        request_id=request_id,
                        from_stage=self.stage.name,
                        chunk=chunk,
                        stage_name=self.stage.name,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Worker stream error for %s: %s", request_id, exc)

    async def _finish_stream_task(self, task: asyncio.Task[None]) -> None:
        """Wait for stream task to finish and cancel if it stalls."""
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Streaming send
    # ------------------------------------------------------------------

    def _enqueue_stream(
        self,
        request_id: str,
        data: Any,
        target_stage: str,
        metadata: dict | None = None,
    ) -> None:
        """Non-blocking enqueue. Must not block — may be called from the event loop."""
        try:
            self._stream_send_queue.put_nowait(
                _PendingStreamItem(
                    request_id=request_id,
                    data=data,
                    target_stage=target_stage,
                    metadata=metadata,
                ),
            )
        except _queue_mod.Full:
            logger.error(
                "Stream send queue full for %s → %s, dropping chunk",
                request_id,
                target_stage,
            )

    def _enqueue_stream_done(self, request_id: str, target_stage: str) -> None:
        """Non-blocking end-of-stream signal."""
        try:
            self._stream_send_queue.put_nowait(
                _PendingStreamItem(
                    request_id=request_id,
                    data=None,
                    target_stage=target_stage,
                    is_done=True,
                ),
            )
        except _queue_mod.Full:
            logger.error(
                "Stream send queue full, dropping done signal for %s", request_id
            )

    def _enqueue_stream_error(
        self, request_id: str, target_stage: str, error: str
    ) -> None:
        """Non-blocking error signal."""
        try:
            self._stream_send_queue.put_nowait(
                _PendingStreamItem(
                    request_id=request_id,
                    data=None,
                    target_stage=target_stage,
                    error=error,
                ),
            )
        except _queue_mod.Full:
            logger.error(
                "Stream send queue full, dropping error signal for %s", request_id
            )

    async def _stream_send_loop(self) -> None:
        """Background async task: drain thread-safe queue, write relay, send messages."""
        loop = asyncio.get_running_loop()
        while self._stream_running or not self._stream_send_queue.empty():
            try:
                item = await loop.run_in_executor(
                    None, lambda: self._stream_send_queue.get(timeout=0.1)
                )
            except _queue_mod.Empty:
                continue
            try:
                await self._do_stream_send(item)
            except Exception as e:
                logger.exception(
                    "Worker: failed to send stream item for %s", item.request_id
                )
                # Propagate failure to downstream stage via error signal
                try:
                    target_endpoint = (
                        self.stage.endpoints.get(item.target_stage)
                        if self.stage
                        else None
                    )
                    error_msg = DataReadyMessage(
                        request_id=item.request_id,
                        from_stage=self.stage.name if self.stage else "",
                        to_stage=item.target_stage,
                        shm_metadata={},
                        error=str(e),
                    )
                    await self._send_stream_control_message(
                        error_msg, item.target_stage, target_endpoint
                    )
                except Exception:
                    logger.debug(
                        "Failed to propagate stream error for %s", item.request_id
                    )

    async def _do_stream_send(self, item: _PendingStreamItem) -> None:
        """Write data to relay and send DataReadyMessage with streaming fields."""
        target_endpoint = (
            self.stage.endpoints.get(item.target_stage) if self.stage else None
        )

        # Handle done/error signals
        if item.is_done or item.error:
            msg = DataReadyMessage(
                request_id=item.request_id,
                from_stage=self.stage.name,
                to_stage=item.target_stage,
                shm_metadata={},
                is_done=item.is_done,
                error=item.error,
            )
            await self._send_stream_control_message(
                msg, item.target_stage, target_endpoint
            )
            key = (item.request_id, item.target_stage)
            self._stream_chunk_counters.pop(key, None)
            return

        # Normal chunk
        key = (item.request_id, item.target_stage)
        chunk_id = self._stream_chunk_counters.get(key, 0)
        self._stream_chunk_counters[key] = chunk_id + 1

        # ── Same-GPU CUDA IPC path: skip relay entirely ──
        if item.target_stage in self._same_gpu_targets:
            ipc_metadata = self._serialize_ipc_chunk(item)
            ipc_metadata["chunk_id"] = chunk_id
            msg = DataReadyMessage(
                request_id=item.request_id,
                from_stage=self.stage.name,
                to_stage=item.target_stage,
                shm_metadata=ipc_metadata,
                chunk_id=chunk_id,
            )
            await self._send_stream_control_message(
                msg, item.target_stage, target_endpoint
            )
            return

        # ── Cross-GPU: use relay ──
        blob_key = (
            f"{item.request_id}:stream:{self.stage.name}:{item.target_stage}:{chunk_id}"
        )

        # Write tensors to relay, then notify receiver BEFORE waiting for completion.
        # NIXL relay has limited credits (default 2). If we wait_for_completion before
        # notifying the receiver, the receiver never starts reading, never sends the
        # notification, and we deadlock. The fix: send the control message first so the
        # receiver starts read_blob (which triggers the RDMA notification), then wait.
        pending_ops: list = []
        relay_metadata, op = await self.data_plane.write_blob(blob_key, item.data)
        pending_ops.append(op)

        # Handle metadata tensors
        if item.metadata:
            from .data_plane import _extract_tensors

            cleaned_meta, tensor_dict = _extract_tensors(item.metadata)
            relay_metadata["chunk_metadata"] = cleaned_meta
            if tensor_dict:
                metadata_refs: dict[str, Any] = {}
                for meta_idx, (tkey, tensor) in enumerate(tensor_dict.items()):
                    meta_blob_key = f"{blob_key}:meta:{meta_idx}"
                    meta_relay_info, meta_op = await self.data_plane.write_blob(
                        meta_blob_key, tensor
                    )
                    pending_ops.append(meta_op)
                    metadata_refs[tkey] = {
                        "blob_key": meta_blob_key,
                        "relay_metadata": meta_relay_info,
                    }
                relay_metadata["chunk_metadata_tensors"] = metadata_refs

        # Send control message FIRST — receiver starts reading immediately
        msg = DataReadyMessage(
            request_id=item.request_id,
            from_stage=self.stage.name,
            to_stage=item.target_stage,
            shm_metadata=relay_metadata,
            chunk_id=chunk_id,
        )
        await self._send_stream_control_message(msg, item.target_stage, target_endpoint)

        # Wait for all writes to complete after notifying receiver.
        # The receiver starts reading (triggering RDMA notification) upon receiving
        # the DataReadyMessage, which helps avoid credit deadlock when credits > 1.
        for pending_op in pending_ops:
            await pending_op.wait_for_completion()

    @staticmethod
    def _ipc_pickle(obj: Any) -> bytes:
        """Serialize an object using ForkingPickler (CUDA IPC for GPU tensors)."""
        buf = io.BytesIO()
        ForkingPickler(buf, 2).dump(obj)
        return buf.getvalue()

    def _serialize_ipc_chunk(self, item: _PendingStreamItem) -> dict[str, Any]:
        """Build IPC metadata dict for a same-GPU stream chunk.

        Uses ``ForkingPickler`` which serialises CUDA tensors via
        ``cudaIpcGetMemHandle`` (~200 bytes per handle, zero data copy).
        The receiver deserialises with plain ``pickle.loads``, getting a tensor
        that points directly to the sender's GPU memory.

        PyTorch's CUDA IPC mechanism uses built-in reference counting: the
        receiver's reconstructed tensor holds a reference to the shared CUDA
        storage, keeping it alive until the receiver is done.
        """
        ipc_metadata: dict[str, Any] = {"_ipc": True}

        # Serialize main tensor / data
        ipc_metadata["tensor_bytes"] = self._ipc_pickle(item.data)

        # Serialize metadata tensors
        if item.metadata:
            serialized_meta: dict[str, Any] = {}
            for mkey, value in item.metadata.items():
                if isinstance(value, torch.Tensor):
                    serialized_meta[mkey] = {
                        "_ipc_tensor": self._ipc_pickle(value),
                    }
                else:
                    serialized_meta[mkey] = value
            ipc_metadata["metadata"] = serialized_meta

        return ipc_metadata

    async def _send_stream_control_message(
        self, msg, target_stage: str, endpoint: str | None
    ) -> None:
        """Send a control message to a specific downstream stage."""
        if self.stage is None:
            logger.warning("Worker: cannot send stream message, stage is None")
            return
        if endpoint is None:
            logger.warning(
                "Worker: no endpoint for stage %s, dropping message", target_stage
            )
            return
        await self.stage.control_plane.send_to_stage(target_stage, endpoint, msg)

    def _get_stream_bootstrap_targets(self) -> list[str]:
        """Return streaming downstream stages that should receive bootstrap data."""
        return list(self._bootstrap_targets)

    def _notify_stream_error(self, request_id: str, error: str) -> None:
        """Propagate error to all downstream streaming targets."""
        for target in self._stream_targets:
            try:
                self._enqueue_stream_error(request_id, target, error)
            except Exception:
                logger.debug(
                    "Worker: failed to propagate stream error for %s", request_id
                )

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
