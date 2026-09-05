# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator

import torch

from sglang_omni.admission import QueueFullError
from sglang_omni.comm import stage_io
from sglang_omni.config.topology import LogicalProcessPlan
from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
from sglang_omni.pipeline.replicas import (
    BindingPolicy,
    ReplicaTopology,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    CompleteMessage,
    DataReadyMessage,
    OmniRequest,
    RequestInfo,
    RequestState,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
    is_update_action,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_EXTERNAL_INPUT_CHUNKS = 4096
DEFAULT_MAX_EXTERNAL_INPUT_BYTES = 64 * 1024 * 1024


@dataclass
class _AdminPendingOperation:
    expected_stages: set[str]
    action: str
    results: dict[str, AdminResult] = field(default_factory=dict)
    future: asyncio.Future | None = None


@dataclass
class _ExternalInputStream:
    entry_stage: str
    entry_endpoint: str
    replica_bindings: dict[str, int] | None
    next_chunk_id: int = 0
    bytes_sent: int = 0
    done: bool = False
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Coordinator:
    """Central coordinator for the multi-stage pipeline.

    Responsibilities:
    - Register stages
    - Submit requests to entry stage
    - Track request state
    - Handle completions
    - Broadcast abort signals
    """

    def __init__(
        self,
        completion_endpoint: str,
        abort_endpoint: str,
        entry_stage: str,
        terminal_stages: list[str] | None = None,
        terminal_stages_resolver: (
            Callable[[OmniRequest], list[str] | None] | None
        ) = None,
        replica_topology: ReplicaTopology | None = None,
        logical_process_plan: LogicalProcessPlan | None = None,
        binding_policy: BindingPolicy | None = None,
        max_in_flight: int | None = None,
        max_external_input_chunks: int = DEFAULT_MAX_EXTERNAL_INPUT_CHUNKS,
        max_external_input_bytes: int = DEFAULT_MAX_EXTERNAL_INPUT_BYTES,
    ):
        """Initialize coordinator.

        Args:
            completion_endpoint: ZMQ endpoint to receive completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            entry_stage: Logical name of the entry stage for new requests
            terminal_stages: Terminal stage names. When multiple are given,
                the coordinator waits for all to complete before resolving.
            replica_topology: Logical stage to expanded instance mapping.
            logical_process_plan: Compiled Process topology; the coordinator
                selects one replica per replicated Process from it.
            max_in_flight: If set, reject new submits once this many requests
                are already tracked. Intended as generation capacity
                (max_running_requests + max_queued_requests).
        """
        self.entry_stage = entry_stage
        self._terminal_stages: set[str] = (
            set(terminal_stages) if terminal_stages else set()
        )
        self._terminal_stages_resolver = terminal_stages_resolver
        self._partial_results: dict[str, dict[str, Any]] = {}
        self._replica_topology = replica_topology or ReplicaTopology()
        self._logical_process_plan = logical_process_plan or LogicalProcessPlan(
            processes=(), stage_to_process={}
        )
        self._binding_policy = binding_policy or RoundRobinBindingPolicy()
        if max_in_flight is None:
            self.max_in_flight = None
        else:
            value = int(max_in_flight)
            if value < 0:
                raise ValueError("max_in_flight must be >= 0")
            self.max_in_flight = value
        self.max_external_input_chunks = int(max_external_input_chunks)
        self.max_external_input_bytes = int(max_external_input_bytes)
        if self.max_external_input_chunks <= 0:
            raise ValueError("max_external_input_chunks must be > 0")
        if self.max_external_input_bytes <= 0:
            raise ValueError("max_external_input_bytes must be > 0")

        # Control plane
        self.control_plane = CoordinatorControlPlane(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Stage registry
        self._stages: dict[str, StageInfo] = {}

        # Request tracking
        self._requests: dict[str, RequestInfo] = {}
        self._completion_futures: dict[str, asyncio.Future] = {}
        self._stream_queues: dict[
            str, asyncio.Queue[CompleteMessage | StreamMessage]
        ] = {}
        # Abort messages carry only the request ID. A strongly held task keeps
        # local admission closed and lets the broadcast survive caller cancellation.
        self._abort_tasks: dict[str, asyncio.Task[bool]] = {}
        self._external_input_streams: dict[str, _ExternalInputStream] = {}
        self._admin_ops: dict[str, _AdminPendingOperation] = {}
        self._admin_lock = asyncio.Lock()

        # State
        self._running = False
        self._fatal_error: str | None = None
        self._external_input_writes_open = True

    def register_stage(self, name: str, endpoint: str) -> None:
        """Register a stage.

        Args:
            name: Stage name
            endpoint: ZMQ endpoint for the stage
        """
        self._stages[name] = StageInfo(name=name, control_endpoint=endpoint)
        logger.info("Coordinator registered stage: %s at %s", name, endpoint)

    async def start(self) -> None:
        """Start the coordinator."""
        await self.control_plane.start()
        self._running = True
        self._external_input_writes_open = True
        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        self._external_input_writes_open = False
        for request_id, stream in list(self._external_input_streams.items()):
            async with stream.write_lock:
                if self._external_input_streams.get(request_id) is stream:
                    self._external_input_streams.pop(request_id, None)
        self.control_plane.close()
        logger.info("Coordinator stopped")

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        """Fail all requests currently owned by the coordinator."""
        self._running = False
        message = str(error)
        self._fatal_error = message
        self._external_input_writes_open = False
        for request_id in list(self._requests):
            stream = self._external_input_streams.get(request_id)
            if stream is None:
                await self._fail_pending_request(request_id, message)
                continue
            async with stream.write_lock:
                await self._fail_pending_request(request_id, message)
        self._partial_results.clear()

    async def _fail_pending_request(self, request_id: str, message: str) -> None:
        """Fail one request while its external-input write lock is held, if any."""
        info = self._requests.get(request_id)
        if info is None:
            return
        info.state = RequestState.FAILED
        info.error = message
        self._reject_completion_future(request_id, RuntimeError(message))
        queue = self._stream_queues.get(request_id)
        if queue is not None:
            await queue.put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error=message,
                )
            )
        self._requests.pop(request_id, None)
        self._partial_results.pop(request_id, None)
        self._external_input_streams.pop(request_id, None)

    async def shutdown_stages(self) -> None:
        """Send shutdown signal to all registered stages."""
        for name, info in self._stages.items():
            try:
                await self.control_plane.send_shutdown(name, info.control_endpoint)
                logger.info("Sent shutdown to stage: %s", name)
            except Exception as e:
                logger.warning("Failed to send shutdown to stage %s: %s", name, e)

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Run an administrative operation against one or more stages."""
        if not self._running:
            raise RuntimeError("Coordinator is not running")

        target_stages = self._resolve_admin_stages(stages)
        if not target_stages:
            raise ValueError("No stages registered for admin operation")

        op_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        pending = _AdminPendingOperation(
            expected_stages=set(target_stages),
            action=action,
            future=loop.create_future(),
        )
        operation = AdminOperation(
            op_id=op_id,
            action=action,
            payload=dict(payload or {}),
            target_stages=list(target_stages),
            timeout_s=timeout_s,
        )

        async with self._admin_lock:
            self._admin_ops[op_id] = pending
            try:
                for stage_name in target_stages:
                    info = self._stages[stage_name]
                    await self.control_plane.send_admin(
                        stage_name,
                        info.control_endpoint,
                        AdminMessage(operation=operation),
                    )

                assert pending.future is not None
                results = await asyncio.wait_for(pending.future, timeout=timeout_s)
            finally:
                self._admin_ops.pop(op_id, None)

        return self._aggregate_admin_results(
            op_id=op_id,
            action=action,
            results=list(results.values()),
        )

    async def model_info(
        self,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "model_info",
            stages=stages,
            timeout_s=timeout_s,
        )

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "pause_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "continue_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_disk",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "init_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "destroy_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_distributed",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "weights_checker",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def submit(self, request_id: str, request: OmniRequest | Any) -> Any:
        """Submit a request to the pipeline and wait for completion."""
        await self._submit_request(request_id, request)

        future = self._completion_futures[request_id]
        try:
            result = await future
            return result
        finally:
            self._completion_futures.pop(request_id, None)

    async def start_input_stream(
        self,
        request_id: str,
        request: OmniRequest | Any,
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and return its output-event iterator."""
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()
        await self._submit_request(
            request_id,
            request,
            stream_queue=stream_queue,
            external_input_stream=True,
        )
        expected = self._expected_terminal_stages(request_id)
        return self._stream_events(request_id, stream_queue, expected)

    async def send_input_chunk(
        self,
        request_id: str,
        data: torch.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Append one bounded CPU tensor to an active entry-stage input stream."""
        if not isinstance(data, torch.Tensor):
            raise TypeError("External input stream chunks must be torch.Tensor values")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("External input stream metadata must be a dict or None")
        data_ref = stage_io.serialize_inline_stream_chunk(data, metadata)
        if data_ref is None:
            raise ValueError(
                "External input stream chunks must be CPU tensors whose serialized "
                f"payload is at most {stage_io.INLINE_STREAM_CHUNK_BYTES_LIMIT} bytes"
            )

        stream = self._active_input_stream(request_id)
        async with stream.write_lock:
            self._ensure_current_input_stream(request_id, stream)
            if stream.done:
                raise RuntimeError(f"Input stream {request_id!r} is already done")
            if stream.next_chunk_id >= self.max_external_input_chunks:
                raise ValueError(
                    f"Input stream {request_id!r} exceeds max_external_input_chunks="
                    f"{self.max_external_input_chunks}"
                )
            chunk_bytes = data.element_size() * data.numel()
            if stream.bytes_sent + chunk_bytes > self.max_external_input_bytes:
                raise ValueError(
                    f"Input stream {request_id!r} exceeds max_external_input_bytes="
                    f"{self.max_external_input_bytes}"
                )
            chunk_id = stream.next_chunk_id
            await self.control_plane.send_input_stream_event(
                stream.entry_stage,
                stream.entry_endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    to_stage=stream.entry_stage,
                    data_ref=data_ref,
                    chunk_id=chunk_id,
                    replica_bindings=stream.replica_bindings,
                ),
            )
            self._ensure_current_input_stream(request_id, stream)
            stream.next_chunk_id += 1
            stream.bytes_sent += chunk_bytes
            return chunk_id

    async def finish_input_stream(self, request_id: str) -> None:
        """Mark an active entry-stage input stream complete."""
        stream = self._active_input_stream(request_id)
        async with stream.write_lock:
            self._ensure_current_input_stream(request_id, stream)
            if stream.done:
                raise RuntimeError(f"Input stream {request_id!r} is already done")
            await self.control_plane.send_input_stream_event(
                stream.entry_stage,
                stream.entry_endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    to_stage=stream.entry_stage,
                    data_ref=None,
                    is_done=True,
                    replica_bindings=stream.replica_bindings,
                ),
            )
            self._ensure_current_input_stream(request_id, stream)
            stream.done = True

    def _active_input_stream(self, request_id: str) -> _ExternalInputStream:
        self._ensure_external_input_writes_open()
        stream = self._external_input_streams.get(request_id)
        if stream is None or request_id not in self._requests:
            raise ValueError(f"No active input stream for request {request_id!r}")
        return stream

    def _ensure_current_input_stream(
        self, request_id: str, stream: _ExternalInputStream
    ) -> None:
        self._ensure_external_input_writes_open()
        if (
            self._external_input_streams.get(request_id) is not stream
            or request_id not in self._requests
        ):
            raise ValueError(f"No active input stream for request {request_id!r}")

    def _ensure_external_input_writes_open(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if not self._external_input_writes_open:
            raise RuntimeError("Coordinator is not accepting external input writes")

    async def stream(
        self, request_id: str, request: OmniRequest | Any
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion."""
        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()

        await self._submit_request(request_id, request, stream_queue=queue)
        expected = self._expected_terminal_stages(request_id)
        events = self._stream_events(request_id, queue, expected)
        async with aclosing(events):
            async for msg in events:
                yield msg

    async def _stream_events(
        self,
        request_id: str,
        queue: asyncio.Queue[CompleteMessage | StreamMessage],
        expected_terminal_stages: set[str],
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        completed_stages: set[str] = set()
        try:
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise QueueFullError.from_message(msg.error)
                    yield msg
                    completed_stages.add(
                        self._replica_topology.logical_name(msg.from_stage)
                    )
                    if (
                        not expected_terminal_stages
                        or completed_stages >= expected_terminal_stages
                    ):
                        return
                else:
                    yield msg
        finally:
            if self._stream_queues.get(request_id) is queue:
                try:
                    if request_id in self._requests:
                        try:
                            await self.abort(request_id)
                        except Exception:
                            pass
                finally:
                    if self._stream_queues.get(request_id) is queue:
                        self._stream_queues.pop(request_id, None)
                        self._completion_futures.pop(request_id, None)
                        self._external_input_streams.pop(request_id, None)

    async def _submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
        external_input_stream: bool = False,
    ) -> None:
        """Submit a request without waiting for completion."""
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if external_input_stream:
            self._ensure_external_input_writes_open()
        if self._request_id_is_reserved(request_id):
            raise ValueError(f"Request {request_id} already exists")

        if self.max_in_flight is not None and len(self._requests) >= self.max_in_flight:
            logger.warning(
                "Rejecting request %s before pipeline submit: in-flight cap "
                "(max_in_flight=%s)",
                request_id,
                self.max_in_flight,
            )
            raise QueueFullError()

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

        replica_bindings = assign_replica_bindings(
            self._logical_process_plan, self._binding_policy, request_id
        )
        bindings = replica_bindings or {}
        entry_instance = (
            self._replica_topology.resolve(self.entry_stage, bindings[self.entry_stage])
            if self._replica_topology.is_replicated(self.entry_stage)
            else self.entry_stage
        )
        if entry_instance not in self._stages:
            raise ValueError(f"Entry stage {entry_instance} not registered")
        entry_info = self._stages[entry_instance]

        # Track request
        self._requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
            terminal_stages=self._resolve_terminal_stages(request),
        )

        # Create future for completion
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._completion_futures[request_id] = future
        if stream_queue is not None:
            self._stream_queues[request_id] = stream_queue
        if external_input_stream:
            self._external_input_streams[request_id] = _ExternalInputStream(
                entry_stage=entry_instance,
                entry_endpoint=entry_info.control_endpoint,
                replica_bindings=replica_bindings,
            )

        payload = StagePayload(
            request_id=request_id,
            request=request,
            data={"raw_inputs": request.inputs},
        )

        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="request_admission",
            metadata={"entry_stage": self.entry_stage},
        )

        try:
            await self.control_plane.submit_to_stage(
                entry_instance,
                entry_info.control_endpoint,
                SubmitMessage(
                    request_id=request_id,
                    data=payload,
                    replica_bindings=replica_bindings,
                    external_input_stream=external_input_stream,
                ),
            )
        except BaseException:
            self._rollback_request_start(request_id)
            raise

        # Update state
        info = self._requests.get(request_id)
        if info is not None:
            info.state = RequestState.RUNNING

        logger.info(
            "Coordinator submitted req=%s to %s at %s bindings=%s",
            request_id,
            entry_instance,
            entry_info.control_endpoint,
            replica_bindings,
        )

    def _rollback_request_start(self, request_id: str) -> None:
        """Release every owner installed before the entry-stage submit."""
        self._requests.pop(request_id, None)
        self._partial_results.pop(request_id, None)
        self._stream_queues.pop(request_id, None)
        self._external_input_streams.pop(request_id, None)
        pending = self._completion_futures.pop(request_id, None)
        if pending is not None and not pending.done():
            pending.cancel()

    def _request_id_is_reserved(self, request_id: str) -> bool:
        """Return whether any coordinator owner still holds this request ID."""
        return (
            request_id in self._requests
            or request_id in self._completion_futures
            or request_id in self._stream_queues
            or request_id in self._abort_tasks
            or request_id in self._external_input_streams
        )

    def _reject_completion_future(
        self,
        request_id: str,
        exc: BaseException,
    ) -> None:
        # Note: (Akazaakane) Non-streaming callers await the completion future,
        # so errors must be propagated with set_exception(). Streaming callers
        # receive errors through the stream queue and never await that future;
        # cancel it instead to avoid "Future exception was never retrieved".
        future = self._completion_futures.get(request_id)
        if future is None or future.done():
            return
        if request_id in self._stream_queues:
            future.cancel()
        else:
            future.set_exception(exc)

    async def abort(self, request_id: str) -> bool:
        """Abort a request.

        Args:
            request_id: Request to abort

        Returns:
            True if aborted, False if not found
        """
        abort_task = self._abort_tasks.get(request_id)
        if abort_task is not None:
            return await asyncio.shield(abort_task)

        info = self._requests.get(request_id)
        if info is None:
            return False

        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False

        abort_task = asyncio.create_task(
            self._run_abort(request_id),
            name=f"coordinator-abort-{request_id}",
        )
        self._abort_tasks[request_id] = abort_task
        abort_task.add_done_callback(
            lambda done, rid=request_id: self._on_abort_task_done(rid, done)
        )
        return await asyncio.shield(abort_task)

    async def _run_abort(
        self,
        request_id: str,
    ) -> bool:
        stream = self._external_input_streams.get(request_id)
        if stream is not None:
            async with stream.write_lock:
                return await self._run_abort_locked(request_id)
        return await self._run_abort_locked(request_id)

    async def _run_abort_locked(self, request_id: str) -> bool:
        await self.control_plane.broadcast_abort(AbortMessage(request_id=request_id))

        info = self._requests.get(request_id)
        if info is None:
            return False

        info.state = RequestState.ABORTED
        self._reject_completion_future(
            request_id, asyncio.CancelledError(f"Request {request_id} aborted")
        )
        stream_queue = self._stream_queues.get(request_id)
        if stream_queue is not None:
            await stream_queue.put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error="aborted",
                )
            )

        self._requests.pop(request_id, None)
        self._partial_results.pop(request_id, None)
        self._external_input_streams.pop(request_id, None)

        logger.info("Coordinator aborted req=%s", request_id)
        return True

    async def close_input_stream(self, request_id: str) -> bool:
        """Abort if active and release every coordinator-side stream owner."""
        aborted = await self.abort(request_id)
        self._stream_queues.pop(request_id, None)
        self._completion_futures.pop(request_id, None)
        self._external_input_streams.pop(request_id, None)
        return aborted

    def _on_abort_task_done(
        self,
        request_id: str,
        task: asyncio.Task[bool],
    ) -> None:
        if self._abort_tasks.get(request_id) is task:
            self._abort_tasks.pop(request_id, None)
        if task.cancelled():
            logger.warning("Coordinator abort task cancelled for req=%s", request_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Failed to abort request %s",
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def run_completion_loop(self) -> None:
        """Run the completion receiving loop.

        This should be run as a background task.
        """
        try:
            while self._running:
                msg = await self.control_plane.recv_event()
                if isinstance(msg, StreamMessage):
                    await self._handle_stream(msg)
                elif isinstance(msg, AdminResultMessage):
                    self._handle_admin_result(msg.result)
                else:
                    await self._handle_completion(msg)
        except asyncio.CancelledError:
            logger.info("Coordinator completion loop cancelled")
        except Exception as e:
            logger.error("Coordinator completion loop error: %s", e)
            raise

    async def _handle_completion(self, msg: CompleteMessage) -> None:
        """Serialize terminal state with external input writes for this request."""
        stream = self._external_input_streams.get(msg.request_id)
        if stream is None:
            await self._handle_completion_locked(msg)
            return
        async with stream.write_lock:
            await self._handle_completion_locked(msg)

    async def _handle_completion_locked(self, msg: CompleteMessage) -> None:
        """Handle a completion message from a stage."""
        request_id = msg.request_id
        logger.debug(
            "Coordinator received completion: req=%s from %s success=%s",
            request_id,
            msg.from_stage,
            msg.success,
        )
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="terminal_response",
            metadata={
                "from_stage": msg.from_stage,
                "success": msg.success,
            },
        )

        if request_id not in self._requests:
            logger.debug(
                "Coordinator ignored completion for inactive req=%s from %s",
                request_id,
                msg.from_stage,
            )
            return

        info = self._requests[request_id]

        # Note (wenyao): the client reads ``from_stage`` off the completion.
        # Observability emits above keep the instance name.
        from_stage = self._replica_topology.logical_name(msg.from_stage)
        if from_stage != msg.from_stage:
            msg = replace(msg, from_stage=from_stage)

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            info.state = RequestState.FAILED
            info.error = msg.error
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id)
            )
            self._partial_results.pop(request_id, None)
            self._reject_completion_future(
                request_id, QueueFullError.from_message(msg.error)
            )
            stream_queue = self._stream_queues.get(request_id)
            if stream_queue is not None:
                await stream_queue.put(msg)
            self._requests.pop(request_id, None)
            self._external_input_streams.pop(request_id, None)
            return

        expected_terminal_stages = self._expected_terminal_stages(request_id)
        if expected_terminal_stages and from_stage not in expected_terminal_stages:
            logger.debug(
                "Coordinator ignoring completion from inactive terminal: "
                "req=%s stage=%s expected=%s",
                request_id,
                msg.from_stage,
                sorted(expected_terminal_stages),
            )
            return

        # Single active terminal (original behavior) or no terminal_stages configured
        if len(expected_terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self._completion_futures:
                future = self._completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
            if request_id in self._stream_queues:
                await self._stream_queues[request_id].put(msg)
            self._requests.pop(request_id, None)
            self._external_input_streams.pop(request_id, None)
            return

        # Multi-terminal: collect partial results
        partials = self._partial_results.setdefault(request_id, {})
        partials[from_stage] = msg.result

        # Forward stream completion per-stage
        if request_id in self._stream_queues:
            await self._stream_queues[request_id].put(msg)

        if set(partials) < expected_terminal_stages:
            return  # still waiting

        # All terminal stages done -> merge and resolve
        merged = dict(partials)
        self._partial_results.pop(request_id)
        info.state = RequestState.COMPLETED
        info.result = merged

        if request_id in self._completion_futures:
            future = self._completion_futures[request_id]
            if not future.done():
                future.set_result(merged)
        self._requests.pop(request_id, None)
        self._external_input_streams.pop(request_id, None)

    async def _handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        request_id = msg.request_id
        if request_id not in self._stream_queues:
            return
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="coordinator_stream_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
            },
        )
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="stage_stream_chunk_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
            },
        )
        # Note (wenyao): normalize both fields -- the client falls back to
        # ``stage_name or from_stage``. Observability emits above keep the
        # instance name.
        logical = self._replica_topology.logical_name(msg.from_stage)
        stage_name = (
            self._replica_topology.logical_name(msg.stage_name)
            if msg.stage_name is not None
            else msg.stage_name
        )
        if logical != msg.from_stage or stage_name != msg.stage_name:
            msg = replace(msg, from_stage=logical, stage_name=stage_name)
        await self._stream_queues[request_id].put(msg)

    def _handle_admin_result(self, result: AdminResult) -> None:
        pending = self._admin_ops.get(result.op_id)
        if pending is None:
            logger.warning(
                "Coordinator received admin result for unknown op=%s stage=%s",
                result.op_id,
                result.stage,
            )
            return
        pending.results[result.stage] = result
        if (
            pending.future is not None
            and pending.results.keys() >= pending.expected_stages
        ):
            if not pending.future.done():
                pending.future.set_result(dict(pending.results))

    def _resolve_admin_stages(self, stages: Sequence[str] | None) -> list[str]:
        if stages is None:
            return sorted(self._stages)
        # Note (wenyao): dedup preserving order so a caller passing both a
        # logical name and one of its instances does not double-send admin ops.
        resolved: list[str] = []
        for name in stages:
            for instance in self._replica_topology.instances(name):
                if instance not in resolved:
                    resolved.append(instance)
        unknown = sorted(set(resolved) - set(self._stages))
        if unknown:
            raise ValueError(f"Unknown admin target stage(s): {unknown}")
        return resolved

    def _aggregate_admin_results(
        self,
        *,
        op_id: str,
        action: str,
        results: list[AdminResult],
    ) -> dict[str, Any]:
        updated_results = [
            item
            for item in results
            if not item.data.get("skipped") and not item.data.get("unsupported")
        ]
        if is_update_action(action):
            success = bool(updated_results) and all(
                item.success for item in updated_results
            )
        else:
            success = all(item.success for item in results)

        errors = [item.error for item in results if item.error]
        if success:
            message = "ok"
        elif errors:
            message = "; ".join(errors)
        else:
            message = "admin operation did not complete successfully"

        return {
            "op_id": op_id,
            "action": action,
            "success": success,
            "message": message,
            "results": [item.to_dict() for item in results],
        }

    def get_request_info(self, request_id: str) -> RequestInfo | None:
        """Get info about a request."""
        return self._requests.get(request_id)

    def _resolve_terminal_stages(self, request: OmniRequest) -> set[str]:
        if self._terminal_stages_resolver is None:
            return set(self._terminal_stages)
        resolved = self._terminal_stages_resolver(request)
        if resolved is None:
            return set(self._terminal_stages)
        if isinstance(resolved, str) or not isinstance(resolved, Sequence):
            raise ValueError(
                "terminal_stages_resolver must return a sequence of terminal "
                "stage names or None"
            )
        if not all(isinstance(stage, str) for stage in resolved):
            raise ValueError(
                "terminal_stages_resolver must return terminal stage names"
            )
        resolved_stages = set(resolved)
        if not resolved_stages:
            raise ValueError("terminal_stages_resolver returned no terminal stages")
        unknown = resolved_stages - self._terminal_stages
        if unknown:
            raise ValueError(
                "terminal_stages_resolver returned stages outside the static "
                f"terminal stages: {sorted(unknown)}. Allowed terminal stages: "
                f"{sorted(self._terminal_stages)}"
            )
        return resolved_stages

    def _expected_terminal_stages(self, request_id: str) -> set[str]:
        info = self._requests.get(request_id)
        if info is None or info.terminal_stages is None:
            return set(self._terminal_stages)
        return info.terminal_stages

    def health(self) -> dict[str, Any]:
        """Return health status."""
        state_counts = {}
        for info in self._requests.values():
            state = info.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "running": self._running,
            "stages": list(self._stages.keys()),
            "entry_stage": self.entry_stage,
            "total_requests": len(self._requests),
            "pending_completions": len(self._completion_futures),
            "request_states": state_counts,
        }
