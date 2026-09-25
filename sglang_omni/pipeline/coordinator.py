# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator

from opentelemetry import trace

from sglang_omni import tracing
from sglang_omni.admission import QueueFullError
from sglang_omni.config.topology import LogicalProcessPlan
from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
from sglang_omni.pipeline.replicas import (
    BindingPolicy,
    ReplicaTopology,
    RoundRobinBindingPolicy,
    assign_replica_bindings,
)
from sglang_omni.pipeline.sessions import CoordinatorSessions
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    CompleteMessage,
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


@dataclass
class AdminPendingOperation:
    expected_stages: set[str]
    action: str
    results: dict[str, AdminResult] = field(default_factory=dict)
    future: asyncio.Future | None = None


class Coordinator(CoordinatorSessions):
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
        tracer: trace.Tracer | None = None,
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
        super().__init__()
        self.traces = tracing.RequestTraces(tracer, "omni.pipeline")
        self.entry_stage = entry_stage
        self.terminal_stages: set[str] = (
            set(terminal_stages) if terminal_stages else set()
        )
        self.terminal_stages_resolver = terminal_stages_resolver
        self.partial_results: dict[str, dict[str, Any]] = {}
        self.replica_topology = replica_topology or ReplicaTopology()
        self.logical_process_plan = logical_process_plan or LogicalProcessPlan(
            processes=(), stage_to_process={}
        )
        self.binding_policy = binding_policy or RoundRobinBindingPolicy()
        if max_in_flight is None:
            self.max_in_flight = None
        else:
            value = int(max_in_flight)
            if value < 0:
                raise ValueError("max_in_flight must be >= 0")
            else:
                pass
            self.max_in_flight = value

        # Control plane
        self.control_plane = CoordinatorControlPlane(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Stage registry
        self.stages: dict[str, StageInfo] = {}

        # Request tracking
        self.requests: dict[str, RequestInfo] = {}
        self.completion_futures: dict[str, asyncio.Future] = {}
        self.stream_queues: dict[
            str, asyncio.Queue[CompleteMessage | StreamMessage]
        ] = {}
        # Abort messages carry only the request ID. A strongly held task keeps
        # local admission closed and lets the broadcast survive caller cancellation.
        self.abort_tasks: dict[str, asyncio.Task[bool]] = {}
        self.admin_ops: dict[str, AdminPendingOperation] = {}
        self.admin_lock = asyncio.Lock()

        # State
        self.running = False
        self.fatal_error: str | None = None

    def register_stage(self, name: str, endpoint: str) -> None:
        """Register a stage.

        Args:
            name: Stage name
            endpoint: ZMQ endpoint for the stage
        """
        self.stages[name] = StageInfo(name=name, control_endpoint=endpoint)
        logger.info("Coordinator registered stage: %s at %s", name, endpoint)

    async def start(self) -> None:
        """Start the coordinator."""
        await self.control_plane.start()
        self.running = True
        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self.traces.shutdown()
        await self.stop_sessions()
        self.running = False
        self.control_plane.close()
        logger.info("Coordinator stopped")

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        """Fail all requests currently owned by the coordinator."""
        self.running = False
        message = str(error)
        self.fatal_error = message
        for request_id, info in list(self.requests.items()):
            self.traces.end(request_id, "error", "engine_failure")
            info.state = RequestState.FAILED
            info.error = message
            self.reject_completion_future(request_id, RuntimeError(message))
            queue = self.stream_queues.get(request_id)
            if queue is not None:
                await queue.put(
                    CompleteMessage(
                        request_id=request_id,
                        from_stage="coordinator",
                        success=False,
                        error=message,
                    )
                )
            else:
                pass
        self.requests.clear()
        self.partial_results.clear()
        # Note (Junnan Li): Session pumps await request futures; wake them before waiting for cleanup.
        await self.fail_sessions(message)

    async def shutdown_stages(self, stage_names: Sequence[str] | None = None) -> None:
        """Send shutdown to registered stages, or only to *stage_names*."""
        selected = None if stage_names is None else set(stage_names)
        await self.shutdown_stage_sessions(selected)
        for name, info in self.stages.items():
            if selected is not None and name not in selected:
                continue
            else:
                pass
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
        if not self.running:
            raise RuntimeError("Coordinator is not running")
        else:
            pass

        target_stages = self.resolve_admin_stages(stages)
        if not target_stages:
            raise ValueError("No stages registered for admin operation")
        else:
            pass

        op_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        pending = AdminPendingOperation(
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

        async with self.admin_lock:
            self.admin_ops[op_id] = pending
            try:
                for stage_name in target_stages:
                    info = self.stages[stage_name]
                    await self.control_plane.send_admin(
                        stage_name,
                        info.control_endpoint,
                        AdminMessage(operation=operation),
                    )

                assert pending.future is not None
                results = await asyncio.wait_for(pending.future, timeout=timeout_s)
            finally:
                self.admin_ops.pop(op_id, None)

        return self.aggregate_admin_results(
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

    async def submit(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        trace_headers: dict[str, str] | None = None,
    ) -> Any:
        """Submit a request to the pipeline and wait for completion."""
        self.reject_session_metadata(request)
        await self.submit_request(request_id, request, trace_headers=trace_headers)

        future = self.completion_futures[request_id]
        try:
            result = await future
            return result
        finally:
            self.completion_futures.pop(request_id, None)

    async def stream(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        trace_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion."""
        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()

        self.reject_session_metadata(request)
        try:
            await self.submit_request(
                request_id, request, stream_queue=queue, trace_headers=trace_headers
            )
            expected_terminal_stages = self.expected_terminal_stages(request_id)

            completed_stages: set[str] = set()
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise QueueFullError.from_message(msg.error)
                    else:
                        pass
                    yield msg
                    completed_stages.add(
                        self.replica_topology.logical_name(msg.from_stage)
                    )
                    if (
                        not expected_terminal_stages
                        or completed_stages >= expected_terminal_stages
                    ):
                        return
                    else:
                        pass
                else:
                    yield msg
        finally:
            if self.stream_queues.get(request_id) is queue:
                try:
                    if request_id in self.requests:
                        try:
                            await self.abort(request_id)
                        except Exception:
                            # The coordinator-owned abort task logs its own failure.
                            # Do not replace the exception already leaving the stream.
                            pass
                    else:
                        pass
                finally:
                    if self.stream_queues.get(request_id) is queue:
                        self.stream_queues.pop(request_id, None)
                        self.completion_futures.pop(request_id, None)
                    else:
                        pass
            else:
                pass

    async def submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
        target_stage: str | None = None,
        terminal_stages: set[str] | None = None,
        replica_bindings: dict[str, int] | None = None,
        should_bypass_admission: bool = False,
        trace_headers: dict[str, str] | None = None,
    ) -> None:
        """Submit a request without waiting for completion."""
        if self.fatal_error is not None:
            raise RuntimeError(self.fatal_error)
        else:
            pass
        if self.request_id_is_reserved(request_id):
            raise ValueError(f"Request {request_id} already exists")
        else:
            pass

        if (
            not should_bypass_admission
            and self.max_in_flight is not None
            and len(self.requests) >= self.max_in_flight
        ):
            logger.warning(
                "Rejecting request %s before pipeline submit: in-flight cap "
                "(max_in_flight=%s)",
                request_id,
                self.max_in_flight,
            )
            raise QueueFullError()
        else:
            pass

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        else:
            pass

        if replica_bindings is None:
            replica_bindings = assign_replica_bindings(
                self.logical_process_plan, self.binding_policy, request_id
            )
        else:
            pass
        bindings = replica_bindings or {}
        entry_instance = target_stage or (
            self.replica_topology.resolve(self.entry_stage, bindings[self.entry_stage])
            if self.replica_topology.is_replicated(self.entry_stage)
            else self.entry_stage
        )
        if entry_instance not in self.stages:
            raise ValueError(f"Entry stage {entry_instance} not registered")
        else:
            pass
        entry_info = self.stages[entry_instance]

        # Track request
        self.requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
            terminal_stages=(
                self.resolve_terminal_stages(request)
                if terminal_stages is None
                else terminal_stages
            ),
        )

        # Create future for completion
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.completion_futures[request_id] = future
        if stream_queue is not None:
            self.stream_queues[request_id] = stream_queue
        else:
            pass

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

        self.traces.start(request_id, trace_headers)
        try:
            await self.control_plane.submit_to_stage(
                entry_instance,
                entry_info.control_endpoint,
                SubmitMessage(
                    request_id=request_id,
                    data=payload,
                    replica_bindings=replica_bindings,
                    trace_headers=self.traces.headers(request_id),
                ),
            )
        except asyncio.CancelledError:
            self.traces.end(request_id, "cancelled")
            raise
        except BaseException:
            self.traces.end(request_id, "error", "submit_failed")
            raise

        # Update state
        info = self.requests.get(request_id)
        if info is not None:
            info.state = RequestState.RUNNING
        else:
            pass

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Coordinator submitted req={request_id} to {entry_instance} "
                f"at {entry_info.control_endpoint} bindings={replica_bindings}"
            )
        else:
            pass

    def request_id_is_reserved(self, request_id: str) -> bool:
        """Return whether any coordinator owner still holds this request ID."""
        return (
            request_id in self.requests
            or request_id in self.completion_futures
            or request_id in self.stream_queues
            or request_id in self.abort_tasks
        )

    def reject_completion_future(
        self,
        request_id: str,
        exc: BaseException,
    ) -> None:
        # Note: (Akazaakane) Non-streaming callers await the completion future,
        # so errors must be propagated with set_exception(). Streaming callers
        # receive errors through the stream queue and never await that future;
        # cancel it instead to avoid "Future exception was never retrieved".
        future = self.completion_futures.get(request_id)
        if future is None or future.done():
            return
        else:
            pass
        if request_id in self.stream_queues:
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
        abort_task = self.abort_tasks.get(request_id)
        if abort_task is not None:
            return await asyncio.shield(abort_task)
        else:
            pass

        info = self.requests.get(request_id)
        if info is None:
            return False
        else:
            pass

        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False
        else:
            pass

        abort_task = asyncio.create_task(
            self.run_abort(request_id),
            name=f"coordinator-abort-{request_id}",
        )
        self.abort_tasks[request_id] = abort_task
        abort_task.add_done_callback(
            lambda done, rid=request_id: self.on_abort_task_done(rid, done)
        )
        return await asyncio.shield(abort_task)

    async def run_abort(
        self,
        request_id: str,
    ) -> bool:
        await self.control_plane.broadcast_abort(AbortMessage(request_id=request_id))

        info = self.requests.get(request_id)
        if info is None:
            return False
        else:
            pass

        self.traces.end(request_id, "cancelled")
        info.state = RequestState.ABORTED
        self.reject_completion_future(
            request_id, asyncio.CancelledError(f"Request {request_id} aborted")
        )
        stream_queue = self.stream_queues.get(request_id)
        if stream_queue is not None:
            await stream_queue.put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error="aborted",
                )
            )
        else:
            pass

        self.requests.pop(request_id, None)
        self.partial_results.pop(request_id, None)

        logger.info("Coordinator aborted req=%s", request_id)
        return True

    def on_abort_task_done(
        self,
        request_id: str,
        task: asyncio.Task[bool],
    ) -> None:
        if self.abort_tasks.get(request_id) is task:
            self.abort_tasks.pop(request_id, None)
        else:
            pass
        if task.cancelled():
            logger.warning("Coordinator abort task cancelled for req=%s", request_id)
            return
        else:
            pass
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Failed to abort request %s",
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        else:
            pass

    async def run_completion_loop(self) -> None:
        """Run the completion receiving loop.

        This should be run as a background task.
        """
        try:
            while self.running:
                msg = await self.control_plane.recv_event()
                if isinstance(msg, StreamMessage):
                    await self.handle_stream(msg)
                elif isinstance(msg, AdminResultMessage):
                    self.handle_admin_result(msg.result)
                else:
                    await self.handle_completion(msg)
        except asyncio.CancelledError:
            logger.info("Coordinator completion loop cancelled")
        except Exception as e:
            logger.error("Coordinator completion loop error: %s", e)
            raise

    async def handle_completion(self, msg: CompleteMessage) -> None:
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

        if request_id not in self.requests:
            logger.debug(
                "Coordinator ignored completion for inactive req=%s from %s",
                request_id,
                msg.from_stage,
            )
            return
        else:
            pass

        info = self.requests[request_id]

        # Note (wenyao): the client reads ``from_stage`` off the completion.
        # Observability emits above keep the instance name.
        from_stage = self.replica_topology.logical_name(msg.from_stage)
        if from_stage != msg.from_stage:
            msg = replace(msg, from_stage=from_stage)
        else:
            pass

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            self.traces.end(request_id, "error", "stage_error")
            info.state = RequestState.FAILED
            info.error = msg.error
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id)
            )
            self.partial_results.pop(request_id, None)
            self.reject_completion_future(
                request_id, QueueFullError.from_message(msg.error)
            )
            stream_queue = self.stream_queues.get(request_id)
            if stream_queue is not None:
                await stream_queue.put(msg)
            else:
                pass
            self.requests.pop(request_id, None)
            return
        else:
            pass

        expected_terminal_stages = self.expected_terminal_stages(request_id)
        if expected_terminal_stages and from_stage not in expected_terminal_stages:
            logger.debug(
                "Coordinator ignoring completion from inactive terminal: "
                "req=%s stage=%s expected=%s",
                request_id,
                msg.from_stage,
                sorted(expected_terminal_stages),
            )
            return
        else:
            pass

        # Single active terminal (original behavior) or no terminal_stages configured
        if len(expected_terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self.completion_futures:
                future = self.completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
                else:
                    pass
            else:
                pass
            if request_id in self.stream_queues:
                await self.stream_queues[request_id].put(msg)
            else:
                pass
            self.traces.end(request_id)
            self.requests.pop(request_id, None)
            return
        else:
            pass

        # Multi-terminal: collect partial results
        partials = self.partial_results.setdefault(request_id, {})
        partials[from_stage] = msg.result

        # Forward stream completion per-stage
        if request_id in self.stream_queues:
            await self.stream_queues[request_id].put(msg)
        else:
            pass

        if set(partials) < expected_terminal_stages:
            return  # still waiting
        else:
            pass

        # All terminal stages done -> merge and resolve
        merged = dict(partials)
        self.partial_results.pop(request_id)
        info.state = RequestState.COMPLETED
        info.result = merged

        if request_id in self.completion_futures:
            future = self.completion_futures[request_id]
            if not future.done():
                future.set_result(merged)
            else:
                pass
        else:
            pass
        self.traces.end(request_id)
        self.requests.pop(request_id, None)

    async def handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        request_id = msg.request_id
        handler = self.session_stream_handlers.get(request_id)
        if handler is not None:
            handler(msg)
            return
        else:
            pass
        if request_id not in self.stream_queues:
            return
        else:
            pass
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
        logical = self.replica_topology.logical_name(msg.from_stage)
        stage_name = (
            self.replica_topology.logical_name(msg.stage_name)
            if msg.stage_name is not None
            else msg.stage_name
        )
        if logical != msg.from_stage or stage_name != msg.stage_name:
            msg = replace(msg, from_stage=logical, stage_name=stage_name)
        else:
            pass
        await self.stream_queues[request_id].put(msg)

    def handle_admin_result(self, result: AdminResult) -> None:
        pending = self.admin_ops.get(result.op_id)
        if pending is None:
            logger.warning(
                "Coordinator received admin result for unknown op=%s stage=%s",
                result.op_id,
                result.stage,
            )
            return
        else:
            pass
        pending.results[result.stage] = result
        if (
            pending.future is not None
            and pending.results.keys() >= pending.expected_stages
        ):
            if not pending.future.done():
                pending.future.set_result(dict(pending.results))
            else:
                pass
        else:
            pass

    def resolve_admin_stages(self, stages: Sequence[str] | None) -> list[str]:
        if stages is None:
            return sorted(self.stages)
        else:
            pass
        # Note (wenyao): dedup preserving order so a caller passing both a
        # logical name and one of its instances does not double-send admin ops.
        resolved: list[str] = []
        for name in stages:
            for instance in self.replica_topology.instances(name):
                if instance not in resolved:
                    resolved.append(instance)
                else:
                    pass
        unknown = sorted(set(resolved) - set(self.stages))
        if unknown:
            raise ValueError(f"Unknown admin target stage(s): {unknown}")
        else:
            pass
        return resolved

    def aggregate_admin_results(
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
        return self.requests.get(request_id)

    def resolve_terminal_stages(self, request: OmniRequest) -> set[str]:
        if self.terminal_stages_resolver is None:
            return set(self.terminal_stages)
        else:
            pass
        resolved = self.terminal_stages_resolver(request)
        if resolved is None:
            return set(self.terminal_stages)
        else:
            pass
        if isinstance(resolved, str) or not isinstance(resolved, Sequence):
            raise ValueError(
                "terminal_stages_resolver must return a sequence of terminal "
                "stage names or None"
            )
        else:
            pass
        if not all(isinstance(stage, str) for stage in resolved):
            raise ValueError(
                "terminal_stages_resolver must return terminal stage names"
            )
        else:
            pass
        resolved_stages = set(resolved)
        if not resolved_stages:
            raise ValueError("terminal_stages_resolver returned no terminal stages")
        else:
            pass
        unknown = resolved_stages - self.terminal_stages
        if unknown:
            raise ValueError(
                "terminal_stages_resolver returned stages outside the static "
                f"terminal stages: {sorted(unknown)}. Allowed terminal stages: "
                f"{sorted(self.terminal_stages)}"
            )
        else:
            pass
        return resolved_stages

    def expected_terminal_stages(self, request_id: str) -> set[str]:
        info = self.requests.get(request_id)
        if info is None or info.terminal_stages is None:
            return set(self.terminal_stages)
        else:
            pass
        return info.terminal_stages

    def health(self) -> dict[str, Any]:
        """Return health status."""
        state_counts = {}
        for info in self.requests.values():
            state = info.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "running": self.running,
            "stages": list(self.stages.keys()),
            "entry_stage": self.entry_stage,
            "total_requests": len(self.requests),
            "pending_completions": len(self.completion_futures),
            "request_states": state_counts,
        }
