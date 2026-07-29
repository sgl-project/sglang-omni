# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable

from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
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
    SessionCommandMessage,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
    is_update_action,
)

logger = logging.getLogger(__name__)


@dataclass
class _AdminPendingOperation:
    expected_stages: set[str]
    action: str
    results: dict[str, AdminResult] = field(default_factory=dict)
    future: asyncio.Future | None = None


@dataclass
class _SessionState:
    generation: int
    input_seq: int = 0
    response_epoch: int = 0
    closing: bool = False
    command_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _SessionEventStream:
    """Async iterator whose ``aclose`` works even before first iteration."""

    def __init__(
        self,
        iterator: AsyncIterator[CompleteMessage | StreamMessage],
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        self._iterator = iterator
        self._cleanup = cleanup
        self._started = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    def __aiter__(self) -> "_SessionEventStream":
        return self

    async def __anext__(self) -> CompleteMessage | StreamMessage:
        if self._closed:
            raise StopAsyncIteration
        self._started = True
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            try:
                if self._started:
                    close = getattr(self._iterator, "aclose", None)
                    if close is not None:
                        await close()
                else:
                    await self._cleanup()
                    close = getattr(self._iterator, "aclose", None)
                    if close is not None:
                        await close()
            finally:
                self._closed = True


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
    ):
        """Initialize coordinator.

        Args:
            completion_endpoint: ZMQ endpoint to receive completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            entry_stage: Name of the entry stage for new requests
            terminal_stages: Terminal stage names. When multiple are given,
                the coordinator waits for all to complete before resolving.
        """
        self.entry_stage = entry_stage
        self._terminal_stages: set[str] = (
            set(terminal_stages) if terminal_stages else set()
        )
        self._terminal_stages_resolver = terminal_stages_resolver
        self._partial_results: dict[str, dict[str, Any]] = {}

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
        self._admin_ops: dict[str, _AdminPendingOperation] = {}
        self._admin_lock = asyncio.Lock()
        self._sessions: dict[str, _SessionState] = {}
        # Unlike ``_sessions``, this lease remains until the session stream
        # iterator is closed. It lets the completion path reject late writes
        # to a bounded session queue without changing legacy stream behavior.
        self._session_stream_ids: set[str] = set()
        self._uncertain_session_ids: set[str] = set()
        self._next_session_generation = 0

        # State
        self._running = False
        self._fatal_error: str | None = None

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
        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        if self._requests:
            await self.fail_pending_requests(RuntimeError("coordinator stopped"))
        else:
            self._running = False
        self.control_plane.close()
        logger.info("Coordinator stopped")

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        """Fail all requests currently owned by the coordinator."""
        self._running = False
        message = str(error)
        self._fatal_error = message
        for request_id, session in list(self._sessions.items()):
            async with session.command_lock:
                if self._sessions.get(request_id) is not session:
                    continue
                await self._fail_pending_request(request_id, message, session)
        for request_id in list(self._requests):
            await self._fail_pending_request(request_id, message, None)

    async def _fail_pending_request(
        self,
        request_id: str,
        message: str,
        session: _SessionState | None,
    ) -> None:
        info = self._requests.get(request_id)
        if info is None:
            return
        info.state = RequestState.FAILED
        info.error = message
        self._reject_completion_future(request_id, RuntimeError(message))
        if request_id in self._stream_queues:
            await self._enqueue_session_message(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error=message,
                    generation=None if session is None else session.generation,
                ),
                terminal=True,
                locked_session=session,
            )
        self._requests.pop(request_id, None)
        self._partial_results.pop(request_id, None)
        self._sessions.pop(request_id, None)

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

    async def stream(
        self, request_id: str, request: OmniRequest | Any
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion."""
        if request_id in self._stream_queues:
            raise ValueError(f"Request {request_id} already streaming")

        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()
        self._stream_queues[request_id] = queue

        try:
            await self._submit_request(request_id, request)
            expected_terminal_stages = self._expected_terminal_stages(request_id)

            completed_stages: set[str] = set()
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise RuntimeError(msg.error or "Unknown error")
                    yield msg
                    completed_stages.add(msg.from_stage)
                    if (
                        not expected_terminal_stages
                        or completed_stages >= expected_terminal_stages
                    ):
                        return
                else:
                    yield msg
        finally:
            self._stream_queues.pop(request_id, None)
            self._completion_futures.pop(request_id, None)

    async def open_session(
        self,
        session_id: str,
        request: OmniRequest | Any,
        *,
        output_queue_size: int = 64,
    ) -> tuple[int, AsyncIterator[CompleteMessage | StreamMessage]]:
        """Open a session whose initial admission uses the normal request path."""
        if not isinstance(session_id, str) or not session_id:
            raise TypeError("session_id must be a non-empty str")
        if type(output_queue_size) is not int or output_queue_size <= 0:
            raise ValueError("output_queue_size must be a positive int")
        if session_id in self._uncertain_session_ids:
            raise RuntimeError(
                f"Session {session_id} cannot be reopened after an uncertain abort"
            )
        if session_id in self._sessions or session_id in self._requests:
            raise ValueError(f"Session {session_id} already exists")
        if session_id in self._stream_queues:
            raise ValueError(f"Session {session_id} already streaming")

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)
        self._next_session_generation += 1
        generation = self._next_session_generation
        metadata = dict(request.metadata)
        metadata["duplex_session"] = {
            "session_id": session_id,
            "generation": generation,
            "input_seq": 0,
            "response_epoch": 0,
        }
        request = OmniRequest(
            inputs=request.inputs,
            params=dict(request.params),
            metadata=metadata,
        )
        terminal_stages = self._resolve_terminal_stages(request)
        if len(terminal_stages) > 1:
            raise ValueError(
                "duplex sessions require exactly one active terminal stage"
            )

        # Keep one slot reserved for a successful terminal message so closing a
        # session never discards already accepted output chunks.
        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue(
            maxsize=output_queue_size + 1
        )
        self._stream_queues[session_id] = queue
        self._session_stream_ids.add(session_id)
        self._sessions[session_id] = _SessionState(generation=generation)
        try:
            await self._submit_request(session_id, request)
        except BaseException:
            try:
                await self.control_plane.broadcast_abort(
                    AbortMessage(request_id=session_id, generation=generation)
                )
            except BaseException as abort_exc:
                self._uncertain_session_ids.add(session_id)
                logger.warning(
                    "Failed to fence cancelled session admission %s generation=%s: %s",
                    session_id,
                    generation,
                    abort_exc,
                )
            finally:
                self._sessions.pop(session_id, None)
                self._session_stream_ids.discard(session_id)
                self._stream_queues.pop(session_id, None)
                self._completion_futures.pop(session_id, None)
                self._requests.pop(session_id, None)
            raise
        expected_terminal_stages = self._expected_terminal_stages(session_id)

        async def _cleanup_session_stream() -> None:
            try:
                if session_id in self._sessions:
                    await self.abort(session_id)
            finally:
                self._session_stream_ids.discard(session_id)
                self._stream_queues.pop(session_id, None)
                self._completion_futures.pop(session_id, None)

        async def _events() -> AsyncIterator[CompleteMessage | StreamMessage]:
            completed_stages: set[str] = set()
            try:
                while True:
                    message = await queue.get()
                    yield message
                    if self._is_terminal_session_stream(message):
                        return
                    if isinstance(message, CompleteMessage):
                        if not message.success:
                            return
                        completed_stages.add(message.from_stage)
                        if (
                            not expected_terminal_stages
                            or completed_stages >= expected_terminal_stages
                        ):
                            return
            finally:
                await _cleanup_session_stream()

        return generation, _SessionEventStream(
            _events(),
            _cleanup_session_stream,
        )

    async def send_session_command(
        self,
        session_id: str,
        generation: int,
        command: str,
        data: dict[str, Any] | None = None,
    ) -> SessionCommandMessage:
        """Order and deliver one command to the active session entry stage."""
        state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session {session_id} is not active")
        if type(generation) is not int or generation != state.generation:
            raise ValueError(
                f"Stale session generation {generation}; active generation is "
                f"{state.generation}"
            )
        if command not in SessionCommandMessage._COMMANDS:
            raise ValueError(
                f"command must be one of {sorted(SessionCommandMessage._COMMANDS)}"
            )
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("session command data must be a dict")

        async with state.command_lock:
            if self._sessions.get(session_id) is not state:
                raise ValueError(f"Session {session_id} is not active")
            if state.closing:
                raise RuntimeError(f"Session {session_id} is closing")

            input_seq = state.input_seq + 1
            response_epoch = state.response_epoch
            if command in {"interrupt", "close"}:
                response_epoch += 1
            message = SessionCommandMessage(
                session_id=session_id,
                generation=generation,
                input_seq=input_seq,
                response_epoch=response_epoch,
                command=command,
                data=data,
            )
            message.to_dict()
            entry_info = self._stages[self.entry_stage]
            try:
                await self.control_plane.submit_to_stage(
                    self.entry_stage,
                    entry_info.control_endpoint,
                    message,
                )
            except BaseException:
                state.closing = True
                cleanup_task = asyncio.create_task(
                    self._abort_request(session_id, state)
                )
                try:
                    await asyncio.shield(cleanup_task)
                except BaseException as cleanup_exc:
                    logger.warning(
                        "Failed to clean up ambiguous session command %s "
                        "generation=%s input_seq=%s: %s",
                        session_id,
                        generation,
                        input_seq,
                        cleanup_exc,
                    )
                raise
            if self._sessions.get(session_id) is not state:
                raise ValueError(f"Session {session_id} is not active")
            state.input_seq = input_seq
            state.response_epoch = response_epoch
            if command == "close":
                state.closing = True
            return message

    async def _submit_request(
        self, request_id: str, request: OmniRequest | Any
    ) -> None:
        """Submit a request without waiting for completion."""
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if request_id in self._requests:
            raise ValueError(f"Request {request_id} already exists")

        if self.entry_stage not in self._stages:
            raise ValueError(f"Entry stage {self.entry_stage} not registered")

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

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

        # Submit to entry stage
        entry_info = self._stages[self.entry_stage]
        await self.control_plane.submit_to_stage(
            self.entry_stage,
            entry_info.control_endpoint,
            SubmitMessage(request_id=request_id, data=payload),
        )

        # Update state
        self._requests[request_id].state = RequestState.RUNNING

        logger.info(
            "Coordinator submitted req=%s to %s at %s",
            request_id,
            self.entry_stage,
            entry_info.control_endpoint,
        )

    def _reject_completion_future(self, request_id: str, exc: BaseException) -> None:
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
        session = self._sessions.get(request_id)
        if session is not None:
            async with session.command_lock:
                if self._sessions.get(request_id) is not session:
                    return False
                return await self._abort_request(request_id, session)
        return await self._abort_request(request_id, None)

    async def _abort_request(
        self,
        request_id: str,
        session: _SessionState | None,
    ) -> bool:
        if request_id not in self._requests:
            return False

        info = self._requests[request_id]
        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False

        transport_error: BaseException | None = None
        try:
            await self.control_plane.broadcast_abort(
                AbortMessage(
                    request_id=request_id,
                    generation=None if session is None else session.generation,
                )
            )
        except BaseException as exc:
            transport_error = exc

        try:
            info.state = RequestState.ABORTED
            self._reject_completion_future(
                request_id, asyncio.CancelledError(f"Request {request_id} aborted")
            )
            if request_id in self._stream_queues:
                await self._enqueue_session_message(
                    CompleteMessage(
                        request_id=request_id,
                        from_stage="coordinator",
                        success=False,
                        error="aborted",
                        generation=(None if session is None else session.generation),
                    ),
                    terminal=True,
                )
        finally:
            self._requests.pop(request_id, None)
            self._partial_results.pop(request_id, None)
            self._sessions.pop(request_id, None)

        logger.info("Coordinator aborted req=%s", request_id)
        if transport_error is not None:
            if session is not None:
                self._uncertain_session_ids.add(request_id)
            raise transport_error
        return True

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
        """Handle a completion message from a stage."""
        request_id = msg.request_id
        logger.debug(
            "Coordinator received completion: req=%s from %s success=%s",
            request_id,
            msg.from_stage,
            msg.success,
        )
        if request_id not in self._requests:
            log = (
                logger.debug
                if request_id in self._session_stream_ids
                else logger.warning
            )
            log("Coordinator received completion for unknown req=%s", request_id)
            return

        session = self._sessions.get(request_id)
        if session is None:
            self._emit_terminal_response_event(msg)
            await self._complete_request(msg)
            return
        if msg.generation != session.generation:
            logger.debug(
                "Coordinator ignored stale session completion req=%s "
                "generation=%s active_generation=%s",
                request_id,
                msg.generation,
                session.generation,
            )
            return
        async with session.command_lock:
            if (
                self._sessions.get(request_id) is not session
                or request_id not in self._requests
                or msg.generation != session.generation
            ):
                logger.debug(
                    "Coordinator ignored completion for inactive session req=%s",
                    request_id,
                )
                return
            self._emit_terminal_response_event(msg)
            await self._complete_request(msg)

    @staticmethod
    def _emit_terminal_response_event(msg: CompleteMessage) -> None:
        _emit_event(
            request_id=msg.request_id,
            stage="coordinator",
            event_name="terminal_response",
            metadata={
                "from_stage": msg.from_stage,
                "success": msg.success,
                "generation": msg.generation,
            },
        )

    async def _complete_request(self, msg: CompleteMessage) -> None:
        """Apply one completion while the session command lock is held."""
        request_id = msg.request_id

        info = self._requests[request_id]

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            info.state = RequestState.FAILED
            info.error = msg.error
            session = self._sessions.get(request_id)
            try:
                await self.control_plane.broadcast_abort(
                    AbortMessage(
                        request_id=request_id,
                        generation=None if session is None else session.generation,
                    )
                )
            except BaseException as exc:
                if session is not None:
                    self._uncertain_session_ids.add(request_id)
                logger.warning(
                    "Failed to broadcast terminal abort for %s: %s",
                    request_id,
                    exc,
                )
            self._partial_results.pop(request_id, None)
            self._reject_completion_future(
                request_id, RuntimeError(msg.error or "Unknown error")
            )
            if request_id in self._stream_queues:
                await self._enqueue_session_message(msg, terminal=True)
            self._requests.pop(request_id, None)
            self._sessions.pop(request_id, None)
            return

        expected_terminal_stages = self._expected_terminal_stages(request_id)
        if expected_terminal_stages and msg.from_stage not in expected_terminal_stages:
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
                await self._enqueue_session_message(msg, terminal=True)
            self._requests.pop(request_id, None)
            self._sessions.pop(request_id, None)
            return

        # Multi-terminal: collect partial results
        partials = self._partial_results.setdefault(request_id, {})
        partials[msg.from_stage] = msg.result

        # Forward stream completion per-stage
        is_final = set(partials) >= expected_terminal_stages
        if request_id in self._stream_queues:
            await self._enqueue_session_message(msg, terminal=is_final)

        if not is_final:
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
        self._sessions.pop(request_id, None)

    async def _handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        request_id = msg.request_id
        if request_id not in self._stream_queues:
            return
        session = self._sessions.get(request_id)
        if session is not None:
            if msg.generation != session.generation:
                logger.debug(
                    "Coordinator ignored stale session stream req=%s "
                    "generation=%s active_generation=%s",
                    request_id,
                    msg.generation,
                    session.generation,
                )
                return
            async with session.command_lock:
                if (
                    self._sessions.get(request_id) is not session
                    or request_id not in self._requests
                    or msg.generation != session.generation
                ):
                    return
                await self._handle_current_session_stream(msg, session)
            return
        if request_id in self._session_stream_ids and request_id not in self._requests:
            return
        await self._enqueue_stream_message(msg)

    async def _handle_current_session_stream(
        self,
        msg: StreamMessage,
        session: _SessionState,
    ) -> None:
        await self._enqueue_stream_message(msg, locked_session=session)
        if not self._is_terminal_session_stream(msg):
            return

        request_id = msg.request_id
        info = self._requests.get(request_id)
        event_type = msg.chunk.get("type")
        if info is not None:
            if event_type == "session.closed":
                info.state = RequestState.COMPLETED
                info.result = msg.chunk
                future = self._completion_futures.get(request_id)
                if future is not None and not future.done():
                    future.set_result(msg.chunk)
            else:
                error = str(msg.chunk.get("error") or "session failed")
                info.state = RequestState.FAILED
                info.error = error
                self._reject_completion_future(request_id, RuntimeError(error))
                try:
                    await self.control_plane.broadcast_abort(
                        AbortMessage(
                            request_id=request_id,
                            generation=session.generation,
                        )
                    )
                except BaseException as exc:
                    self._uncertain_session_ids.add(request_id)
                    logger.warning(
                        "Failed to broadcast session error abort for %s: %s",
                        request_id,
                        exc,
                    )
        self._partial_results.pop(request_id, None)
        self._requests.pop(request_id, None)
        self._sessions.pop(request_id, None)

    async def _enqueue_stream_message(
        self,
        msg: StreamMessage,
        *,
        locked_session: _SessionState | None = None,
    ) -> None:
        request_id = msg.request_id
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
        terminal = self._is_terminal_session_stream(msg)
        await self._enqueue_session_message(
            msg,
            terminal=terminal,
            locked_session=locked_session,
        )

    @staticmethod
    def _is_terminal_session_stream(
        msg: CompleteMessage | StreamMessage,
    ) -> bool:
        return (
            isinstance(msg, StreamMessage)
            and isinstance(msg.chunk, dict)
            and msg.chunk.get("type") in {"session.closed", "session.error"}
        )

    async def _enqueue_session_message(
        self,
        msg: CompleteMessage | StreamMessage,
        *,
        terminal: bool,
        locked_session: _SessionState | None = None,
    ) -> None:
        request_id = msg.request_id
        queue = self._stream_queues.get(request_id)
        if queue is None:
            return
        if request_id not in self._sessions:
            await queue.put(msg)
            return
        if terminal:
            successful_terminal = (
                isinstance(msg, CompleteMessage)
                and msg.success
                or isinstance(msg, StreamMessage)
                and isinstance(msg.chunk, dict)
                and msg.chunk.get("type") == "session.closed"
            )
            if successful_terminal:
                try:
                    queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.debug(
                        "Coordinator dropped duplicate successful terminal for %s",
                        request_id,
                    )
            else:
                self._replace_session_queue_with_terminal(queue, msg)
            return
        if queue.qsize() >= queue.maxsize - 1:
            if locked_session is None:
                await self._fail_session_output_overflow(request_id, queue)
            else:
                await self._fail_session_output_overflow_locked(
                    request_id,
                    queue,
                    locked_session,
                )
            return
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            if locked_session is None:
                await self._fail_session_output_overflow(request_id, queue)
            else:
                await self._fail_session_output_overflow_locked(
                    request_id,
                    queue,
                    locked_session,
                )

    @staticmethod
    def _replace_session_queue_with_terminal(
        queue: asyncio.Queue[CompleteMessage | StreamMessage],
        msg: CompleteMessage | StreamMessage,
    ) -> None:
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(msg)

    async def _fail_session_output_overflow(
        self,
        request_id: str,
        queue: asyncio.Queue[CompleteMessage | StreamMessage],
    ) -> None:
        state = self._sessions.get(request_id)
        if state is None:
            return
        async with state.command_lock:
            if self._sessions.get(request_id) is not state:
                return
            await self._fail_session_output_overflow_locked(request_id, queue, state)

    async def _fail_session_output_overflow_locked(
        self,
        request_id: str,
        queue: asyncio.Queue[CompleteMessage | StreamMessage],
        state: _SessionState,
    ) -> None:
        error = "duplex session output queue overflow"
        try:
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id, generation=state.generation)
            )
        except BaseException as exc:
            self._uncertain_session_ids.add(request_id)
            logger.warning(
                "Failed to broadcast overflow abort for %s: %s",
                request_id,
                exc,
            )
        finally:
            info = self._requests.get(request_id)
            if info is not None:
                info.state = RequestState.FAILED
                info.error = error
            self._reject_completion_future(request_id, RuntimeError(error))
            self._replace_session_queue_with_terminal(
                queue,
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error=error,
                    generation=state.generation,
                ),
            )
            self._partial_results.pop(request_id, None)
            self._requests.pop(request_id, None)
            self._sessions.pop(request_id, None)

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
        resolved = list(stages)
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
            "active_sessions": len(self._sessions),
            "uncertain_sessions": len(self._uncertain_session_ids),
            "request_states": state_counts,
        }


async def run_coordinator(
    completion_endpoint: str,
    abort_endpoint: str,
    entry_stage: str,
    stages: dict[str, str],  # name -> endpoint
    terminal_stages: list[str] | None = None,
    terminal_stages_resolver: Callable[[OmniRequest], list[str] | None] | None = None,
) -> Coordinator:
    """Create and start a coordinator.

    Args:
        completion_endpoint: ZMQ endpoint to receive completions
        abort_endpoint: ZMQ endpoint for abort broadcasts
        entry_stage: Name of the entry stage
        stages: Dict of stage_name -> stage_endpoint
        terminal_stages: Optional list of terminal stage names for multi-terminal merge

    Returns:
        Started Coordinator instance
    """
    coordinator = Coordinator(
        completion_endpoint=completion_endpoint,
        abort_endpoint=abort_endpoint,
        entry_stage=entry_stage,
        terminal_stages=terminal_stages,
        terminal_stages_resolver=terminal_stages_resolver,
    )

    # Register stages
    for name, endpoint in stages.items():
        coordinator.register_stage(name, endpoint)

    # Start
    await coordinator.start()

    return coordinator
