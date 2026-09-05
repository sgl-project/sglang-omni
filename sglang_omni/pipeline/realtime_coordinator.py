# SPDX-License-Identifier: Apache-2.0
"""Coordinator extension for bidirectional realtime requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from sglang_omni.config.topology import LogicalProcessPlan
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.replicas import BindingPolicy, ReplicaTopology
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import (
    AbortMessage,
    CompleteMessage,
    OmniRequest,
    RequestInfo,
    RequestState,
    StagePayload,
    StreamMessage,
    SubmitMessage,
)

logger = logging.getLogger(__name__)

RealtimeOutput = CompleteMessage | StreamMessage


@dataclass(slots=True)
class _RealtimeRequestContext:
    info: RequestInfo
    output_queue: asyncio.Queue[RealtimeOutput] = field(default_factory=asyncio.Queue)
    partial_results: dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        return self.info.request_id

    @property
    def is_running(self) -> bool:
        return self.info.state is RequestState.RUNNING

    @property
    def terminal_stages(self) -> set[str]:
        return set(self.info.terminal_stages or ())

    def publish(self, message: RealtimeOutput) -> None:
        self.output_queue.put_nowait(message)

    def complete(self, result: Any) -> None:
        self.info.state = RequestState.COMPLETED
        self.info.result = result

    def fail(self, error: str, message: CompleteMessage) -> None:
        self.info.state = RequestState.FAILED
        self.info.error = error
        self.publish(message)

    def abort(self, message: CompleteMessage) -> None:
        self.info.state = RequestState.ABORTED
        self.publish(message)

    def fail_submission(self, error: BaseException) -> None:
        if not self.is_running:
            return
        self.info.state = RequestState.FAILED
        self.info.error = str(error)


class RealtimeStream(AsyncIterator[RealtimeOutput]):
    """Turn-scoped bidirectional stream owned by RealtimeCoordinator."""

    def __init__(
        self,
        context: _RealtimeRequestContext,
        abort: Callable[[], Awaitable[bool]],
        send_input: Callable[[Any], Awaitable[None]],
        *,
        session_id: str,
        turn_id: str,
        input_stage: str,
    ) -> None:
        self._context = context
        self._abort = abort
        self._send_input = send_input
        self._send_lock = asyncio.Lock()
        self._completed_stages: set[str] = set()
        self._closed = False
        self.session_id = session_id
        self.turn_id = turn_id
        self.input_stage = input_stage

    @property
    def request_id(self) -> str:
        return self._context.request_id

    def __aiter__(self) -> "RealtimeStream":
        return self

    async def __anext__(self) -> RealtimeOutput:
        if self._closed:
            raise StopAsyncIteration

        message = await self._context.output_queue.get()
        if not isinstance(message, CompleteMessage):
            return message
        if not message.success:
            self._closed = True
            raise RuntimeError(message.error or "Unknown error")

        self._completed_stages.add(message.from_stage)
        expected = self._context.terminal_stages
        if not expected or self._completed_stages >= expected:
            self._closed = True
        return message

    async def send_input(self, message: Any) -> None:
        self._ensure_active()
        self._validate_identity(message)
        async with self._send_lock:
            self._ensure_active()
            await self._send_input(message)

    async def aclose(self) -> None:
        if self._closed:
            return
        try:
            if self._context.is_running:
                await self._abort()
        finally:
            self._closed = True

    def _ensure_active(self) -> None:
        if self._closed or not self._context.is_running:
            raise ValueError(f"Realtime request {self.request_id} is not active")

    def _validate_identity(self, message: Any) -> None:
        if getattr(message, "request_id", None) != self.request_id:
            raise ValueError(
                "realtime input message request_id must match its request handle"
            )
        for name, expected in (
            ("session_id", self.session_id),
            ("turn_id", self.turn_id),
        ):
            if getattr(message, name, expected) != expected:
                raise ValueError(
                    f"realtime input message {name} must match its request handle"
                )


class RealtimeCoordinator(Coordinator):
    """Coordinator variant selected only for realtime-capable pipelines."""

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
    ) -> None:
        super().__init__(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
            entry_stage=entry_stage,
            terminal_stages=terminal_stages,
            terminal_stages_resolver=terminal_stages_resolver,
            replica_topology=replica_topology,
            logical_process_plan=logical_process_plan,
            binding_policy=binding_policy,
            max_in_flight=max_in_flight,
        )
        self._realtime_requests: dict[str, _RealtimeRequestContext] = {}

    async def open_realtime(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        session_id: str,
        turn_id: str,
        input_stage: str,
    ) -> RealtimeStream:
        for name, value in (
            ("session_id", session_id),
            ("turn_id", turn_id),
            ("input_stage", input_stage),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if self._request_id_is_reserved(request_id):
            raise ValueError(f"Request {request_id} already exists")
        if self.entry_stage not in self._stages:
            raise ValueError(f"Entry stage {self.entry_stage} not registered")
        if input_stage not in self._stages:
            raise ValueError(f"Realtime input stage {input_stage!r} is not registered")

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

        context = _RealtimeRequestContext(
            info=RequestInfo(
                request_id=request_id,
                state=RequestState.RUNNING,
                current_stage=self.entry_stage,
                terminal_stages=self._resolve_terminal_stages(request),
            )
        )
        self._realtime_requests[request_id] = context

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

        entry_info = self._stages[self.entry_stage]
        try:
            await self.control_plane.submit_to_stage(
                self.entry_stage,
                entry_info.control_endpoint,
                SubmitMessage(request_id=request_id, data=payload),
            )
        except BaseException as error:
            self._release_realtime_context(context)
            context.fail_submission(error)
            raise

        input_stage_info = self._stages[input_stage]

        async def send_input(message: Any) -> None:
            await self.control_plane.submit_to_stage(
                input_stage,
                input_stage_info.control_endpoint,
                message,
            )

        logger.info(
            "RealtimeCoordinator submitted req=%s to %s at %s",
            request_id,
            self.entry_stage,
            entry_info.control_endpoint,
        )
        return RealtimeStream(
            context,
            lambda: self._abort_realtime_context(context),
            send_input,
            session_id=session_id,
            turn_id=turn_id,
            input_stage=input_stage,
        )

    async def _submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
    ) -> None:
        await super()._submit_request(
            request_id,
            request,
            stream_queue=stream_queue,
        )

    def _request_id_is_reserved(self, request_id: str) -> bool:
        return request_id in self._realtime_requests or super()._request_id_is_reserved(
            request_id
        )

    async def abort(self, request_id: str) -> bool:
        context = self._realtime_requests.get(request_id)
        if context is not None:
            return await self._abort_realtime_context(context)
        return await super().abort(request_id)

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        message = str(error)
        for context in list(self._realtime_requests.values()):
            context.fail(
                message,
                CompleteMessage(
                    request_id=context.request_id,
                    from_stage="coordinator",
                    success=False,
                    error=message,
                ),
            )
            self._release_realtime_context(context)
        await super().fail_pending_requests(error)

    async def _handle_completion(self, message: CompleteMessage) -> None:
        context = self._realtime_requests.get(message.request_id)
        if context is None:
            await super()._handle_completion(message)
            return

        request_id = message.request_id
        logger.debug(
            "RealtimeCoordinator received completion: req=%s from %s success=%s",
            request_id,
            message.from_stage,
            message.success,
        )
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="terminal_response",
            metadata={
                "from_stage": message.from_stage,
                "success": message.success,
            },
        )

        if not message.success:
            context.fail(message.error or "Unknown error", message)
            self._release_realtime_context(context)
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id)
            )
            return

        terminal_stages = context.terminal_stages
        if terminal_stages and message.from_stage not in terminal_stages:
            logger.debug(
                "RealtimeCoordinator ignoring completion from inactive terminal: "
                "req=%s stage=%s expected=%s",
                request_id,
                message.from_stage,
                sorted(terminal_stages),
            )
            return

        if len(terminal_stages) <= 1:
            context.complete(message.result)
            context.publish(message)
            self._release_realtime_context(context)
            return

        context.partial_results[message.from_stage] = message.result
        context.publish(message)
        if set(context.partial_results) < terminal_stages:
            return

        context.complete(dict(context.partial_results))
        self._release_realtime_context(context)

    async def _handle_stream(self, message: StreamMessage) -> None:
        context = self._realtime_requests.get(message.request_id)
        if context is None:
            await super()._handle_stream(message)
            return

        _emit_event(
            request_id=message.request_id,
            stage="coordinator",
            event_name="coordinator_stream_received",
            metadata={
                "from_stage": message.from_stage,
                "chunk_id": message.chunk_id,
                "modality": message.modality,
            },
        )
        _emit_event(
            request_id=message.request_id,
            stage="coordinator",
            event_name="stage_stream_chunk_received",
            metadata={
                "from_stage": message.from_stage,
                "chunk_id": message.chunk_id,
                "modality": message.modality,
            },
        )
        context.publish(message)

    def get_request_info(self, request_id: str) -> RequestInfo | None:
        context = self._realtime_requests.get(request_id)
        if context is not None:
            return context.info
        return super().get_request_info(request_id)

    def health(self) -> dict[str, Any]:
        status = super().health()
        realtime_count = len(self._realtime_requests)
        status["total_requests"] += realtime_count
        status["pending_completions"] += realtime_count
        for context in self._realtime_requests.values():
            state = context.info.state.value
            state_counts = status["request_states"]
            state_counts[state] = state_counts.get(state, 0) + 1
        return status

    async def _abort_realtime_context(self, context: _RealtimeRequestContext) -> bool:
        request_id = context.request_id
        abort_task = self._abort_tasks.get(request_id)
        if abort_task is not None:
            return await asyncio.shield(abort_task)
        if (
            self._realtime_requests.get(request_id) is not context
            or not context.is_running
        ):
            return False

        abort_task = asyncio.create_task(
            self._run_realtime_abort(context),
            name=f"realtime-coordinator-abort-{request_id}",
        )
        self._abort_tasks[request_id] = abort_task
        abort_task.add_done_callback(
            lambda done, rid=request_id: self._on_abort_task_done(rid, done)
        )
        return await asyncio.shield(abort_task)

    async def _run_realtime_abort(
        self,
        context: _RealtimeRequestContext,
    ) -> bool:
        request_id = context.request_id
        await self.control_plane.broadcast_abort(AbortMessage(request_id=request_id))
        if (
            self._realtime_requests.get(request_id) is not context
            or not context.is_running
        ):
            return False

        context.abort(
            CompleteMessage(
                request_id=request_id,
                from_stage="coordinator",
                success=False,
                error="aborted",
            )
        )
        self._release_realtime_context(context)
        logger.info("RealtimeCoordinator aborted req=%s", request_id)
        return True

    def _release_realtime_context(self, context: _RealtimeRequestContext) -> bool:
        if self._realtime_requests.get(context.request_id) is not context:
            return False
        self._realtime_requests.pop(context.request_id)
        return True
