# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
from typing import Any, AsyncIterator

from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
from sglang_omni.proto import (
    AbortMessage,
    CompleteMessage,
    OmniRequest,
    RequestInfo,
    RequestState,
    StageInfo,
    StagePayload,
    StreamMessage,
    SubmitMessage,
)

logger = logging.getLogger(__name__)


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

        # State
        self._running = False

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
        self._running = False
        self.control_plane.close()
        logger.info("Coordinator stopped")

    async def shutdown_stages(self) -> None:
        """Send shutdown signal to all registered stages."""
        for name, info in self._stages.items():
            try:
                await self.control_plane.send_shutdown(name, info.control_endpoint)
                logger.info("Sent shutdown to stage: %s", name)
            except Exception as e:
                logger.warning("Failed to send shutdown to stage %s: %s", name, e)

    async def submit(self, request_id: str, request: OmniRequest | Any) -> Any:
        """Submit a request to the pipeline and wait for completion."""
        await self._submit_request(request_id, request)

        future = self._completion_futures[request_id]
        result = await future
        del self._completion_futures[request_id]
        return result

    async def stream(
        self, request_id: str, request: OmniRequest | Any
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion."""
        if request_id in self._stream_queues:
            raise ValueError(f"Request {request_id} already streaming")

        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()
        self._stream_queues[request_id] = queue

        await self._submit_request(request_id, request)

        completed_stages: set[str] = set()
        try:
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise RuntimeError(msg.error or "Unknown error")
                    yield msg
                    completed_stages.add(msg.from_stage)
                    if (
                        not self._terminal_stages
                        or completed_stages >= self._terminal_stages
                    ):
                        return
                else:
                    yield msg
        finally:
            self._stream_queues.pop(request_id, None)
            self._completion_futures.pop(request_id, None)

    async def _submit_request(
        self, request_id: str, request: OmniRequest | Any
    ) -> None:
        """Submit a request without waiting for completion."""
        if request_id in self._requests:
            raise ValueError(f"Request {request_id} already exists")

        if self.entry_stage not in self._stages:
            raise ValueError(f"Entry stage {self.entry_stage} not registered")

        # Track request
        self._requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
        )

        # Create future for completion
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._completion_futures[request_id] = future

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

        payload = StagePayload(
            request_id=request_id,
            request=request,
            data={"raw_inputs": request.inputs},
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

    async def abort(self, request_id: str) -> bool:
        """Abort a request.

        Args:
            request_id: Request to abort

        Returns:
            True if aborted, False if not found
        """
        if request_id not in self._requests:
            return False

        info = self._requests[request_id]
        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False

        # Broadcast abort to all stages
        await self.control_plane.broadcast_abort(AbortMessage(request_id=request_id))

        # Update state
        info.state = RequestState.ABORTED

        # Resolve future with error
        if request_id in self._completion_futures:
            self._completion_futures[request_id].set_exception(
                asyncio.CancelledError(f"Request {request_id} aborted")
            )
        if request_id in self._stream_queues:
            await self._stream_queues[request_id].put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error="aborted",
                )
            )

        logger.info("Coordinator aborted req=%s", request_id)
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
            logger.warning(
                "Coordinator received completion for unknown req=%s", request_id
            )
            return

        info = self._requests[request_id]

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            info.state = RequestState.FAILED
            info.error = msg.error
            self._partial_results.pop(request_id, None)
            if request_id in self._completion_futures:
                future = self._completion_futures[request_id]
                if not future.done():
                    future.set_exception(RuntimeError(msg.error or "Unknown error"))
            if request_id in self._stream_queues:
                await self._stream_queues[request_id].put(msg)
            return

        # Single terminal (original behavior) or no terminal_stages configured
        if len(self._terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self._completion_futures:
                future = self._completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
            if request_id in self._stream_queues:
                await self._stream_queues[request_id].put(msg)
            return

        # Multi-terminal: collect partial results
        partials = self._partial_results.setdefault(request_id, {})
        partials[msg.from_stage] = msg.result

        # Forward stream completion per-stage
        if request_id in self._stream_queues:
            await self._stream_queues[request_id].put(msg)

        if len(partials) < len(self._terminal_stages):
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

    async def _handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        request_id = msg.request_id
        if request_id not in self._stream_queues:
            return
        await self._stream_queues[request_id].put(msg)

    def get_request_info(self, request_id: str) -> RequestInfo | None:
        """Get info about a request."""
        return self._requests.get(request_id)

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


async def run_coordinator(
    completion_endpoint: str,
    abort_endpoint: str,
    entry_stage: str,
    stages: dict[str, str],  # name -> endpoint
    terminal_stages: list[str] | None = None,
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
    )

    # Register stages
    for name, endpoint in stages.items():
        coordinator.register_stage(name, endpoint)

    # Start
    await coordinator.start()

    return coordinator
