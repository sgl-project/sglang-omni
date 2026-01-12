# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
from typing import Any

from sglang_omni.core.types import (
    AbortMessage,
    CompleteMessage,
    RequestInfo,
    RequestState,
    StageInfo,
    SubmitMessage,
)
from sglang_omni.transport.control_plane import CoordinatorControlPlane

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
    ):
        """Initialize coordinator.

        Args:
            completion_endpoint: ZMQ endpoint to receive completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            entry_stage: Name of the entry stage for new requests
        """
        self.entry_stage = entry_stage

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

    async def submit(self, request_id: str, data: Any) -> Any:
        """Submit a request to the pipeline.

        Args:
            request_id: Unique request identifier
            data: Input data

        Returns:
            Result from the pipeline
        """
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

        # Submit to entry stage
        entry_info = self._stages[self.entry_stage]
        await self.control_plane.submit_to_stage(
            self.entry_stage,
            entry_info.control_endpoint,
            SubmitMessage(request_id=request_id, data=data),
        )

        # Update state
        self._requests[request_id].state = RequestState.RUNNING

        logger.debug("Coordinator submitted req=%s to %s", request_id, self.entry_stage)

        # Wait for completion
        result = await future

        # Cleanup
        del self._completion_futures[request_id]

        return result

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

        logger.info("Coordinator aborted req=%s", request_id)
        return True

    async def run_completion_loop(self) -> None:
        """Run the completion receiving loop.

        This should be run as a background task.
        """
        try:
            while self._running:
                msg = await self.control_plane.recv_completion()
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

        if msg.success:
            info.state = RequestState.COMPLETED
            info.result = msg.result
        else:
            info.state = RequestState.FAILED
            info.error = msg.error

        # Resolve future (if not already done, e.g., by abort)
        if request_id in self._completion_futures:
            future = self._completion_futures[request_id]
            if not future.done():
                if msg.success:
                    future.set_result(msg.result)
                else:
                    future.set_exception(RuntimeError(msg.error or "Unknown error"))

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
) -> Coordinator:
    """Create and start a coordinator.

    Args:
        completion_endpoint: ZMQ endpoint to receive completions
        abort_endpoint: ZMQ endpoint for abort broadcasts
        entry_stage: Name of the entry stage
        stages: Dict of stage_name -> stage_endpoint

    Returns:
        Started Coordinator instance
    """
    coordinator = Coordinator(
        completion_endpoint=completion_endpoint,
        abort_endpoint=abort_endpoint,
        entry_stage=entry_stage,
    )

    # Register stages
    for name, endpoint in stages.items():
        coordinator.register_stage(name, endpoint)

    # Start
    await coordinator.start()

    return coordinator
