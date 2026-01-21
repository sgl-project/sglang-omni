# SPDX-License-Identifier: Apache-2.0
"""Generic Scheduler - model-agnostic request lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from .types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

if TYPE_CHECKING:
    from .runtime.interfaces import BatchPlanner, IterationController, ResourceManager

logger = logging.getLogger(__name__)


class Scheduler:
    """Generic request scheduler.

    Responsibilities:
    - Manage request lifecycle (WAITING -> RUNNING -> FINISHED/ABORTED)
    - Delegate selection to BatchPlanner and allocation to ResourceManager
    - Delegate per-request updates to IterationController
    - Produce SchedulerOutput for ModelRunner

    Does NOT know about:
    - Input formats (tokens, latents, etc.)
    - Model-specific batching logic
    - Resource details (KV cache, etc.)
    """

    def __init__(
        self,
        batch_planner: BatchPlanner,
        resource_manager: ResourceManager,
        iteration_controller: IterationController,
        stream_adapter: Callable[[SchedulerRequest, RequestOutput], Any] | None = None,
    ):
        self.batch_planner = batch_planner
        self.resource_manager = resource_manager
        self.iteration_controller = iteration_controller
        self._stream_adapter = stream_adapter

        # Scheduler request state
        self.requests: dict[str, SchedulerRequest] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []

        # Result futures (created lazily in get_result)
        self._futures: dict[str, asyncio.Future[SchedulerRequest]] = {}
        self._step_id = 0
        self._aborted_this_step: set[str] = set()
        self._stream_queues: dict[str, asyncio.Queue[Any]] = {}
        self._stream_done = object()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_request(self, request_id: str, data: Any) -> None:
        """Add a new request with model-specific data."""
        request = SchedulerRequest(
            request_id=request_id,
            data=data,
            arrival_time=time.time(),
        )
        self.requests[request_id] = request
        self.waiting.append(request_id)
        # Note: Future created lazily in get_result() to avoid event loop issues

    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_this_step.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED)

    def has_requests(self) -> bool:
        """Check if there are any requests to process."""
        return len(self.waiting) > 0 or len(self.running) > 0

    async def get_result(self, request_id: str) -> SchedulerRequest:
        """Wait for a request to complete."""
        if request_id not in self.requests:
            raise KeyError(f"Unknown request: {request_id}")

        # Create future lazily (requires running event loop)
        if request_id not in self._futures:
            self._futures[request_id] = asyncio.get_running_loop().create_future()

        # If already finished or aborted, resolve immediately
        request = self.requests[request_id]
        if request.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
            return request

        return await self._futures[request_id]

    async def stream(self, request_id: str) -> AsyncIterator[Any]:
        """Yield per-step stream data for a request."""
        queue = self._subscribe_stream(request_id)
        while True:
            item = await queue.get()
            if item is self._stream_done:
                return
            yield item

    def _subscribe_stream(self, request_id: str) -> asyncio.Queue[Any]:
        if request_id not in self.requests:
            raise KeyError(f"Unknown request: {request_id}")
        queue = self._stream_queues.get(request_id)
        if queue is None:
            queue = asyncio.Queue()
            self._stream_queues[request_id] = queue
        request = self.requests[request_id]
        if request.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
            queue.put_nowait(self._stream_done)
        return queue

    # -------------------------------------------------------------------------
    # Core Scheduling
    # -------------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput | None:
        """Schedule next batch. Returns None if no work."""
        if not self.waiting and not self.running:
            return None

        self._step_id += 1
        self._aborted_this_step.clear()

        waiting_reqs = [self.requests[req_id] for req_id in self.waiting]
        running_reqs = [self.requests[req_id] for req_id in self.running]

        selected = self.batch_planner.select_requests(
            waiting_reqs,
            running_reqs,
            self.resource_manager,
        )

        if not selected:
            return None

        for request in selected:
            if request.request_id in self.waiting:
                self.waiting.remove(request.request_id)
                self.running.append(request.request_id)
                request.status = SchedulerStatus.RUNNING

        batch_data = self.batch_planner.build_batch(selected)

        return SchedulerOutput(
            requests=selected,
            batch_data=batch_data,
            step_id=self._step_id,
        )

    def update(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[SchedulerRequest]:
        """Update state from model output.

        Returns list of finished requests.
        """
        finished: list[SchedulerRequest] = []

        for request in scheduler_output.requests:
            if request.request_id in self._aborted_this_step:
                continue

            output = model_output.outputs.get(request.request_id)
            if output is None:
                logger.warning("Missing output for request_id=%s", request.request_id)
                continue

            # Update via iteration controller (model-specific)
            self.iteration_controller.update_request(request, output)
            self._emit_stream(request, output)

            # Check completion via iteration controller
            if self.iteration_controller.is_finished(request, output):
                self._finish_request(request)
                finished.append(request)

        return finished

    def _emit_stream(self, request: SchedulerRequest, output: RequestOutput) -> None:
        if self._stream_adapter is None:
            return
        queue = self._stream_queues.get(request.request_id)
        if queue is None:
            return
        item = self._stream_adapter(request, output)
        if item is None:
            return
        queue.put_nowait(item)

    def _finish_request(
        self,
        request: SchedulerRequest,
        status: SchedulerStatus = SchedulerStatus.FINISHED,
    ) -> None:
        """Clean up finished request."""
        was_running = request.status == SchedulerStatus.RUNNING
        request.status = status
        request.finish_time = time.time()

        if was_running:
            self.resource_manager.free(request)

        # Remove from queues
        if request.request_id in self.running:
            self.running.remove(request.request_id)
        if request.request_id in self.waiting:
            self.waiting.remove(request.request_id)

        # Resolve future if someone is waiting
        if request.request_id in self._futures:
            future = self._futures[request.request_id]
            if not future.done():
                future.set_result(request)

        queue = self._stream_queues.pop(request.request_id, None)
        if queue is not None:
            queue.put_nowait(self._stream_done)
