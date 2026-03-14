# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine combining Scheduler and ModelRunner."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, Optional

from ..base import Engine
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .types import ModelRunnerOutput, SchedulerOutput

if TYPE_CHECKING:
    from .runtime.interfaces import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class _PendingResult:
    """Buffered result from a previous step, awaiting CPU processing."""

    scheduler_output: SchedulerOutput
    model_output: ModelRunnerOutput


class OmniEngine(Engine):
    """Unified engine for all model types.

    Combines:
    - Scheduler (owns state, makes scheduling decisions)
    - ModelRunner (stateless executor)
    - CacheManager (optional, manages output caching)

    Execution model (normal):
    - Busy loop: schedule() -> [check cache] -> execute() -> [update cache] -> update()
    - Async-friendly: add_request() and get_result() are async

        schedule(N) -> execute(N) -> update(N) -> schedule(N+1) -> ...

    Execution model (overlap):
    - Step N:   schedule() -> execute(N) [GPU async] -> update(N-1) [CPU, overlaps with GPU]
    - This overlaps CPU processing of the previous step with GPU computation of the current step.
    - Improves throughput by hiding CPU overhead behind GPU computation.

        Step N:   schedule(N) -> launch_execute(N) ─┐
                                                    ├── concurrent
        Step N:   update(N-1) ──────────────────────┘
        Step N:   await execute(N) -> buffer result(N)
        Step N+1: schedule(N+1) -> launch_execute(N+1) ─┐
                                                        ├── concurrent
        Step N+1: update(N) ────────────────────────────┘
        ...

    The overlap is achieved by:
    - Launching GPU execution via run_in_executor (non-blocking)
    - While GPU is busy, processing the previous step's update() on CPU
    - Awaiting GPU completion
    - Buffering the result for next step's CPU processing
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: ModelRunner,
        cache_manager: CacheManager | None = None,
        enable_overlap: bool = False,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.cache_manager = cache_manager
        self.enable_overlap = enable_overlap

        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

        # Overlap scheduling state
        self._result_queue: Deque[_PendingResult] = deque()
        self._last_scheduler_output: Optional[SchedulerOutput] = None

    # -------------------------------------------------------------------------
    # Engine ABC Implementation
    # -------------------------------------------------------------------------

    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        self.scheduler.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        request = await self.scheduler.get_result(request_id)
        return request.data

    async def stream(self, request_id: str):
        """Stream per-step outputs for a request."""
        async for item in self.scheduler.stream(request_id):
            yield item

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        self.scheduler.abort_request(request_id)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the engine processing loop."""
        if self._running:
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("OmniEngine started (overlap=%s)", self.enable_overlap)

    async def stop(self) -> None:
        """Stop the engine processing loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("OmniEngine stopped")

        # Drain any pending results
        self._drain_pending_results()
        logger.info("OmniEngine stopped")

    # -------------------------------------------------------------------------
    # Processing Loop
    # -------------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            if self.enable_overlap:
                await self._step_overlap()
            else:
                await self._step_normal()
            await asyncio.sleep(0)  # Yield to other coroutines

    # -------------------------------------------------------------------------
    # Normal Step (no overlap)
    # -------------------------------------------------------------------------

    async def _step_normal(self) -> bool:
        """Execute one step in normal (non-overlap) mode."""
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            await asyncio.sleep(0.001)
            return False

        try:
            # Check cache
            if self.cache_manager is not None:
                scheduler_output = await self._filter_cached(scheduler_output)
                if scheduler_output is None:
                    return True

            # Execute
            model_output = await self._execute_async(scheduler_output)

            # Update cache
            if self.cache_manager is not None:
                self._update_cache_sync(scheduler_output, model_output)

            # Update state
            finished = self.scheduler.update(scheduler_output, model_output)
            if finished:
                for req in finished:
                    logger.debug("Request %s finished", req.request_id)

        except Exception as e:
            logger.exception(
                "OmniEngine step failed, failing %d request(s)",
                len(scheduler_output.requests),
            )
            self._fail_requests(scheduler_output, e)
            return False

        return True

    # -------------------------------------------------------------------------
    # Overlap Step
    # -------------------------------------------------------------------------

    async def _step_overlap(self) -> bool:
        """Execute one step with overlap scheduling.

        Key insight: We launch GPU execution via run_in_executor, which returns
        a Future. While the Future is pending (GPU is computing), we process
        the previous step's results on the CPU. Then we await the Future to
        get the current step's results.

        This achieves true concurrency between:
        - GPU: executing current step's forward pass
        - CPU: processing previous step's update (token append, finish check, etc.)
        """
        # 1. Schedule next batch
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            # No new work. Process any pending results.
            if self._result_queue:
                self._process_pending_result()
            else:
                await asyncio.sleep(0.001)
            self._last_scheduler_output = None
            return False

        try:
            # 2. Check if overlap should be disabled
            disable_overlap = self._should_disable_overlap(scheduler_output)

            # 3. If disabled, synchronously process previous results first
            if disable_overlap and self._result_queue:
                self._process_pending_result()

            # 4. Handle cache filtering
            if self.cache_manager is not None:
                scheduler_output = await self._filter_cached(scheduler_output)
                if scheduler_output is None:
                    if not disable_overlap and self._result_queue:
                        self._process_pending_result()
                    self._last_scheduler_output = None
                    return True

            # 5. Determine execution strategy
            execute_in_thread = self._should_execute_in_thread()

            if execute_in_thread and not disable_overlap:
                # ═══ OVERLAP PATH ═══
                # Launch GPU execution as a background task
                loop = asyncio.get_running_loop()
                execute_future = loop.run_in_executor(
                    None,
                    self.model_runner.execute,
                    scheduler_output,
                )

                # While GPU is busy, process previous step's result on CPU
                if self._result_queue:
                    self._process_pending_result()

                # Now wait for GPU to finish
                model_output = await execute_future
            else:
                # ═══ SYNCHRONOUS PATH ═══
                # Either CPU device or overlap disabled for this step.
                # Process pending results BEFORE execution to minimize latency.
                if (
                    not disable_overlap
                    and self._last_scheduler_output is not None
                    and self._result_queue
                ):
                    self._process_pending_result()

                if execute_in_thread:
                    loop = asyncio.get_running_loop()
                    model_output = await loop.run_in_executor(
                        None,
                        self.model_runner.execute,
                        scheduler_output,
                    )
                else:
                    model_output = self.model_runner.execute(scheduler_output)

            # 6. Buffer current result for next step's CPU processing
            #    (cache update is deferred to _process_pending_result to keep
            #     it co-located with scheduler.update for consistency)
            self._result_queue.append(
                _PendingResult(
                    scheduler_output=scheduler_output,
                    model_output=model_output,
                )
            )

            # 7. Track last output for prefill-detection heuristic
            self._last_scheduler_output = scheduler_output

        except Exception as e:
            logger.exception(
                "OmniEngine overlap step failed, failing %d request(s)",
                len(scheduler_output.requests),
            )
            self._fail_requests(scheduler_output, e)
            return False

        return True

    # -------------------------------------------------------------------------
    # Overlap Helpers
    # -------------------------------------------------------------------------

    def _should_disable_overlap(self, current_output: SchedulerOutput) -> bool:
        """Determine if overlap should be disabled for the current step.

        Overlap is disabled when:
        1. Two consecutive prefill batches - to improve TTFT of the first batch.
           Processing the first prefill's result immediately means the tokens
           are available sooner.
        2. No previous batch exists (first step).

        For SGLang backend, we check if both the current and last batch are
        in extend (prefill) mode.
        """
        if self._last_scheduler_output is None:
            return False

        # Check for consecutive prefills (SGLang backend)
        last_batch = getattr(self._last_scheduler_output, "batch_data", None)
        curr_batch = getattr(current_output, "batch_data", None)

        if last_batch is not None and curr_batch is not None:
            last_is_prefill = _is_prefill_batch(last_batch)
            curr_is_prefill = _is_prefill_batch(curr_batch)
            if last_is_prefill and curr_is_prefill:
                return True

        return False

    def _should_execute_in_thread(self) -> bool:
        """Determine if model execution should run in a thread pool."""
        execute_in_thread = getattr(self.model_runner, "execute_in_thread", None)
        if execute_in_thread is not None:
            return execute_in_thread

        device = getattr(self.model_runner, "device", None)
        device_type = getattr(device, "type", str(device) if device is not None else "")
        return str(device_type) != "cpu"

    def _process_pending_result(self) -> None:
        """Process the oldest pending result from the result queue."""
        if not self._result_queue:
            return

        pending = self._result_queue.popleft()

        try:
            # Update cache (if enabled)
            if self.cache_manager is not None:
                # Note: we do sync cache update here since we're on CPU
                for request in pending.scheduler_output.requests:
                    output = pending.model_output.outputs.get(request.request_id)
                    if output is not None:
                        self.cache_manager.put(request, output)

            # Update scheduler state
            finished = self.scheduler.update(
                pending.scheduler_output, pending.model_output
            )

            if finished:
                for req in finished:
                    logger.debug("Request %s finished (overlap)", req.request_id)

        except Exception as e:
            logger.exception(
                "Failed to process pending result for %d request(s)",
                len(pending.scheduler_output.requests),
            )
            for request in pending.scheduler_output.requests:
                try:
                    self.scheduler.fail_request(request.request_id, e)
                except Exception:
                    pass

    def _drain_pending_results(self) -> None:
        """Process all pending results. Called during shutdown."""
        while self._result_queue:
            self._process_pending_result()
        self._last_scheduler_output = None

    # -------------------------------------------------------------------------
    # Shared Helpers
    # -------------------------------------------------------------------------

    async def _execute_async(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Execute model forward pass asynchronously."""
        if self._should_execute_in_thread():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self.model_runner.execute,
                scheduler_output,
            )
        else:
            return self.model_runner.execute(scheduler_output)

    def _update_cache_sync(
        self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        """Synchronously update cache with model outputs."""
        assert self.cache_manager is not None
        for request in scheduler_output.requests:
            output = model_output.outputs.get(request.request_id)
            if output is not None:
                self.cache_manager.put(request, output)

    def _fail_requests(
        self, scheduler_output: SchedulerOutput, error: Exception
    ) -> None:
        """Fail all requests in a scheduler output."""
        for request in scheduler_output.requests:
            try:
                self.scheduler.fail_request(request.request_id, error)
            except Exception:
                pass

    async def _filter_cached(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput | None:
        """Check cache and filter out cached requests."""
        assert self.cache_manager is not None

        cached_outputs = {}
        uncached_requests = []

        for request in scheduler_output.requests:
            cached = self.cache_manager.get(request)
            if cached is not None:
                cached_outputs[request.request_id] = cached
            else:
                uncached_requests.append(request)

        if not uncached_requests:
            req_ids = [req.request_id for req in scheduler_output.requests]
            req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
            model_output = ModelRunnerOutput(
                outputs=cached_outputs,
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
            )
            self.scheduler.update(scheduler_output, model_output)
            return None

        batch_data = self.scheduler.batch_planner.build_batch(uncached_requests)
        return SchedulerOutput(
            requests=uncached_requests,
            batch_data=batch_data,
            step_id=scheduler_output.step_id,
        )


def _is_prefill_batch(batch_data: Any) -> bool:
    """Check if a batch_data represents a prefill/extend batch."""
    forward_mode = getattr(batch_data, "forward_mode", None)
    if forward_mode is not None:
        is_extend = getattr(forward_mode, "is_extend", None)
        if callable(is_extend):
            return is_extend()
    return False
