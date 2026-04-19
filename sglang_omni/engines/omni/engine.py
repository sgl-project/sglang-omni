# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine combining Scheduler and ModelRunner."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, Optional

from ..base import Engine
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage.stream_queue import StreamQueue

    from .runtime.interfaces import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class _PendingResult:
    """Buffered result awaiting CPU-side scheduler.update processing."""

    scheduler_output: SchedulerOutput
    model_output: ModelRunnerOutput
    update_cache: bool = True


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
        feedback_mailbox: StreamQueue | None = None,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.cache_manager = cache_manager
        self.enable_overlap = enable_overlap
        self._feedback_mailbox = feedback_mailbox

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

    def prepare_stream(self, request_id: str) -> None:
        """Pre-register stream delivery before request execution starts."""
        self.scheduler.prepare_stream(request_id)

    def discard_stream(self, request_id: str) -> None:
        """Discard a pre-registered stream queue for failed submissions."""
        self.scheduler.discard_stream(request_id)

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
            if self._feedback_mailbox is not None:
                self._check_feedback()
            await asyncio.sleep(0.001)
            return False

        try:
            if self.cache_manager is not None:
                uncached_output, cached_pending = self._filter_cached(scheduler_output)
                if cached_pending is not None:
                    self._apply_pending_result(cached_pending)
                if uncached_output is None:
                    return True
                scheduler_output = uncached_output

            execute_in_thread = getattr(self.model_runner, "execute_in_thread", None)
            if execute_in_thread is None:
                device = getattr(self.model_runner, "device", None)
                device_type = getattr(
                    device, "type", str(device) if device is not None else ""
                )
                execute_in_thread = str(device_type) != "cpu"

            if execute_in_thread:
                loop = asyncio.get_running_loop()
                model_output = await loop.run_in_executor(
                    None,
                    self.model_runner.execute,
                    scheduler_output,
                )
            else:
                model_output = self.model_runner.execute(scheduler_output)

            # 4. Update cache (if enabled)
            if self.cache_manager is not None:
                await self._update_cache(scheduler_output, model_output)

            # 5. Update state
            finished = self.scheduler.update(scheduler_output, model_output)

            if finished:
                for req in finished:
                    logger.debug("Request %s finished", req.request_id)

        except Exception as e:
            logger.exception(
                "OmniEngine step failed, failing %d request(s)",
                len(scheduler_output.requests),
            )
            for request in scheduler_output.requests:
                try:
                    self.scheduler.fail_request(request.request_id, e)
                except Exception:
                    pass
            return False

        # 6. Check feedback needs — set WAITING_FEEDBACK for requests needing it
        iter_ctrl = self.scheduler.iteration_controller
        if hasattr(iter_ctrl, "needs_feedback"):
            for request in scheduler_output.requests:
                if request.status in (
                    SchedulerStatus.FINISHED,
                    SchedulerStatus.ABORTED,
                ):
                    continue
                output = model_output.outputs.get(request.request_id)
                if output is not None and iter_ctrl.needs_feedback(request, output):
                    request.status = SchedulerStatus.WAITING_FEEDBACK
                    request._feedback_wait_start = time.time()

        # 7. Check for arrived feedback — resume WAITING_FEEDBACK requests
        if self._feedback_mailbox is not None:
            self._check_feedback()

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
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            if self._result_queue:
                self._process_pending_result()
            elif self._feedback_mailbox is not None:
                self._check_feedback()
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0.001)
            self._last_scheduler_output = None
            return False

        try:
            disable_overlap = self._should_disable_overlap(scheduler_output)

            if disable_overlap and self._result_queue:
                self._process_pending_result()

            if self.cache_manager is not None:
                uncached_output, cached_pending = self._filter_cached(scheduler_output)
                if cached_pending is not None:
                    self._result_queue.append(cached_pending)
                if uncached_output is None:
                    if not disable_overlap and self._result_queue:
                        self._process_pending_result()
                    self._last_scheduler_output = None
                    return True
                scheduler_output = uncached_output

            execute_in_thread = self._should_execute_in_thread()

            if execute_in_thread and not disable_overlap:
                loop = asyncio.get_running_loop()
                execute_future = loop.run_in_executor(
                    None,
                    self.model_runner.execute,
                    scheduler_output,
                )

                if self._result_queue:
                    self._process_pending_result()

                model_output = await execute_future
            else:
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

            self._result_queue.append(
                _PendingResult(
                    scheduler_output=scheduler_output,
                    model_output=model_output,
                )
            )

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
        self._apply_pending_result(self._result_queue.popleft())

    def _apply_pending_result(self, pending: _PendingResult) -> None:
        """Apply a pending result: cache put (if enabled) + scheduler.update + feedback housekeeping."""
        try:
            if pending.update_cache and self.cache_manager is not None:
                self._put_outputs_in_cache(
                    pending.scheduler_output, pending.model_output
                )

            finished = self.scheduler.update(
                pending.scheduler_output, pending.model_output
            )

            if finished:
                for req in finished:
                    logger.debug("Request %s finished (overlap)", req.request_id)

            iter_ctrl = self.scheduler.iteration_controller
            if hasattr(iter_ctrl, "needs_feedback"):
                for request in pending.scheduler_output.requests:
                    if request.status in (
                        SchedulerStatus.FINISHED,
                        SchedulerStatus.ABORTED,
                    ):
                        continue
                    output = pending.model_output.outputs.get(request.request_id)
                    if output is not None and iter_ctrl.needs_feedback(request, output):
                        request.status = SchedulerStatus.WAITING_FEEDBACK
                        request._feedback_wait_start = time.time()

            if self._feedback_mailbox is not None:
                self._check_feedback()

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

    def _put_outputs_in_cache(
        self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        """Store every per-request output in the cache. No-op if disabled."""
        if self.cache_manager is None:
            return
        for request in scheduler_output.requests:
            output = model_output.outputs.get(request.request_id)
            if output is not None:
                self.cache_manager.put(request, output)

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
        self._put_outputs_in_cache(scheduler_output, model_output)

    def _fail_requests(
        self, scheduler_output: SchedulerOutput, error: Exception
    ) -> None:
        """Fail all requests in a scheduler output."""
        for request in scheduler_output.requests:
            try:
                self.scheduler.fail_request(request.request_id, error)
            except Exception:
                pass

    def _filter_cached(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[SchedulerOutput | None, _PendingResult | None]:
        """Partition a scheduler output into cached and uncached halves."""
        assert self.cache_manager is not None

        cached_outputs: dict[str, RequestOutput] = {}
        cached_requests: list[SchedulerRequest] = []
        uncached_requests: list[SchedulerRequest] = []

        for request in scheduler_output.requests:
            cached = self.cache_manager.get(request)
            if cached is not None:
                cached_outputs[request.request_id] = cached
                cached_requests.append(request)
            else:
                uncached_requests.append(request)

        has_cached = bool(cached_requests)
        has_uncached = bool(uncached_requests)

        cached_pending = (
            self._build_cached_pending(
                cached_requests, cached_outputs, scheduler_output
            )
            if has_cached
            else None
        )

        if not has_cached and not has_uncached:
            return None, None
        elif has_cached and not has_uncached:
            return None, cached_pending
        elif not has_cached and has_uncached:
            return (
                self._build_uncached_output(uncached_requests, scheduler_output),
                None,
            )
        elif has_cached and has_uncached:
            return (
                self._build_uncached_output(uncached_requests, scheduler_output),
                cached_pending,
            )
        else:
            raise AssertionError("unreachable: _filter_cached logic error")

    def _build_cached_pending(
        self,
        cached_requests: list[SchedulerRequest],
        cached_outputs: dict[str, RequestOutput],
        scheduler_output: SchedulerOutput,
    ) -> _PendingResult:
        """Package cached outputs as a _PendingResult."""
        for req in cached_requests:
            if req.status != SchedulerStatus.RUNNING:
                logger.error(
                    "Cached request %s is in status %s, not RUNNING. "
                    "scheduler.update would silently skip it; failing explicitly.",
                    req.request_id,
                    req.status,
                )
                self.scheduler.fail_request(
                    req.request_id,
                    RuntimeError(
                        f"Cached request {req.request_id} in non-RUNNING status "
                        f"{req.status} at cache-resolution time"
                    ),
                )

        req_ids = [req.request_id for req in cached_requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
        model_output = ModelRunnerOutput(
            outputs=cached_outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )
        scheduler_output_for_cached = SchedulerOutput(
            requests=cached_requests,
            batch_data=None,
            step_id=scheduler_output.step_id,
        )
        return _PendingResult(
            scheduler_output=scheduler_output_for_cached,
            model_output=model_output,
            update_cache=False,
        )

    def _build_uncached_output(
        self,
        uncached_requests: list[SchedulerRequest],
        scheduler_output: SchedulerOutput,
    ) -> SchedulerOutput:
        """Build a SchedulerOutput for the uncached subset, for model execution."""
        assert uncached_requests, "_build_uncached_output called with empty list"
        batch_data = self.scheduler.batch_planner.build_batch(uncached_requests)
        return SchedulerOutput(
            requests=uncached_requests,
            batch_data=batch_data,
            step_id=scheduler_output.step_id,
        )

    def _check_feedback(self) -> None:
        """Check feedback mailbox for arrived feedback and resume requests."""
        assert self._feedback_mailbox is not None
        from sglang_omni.pipeline.stage.stream_queue import StreamSignal

        for req_id, request in list(self.scheduler.requests.items()):
            if request.status != SchedulerStatus.WAITING_FEEDBACK:
                continue
            if not self._feedback_mailbox.has(req_id):
                continue
            # Try non-blocking get from the queue
            queue = self._feedback_mailbox._queues.get(req_id)
            if queue is not None and not queue.empty():
                try:
                    item = queue.get_nowait()
                    if isinstance(item, BaseException):
                        logger.error(
                            "Feedback exception for request %s: %s", req_id, item
                        )
                        self.scheduler.fail_request(
                            req_id,
                            (
                                item
                                if isinstance(item, Exception)
                                else RuntimeError(str(item))
                            ),
                        )
                        continue
                    if isinstance(item, StreamSignal):
                        if item.error is not None:
                            logger.error(
                                "Feedback error for request %s: %s", req_id, item.error
                            )
                            err = (
                                item.error
                                if isinstance(item.error, Exception)
                                else RuntimeError(str(item.error))
                            )
                            self.scheduler.fail_request(req_id, err)
                            continue
                        if item.is_done:
                            logger.debug("Feedback done for request %s", req_id)
                            self.scheduler.resume_request(req_id)
                            continue
                    if not hasattr(item, "data"):
                        continue
                    # Apply feedback
                    iter_ctrl = self.scheduler.iteration_controller
                    if hasattr(iter_ctrl, "apply_feedback"):
                        iter_ctrl.apply_feedback(request, item.data)
                    self.scheduler.resume_request(req_id)
                except Exception as e:
                    logger.error(
                        "Feedback handling failed for %s, aborting: %s", req_id, e
                    )
                    try:
                        self.scheduler.fail_request(
                            req_id,
                            e if isinstance(e, Exception) else RuntimeError(str(e)),
                        )
                    except Exception:
                        pass

    async def _update_cache(self, scheduler_output: SchedulerOutput, model_output: Any):
        """Update cache with fresh model outputs."""
        assert self.cache_manager is not None
        self._put_outputs_in_cache(scheduler_output, model_output)


def _is_prefill_batch(batch_data: Any) -> bool:
    """Check if a batch_data represents a prefill/extend batch."""
    forward_mode = getattr(batch_data, "forward_mode", None)
    if forward_mode is not None:
        is_extend = getattr(forward_mode, "is_extend", None)
        if callable(is_extend):
            return is_extend()
    return False
