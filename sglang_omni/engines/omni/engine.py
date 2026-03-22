# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine combining Scheduler and ModelRunner."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.pipeline.chunk.mailbox import ChunkItem

from ..base import Engine
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .types import SchedulerOutput, SchedulerStatus

if TYPE_CHECKING:
    from sglang_omni.pipeline.chunk.mailbox import ChunkMailbox

    from .runtime.interfaces import CacheManager

logger = logging.getLogger(__name__)


class OmniEngine(Engine):
    """Unified engine for all model types.

    Combines:
    - Scheduler (owns state, makes scheduling decisions)
    - ModelRunner (stateless executor)
    - CacheManager (optional, manages output caching)

    Execution model:
    - Busy loop: schedule() -> [check cache] -> execute() -> [update cache] -> update()
    - Async-friendly: add_request() and get_result() are async
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: ModelRunner,
        cache_manager: CacheManager | None = None,
        feedback_mailbox: ChunkMailbox | None = None,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.cache_manager = cache_manager
        self._feedback_mailbox = feedback_mailbox

        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

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
    # CUDA Graph
    # -------------------------------------------------------------------------

    def _get_sglang_model_runner(self):
        """Return the underlying SGLang ModelRunner, or None."""
        model_worker = getattr(self.model_runner, "model_worker", None)
        return getattr(model_worker, "model_runner", None) if model_worker else None

    def enable_cuda_graph(
        self,
        bs: list[int] | None = None,
        capture_hidden: bool = False,
    ) -> None:
        """Capture CUDA graphs and configure hidden-state mode.

        Call after model buffers are fully allocated (e.g. after
        setup_code_predictor_decode).

        Args:
            bs: Batch sizes to capture (default [1]).
            capture_hidden: Set capture_hidden_mode=LAST so decode batches
                requesting hidden states use the graph instead of eager.
        """
        mr = self._get_sglang_model_runner()
        if mr is None:
            logger.warning("enable_cuda_graph: not an SGLang-backed engine, skipping")
            return

        if bs is None:
            bs = [1]
        mr.server_args.disable_cuda_graph = False
        mr.server_args.cuda_graph_bs = bs
        mr.server_args.cuda_graph_max_bs = max(bs)
        mr.init_device_graphs()

        if capture_hidden:
            self._set_graph_capture_hidden_mode()

    def _set_graph_capture_hidden_mode(self) -> None:
        """Mark the graph runner to accept capture_hidden_mode=LAST batches."""
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        mr = self._get_sglang_model_runner()
        if mr is None:
            return
        graph_runner = getattr(mr, "graph_runner", None)
        if graph_runner is not None:
            graph_runner.capture_hidden_mode = CaptureHiddenMode.LAST
            logger.info("CUDA graph: set capture_hidden_mode=LAST")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the engine processing loop."""
        if self._running:
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("OmniEngine started")

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

    # -------------------------------------------------------------------------
    # Processing Loop
    # -------------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            await self._step()
            await asyncio.sleep(0)  # Yield to other coroutines

    async def _step(self) -> bool:
        """Execute one step. Returns True if work was done."""
        # 1. Schedule
        torch.cuda.nvtx.range_push("schedule")
        scheduler_output = self.scheduler.schedule()
        torch.cuda.nvtx.range_pop()

        if scheduler_output is None:
            # Check for arrived feedback even when idle
            if self._feedback_mailbox is not None:
                self._check_feedback()
            await asyncio.sleep(0.001)  # Brief sleep when idle
            return False

        try:
            # 2. Check cache (if enabled)
            if self.cache_manager is not None:
                scheduler_output = await self._filter_cached(scheduler_output)
                if scheduler_output is None:
                    return True  # All cached, no execution needed

            # 3. Execute
            # Run CPU model runners inline to avoid threadpool hangs with
            # non-thread-safe mock/model outputs. Keep threaded execution for
            # accelerator-backed runners by default.
            execute_in_thread = getattr(self.model_runner, "execute_in_thread", None)
            if execute_in_thread is None:
                device = getattr(self.model_runner, "device", None)
                device_type = getattr(
                    device, "type", str(device) if device is not None else ""
                )
                execute_in_thread = str(device_type) != "cpu"

            torch.cuda.nvtx.range_push("model_forward")
            if execute_in_thread:
                loop = asyncio.get_running_loop()
                model_output = await loop.run_in_executor(
                    None,
                    self.model_runner.execute,
                    scheduler_output,
                )
            else:
                model_output = self.model_runner.execute(scheduler_output)
            torch.cuda.nvtx.range_pop()

            # 4. Update cache (if enabled)
            if self.cache_manager is not None:
                await self._update_cache(scheduler_output, model_output)

            # 5. Update state
            torch.cuda.nvtx.range_push("scheduler_update")
            finished = self.scheduler.update(scheduler_output, model_output)
            torch.cuda.nvtx.range_pop()

            if finished:
                for req in finished:
                    logger.info("Request %s finished", req.request_id)

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

        # 6. Check feedback needs + apply arrived feedback
        torch.cuda.nvtx.range_push("feedback")
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

        if self._feedback_mailbox is not None:
            self._check_feedback()
        torch.cuda.nvtx.range_pop()

        return True

    async def _filter_cached(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput | None:
        """Check cache and filter out cached requests. Returns None if all cached."""
        assert self.cache_manager is not None

        cached_outputs = {}
        uncached_requests = []

        for request in scheduler_output.requests:
            cached = self.cache_manager.get(request)
            if cached is not None:
                cached_outputs[request.request_id] = cached
            else:
                uncached_requests.append(request)

        # If all cached, update scheduler directly and skip execution
        if not uncached_requests:
            from .types import ModelRunnerOutput

            req_ids = [req.request_id for req in scheduler_output.requests]
            req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}
            model_output = ModelRunnerOutput(
                outputs=cached_outputs,
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
            )
            self.scheduler.update(scheduler_output, model_output)
            return None

        # Rebuild batch_data for uncached requests only
        batch_data = self.scheduler.batch_planner.build_batch(uncached_requests)

        return SchedulerOutput(
            requests=uncached_requests,
            batch_data=batch_data,
            step_id=scheduler_output.step_id,
        )

    def _check_feedback(self) -> None:
        """Check feedback mailbox and resume WAITING_FEEDBACK requests."""
        mb = self._feedback_mailbox
        assert mb is not None
        iter_ctrl = self.scheduler.iteration_controller
        for req_id, request in list(self.scheduler.requests.items()):
            if request.status != SchedulerStatus.WAITING_FEEDBACK:
                continue
            queue = mb._queues.get(req_id)
            if queue is None or queue.empty():
                continue
            try:
                item = queue.get_nowait()
            except Exception:
                continue
            if not isinstance(item, ChunkItem):
                continue
            if hasattr(iter_ctrl, "apply_feedback"):
                iter_ctrl.apply_feedback(request, item.tensor)
            self.scheduler.resume_request(req_id)

    async def _update_cache(self, scheduler_output: SchedulerOutput, model_output: Any):
        """Update cache with fresh model outputs."""
        assert self.cache_manager is not None

        for request in scheduler_output.requests:
            output = model_output.outputs.get(request.request_id)
            if output is not None:
                self.cache_manager.put(request, output)
