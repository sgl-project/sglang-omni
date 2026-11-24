# SPDX-License-Identifier: Apache-2.0
"""Schedule settled tensor batches while retaining request-owned state."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from sglang_omni.scheduling.step_scheduler import (
    ShutdownCallback,
    StepRequest,
    StepResult,
    StepScheduler,
    StepTask,
    TaskBuilder,
)


@dataclass(kw_only=True)
class BatchRequestStep:
    request_id: str
    task: StepTask


class StepBatchRunner(Protocol):
    """Run compatible request steps and route their results by request ID."""

    def supports_batching(self, task: StepTask) -> bool: ...

    def __call__(
        self, request_steps: list[BatchRequestStep]
    ) -> dict[str, StepResult | Exception]: ...


class BatchedStepScheduler(StepScheduler):
    """Advance one shared batch at a time and refill completed request slots."""

    def __init__(
        self,
        build_task: TaskBuilder,
        *,
        run_batch: StepBatchRunner,
        max_batch_size: int,
        max_running_requests: int,
        max_queued_requests: int,
        shutdown_callback: ShutdownCallback,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        else:
            pass
        super().__init__(
            build_task,
            max_workers=1,
            max_running_requests=max_running_requests,
            max_queued_requests=max_queued_requests,
            shutdown_callback=shutdown_callback,
        )
        self.run_batch: StepBatchRunner = run_batch
        self.max_batch_size: int = max_batch_size
        self.batch_future: Future[dict[str, StepResult | Exception]] | None = None
        self.batch_request_ids: tuple[str, ...] = ()
        self.dispatched_batch_count: int = 0
        self.last_dispatch_steps: dict[str, int] = {}

    def finish_request(
        self,
        request_id: str,
        *,
        result: StepResult | None = None,
        error: Exception | None = None,
        aborted: bool = False,
    ) -> None:
        self.last_dispatch_steps.pop(request_id, None)
        super().finish_request(request_id, result=result, error=error, aborted=aborted)

    def collect_results(self) -> None:
        batch_future = self.batch_future
        if batch_future is not None and batch_future.done():
            request_ids = self.batch_request_ids
            self.batch_future = None
            self.batch_request_ids = ()
            try:
                results = batch_future.result()
            except Exception as error:
                results = {request_id: error for request_id in request_ids}
            for request_id in request_ids:
                request = self.requests[request_id]
                result = results.get(
                    request_id, RuntimeError("Missing batched step result")
                )
                if request.aborted or request_id in self.aborted_request_ids:
                    self.finish_request(request_id, aborted=True)
                elif isinstance(result, Exception):
                    self.finish_request(request_id, error=result)
                elif result.done:
                    self.finish_request(request_id, result=result)
                else:
                    pass
        else:
            pass
        for request_id, request in list(self.requests.items()):
            # note (eranik): Shared device work must settle before an aborted task releases its tensors.
            if request_id not in self.batch_request_ids and (
                request.aborted or request_id in self.aborted_request_ids
            ):
                self.finish_request(request_id, aborted=True)
            else:
                pass

    def launch_steps(self, executor: ThreadPoolExecutor) -> None:
        if self.batch_future is not None:
            return
        else:
            pass
        for request_id in self.requests:
            if len(self.active) >= self.max_running_requests:
                break
            else:
                self.active.add(request_id)
        ready_requests = [
            (request_id, request)
            for request_id, request in self.requests.items()
            if request_id in self.active
        ]
        if not ready_requests:
            return
        else:
            pass

        for request_id, _ in ready_requests:
            self.last_dispatch_steps.setdefault(request_id, self.dispatched_batch_count)

        def dispatch_priority(
            request_entry: tuple[str, StepRequest],
        ) -> tuple[int, bool]:
            request_id, request = request_entry
            return (
                self.last_dispatch_steps[request_id],
                request.started,
            )

        ready_requests.sort(key=dispatch_priority)
        if self.run_batch.supports_batching(ready_requests[0][1].task):
            selected_requests = [
                (request_id, request)
                for request_id, request in ready_requests
                if self.run_batch.supports_batching(request.task)
            ][: self.max_batch_size]
        else:
            selected_requests = ready_requests[:1]
        self.dispatched_batch_count += 1
        for request_id, request in selected_requests:
            request.started = True
            self.last_dispatch_steps[request_id] = self.dispatched_batch_count
        self.batch_request_ids = tuple(
            request_id for request_id, _ in selected_requests
        )
        self.batch_future = executor.submit(
            self.run_batch,
            [
                BatchRequestStep(request_id=request_id, task=request.task)
                for request_id, request in selected_requests
            ],
        )

        def wake_scheduler(
            completed_batch: Future[dict[str, StepResult | Exception]],
        ) -> None:
            assert completed_batch.done()
            self.wakeup.set()

        self.batch_future.add_done_callback(wake_scheduler)
