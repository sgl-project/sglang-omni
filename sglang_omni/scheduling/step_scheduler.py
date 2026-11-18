# SPDX-License-Identifier: Apache-2.0
"""Cooperative scheduling for stages that advance requests in settled steps.

Stages such as diarization advance a request by one model window per step and
keep request-local state between steps. Requests become active in arrival order
up to max_running_requests. A newly active request runs its first step before
active requests continue, and active requests then finish in arrival order.
Each step settles its device work before its task can move to another worker or
close. The module imports no SGLang runtime, so such a stage pays no
autoregressive dependency at startup.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

ABORTED_REQUEST_ID_LIMIT = 10000


class StepInbox(queue.Queue):
    """Wake the scheduler loop when a stage submits work."""

    def __init__(self, wakeup: threading.Event) -> None:
        super().__init__()
        self.wakeup = wakeup

    def put(
        self, item: IncomingMessage, block: bool = True, timeout: float | None = None
    ) -> None:
        super().put(item, block=block, timeout=timeout)
        self.wakeup.set()


@dataclass
class StepResult:
    done: bool = False
    output: StagePayload | None = None


class StepTask(Protocol):
    """One request whose device work settles before each step returns.

    Construction must not run inference. A step may run on a different worker
    from the previous step, so implementations own stream ordering. close runs
    exactly once, after every step of the task has settled. When
    bounded_by_stage is true, the stage bounds how many such tasks exist, so
    admission never rejects them and they do not count against the limits.
    """

    bounded_by_stage: bool

    def step(self) -> StepResult: ...

    def close(self, *, aborted: bool) -> None: ...


class TaskBuilder(Protocol):
    """Build the task of one admitted request."""

    def __call__(self, payload: StagePayload) -> StepTask: ...


class ShutdownCallback(Protocol):
    """Release stage resources after the scheduler loop exits."""

    def __call__(self) -> None: ...


@dataclass
class StepRequest:
    task: StepTask
    future: Future[StepResult] | None = None
    started: bool = False
    aborted: bool = False


class StepScheduler:
    """Advance admitted tasks one settled step at a time on a bounded worker pool."""

    def __init__(
        self,
        build_task: TaskBuilder,
        *,
        max_workers: int,
        max_running_requests: int,
        max_queued_requests: int,
        shutdown_callback: ShutdownCallback,
    ) -> None:
        self.build_task = build_task
        self.max_workers = max_workers
        self.max_running_requests = max_running_requests
        self.max_queued_requests = max_queued_requests
        self.shutdown_callback = shutdown_callback
        self.requests: dict[str, StepRequest] = {}
        self.active: set[str] = set()
        # note (Richard Wang): A bounded window of aborted ids drops their late messages.
        self.aborted_request_ids: dict[str, None] = {}
        self.wakeup = threading.Event()
        self.inbox = StepInbox(self.wakeup)
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.lock = threading.Lock()
        # note (Richard Wang): A stop before the stage thread calls start must hold.
        self.running = True

    def warm_up_serving_thread(self) -> None:
        pass

    def start(self) -> None:
        """Run the scheduler loop on the calling thread until stop."""
        try:
            # note (Richard Wang): Leaving the pool waits for in-flight steps before tasks close.
            with ThreadPoolExecutor(
                self.max_workers, thread_name_prefix="omni-step"
            ) as executor:
                while self.running:
                    self.wakeup.clear()
                    with self.lock:
                        self.admit_messages()
                        self.collect_results()
                        self.launch_steps(executor)
                    if self.running:
                        self.wakeup.wait()
                    else:
                        pass
        finally:
            with self.lock:
                for request_id in list(self.requests):
                    self.finish_request(request_id, aborted=True)
            self.shutdown_callback()

    def stop(self) -> None:
        self.running = False
        self.wakeup.set()

    def abort(self, request_id: str) -> None:
        with self.lock:
            request = self.requests.get(request_id)
            if request is not None:
                request.aborted = True
            else:
                pass
            self.aborted_request_ids[request_id] = None
            if len(self.aborted_request_ids) > ABORTED_REQUEST_ID_LIMIT:
                self.aborted_request_ids.pop(next(iter(self.aborted_request_ids)))
            else:
                pass
        self.wakeup.set()

    def finish_request(
        self,
        request_id: str,
        *,
        result: StepResult | None = None,
        error: Exception | None = None,
        aborted: bool = False,
    ) -> None:
        request = self.requests.pop(request_id)
        self.active.discard(request_id)
        aborted = aborted or request.aborted or request_id in self.aborted_request_ids
        try:
            request.task.close(aborted=aborted or error is not None)
        except Exception as close_error:
            logger.exception(f"Step task cleanup failed for {request_id}")
            error = error or close_error
        if aborted:
            pass
        elif error is not None:
            self.outbox.put(OutgoingMessage(request_id, "error", data=error))
        else:
            self.outbox.put(OutgoingMessage(request_id, "result", data=result.output))

    def admit_messages(self) -> None:
        while True:
            try:
                message = self.inbox.get_nowait()
            except queue.Empty:
                return
            request_id = message.request_id
            if request_id in self.aborted_request_ids or request_id in self.requests:
                continue
            else:
                pass
            try:
                task = self.build_task(message.data)
            except Exception as build_error:
                self.outbox.put(OutgoingMessage(request_id, "error", data=build_error))
                continue
            self.requests[request_id] = StepRequest(task)
            if not task.bounded_by_stage and sum(
                not request.task.bounded_by_stage for request in self.requests.values()
            ) > (self.max_running_requests + self.max_queued_requests):
                # note (Richard Wang): A rejected live operation may own session state, so close it.
                self.finish_request(request_id, error=QueueFullError())
            else:
                pass

    def collect_results(self) -> None:
        for request_id, request in list(self.requests.items()):
            future = request.future
            if future is not None and not future.done():
                continue
            elif request.aborted or request_id in self.aborted_request_ids:
                self.finish_request(request_id, aborted=True)
                continue
            elif future is None:
                continue
            else:
                request.future = None
            try:
                result = future.result()
            except Exception as step_error:
                self.finish_request(request_id, error=step_error)
                continue
            if result.done:
                self.finish_request(request_id, result=result)
            else:
                pass

    def launch_steps(self, executor: ThreadPoolExecutor) -> None:
        for request_id in self.requests:
            if len(self.active) >= self.max_running_requests:
                break
            else:
                self.active.add(request_id)
        running = sum(request.future is not None for request in self.requests.values())
        while running < self.max_workers:
            # note (Richard Wang): A newly active request steps once before others continue.
            request_id = None
            for candidate, request in self.requests.items():
                if candidate not in self.active or request.future is not None:
                    continue
                elif not request.started:
                    request_id = candidate
                    break
                elif request_id is None:
                    request_id = candidate
                else:
                    pass
            if request_id is None:
                return
            else:
                pass
            request = self.requests[request_id]
            request.started = True
            request.future = executor.submit(request.task.step)
            request.future.add_done_callback(lambda _: self.wakeup.set())
            running += 1
