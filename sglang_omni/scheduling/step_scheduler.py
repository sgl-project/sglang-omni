# SPDX-License-Identifier: Apache-2.0
"""StepScheduler: cooperative scheduling for stages that work in settled steps.

Stages such as diarization advance a request by one model window per step and
keep request-local state between steps. This scheduler admits requests under
bounded running and queued capacity, runs a newly admitted request's first step
before earlier requests continue, then finishes earlier requests before later
ones advance. Every step settles its device work before the task can migrate to
another worker or close.

The module imports no SGLang runtime, so a stage that never allocates a KV cache
pays no autoregressive dependency at startup. Same inbox/outbox contract as
SimpleScheduler and OmniScheduler so Stage does not branch.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sglang_omni.admission import QueueFullError
from sglang_omni.profiler.event_recorder import (
    emit_model_path_end as _emit_model_path_end,
)
from sglang_omni.profiler.event_recorder import (
    emit_model_path_start as _emit_model_path_start,
)
from sglang_omni.proto.admin import ADMIN_MODEL_INFO
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_ABORTED_REQUEST_ID_LIMIT = 10000


class StepInbox(_queue_mod.Queue):
    """Wake the cooperative loop when a stage submits work."""

    def __init__(self, wakeup: threading.Event):
        super().__init__()
        self.wakeup = wakeup

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        self.wakeup.set()


@dataclass
class StepResult:
    done: bool = False
    output: Any = None


class StepTask(Protocol):
    """One request whose device work settles before each step returns.

    Construction must not run inference. A step may execute on a different
    worker from the preceding step, so implementations own stream ordering.
    close runs exactly once, after all of this task's steps have settled.
    """

    def step(self) -> StepResult: ...

    def close(self, *, aborted: bool) -> None: ...


@dataclass
class StepRequest:
    task: StepTask
    future: Future | None = None
    started: bool = False
    aborted: bool = False


class StepScheduler:
    """Advance admitted tasks one settled step at a time on a bounded worker pool.

    Public contract (used by Stage):
        ``inbox``, ``outbox``, ``start()``, ``stop()``, ``abort(request_id)``,
        ``admin(action)``
    """

    def __init__(
        self,
        step_request_builder: Callable[[Any], StepTask],
        *,
        max_workers: int = 1,
        max_running_requests: int = 16,
        max_queued_requests: int = 64,
        abort_callback: Callable[[str], None] | None = None,
        request_finished_callback: Callable[[str], None] | None = None,
        shutdown_callback: Callable[[], None] | None = None,
    ):
        for name, value in (
            ("max_workers", max_workers),
            ("max_running_requests", max_running_requests),
            ("max_queued_requests", max_queued_requests),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_workers > max_running_requests:
            raise ValueError("max_workers cannot exceed max_running_requests")
        self._build = step_request_builder
        self._max_workers = max_workers
        self._max_running = max_running_requests
        self._max_queued = max_queued_requests
        self._requests: dict[str, StepRequest] = {}
        self._active: set[str] = set()
        self._wakeup = threading.Event()
        self.inbox: _queue_mod.Queue[IncomingMessage] = StepInbox(self._wakeup)
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = False
        self._abort_callback = abort_callback
        self._request_finished_callback = request_finished_callback
        self._shutdown_callback = shutdown_callback
        self._shutdown_lock = threading.Lock()
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread_id: int | None = None
        # Aborts can arrive before admission or after completion. Keep a
        # bounded window of ids so a late message for an aborted request is
        # dropped instead of admitted.
        self._aborted_request_ids: set[str] = set()
        self._aborted_request_id_order: deque[str] = deque()

    def start(self) -> None:
        """Run the cooperative loop on the calling thread until ``stop``."""
        self._scheduler_thread_id = threading.get_ident()
        self._running = True
        try:
            with ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="omni-step"
            ) as executor:
                try:
                    while self._running:
                        self._wakeup.clear()
                        with self._lock:
                            self.admit_messages()
                            self.collect_results()
                            self.launch_work(executor)
                        if self._running:
                            self._wakeup.wait()
                finally:
                    # A task retains its state until its device work settles.
                    executor.shutdown(wait=True, cancel_futures=True)
                    with self._lock:
                        for request_id in list(self._requests):
                            self.finish_request(request_id, aborted=True)
                        self._active.clear()
        finally:
            self._running = False
            self._scheduler_thread_id = None
            self.shutdown_resources()

    def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        if self._scheduler_thread_id is None:
            self.shutdown_resources()

    def shutdown_resources(self) -> None:
        with self._shutdown_lock:
            callback = self._shutdown_callback
            self._shutdown_callback = None
        if callback is not None:
            callback()

    def abort(self, request_id: str) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is not None:
                request.aborted = True
            if request_id not in self._aborted_request_ids:
                self._aborted_request_ids.add(request_id)
                self._aborted_request_id_order.append(request_id)
            while len(self._aborted_request_id_order) > _ABORTED_REQUEST_ID_LIMIT:
                self._aborted_request_ids.discard(
                    self._aborted_request_id_order.popleft()
                )
        self._wakeup.set()

    def admin(
        self, action: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        del payload
        if action == ADMIN_MODEL_INFO:
            with self._lock:
                return {
                    "success": True,
                    "data": {
                        "scheduler": type(self).__name__,
                        "mode": "cooperative_steps",
                        "active_requests": len(self._active),
                        "queued_requests": len(self._requests) - len(self._active),
                        "inflight_steps": sum(
                            item.future is not None for item in self._requests.values()
                        ),
                        "max_running_requests": self._max_running,
                        "max_queued_requests": self._max_queued,
                        "max_workers": self._max_workers,
                    },
                }
        return {
            "success": True,
            "message": "stage does not support admin operations",
            "data": {"skipped": True, "unsupported": True},
        }

    def emit_request_error(self, request_id: str, error: Exception) -> None:
        self.outbox.put(
            OutgoingMessage(request_id=request_id, type="error", data=error)
        )

    def run_abort_callback(self, request_id: str) -> None:
        callback = self._abort_callback
        if callback is None:
            return
        try:
            callback(request_id)
        except Exception:
            logger.exception("StepScheduler: abort cleanup failed for %s", request_id)

    def run_request_finished_callback(self, request_id: str) -> Exception | None:
        callback = self._request_finished_callback
        if callback is None:
            return None
        try:
            callback(request_id)
        except Exception as exc:
            logger.exception(
                "StepScheduler: terminal cleanup failed for %s", request_id
            )
            return exc
        return None

    def drain_inbox(self) -> list[IncomingMessage]:
        messages: list[IncomingMessage] = []
        while True:
            try:
                messages.append(self.inbox.get_nowait())
            except _queue_mod.Empty:
                return messages

    def finish_request(
        self,
        request_id: str,
        *,
        result: StepResult | None = None,
        error: Exception | None = None,
        aborted: bool = False,
    ) -> None:
        request = self._requests.pop(request_id)
        self._active.discard(request_id)
        aborted = aborted or request.aborted or request_id in self._aborted_request_ids
        try:
            request.task.close(aborted=aborted or error is not None)
        except Exception as exc:
            logger.exception("StepScheduler: step cleanup failed for %s", request_id)
            error = error or exc
        if request.started:
            _emit_model_path_end(
                request_id,
                status="aborted" if aborted else "error" if error else "completed",
            )
        if aborted:
            self.run_abort_callback(request_id)
        elif error is not None:
            self.emit_request_error(request_id, error)
        elif result is not None:
            callback_error = self.run_request_finished_callback(request_id)
            if callback_error is not None:
                self.emit_request_error(request_id, callback_error)
            else:
                self.outbox.put(
                    OutgoingMessage(request_id, "result", data=result.output)
                )

    def admit_messages(self) -> None:
        for message in self.drain_inbox():
            request_id = message.request_id
            if request_id in self._aborted_request_ids:
                continue
            if message.type != "new_request":
                self.emit_request_error(
                    request_id,
                    ValueError("Cooperative tasks accept new_request messages only"),
                )
                continue
            if request_id in self._requests:
                continue
            try:
                task = self._build(message.data)
            except Exception as exc:
                self.emit_request_error(request_id, exc)
                continue
            self._requests[request_id] = StepRequest(task)
            if len(self._requests) > self._max_running + self._max_queued:
                # Rejected control operations may own persistent session state.
                self.finish_request(request_id, error=QueueFullError())

    def collect_results(self) -> None:
        for request_id, request in list(self._requests.items()):
            future = request.future
            if future is not None and not future.done():
                continue
            if request.aborted or request_id in self._aborted_request_ids:
                self.finish_request(request_id, aborted=True)
                continue
            if future is None:
                continue
            request.future = None
            try:
                result = future.result()
                if not isinstance(result, StepResult):
                    raise TypeError("A cooperative step must return StepResult")
            except Exception as exc:
                self.finish_request(request_id, error=exc)
                continue
            if result.done:
                self.finish_request(request_id, result=result)

    def next_request(self) -> str | None:
        """Run first steps promptly, then finish uploads in admission order."""
        chosen = None
        for request_id, request in self._requests.items():
            if (
                request_id in self._active
                and request.future is None
                and (chosen is None or not request.started)
            ):
                chosen = request_id
                if not request.started:
                    break
        return chosen

    def launch_work(self, executor: ThreadPoolExecutor) -> None:
        for request_id in self._requests:
            if len(self._active) >= self._max_running:
                break
            self._active.add(request_id)
        running = sum(item.future is not None for item in self._requests.values())
        while running < self._max_workers:
            request_id = self.next_request()
            if request_id is None:
                break
            request = self._requests[request_id]
            if not request.started:
                _emit_model_path_start(request_id)
                request.started = True
            request.future = executor.submit(request.task.step)
            request.future.add_done_callback(lambda _: self._wakeup.set())
            running += 1
