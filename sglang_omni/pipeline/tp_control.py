# SPDX-License-Identifier: Apache-2.0
"""Stage fanout shared by TP and SP (legacy TP names remain source-compatible).

These helpers sit above the per-rank SGLang worker layer and below the
pipeline stage abstraction. They mirror stage-control messages and, for
non-SGLang schedulers (e.g. SimpleScheduler-based image encoders),
replicate work payloads from the leader to follower ranks so that NCCL
collectives in TP-parallel forward passes do not deadlock.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import queue as queue_mod
from collections import deque
from dataclasses import dataclass
from typing import Any

from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminResultMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
)

logger = logging.getLogger(__name__)

_WORK_POLL_SECONDS = 0.1


@dataclass
class TPWorkMessage:
    """Payload replicated from a stage leader to follower schedulers."""

    request_id: str
    data: Any
    dispatch_id: int | None = None


@dataclass(frozen=True)
class ParallelAbortMessage:
    request_id: str
    dispatch_id: int

    def __post_init__(self) -> None:
        if not self.request_id or self.dispatch_id < 1:
            raise ValueError(
                "Parallel abort requires a request id and positive dispatch id"
            )


class RequestDispatchTracker:
    """Correlate terminals and cross-queue aborts, including sequential RID reuse.

    Terminals for one RID must remain FIFO. Completed ranges compress late-abort
    tombstones without forgetting old dispatches or retaining every request ID.
    """

    def __init__(self) -> None:
        self._active: dict[str, deque[int]] = {}
        self._aborted: set[tuple[str, int]] = set()
        self._watermark = 0
        self._completed: list[tuple[int, int]] = []

    def is_completed(self, dispatch_id: int) -> bool:
        if dispatch_id < 1:
            raise ValueError("Dispatch id must be positive")
        if dispatch_id <= self._watermark:
            return True
        index = bisect.bisect_right(self._completed, (dispatch_id, float("inf"))) - 1
        return index >= 0 and dispatch_id <= self._completed[index][1]

    def register_work(self, request_id: str, dispatch_id: int) -> bool:
        if self.is_completed(dispatch_id):
            return False
        active = self._active.setdefault(request_id, deque())
        if dispatch_id in active:
            return False
        if active and dispatch_id < active[-1]:
            raise RuntimeError("Out-of-order parallel work dispatch")
        active.append(dispatch_id)
        return True

    def current(self, request_id: str) -> int | None:
        active = self._active.get(request_id)
        return active[-1] if active else None

    def record_abort(self, request_id: str, dispatch_id: int) -> bool:
        key = (request_id, dispatch_id)
        if self.is_completed(dispatch_id) or key in self._aborted:
            return False
        self._aborted.add(key)
        return True

    def finish_terminal(self, request_id: str) -> int | None:
        active = self._active.get(request_id)
        if not active:
            return None
        dispatch_id = active.popleft()
        if not active:
            del self._active[request_id]
        self._aborted.discard((request_id, dispatch_id))
        ranges = self._completed
        index = bisect.bisect_left(ranges, (dispatch_id, dispatch_id))
        start = end = dispatch_id
        if index and ranges[index - 1][1] + 1 >= start:
            index -= 1
            start, previous_end = ranges.pop(index)
            end = max(end, previous_end)
        while index < len(ranges) and ranges[index][0] <= end + 1:
            end = max(end, ranges.pop(index)[1])
        ranges.insert(index, (start, end))
        if ranges[0][0] == self._watermark + 1:
            _, self._watermark = ranges.pop(0)
        return dispatch_id


class TPLeaderFanout:
    """Broadcast leader-owned stage events to TP or SP followers."""

    def __init__(
        self,
        stage_name: str,
        *,
        follower_work_queues: list[Any],
        follower_abort_queues: list[Any],
        follower_admin_result_queues: list[Any] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self._follower_work_queues = list(follower_work_queues)
        self._follower_abort_queues = list(follower_abort_queues)
        self._follower_admin_result_queues = list(follower_admin_result_queues or [])
        self._next_dispatch_id = 1

    async def fanout_control(
        self,
        msg: (
            ShutdownMessage | ProfilerStartMessage | ProfilerStopMessage | AdminMessage
        ),
    ) -> None:
        for q in self._follower_work_queues:
            q.put_nowait(msg)

    def fanout_work(self, payload: Any, *, track_dispatch: bool = False) -> int | None:
        dispatch_id = None
        if track_dispatch:
            dispatch_id = self._next_dispatch_id
            self._next_dispatch_id += 1
        msg = TPWorkMessage(
            request_id=payload.request_id, data=payload, dispatch_id=dispatch_id
        )
        for q in self._follower_work_queues:
            q.put_nowait(msg)
        return dispatch_id

    async def fanout_abort(self, msg: AbortMessage | ParallelAbortMessage) -> None:
        for q in self._follower_abort_queues:
            q.put_nowait(msg)

    async def collect_admin_results(
        self,
        op_id: str,
        *,
        timeout_s: float = 60.0,
    ) -> list[AdminResultMessage]:
        """Collect one admin result from every TP follower."""
        if not self._follower_admin_result_queues:
            return []

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda q=q: q.get(timeout=timeout_s),
            )
            for q in self._follower_admin_result_queues
        ]
        raw_results = await asyncio.gather(*tasks)
        results: list[AdminResultMessage] = []
        for msg in raw_results:
            if not isinstance(msg, AdminResultMessage):
                raise ValueError(
                    f"Unexpected TP follower admin result: {type(msg).__name__}"
                )
            if msg.result.op_id != op_id:
                raise ValueError(
                    f"Unexpected TP follower admin op id: {msg.result.op_id} != {op_id}"
                )
            results.append(msg)
        return results

    def close(self) -> None:
        self._follower_work_queues.clear()
        self._follower_abort_queues.clear()
        self._follower_admin_result_queues.clear()


class TPFollowerControlPlane:
    """Follower-side control plane backed by multiprocessing queues."""

    def __init__(
        self,
        *,
        stage_name: str,
        recv_endpoint: str = "",
        work_queue: Any,
        abort_queue: Any,
        admin_result_queue: Any | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.recv_endpoint = recv_endpoint
        self._work_queue = work_queue
        self._abort_queue = abort_queue
        self._admin_result_queue = admin_result_queue
        self._closed = False

    async def start(self) -> None:
        logger.info("TP follower control plane started for stage %s", self.stage_name)

    async def recv(
        self,
    ) -> (
        AdminMessage
        | ShutdownMessage
        | ProfilerStartMessage
        | ProfilerStopMessage
        | TPWorkMessage
    ):
        msg = await self.recv_from_queue(self._work_queue)
        if isinstance(
            msg,
            (
                AdminMessage,
                ShutdownMessage,
                ProfilerStartMessage,
                ProfilerStopMessage,
                TPWorkMessage,
            ),
        ):
            return msg
        raise ValueError(f"Unexpected TP follower work message: {type(msg)}")

    async def recv_abort(self) -> AbortMessage | ParallelAbortMessage:
        msg = await self.recv_from_queue(self._abort_queue)
        if isinstance(msg, (AbortMessage, ParallelAbortMessage)):
            return msg
        raise ValueError(f"Unexpected TP follower abort message: {type(msg)}")

    async def send_admin_result(self, msg: AdminResultMessage) -> None:
        if self._admin_result_queue is None:
            raise RuntimeError(
                f"TP follower stage {self.stage_name} has no admin result queue"
            )
        self._admin_result_queue.put_nowait(msg)

    async def recv_from_queue(self, q: Any) -> Any:
        loop = asyncio.get_running_loop()
        while True:
            if self._closed:
                raise RuntimeError(
                    f"TP follower control plane closed for stage {self.stage_name}"
                )
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: q.get(timeout=_WORK_POLL_SECONDS),
                )
            except queue_mod.Empty:
                continue

    def close(self) -> None:
        self._closed = True
