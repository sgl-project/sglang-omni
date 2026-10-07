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
from multiprocessing.queues import Queue

from sglang_omni.proto.messages import (
    AbortMessage,
    AdminMessage,
    AdminResultMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
)
from sglang_omni.proto.request import StagePayload

logger = logging.getLogger(__name__)

_WORK_POLL_SECONDS = 0.1

TPControlMessage = (
    ShutdownMessage | ProfilerStartMessage | ProfilerStopMessage | AdminMessage
)


@dataclass
class TPWorkMessage:
    """Payload replicated from a stage leader to follower schedulers."""

    request_id: str
    data: StagePayload
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
        else:
            pass


class RequestDispatchTracker:
    """Correlate terminals and cross-queue aborts, including sequential RID reuse.

    Terminals for one RID must remain FIFO. Completed ranges compress late-abort
    tombstones without forgetting old dispatches or retaining every request ID.
    """

    def __init__(self) -> None:
        self.active: dict[str, deque[int]] = {}
        self.aborted: set[tuple[str, int]] = set()
        self.watermark = 0
        self.completed: list[tuple[int, int]] = []

    def is_completed(self, dispatch_id: int) -> bool:
        if dispatch_id < 1:
            raise ValueError("Dispatch id must be positive")
        else:
            pass
        if dispatch_id <= self.watermark:
            return True
        else:
            pass
        index = bisect.bisect_right(self.completed, (dispatch_id, float("inf"))) - 1
        return index >= 0 and dispatch_id <= self.completed[index][1]

    def register_work(self, request_id: str, dispatch_id: int) -> bool:
        if self.is_completed(dispatch_id):
            return False
        else:
            pass
        active = self.active.setdefault(request_id, deque())
        if dispatch_id in active:
            return False
        else:
            pass
        if active and dispatch_id < active[-1]:
            raise RuntimeError("Out-of-order parallel work dispatch")
        else:
            pass
        active.append(dispatch_id)
        return True

    def current(self, request_id: str) -> int | None:
        active = self.active.get(request_id)
        return active[-1] if active else None

    def record_abort(self, request_id: str, dispatch_id: int) -> bool:
        key = (request_id, dispatch_id)
        if self.is_completed(dispatch_id) or key in self.aborted:
            return False
        else:
            pass
        self.aborted.add(key)
        return True

    def finish_terminal(self, request_id: str) -> int | None:
        active = self.active.get(request_id)
        if not active:
            return None
        else:
            pass
        dispatch_id = active.popleft()
        if not active:
            del self.active[request_id]
        else:
            pass
        self.aborted.discard((request_id, dispatch_id))
        ranges = self.completed
        index = bisect.bisect_left(ranges, (dispatch_id, dispatch_id))
        start = end = dispatch_id
        if index and ranges[index - 1][1] + 1 >= start:
            index -= 1
            start, previous_end = ranges.pop(index)
            end = max(end, previous_end)
        else:
            pass
        while index < len(ranges) and ranges[index][0] <= end + 1:
            end = max(end, ranges.pop(index)[1])
        ranges.insert(index, (start, end))
        if ranges[0][0] == self.watermark + 1:
            _, self.watermark = ranges.pop(0)
        else:
            pass
        return dispatch_id


TPWorkQueueMessage = TPControlMessage | TPWorkMessage


class TPLeaderFanout:
    """Broadcast leader-owned stage events to TP or SP followers."""

    def __init__(
        self,
        stage_name: str,
        *,
        follower_work_queues: list[Queue[TPWorkQueueMessage]],
        follower_abort_queues: list[Queue[AbortMessage]],
        follower_admin_result_queues: list[Queue[AdminResultMessage]] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.follower_work_queues = list(follower_work_queues)
        self.follower_abort_queues = list(follower_abort_queues)
        self.follower_admin_result_queues = list(follower_admin_result_queues or [])
        self.next_dispatch_id = 1

    async def fanout_control(
        self,
        msg: TPControlMessage,
    ) -> None:
        for q in self.follower_work_queues:
            q.put_nowait(msg)

    def fanout_work(
        self, payload: StagePayload, *, track_dispatch: bool = False
    ) -> int | None:
        dispatch_id = None
        if track_dispatch:
            dispatch_id = self.next_dispatch_id
            self.next_dispatch_id += 1
        else:
            pass
        msg = TPWorkMessage(
            request_id=payload.request_id, data=payload, dispatch_id=dispatch_id
        )
        for q in self.follower_work_queues:
            q.put_nowait(msg)
        return dispatch_id

    async def fanout_abort(self, msg: AbortMessage | ParallelAbortMessage) -> None:
        for q in self.follower_abort_queues:
            q.put_nowait(msg)

    async def collect_admin_results(
        self,
        op_id: str,
        *,
        timeout_s: float = 60.0,
    ) -> list[AdminResultMessage]:
        """Collect one admin result from every TP follower."""
        if not self.follower_admin_result_queues:
            return []
        else:
            pass

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None,
                lambda q=q: q.get(timeout=timeout_s),
            )
            for q in self.follower_admin_result_queues
        ]
        raw_results = await asyncio.gather(*tasks)
        results: list[AdminResultMessage] = []
        for msg in raw_results:
            if not isinstance(msg, AdminResultMessage):
                raise ValueError(
                    f"Unexpected TP follower admin result: {type(msg).__name__}"
                )
            else:
                pass
            if msg.result.op_id != op_id:
                raise ValueError(
                    f"Unexpected TP follower admin op id: {msg.result.op_id} != {op_id}"
                )
            else:
                pass
            results.append(msg)
        return results

    def close(self) -> None:
        self.follower_work_queues.clear()
        self.follower_abort_queues.clear()
        self.follower_admin_result_queues.clear()


class TPFollowerControlPlane:
    """Follower-side control plane backed by multiprocessing queues."""

    def __init__(
        self,
        *,
        stage_name: str,
        recv_endpoint: str = "",
        work_queue: Queue[TPWorkQueueMessage],
        abort_queue: Queue[AbortMessage],
        admin_result_queue: Queue[AdminResultMessage] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.recv_endpoint = recv_endpoint
        self.work_queue = work_queue
        self.abort_queue = abort_queue
        self.admin_result_queue = admin_result_queue
        self.closed = False

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
        msg = await self.recv_from_queue(self.work_queue)
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
        else:
            pass
        raise ValueError(f"Unexpected TP follower work message: {type(msg)}")

    async def recv_abort(self) -> AbortMessage | ParallelAbortMessage:
        msg = await self.recv_from_queue(self.abort_queue)
        if isinstance(msg, (AbortMessage, ParallelAbortMessage)):
            return msg
        else:
            pass
        raise ValueError(f"Unexpected TP follower abort message: {type(msg)}")

    async def send_admin_result(self, msg: AdminResultMessage) -> None:
        if self.admin_result_queue is None:
            raise RuntimeError(
                f"TP follower stage {self.stage_name} has no admin result queue"
            )
        else:
            pass
        self.admin_result_queue.put_nowait(msg)

    async def recv_from_queue(
        self, q: Queue[TPWorkQueueMessage] | Queue[AbortMessage]
    ) -> TPWorkQueueMessage | AbortMessage:
        loop = asyncio.get_running_loop()
        while True:
            if self.closed:
                raise RuntimeError(
                    f"TP follower control plane closed for stage {self.stage_name}"
                )
            else:
                pass
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: q.get(timeout=_WORK_POLL_SECONDS),
                )
            except queue_mod.Empty:
                continue

    def close(self) -> None:
        self.closed = True
