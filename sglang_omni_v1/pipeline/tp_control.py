# SPDX-License-Identifier: Apache-2.0
"""Internal TP control helpers.

These helpers sit above the per-rank SGLang worker layer and below the
pipeline stage abstraction. They only mirror stage-control messages from
the leader to follower ranks; request replication remains owned by
SGLang itself.
"""

from __future__ import annotations

import asyncio
import logging
import queue as queue_mod
from typing import Any

from sglang_omni_v1.proto import (
    AbortMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
    ShutdownMessage,
)

logger = logging.getLogger(__name__)

_WORK_POLL_SECONDS = 0.1


class TPLeaderFanout:
    """Broadcast leader-owned stage events to TP followers."""

    def __init__(
        self,
        stage_name: str,
        *,
        follower_work_queues: list[Any],
        follower_abort_queues: list[Any],
    ) -> None:
        self.stage_name = stage_name
        self._follower_work_queues = list(follower_work_queues)
        self._follower_abort_queues = list(follower_abort_queues)

    async def fanout_control(
        self,
        msg: ShutdownMessage | ProfilerStartMessage | ProfilerStopMessage,
    ) -> None:
        for q in self._follower_work_queues:
            q.put_nowait(msg)

    async def fanout_abort(self, msg: AbortMessage) -> None:
        for q in self._follower_abort_queues:
            q.put_nowait(msg)

    def close(self) -> None:
        self._follower_work_queues.clear()
        self._follower_abort_queues.clear()


class TPFollowerControlPlane:
    """Follower-side control plane backed by multiprocessing queues."""

    def __init__(
        self,
        *,
        stage_name: str,
        recv_endpoint: str = "",
        work_queue: Any,
        abort_queue: Any,
    ) -> None:
        self.stage_name = stage_name
        self.recv_endpoint = recv_endpoint
        self._work_queue = work_queue
        self._abort_queue = abort_queue
        self._closed = False

    async def start(self) -> None:
        logger.info("TP follower control plane started for stage %s", self.stage_name)

    async def recv(
        self,
    ) -> ShutdownMessage | ProfilerStartMessage | ProfilerStopMessage:
        msg = await self._recv_from_queue(self._work_queue)
        if isinstance(
            msg,
            (
                ShutdownMessage,
                ProfilerStartMessage,
                ProfilerStopMessage,
            ),
        ):
            return msg
        raise ValueError(f"Unexpected TP follower work message: {type(msg)}")

    async def recv_abort(self) -> AbortMessage:
        msg = await self._recv_from_queue(self._abort_queue)
        if isinstance(msg, AbortMessage):
            return msg
        raise ValueError(f"Unexpected TP follower abort message: {type(msg)}")

    async def _recv_from_queue(self, q: Any) -> Any:
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
