# SPDX-License-Identifier: Apache-2.0
"""Internal TP control helpers.

These helpers sit above the per-rank SGLang worker layer and below the
pipeline stage abstraction. They mirror stage-control messages and, for
non-SGLang schedulers (e.g. SimpleScheduler-based image encoders),
replicate work payloads from the leader to follower ranks so that NCCL
collectives in TP-parallel forward passes do not deadlock.
"""

from __future__ import annotations

import asyncio
import logging
import queue as queue_mod
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
    """Payload replicated from the TP leader to follower schedulers."""

    request_id: str
    data: Any


class TPLeaderFanout:
    """Broadcast leader-owned stage events to TP followers."""

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

    async def fanout_control(
        self,
        msg: (
            ShutdownMessage | ProfilerStartMessage | ProfilerStopMessage | AdminMessage
        ),
    ) -> None:
        for q in self._follower_work_queues:
            q.put_nowait(msg)

    def fanout_work(self, payload: Any) -> None:
        msg = TPWorkMessage(request_id=getattr(payload, "request_id", ""), data=payload)
        for q in self._follower_work_queues:
            q.put_nowait(msg)

    async def fanout_abort(self, msg: AbortMessage) -> None:
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
                    "Unexpected TP follower admin op id: "
                    f"{msg.result.op_id} != {op_id}"
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
        msg = await self._recv_from_queue(self._work_queue)
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

    async def recv_abort(self) -> AbortMessage:
        msg = await self._recv_from_queue(self._abort_queue)
        if isinstance(msg, AbortMessage):
            return msg
        raise ValueError(f"Unexpected TP follower abort message: {type(msg)}")

    async def send_admin_result(self, msg: AdminResultMessage) -> None:
        if self._admin_result_queue is None:
            raise RuntimeError(
                f"TP follower stage {self.stage_name} has no admin result queue"
            )
        self._admin_result_queue.put_nowait(msg)

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
