# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest

from sglang_omni.pipeline.tp_control import TPLeaderFanout
from sglang_omni.proto import AdminResult, AdminResultMessage


def _result(op_id: str) -> AdminResultMessage:
    return AdminResultMessage(
        AdminResult(
            op_id=op_id,
            stage="decode",
            action="model_info",
            success=True,
        )
    )


def test_collect_admin_results_discards_result_from_timed_out_operation() -> None:
    async def _run() -> None:
        result_queue: queue.Queue = queue.Queue()
        fanout = TPLeaderFanout(
            "decode",
            follower_work_queues=[],
            follower_abort_queues=[],
            follower_admin_result_queues=[result_queue],
        )

        with pytest.raises(queue.Empty):
            await fanout.collect_admin_results("op-1", timeout_s=0.01)

        result_queue.put_nowait(_result("op-1"))
        result_queue.put_nowait(_result("op-2"))
        results = await fanout.collect_admin_results("op-2", timeout_s=1.0)
        assert [msg.result.op_id for msg in results] == ["op-2"]

        result_queue.put_nowait(_result("op-3"))
        results = await fanout.collect_admin_results("op-3", timeout_s=1.0)
        assert [msg.result.op_id for msg in results] == ["op-3"]

    asyncio.run(_run())


def test_collect_admin_results_rejects_unexpected_message() -> None:
    async def _run() -> None:
        result_queue: queue.Queue = queue.Queue()
        result_queue.put_nowait(object())
        fanout = TPLeaderFanout(
            "decode",
            follower_work_queues=[],
            follower_abort_queues=[],
            follower_admin_result_queues=[result_queue],
        )

        with pytest.raises(
            ValueError,
            match="Unexpected TP follower admin result: object",
        ):
            await fanout.collect_admin_results("op-1", timeout_s=1.0)

    asyncio.run(_run())


def test_collect_admin_results_keeps_deadline_after_stale_result() -> None:
    class StaleThenEmptyQueue:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def get(self, *, timeout: float) -> AdminResultMessage:
            self.timeouts.append(timeout)
            if len(self.timeouts) == 1:
                time.sleep(0.02)
                return _result("op-1")
            raise queue.Empty

    async def _run() -> None:
        result_queue = StaleThenEmptyQueue()
        fanout = TPLeaderFanout(
            "decode",
            follower_work_queues=[],
            follower_abort_queues=[],
            follower_admin_result_queues=[result_queue],
        )

        with pytest.raises(queue.Empty):
            await fanout.collect_admin_results("op-2", timeout_s=0.1)

        assert len(result_queue.timeouts) == 2
        assert result_queue.timeouts[1] < result_queue.timeouts[0] - 0.01

    asyncio.run(_run())


def test_invalid_result_does_not_leave_other_follower_collector_running() -> None:
    class GatedFirstGetQueue:
        def __init__(self) -> None:
            self._items: queue.Queue = queue.Queue()
            self._lock = threading.Lock()
            self._calls = 0
            self.first_get_started = threading.Event()
            self.release_first_get = threading.Event()
            self.first_get_finished = threading.Event()

        def put_nowait(self, item: AdminResultMessage) -> None:
            self._items.put_nowait(item)

        def get(self, *, timeout: float) -> AdminResultMessage:
            with self._lock:
                self._calls += 1
                call_number = self._calls
            if call_number == 1:
                self.first_get_started.set()
                if not self.release_first_get.wait(timeout):
                    self.first_get_finished.set()
                    raise queue.Empty
            try:
                return self._items.get(timeout=timeout)
            finally:
                if call_number == 1:
                    self.first_get_finished.set()

    async def _run() -> None:
        invalid_queue: queue.Queue = queue.Queue()
        gated_queue = GatedFirstGetQueue()
        fanout = TPLeaderFanout(
            "decode",
            follower_work_queues=[],
            follower_abort_queues=[],
            follower_admin_result_queues=[invalid_queue, gated_queue],
        )

        collection = asyncio.create_task(
            fanout.collect_admin_results("op-1", timeout_s=0.1)
        )
        while not gated_queue.first_get_started.is_set():
            await asyncio.sleep(0)
        invalid_queue.put_nowait(object())

        with pytest.raises(
            ValueError,
            match="Unexpected TP follower admin result: object",
        ):
            await collection

        invalid_queue.put_nowait(_result("op-2"))
        gated_queue.put_nowait(_result("op-2"))
        gated_queue.release_first_get.set()
        while not gated_queue.first_get_finished.is_set():
            await asyncio.sleep(0)

        results = await fanout.collect_admin_results("op-2", timeout_s=0.1)
        assert [msg.result.op_id for msg in results] == ["op-2", "op-2"]

    asyncio.run(_run())
