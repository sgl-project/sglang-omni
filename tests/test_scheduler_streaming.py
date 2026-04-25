# SPDX-License-Identifier: Apache-2.0
"""Scheduler streaming regression tests."""

from __future__ import annotations

import asyncio

from sglang_omni.engines.omni.scheduler import Scheduler
from sglang_omni.engines.omni.types import RequestOutput


class _DummyBatchPlanner:
    def select_requests(self, waiting_reqs, running_reqs, resource_manager):
        del waiting_reqs, running_reqs, resource_manager
        return []

    def build_batch(self, selected):
        del selected
        return None


class _DummyResourceManager:
    def free(self, request):
        del request


class _DummyIterationController:
    def update_request(self, request, output):
        del request, output

    def is_finished(self, request, output):
        del request, output
        return False


def test_scheduler_prepare_stream_preserves_early_chunks() -> None:
    asyncio.run(_run_scheduler_prepare_stream_preserves_early_chunks())


async def _run_scheduler_prepare_stream_preserves_early_chunks() -> None:
    scheduler = Scheduler(
        batch_planner=_DummyBatchPlanner(),
        resource_manager=_DummyResourceManager(),
        iteration_controller=_DummyIterationController(),
        stream_adapter=lambda request, output: output.data,
    )

    request_id = "req-1"
    scheduler.prepare_stream(request_id)
    scheduler.add_request(request_id, data={"stream": True})

    request = scheduler.requests[request_id]
    scheduler._emit_stream(
        request,
        RequestOutput(request_id=request_id, data={"audio_data": [1, 2, 3]}),
    )
    scheduler._finish_request(request)

    chunks = []
    async for chunk in scheduler.stream(request_id):
        chunks.append(chunk)

    assert chunks == [{"audio_data": [1, 2, 3]}]


def test_scheduler_completed_stream_queue_is_bounded() -> None:
    scheduler = Scheduler(
        batch_planner=_DummyBatchPlanner(),
        resource_manager=_DummyResourceManager(),
        iteration_controller=_DummyIterationController(),
        stream_adapter=lambda request, output: output.data,
    )

    total_requests = 12000
    for i in range(total_requests):
        request_id = f"req-{i}"
        scheduler.prepare_stream(request_id)
        scheduler.add_request(request_id, data={"stream": False})
        request = scheduler.requests[request_id]
        scheduler._finish_request(request)

    assert (
        len(scheduler._completed_stream_queues)
        <= Scheduler._COMPLETED_RETENTION_HARD_LIMIT
    )
    assert "req-0" not in scheduler._completed_stream_queues
    latest_request_id = f"req-{total_requests - 1}"
    assert latest_request_id in scheduler._completed_stream_queues


def test_scheduler_reused_request_id_does_not_replay_old_stream_done() -> None:
    asyncio.run(_run_scheduler_reused_request_id_does_not_replay_old_stream_done())


async def _run_scheduler_reused_request_id_does_not_replay_old_stream_done() -> None:
    scheduler = Scheduler(
        batch_planner=_DummyBatchPlanner(),
        resource_manager=_DummyResourceManager(),
        iteration_controller=_DummyIterationController(),
        stream_adapter=lambda request, output: output.data,
    )

    request_id = "req-reused"
    scheduler.prepare_stream(request_id)
    scheduler.add_request(request_id, data={"stream": True})
    first_request = scheduler.requests[request_id]
    scheduler._finish_request(first_request)

    scheduler.prepare_stream(request_id)
    scheduler.add_request(request_id, data={"stream": True})
    second_request = scheduler.requests[request_id]
    scheduler._emit_stream(
        second_request,
        RequestOutput(request_id=request_id, data={"audio_data": [4, 5, 6]}),
    )
    scheduler._finish_request(second_request)

    chunks = []
    async for chunk in scheduler.stream(request_id):
        chunks.append(chunk)

    assert chunks == [{"audio_data": [4, 5, 6]}]
