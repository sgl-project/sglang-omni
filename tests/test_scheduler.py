# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.engines.omni.runtime.encoder import EncoderBatchPlanner
from sglang_omni.engines.omni.scheduler import Scheduler
from sglang_omni.engines.omni.types import SchedulerRequest, SchedulerStatus


class _DummyBatchPlanner:
    def select_requests(self, *args, **kwargs):
        return []

    def build_batch(self, *args, **kwargs):
        return None


class _DummyResourceManager:
    def can_allocate(self, request):
        del request
        return True

    def allocate(self, request):
        del request

    def free(self, request):
        del request


class _DummyIterationController:
    def update_request(self, request, output):
        del request, output

    def is_finished(self, request, output):
        del request, output
        return False


@pytest.mark.asyncio
async def test_get_result_keeps_terminal_request_streamable() -> None:
    scheduler = Scheduler(
        _DummyBatchPlanner(),
        _DummyResourceManager(),
        _DummyIterationController(),
    )
    scheduler.add_request("req-1", {"value": 1})
    request = scheduler.requests["req-1"]
    scheduler._finish_request(request, status=SchedulerStatus.FINISHED)

    start_stream = asyncio.Event()

    async def _consume_stream() -> list[object]:
        await start_stream.wait()
        items = []
        async for item in scheduler.stream("req-1"):
            items.append(item)
        return items

    stream_task = asyncio.create_task(_consume_stream())
    result = await scheduler.get_result("req-1")
    start_stream.set()

    assert result.status == SchedulerStatus.FINISHED
    assert await stream_task == []


def test_encoder_batch_planner_respects_request_cost_budget() -> None:
    planner = EncoderBatchPlanner(
        max_batch_size=4,
        request_cost_fn=lambda request: request.data["cost"],
        max_batch_cost=10,
    )
    requests = [
        SchedulerRequest(request_id="r1", data={"cost": 4}),
        SchedulerRequest(request_id="r2", data={"cost": 5}),
        SchedulerRequest(request_id="r3", data={"cost": 3}),
    ]

    selected = planner.select_requests(requests, [], _DummyResourceManager())

    assert [request.request_id for request in selected] == ["r1", "r2"]


def test_encoder_batch_planner_allows_single_oversized_request() -> None:
    planner = EncoderBatchPlanner(
        max_batch_size=4,
        request_cost_fn=lambda request: request.data["cost"],
        max_batch_cost=10,
    )
    requests = [
        SchedulerRequest(request_id="large", data={"cost": 99}),
        SchedulerRequest(request_id="next", data={"cost": 1}),
    ]

    selected = planner.select_requests(requests, [], _DummyResourceManager())

    assert [request.request_id for request in selected] == ["large"]
