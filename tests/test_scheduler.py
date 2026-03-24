# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.engines.omni.scheduler import Scheduler
from sglang_omni.engines.omni.types import SchedulerStatus

from tests.dummy import DummyBatchPlanner, DummyIterationController, DummyResourceManager


@pytest.mark.asyncio
async def test_get_result_keeps_terminal_request_streamable() -> None:
    scheduler = Scheduler(
        DummyBatchPlanner(),
        DummyResourceManager(),
        DummyIterationController(),
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
