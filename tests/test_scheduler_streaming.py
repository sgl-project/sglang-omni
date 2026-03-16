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
