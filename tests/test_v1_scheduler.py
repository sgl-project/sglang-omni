# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

from sglang_omni_v1.pipeline.coordinator import Coordinator
from sglang_omni_v1.proto import CompleteMessage, RequestInfo, RequestState
from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.simple_scheduler import SimpleScheduler


def _costed_scheduler() -> SimpleScheduler:
    return SimpleScheduler(
        compute_fn=lambda payload: payload,
        batch_compute_fn=lambda payloads: payloads,
        max_batch_size=4,
        request_cost_fn=lambda payload: payload["cost"],
        max_batch_cost=10,
    )


def test_v1_simple_scheduler_batch_cost_budget_defers_next_request() -> None:
    scheduler = _costed_scheduler()
    first = IncomingMessage("r1", "new_request", {"cost": 4})
    scheduler.inbox.put(IncomingMessage("r2", "new_request", {"cost": 5}))
    scheduler.inbox.put(IncomingMessage("r3", "new_request", {"cost": 3}))

    batch = scheduler._collect_batch(first)

    assert [msg.request_id for msg in batch] == ["r1", "r2"]
    assert [msg.request_id for msg in scheduler._pending_messages] == ["r3"]


def test_v1_simple_scheduler_batch_cost_allows_single_oversized_request() -> None:
    scheduler = _costed_scheduler()
    first = IncomingMessage("large", "new_request", {"cost": 99})
    scheduler.inbox.put(IncomingMessage("next", "new_request", {"cost": 1}))

    batch = scheduler._collect_batch(first)

    assert [msg.request_id for msg in batch] == ["large"]
    assert [msg.request_id for msg in scheduler._pending_messages] == ["next"]


def test_v1_coordinator_terminal_completion_reaches_stream_queue() -> None:
    asyncio.run(_run_terminal_completion_reaches_stream_queue())


async def _run_terminal_completion_reaches_stream_queue() -> None:
    coordinator = Coordinator(
        completion_endpoint="inproc://complete",
        abort_endpoint="inproc://abort",
        entry_stage="preprocess",
        terminal_stages=["decode"],
    )
    loop = asyncio.get_running_loop()
    coordinator._requests["req-1"] = RequestInfo(
        request_id="req-1",
        state=RequestState.RUNNING,
    )
    coordinator._completion_futures["req-1"] = loop.create_future()
    stream_queue = asyncio.Queue()
    coordinator._stream_queues["req-1"] = stream_queue
    completion = CompleteMessage(
        request_id="req-1",
        from_stage="decode",
        success=True,
        result={"text": "done"},
    )

    await coordinator._handle_completion(completion)

    assert await coordinator._completion_futures["req-1"] == {"text": "done"}
    assert await stream_queue.get() is completion
    assert "req-1" not in coordinator._requests
