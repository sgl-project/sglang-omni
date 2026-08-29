# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def test_batch_compute_can_return_a_request_specific_error() -> None:
    scheduler = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: [ValueError("bad input"), payloads[1]],
        max_batch_size=2,
    )
    batch = [
        IncomingMessage("bad", "new_request", "bad"),
        IncomingMessage("good", "new_request", "good"),
    ]
    loop = asyncio.new_event_loop()
    try:
        scheduler._run_batch(batch, loop)
    finally:
        loop.close()

    error = scheduler.outbox.get_nowait()
    result = scheduler.outbox.get_nowait()
    assert error.request_id == "bad"
    assert error.type == "error"
    assert isinstance(error.data, ValueError)
    assert result.request_id == "good"
    assert result.type == "result"
    assert result.data == "good"
