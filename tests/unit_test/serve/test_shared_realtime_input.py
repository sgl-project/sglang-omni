# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from sglang_omni.serve.realtime.control import Drained, UnitCompleted
from sglang_omni.serve.realtime.runtime import (
    Capabilities,
    ProtocolError,
    RuntimeLimits,
    SessionRuntime,
)
from tests.unit_test.fixtures.realtime_websocket import Adapter, next_event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,padding,pcm_bytes", [("flush", 0, 160), ("pad", 15, 640)]
)
async def test_accepted_tail_reports_only_real_consumption(policy, padding, pcm_bytes):
    class TailAdapter(Adapter):
        async def process(self, unit, epoch):
            assert len(unit.pcm) == pcm_bytes
            return unit.real_samples

    runtime = SessionRuntime(
        "mock", Capabilities(tail_policy=policy), TailAdapter, RuntimeLimits()
    )
    outputs = runtime.outputs()
    try:
        await runtime.update({}, "open")
        await runtime.append(b"\0\0" * 80, 0, 0, "append")
        await runtime.end("end")
        event = (await next_event(outputs, Drained)).event
        assert (event.consumed_ms, event.padding_ms) == (5, padding)
    finally:
        await runtime.close("client_closed")
        await outputs.aclose()


@pytest.mark.asyncio
async def test_rejected_tail_leaves_input_open_for_retry():
    runtime = SessionRuntime(
        "mock", Capabilities(tail_policy="reject"), Adapter, RuntimeLimits()
    )
    outputs = runtime.outputs()
    try:
        await runtime.update({}, "open")
        await runtime.append(b"\0\0" * 80, 0, 0, "append")
        with pytest.raises(ProtocolError, match="partial native unit"):
            await runtime.end("end")
        await runtime.append(b"\0\0" * 240, 1, 5, "retry")
        await runtime.end("end-retry")
        assert (await next_event(outputs, Drained)).event.consumed_ms == 20
    finally:
        await runtime.close("client_closed")
        await outputs.aclose()


@pytest.mark.asyncio
async def test_unit_done_follows_successful_processing():
    entered, release, observed = (asyncio.Event() for _ in range(3))

    class BlockedAdapter(Adapter):
        async def process(self, unit, epoch):
            entered.set()
            await release.wait()
            return unit.real_samples

    runtime = SessionRuntime("mock", Capabilities(), BlockedAdapter, RuntimeLimits())
    outputs = runtime.outputs()

    async def consume():
        envelope = await next_event(outputs, UnitCompleted)
        observed.set()
        return envelope.event

    task = asyncio.create_task(consume())
    try:
        await runtime.update({}, "open")
        await runtime.append(b"\0\0" * 320, 0, 0, "append")
        await asyncio.wait_for(entered.wait(), 5)
        assert not observed.is_set()
        release.set()
        assert (await asyncio.wait_for(task, 5)).unit_id == "unit_0"
    finally:
        release.set()
        await runtime.close("client_closed")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await outputs.aclose()
