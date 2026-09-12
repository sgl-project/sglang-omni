# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.proto import OmniRequest
from tests.unit_test.fixtures.session_pipeline import block_async_call, chunk, pipeline


@pytest.mark.asyncio
async def test_abort_epoch_suppression_and_reuse(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cadence": 5, "delay": 0.05}),
            stages=["source", "sink"],
            session_id="reusable",
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0))
        assert (await asyncio.wait_for(anext(output), 5)).ref == ref
        new_ref = await coordinator.abort_session(ref)
        assert new_ref.epoch == ref.epoch + 1
        with pytest.raises(ValueError, match="stale"):
            await coordinator.append_session(ref, chunk(0))
        await coordinator.append_session(new_ref, chunk(1, eos=True))
        receipt = await asyncio.wait_for(anext(output), 5)
        assert receipt.kind == "input_done" and receipt.input_seq == 0
        result = await asyncio.wait_for(anext(output), 5)
        assert result.ref == new_ref and result.seq >= 1
        await output.aclose()
        reused = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="reusable"
        )
        assert reused.incarnation != ref.incarnation
        await coordinator.close_session(reused)


@pytest.mark.asyncio
async def test_abort_keeps_queued_completion_receipt_but_drops_old_data(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"]
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0))
        assert (await asyncio.wait_for(anext(output), 5)).kind == "data"
        assert (await asyncio.wait_for(anext(output), 5)).kind == "input_done"
        await coordinator.append_session(ref, chunk(1))
        session = coordinator._sessions[ref.session_id]
        async with asyncio.timeout(5):
            while session.pending_count:
                await asyncio.sleep(0.01)
        assert [item.kind for item, _ in session.outputs] == ["data", "input_done"]
        new_ref = await coordinator.abort_session(ref)
        receipt = await asyncio.wait_for(anext(output), 5)
        assert receipt.kind == "input_done" and receipt.input_seq == 1
        assert receipt.ref == ref
        await coordinator.append_session(new_ref, chunk(2))
        data = await asyncio.wait_for(anext(output), 5)
        assert data.kind == "data" and data.ref == new_ref and data.input_seq == 2
        await output.aclose()


@pytest.mark.asyncio
async def test_cancel_finishes_active_unit_and_preserves_consumption(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cadence": 3, "delay": 0.08}),
            stages=["source", "sink"],
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0))
        assert (await asyncio.wait_for(anext(output), 5)).kind == "data"
        await coordinator.append_session(ref, chunk(1, eos=True))
        new_ref = await coordinator.abort_session(ref)
        receipt = await asyncio.wait_for(anext(output), 5)
        assert receipt.kind == "input_done" and receipt.input_seq == 0
        assert receipt.ref == ref
        later = await asyncio.wait_for(anext(output), 5)
        assert later.kind == "data" and later.input_seq == 1
        assert later.ref == new_ref and later.payload[0] == 2
        await output.aclose()
        log = []
        while not events.empty():
            log.append(events.get(timeout=1))
        assert not any(e[0] == "cancelled" for e in log)


@pytest.mark.asyncio
async def test_close_previous_epoch_after_cancelled_abort_waiter(tmp_path, monkeypatch):
    async with pipeline(tmp_path) as (coordinator, _, _):
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="reused"
        )
        entered, release, completed = block_async_call(
            monkeypatch, coordinator, "_abort_session"
        )
        waiter = asyncio.create_task(coordinator.abort_session(ref))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        finally:
            release.set()
        await asyncio.wait_for(completed.wait(), 5)
        await coordinator.close_session(ref)
        current = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="reused"
        )
        with pytest.raises(ValueError, match="stale"):
            await coordinator.close_session(ref)
        await coordinator.append_session(current, chunk(0))
        await coordinator.close_session(current)
