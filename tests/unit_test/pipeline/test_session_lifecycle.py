# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionLimits, TimedChunk
from tests.unit_test.fixtures.session_pipeline import block_async_call, chunk, pipeline


def drain(events):
    log = []
    while not events.empty():
        log.append(events.get(timeout=1))
    return log


@pytest.mark.asyncio
async def test_timeout_cancel_noop_waits_before_close(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"ignore_cancel": True, "delay": 0.3}),
            stages=["source", "sink"],
        )
        coordinator._sessions[ref.session_id].limits = SessionLimits(
            command_timeout_s=0.1
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(output), 5)
        # Note (Junnan Li): A failed close retains the session reservation until worker teardown.
        assert coordinator._sessions[ref.session_id].cleanup_error is not None
        await asyncio.sleep(0.4)
        log = []
        while not events.empty():
            log.append(events.get(timeout=1))
        finished = next(i for i, e in enumerate(log) if e[:2] == ("finished", "sink"))
        # Note (Junnan Li): A timed-out queued close may be dropped by request abort. If it ran,
        # it must follow completion; otherwise scheduler.stop owns reclamation.
        close_positions = [i for i, e in enumerate(log) if e[:2] == ("close", "sink")]
        assert all(i > finished for i in close_positions)
        assert not any(e[:2] == ("close", "source") for e in log)


@pytest.mark.asyncio
async def test_open_timeout_quarantines_and_worker_shutdown_releases(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        with pytest.raises(TimeoutError):
            await coordinator.open_session(
                OmniRequest(None, {"open_delay": 0.3}),
                stages=["source", "sink"],
                session_id="slow-open",
                limits=SessionLimits(command_timeout_s=0.08),
            )
        assert coordinator._sessions["slow-open"].cleanup_error is not None
        await asyncio.sleep(0.4)
        with pytest.raises(RuntimeError, match="capacity remains reserved"):
            await coordinator.close_session(coordinator._sessions["slow-open"].ref)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["close_rejected", "pump_unresponsive"])
async def test_unconfirmed_owner_release_blocks_new_sessions(tmp_path, reason):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        params = {"fail_close_once": "source"}
        if reason == "pump_unresponsive":
            params = {"ignore_cancel": True, "delay": 0.3}
        ref = await coordinator.open_session(
            OmniRequest(None, params), stages=["source", "sink"], session_id="held"
        )
        if reason == "pump_unresponsive":
            coordinator._sessions["held"].limits = SessionLimits(command_timeout_s=0.1)
            output = coordinator.session_outputs(ref)
            await coordinator.append_session(ref, chunk(0))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(anext(output), 5)
            await output.aclose()
        else:
            with pytest.raises(RuntimeError, match="cleanup incomplete"):
                await coordinator.close_session(ref)
        assert coordinator._sessions["held"].cleanup_error is not None
        with pytest.raises(ValueError, match="unavailable owner"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
        with pytest.raises(ValueError, match="already reserved"):
            await coordinator.open_session(
                OmniRequest(None), stages=["source", "sink"], session_id="held"
            )
        with pytest.raises(RuntimeError, match="capacity remains reserved"):
            await coordinator.close_session(ref)
        if reason == "close_rejected":
            # Note (Junnan Li): Only the owner whose close was rejected, and owners upstream of
            # it, stay unconfirmed; the downstream owner acknowledged its close.
            closed = [e[1] for e in drain(events) if e[0] == "close"]
            assert closed == ["sink", "source"]
        await asyncio.sleep(0.4)


@pytest.mark.asyncio
async def test_worker_failure_wakes_output_and_fails_session(tmp_path, monkeypatch):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"delay": 30}), stages=["source", "sink"]
        )
        output = coordinator.session_outputs(ref)
        waiting = asyncio.create_task(anext(output))
        await coordinator.append_session(ref, chunk(0))
        for _ in range(100):
            if any(e[:2] == ("append", "sink") for e in drain(events)):
                break
            await asyncio.sleep(0.05)
        processes[-1].kill()
        processes[-1].expected_exitcode = -9
        await asyncio.to_thread(processes[-1].join, 5)
        futures = list(coordinator._completion_futures.values())
        assert futures and not any(future.done() for future in futures)
        # Note (Junnan Li): The pump is parked on the unit's completion future; cleanup waits
        # for the pump, so request waiters must be failed before cleanup is entered.
        entered, release, _ = block_async_call(
            monkeypatch, coordinator, "_cleanup_session"
        )
        failing = asyncio.create_task(
            coordinator.fail_pending_requests("session worker exited")
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert all(future.done() for future in futures)
        release.set()
        await asyncio.wait_for(failing, 5)
        with pytest.raises(RuntimeError, match="worker exited"):
            await asyncio.wait_for(waiting, 5)
        with pytest.raises(RuntimeError, match="worker exited"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])


@pytest.mark.asyncio
async def test_public_subset_shutdown_closes_owners_despite_full_admission(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        refs = [
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
            for _ in range(3)
        ]
        with pytest.raises(QueueFullError):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
        await coordinator.shutdown_stages([])
        assert len(coordinator._sessions) == 3
        coordinator.max_in_flight = 0
        await coordinator.shutdown_stages(["sink"])
        assert not coordinator._sessions
        assert processes[0].is_alive()
        await asyncio.to_thread(processes[1].join, 5)
        assert processes[1].exitcode == 0
        with pytest.raises(ValueError, match="unregistered owner"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
        for ref in refs:
            await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_partial_open_releases_previously_opened_owner(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        with pytest.raises(RuntimeError, match="open failed"):
            await coordinator.open_session(
                OmniRequest(None, {"fail_open": "sink"}),
                stages=["source", "sink"],
                session_id="reused",
            )
        log = [await asyncio.to_thread(events.get, True, 1) for _ in range(3)]
        assert [entry[1] for entry in log if entry[0] == "close"] == ["source"]
        reopened = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="reused"
        )
        await coordinator.close_session(reopened)


@pytest.mark.asyncio
async def test_cancel_failure_closes_owners_in_reverse_order(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cannot_abort": True}),
            stages=["source", "sink"],
            session_id="reused",
        )
        with pytest.raises(RuntimeError, match="cannot be retained"):
            await coordinator.abort_session(ref)
        log = [await asyncio.to_thread(events.get, True, 1) for _ in range(5)]
        assert [entry[1] for entry in log if entry[0] == "close"] == ["sink", "source"]
        reopened = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="reused"
        )
        await coordinator.close_session(reopened)


@pytest.mark.asyncio
async def test_pending_input_limit_rejects_extra_unit(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(max_pending_chunks=1),
        )
        await coordinator.append_session(ref, chunk(0))
        with pytest.raises(QueueFullError):
            await coordinator.append_session(ref, chunk(1))
        await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_input_sequence_rejects_a_gap(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"]
        )
        await coordinator.append_session(ref, chunk(0))
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(ref, chunk(2))
        await coordinator.append_session(ref, chunk(1, eos=True))
        await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_output_overflow_closes_session(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None, {"cadence": 3}),
            stages=["source", "sink"],
            limits=SessionLimits(max_output_chunks=1),
        )
        state = coordinator._sessions[ref.session_id]
        await coordinator.append_session(ref, chunk(0))
        async with asyncio.timeout(5):
            while ref.session_id in coordinator._sessions:
                await asyncio.sleep(0.01)
        assert isinstance(state.error, QueueFullError)


@pytest.mark.asyncio
async def test_idle_timeout_closes_session_and_wakes_reader(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(idle_timeout_s=0.3),
            session_id="reused",
        )
        output = coordinator.session_outputs(ref)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(output), 5)
        reopened = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="reused"
        )
        await coordinator.close_session(reopened)


@pytest.mark.asyncio
async def test_cross_modality_order_and_rejected_input_retry(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        ref = await coordinator.open_session(
            OmniRequest(None),
            stages=["source", "sink"],
            limits=SessionLimits(max_pending_chunks=1),
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0, eos=True))
        text = TimedChunk("text", 0, 0, 1, "hello", eos=True)
        with pytest.raises(QueueFullError):
            await coordinator.append_session(ref, text)
        for kind in ("data", "input_done"):
            result = await asyncio.wait_for(anext(output), 5)
            assert (result.kind, result.input_seq) == (kind, 0)
        assert await coordinator.append_session(ref, text) == 1
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(ref, text)
        for kind in ("data", "input_done"):
            result = await asyncio.wait_for(anext(output), 5)
            assert (result.kind, result.input_seq) == (kind, 1)
        with pytest.raises(ValueError, match="EOS"):
            await coordinator.append_session(ref, chunk(2))
        await output.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["close", "shutdown", "idle", "command_timeout"])
async def test_closing_rejects_input_before_cleanup(tmp_path, monkeypatch, trigger):
    async with pipeline(tmp_path) as (coordinator, _, _):
        params = (
            {"ignore_cancel": True, "delay": 0.3}
            if trigger == "command_timeout"
            else {}
        )
        ref = await coordinator.open_session(
            OmniRequest(None, params),
            stages=["source", "sink"],
            limits=SessionLimits(
                idle_timeout_s=0.2 if trigger == "idle" else 300,
                command_timeout_s=0.1 if trigger == "command_timeout" else 30,
            ),
        )
        # Note (Junnan Li): A failed command finalizes through request abort before the
        # pump can start cleanup; admission must already be closed at that seam.
        seam = "abort" if trigger == "command_timeout" else "_cleanup_session"
        entered, release, _ = block_async_call(monkeypatch, coordinator, seam)
        task = None
        if trigger == "close":
            task = asyncio.create_task(coordinator.close_session(ref))
        elif trigger == "shutdown":
            task = asyncio.create_task(coordinator.shutdown_stages(["sink"]))
        elif trigger == "command_timeout":
            await coordinator.append_session(ref, chunk(0))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(RuntimeError, match="closing"):
                await coordinator.append_session(ref, chunk(1))
        finally:
            release.set()
            if task is not None:
                await asyncio.wait_for(task, 5)
            elif trigger == "command_timeout":
                # Note (Junnan Li): The non-preemptible hook outlives the command timeout, so
                # this close reports an incomplete cleanup; wait for the hook before teardown.
                with pytest.raises(RuntimeError, match="capacity remains reserved"):
                    await asyncio.wait_for(coordinator.close_session(ref), 5)
                await asyncio.sleep(0.4)
            else:
                await asyncio.wait_for(coordinator.close_session(ref), 5)
