# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.config.schema import replica_instance_name
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionLimits, TimedChunk
from tests.unit_test.fixtures.session_pipeline import (
    PipelineResources,
    block_request_abort,
    block_session_cleanup,
    chunk,
    event_log,
    pipeline,
    wait_until,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_timeout_cancel_noop_waits_before_close(linear_pair):
    coordinator, events, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None, {"ignore_cancel": True, "delay": 0.3}),
        stages=["source", "sink"],
    )
    coordinator.sessions[session_identity.session_id].limits = SessionLimits(
        operation_timeout_s=0.1
    )
    output = coordinator.session_outputs(session_identity)
    await coordinator.append_session(session_identity, chunk(0))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(output), 5)
    await asyncio.wait_for(coordinator.close_session(session_identity), 5)
    assert session_identity.session_id not in coordinator.sessions
    log = event_log(events)
    finished = next(i for i, e in enumerate(log) if e[:2] == ("finished", "sink"))
    # Note (Junnan Li): Close waits for the hook that outlived the command timeout, then closes upstream.
    closed = [(i, e[1]) for i, e in enumerate(log) if e[0] == "close"]
    assert [owner for _, owner in closed] == ["sink", "source"]
    assert closed[0][0] > finished
    second = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"]
    )
    await coordinator.close_session(second)


@pytest.mark.asyncio(loop_scope="session")
async def test_open_timeout_closes_the_opened_owner(linear_pair):
    coordinator, events, _ = linear_pair
    with pytest.raises(TimeoutError):
        await coordinator.open_session(
            OmniRequest(None, {"open_delay": 0.3}),
            stages=["source", "sink"],
            session_id="slow-open",
            limits=SessionLimits(operation_timeout_s=0.08),
        )
    assert "slow-open" not in coordinator.sessions
    assert [e[1] for e in event_log(events) if e[0] == "close"] == ["source"]
    second = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"], session_id="slow-open"
    )
    await coordinator.close_session(second)


@pytest.mark.asyncio(loop_scope="session")
async def test_unconfirmed_owner_release_blocks_new_sessions(linear_pair):
    coordinator, events, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None, {"fail_close_once": "source"}),
        stages=["source", "sink"],
        session_id="held",
    )
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        await coordinator.close_session(session_identity)
    assert coordinator.sessions["held"].cleanup_error is not None
    with pytest.raises(ValueError, match="unavailable owner"):
        await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
    with pytest.raises(ValueError, match="already reserved"):
        await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="held"
        )
    with pytest.raises(RuntimeError, match="capacity remains reserved"):
        await coordinator.close_session(session_identity)
    # Note (Junnan Li): Owners close downstream first; the rejected owner and those upstream stay unconfirmed.
    closed = [e[1] for e in event_log(events) if e[0] == "close"]
    assert closed == ["sink", "source"]


@pytest.mark.asyncio
async def test_worker_failure_wakes_output_and_fails_session(tmp_path, monkeypatch):
    async with pipeline(tmp_path) as (coordinator, events, processes):
        session_identity = await coordinator.open_session(
            OmniRequest(None, {"delay": 30}), stages=["source", "sink"]
        )
        output = coordinator.session_outputs(session_identity)
        waiting = asyncio.create_task(anext(output))
        await coordinator.append_session(session_identity, chunk(0))
        for _ in range(100):
            if any(e[:2] == ("append", "sink") for e in event_log(events)):
                break
            await asyncio.sleep(0.05)
        processes[-1].kill()
        processes[-1].expected_exitcode = -9
        await asyncio.to_thread(processes[-1].join, 5)
        futures = list(coordinator.completion_futures.values())
        assert futures and not any(future.done() for future in futures)
        # Note (Junnan Li): Cleanup waits for the pump, which waits on this future; fail the waiters first.
        entered, release = block_session_cleanup(monkeypatch, coordinator)
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
        session_identities = [
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
            for _ in range(3)
        ]
        await coordinator.shutdown_stages([])
        assert len(coordinator.sessions) == 3
        coordinator.max_in_flight = 0
        await coordinator.shutdown_stages(["sink"])
        assert not coordinator.sessions
        assert processes[0].is_alive()
        await asyncio.to_thread(processes[1].join, 5)
        assert processes[1].exitcode == 0
        with pytest.raises(ValueError, match="unregistered owner"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
        for session_identity in session_identities:
            await coordinator.close_session(session_identity)


@pytest.mark.asyncio(loop_scope="session")
async def test_partial_open_releases_previously_opened_owner(linear_pair):
    coordinator, events, _ = linear_pair
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


@pytest.mark.asyncio(loop_scope="session")
async def test_output_overflow_closes_session(linear_pair):
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None, {"cadence": 3}),
        stages=["source", "sink"],
        limits=SessionLimits(max_output_chunks=1),
    )
    state = coordinator.sessions[session_identity.session_id]
    await coordinator.append_session(session_identity, chunk(0))
    await wait_until(lambda: session_identity.session_id not in coordinator.sessions)
    assert isinstance(state.error, QueueFullError)


@pytest.mark.asyncio(loop_scope="session")
async def test_idle_timeout_closes_session_and_wakes_reader(linear_pair):
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None),
        stages=["source", "sink"],
        limits=SessionLimits(idle_timeout_s=0.3),
        session_id="reused",
    )
    output = coordinator.session_outputs(session_identity)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(output), 5)
    reopened = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"], session_id="reused"
    )
    await coordinator.close_session(reopened)


@pytest.mark.asyncio(loop_scope="session")
async def test_cross_modality_order_and_rejected_input_retry(linear_pair):
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None),
        stages=["source", "sink"],
        limits=SessionLimits(max_pending_chunks=1),
    )
    output = coordinator.session_outputs(session_identity)
    await coordinator.append_session(session_identity, chunk(0, eos=True))
    text = TimedChunk("text", 0, 0, 1, b"hello", eos=True)
    with pytest.raises(QueueFullError):
        await coordinator.append_session(session_identity, text)
    for kind in ("data", "input_done"):
        result = await asyncio.wait_for(anext(output), 5)
        assert (result.kind, result.input_seq) == (kind, 0)
    assert await coordinator.append_session(session_identity, text) == 1
    with pytest.raises(ValueError, match="contiguous"):
        await coordinator.append_session(session_identity, text)
    for kind in ("data", "input_done"):
        result = await asyncio.wait_for(anext(output), 5)
        assert (result.kind, result.input_seq) == (kind, 1)
    with pytest.raises(ValueError, match="EOS"):
        await coordinator.append_session(session_identity, chunk(2))
    await output.aclose()


@pytest.mark.asyncio(loop_scope="session")
async def test_oversize_chunk_is_a_permanent_error(linear_pair):
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None),
        stages=["source", "sink"],
        limits=SessionLimits(max_chunk_bytes=256),
    )
    oversize = TimedChunk("audio", 0, 20, 0, b"x" * 512)
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        await coordinator.append_session(session_identity, oversize)
    assert await coordinator.append_session(session_identity, chunk(0, eos=True)) == 0
    await coordinator.close_session(session_identity)


async def assert_closing_rejects_new_input(
    coordinator: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
    trigger: Literal["close", "shutdown", "idle", "operation_timeout"],
) -> None:
    request_params = (
        {"ignore_cancel": True, "delay": 0.3} if trigger == "operation_timeout" else {}
    )
    session_identity = await coordinator.open_session(
        OmniRequest(None, request_params),
        stages=["source", "sink"],
        limits=SessionLimits(
            idle_timeout_s=0.2 if trigger == "idle" else 300,
            operation_timeout_s=0.1 if trigger == "operation_timeout" else 30,
        ),
    )
    # Note (Junnan Li): A failed operation aborts its request before the pump cleans up; admission is closed by then.
    if trigger == "operation_timeout":
        entered, release = block_request_abort(monkeypatch, coordinator)
    else:
        entered, release = block_session_cleanup(monkeypatch, coordinator)
    if trigger == "close":
        close_task = asyncio.create_task(coordinator.close_session(session_identity))
    elif trigger == "shutdown":
        close_task = asyncio.create_task(coordinator.shutdown_stages(["sink"]))
    elif trigger == "operation_timeout":
        await coordinator.append_session(session_identity, chunk(0))
        close_task = None
    else:
        assert trigger == "idle"
        close_task = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(RuntimeError, match="closing"):
            await coordinator.append_session(session_identity, chunk(1))
    finally:
        release.set()
        if close_task is not None:
            await asyncio.wait_for(close_task, 5)
        else:
            await asyncio.wait_for(coordinator.close_session(session_identity), 5)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("trigger", ["close", "idle", "operation_timeout"])
async def test_closing_rejects_input_before_cleanup(
    linear_pair: PipelineResources,
    monkeypatch: pytest.MonkeyPatch,
    trigger: Literal["close", "idle", "operation_timeout"],
) -> None:
    coordinator, _, _ = linear_pair
    await assert_closing_rejects_new_input(coordinator, monkeypatch, trigger)


@pytest.mark.asyncio
async def test_shutdown_rejects_input_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with pipeline(tmp_path) as (coordinator, _, _):
        await assert_closing_rejects_new_input(coordinator, monkeypatch, "shutdown")


@pytest.mark.asyncio(loop_scope="session")
async def test_public_submit_rejects_session_metadata(linear_pair):
    coordinator, _, _ = linear_pair
    request = OmniRequest(None, metadata={"omni_session": {}})
    with pytest.raises(ValueError, match="reserved"):
        await coordinator.submit("rogue", request)
    with pytest.raises(ValueError, match="reserved"):
        await anext(coordinator.stream("rogue", request))
    assert "rogue" not in coordinator.requests


@pytest.mark.asyncio(loop_scope="session")
async def test_close_fences_queued_outputs(linear_pair):
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"]
    )
    output = coordinator.session_outputs(session_identity)
    await coordinator.append_session(session_identity, chunk(0))
    first = await asyncio.wait_for(anext(output), 5)
    assert first.kind == "data"
    session = coordinator.sessions[session_identity.session_id]
    await coordinator.append_session(session_identity, chunk(1, eos=True))
    for _ in range(500):
        if session.outputs:
            break
        await asyncio.sleep(0.01)
    assert session.outputs
    await coordinator.close_session(session_identity)
    assert not session.outputs and session.output_bytes == 0
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(output), 5)


@pytest.mark.asyncio(loop_scope="session")
async def test_stale_incarnation_cannot_address_the_reopened_session(
    linear_pair,
) -> None:
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"], session_id="again"
    )
    await coordinator.close_session(session_identity)
    reopened = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"], session_id="again"
    )
    assert reopened.incarnation != session_identity.incarnation
    with pytest.raises(ValueError, match="stale"):
        await coordinator.append_session(session_identity, chunk(0))
    with pytest.raises(ValueError, match="stale"):
        await coordinator.close_session(session_identity)
    assert "again" in coordinator.sessions
    await coordinator.close_session(reopened)


@pytest.mark.asyncio(loop_scope="session")
async def test_acknowledged_downstream_owner_is_not_quarantined(linear_triple) -> None:
    coordinator, _, _ = linear_triple
    session_identity = await coordinator.open_session(
        OmniRequest(None, {"fail_close_once": "middle"}),
        stages=["source", "middle", "sink"],
        session_id="held",
    )
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        await coordinator.close_session(session_identity)
    unavailable = coordinator.session_unavailable_stages
    assert unavailable.issuperset({"source", "middle"})
    assert "sink" not in unavailable
    with pytest.raises(ValueError, match="unavailable owner"):
        await coordinator.open_session(
            OmniRequest(None), stages=["source", "middle", "sink"]
        )


@pytest.mark.asyncio
async def test_healthy_replica_accepts_a_new_session_after_unconfirmed_close(
    tmp_path,
) -> None:
    source_owner = replica_instance_name("source", 0)
    sink_owner = replica_instance_name("sink", 0)
    async with pipeline(tmp_path, replicated=True, replicate_entry=True) as (
        coordinator,
        events,
        _,
    ):
        failed_ref = await coordinator.open_session(
            OmniRequest(None, {"fail_close_once": sink_owner}),
            stages=["source", "sink"],
            session_id="failed",
        )
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            await coordinator.close_session(failed_ref)
        opened_owners = [
            event[1]
            for event in event_log(events)
            if event[0] == "open" and event[2] == "failed"
        ]
        assert opened_owners == [source_owner, sink_owner]
        healthy_ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="healthy"
        )
        outputs = coordinator.session_outputs(healthy_ref)
        await coordinator.append_session(healthy_ref, chunk(0, eos=True))
        output_chunk = await asyncio.wait_for(anext(outputs), 5)
        assert output_chunk.kind == "data"
        await outputs.aclose()
        with pytest.raises(ValueError, match="unavailable owner"):
            await coordinator.open_session(OmniRequest(None), stages=["source", "sink"])
