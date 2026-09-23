# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from multiprocessing.queues import Queue
from typing import Literal

import pytest

from sglang_omni.pipeline.sessions import Session
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import TimedChunk, find_session_operation
from tests.unit_test.fixtures.session_pipeline import chunk, event_log, pipeline

IN_FLIGHT_ACCEPT_TIMEOUT_S = 1
STAGE_REPLY_TIMEOUT_S = 5


def append_owners(events: Queue) -> list[str]:
    return [event[1] for event in event_log(events) if event[0] == "append"]


@pytest.mark.asyncio(loop_scope="session")
async def test_interleaved_cadences_input_during_output_eos_and_disconnect(linear_pair):
    coordinator, _, _ = linear_pair
    a = await coordinator.open_session(
        OmniRequest(None, {"cadence": 3, "delay": 0.05}), stages=["source", "sink"]
    )
    b = await coordinator.open_session(
        OmniRequest(None, {"emit_every": 2}), stages=["source", "sink"]
    )
    output_a = coordinator.session_outputs(a)
    output_b = coordinator.session_outputs(b)
    await coordinator.append_session(a, chunk(0))
    first = await asyncio.wait_for(anext(output_a), 5)
    assert first.payload == {"count": 1, "index": 0}
    await coordinator.append_session(a, chunk(1, eos=True))
    await coordinator.append_session(b, chunk(0))
    receipt = await asyncio.wait_for(anext(output_b), 5)
    assert receipt.kind == "input_done"
    await coordinator.append_session(b, chunk(1, eos=True))
    other = await asyncio.wait_for(anext(output_b), 5)
    assert other.payload == {"count": 2, "index": 0} and other.eos
    rest = [await asyncio.wait_for(anext(output_a), 5) for _ in range(7)]
    assert [item.seq for item in [first, *rest]] == list(range(8))
    assert [item.input_seq for item in rest] == [0, 0, 0, 1, 1, 1, 1]
    assert rest[-1].eos
    with pytest.raises(ValueError, match="EOS"):
        await coordinator.append_session(a, chunk(2))
    await output_a.aclose()
    await output_b.aclose()
    assert not coordinator.sessions
    assert not coordinator._requests


@pytest.mark.asyncio
async def test_configured_singleton_list_route_with_three_stages(tmp_path):
    async with pipeline(tmp_path, stage_count=3, list_next=True) as (
        coordinator,
        events,
        processes,
    ):
        session_identity = await coordinator.open_session(
            OmniRequest(None), stages=["source", "middle", "sink"]
        )
        output = coordinator.session_outputs(session_identity)
        await coordinator.append_session(session_identity, chunk(0, eos=True))
        data = await asyncio.wait_for(anext(output), 5)
        assert data.kind == "data" and data.payload == {"count": 1, "index": 0}
        receipt = await asyncio.wait_for(anext(output), 5)
        assert receipt.kind == "input_done" and receipt.eos
        await output.aclose()


@pytest.mark.asyncio
async def test_replica_owner_survives_units_abort_and_scoped_shutdown(tmp_path):
    async with pipeline(tmp_path, replicated=True) as (coordinator, events, processes):
        first = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="first"
        )
        second = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"], session_id="second"
        )
        first_output = coordinator.session_outputs(first)
        second_output = coordinator.session_outputs(second)

        async def unit(session_identity, output, seq):
            await coordinator.append_session(session_identity, chunk(seq))
            data = await asyncio.wait_for(anext(output), 5)
            receipt = await asyncio.wait_for(anext(output), 5)
            assert data.kind == "data" and receipt.kind == "input_done"
            assert receipt.input_seq == seq

        await unit(first, first_output, 0)
        await unit(second, second_output, 0)
        await unit(first, first_output, 1)
        await unit(second, second_output, 1)
        await coordinator.shutdown_stages(["sink@r0"])
        assert processes[0].is_alive() and processes[2].is_alive()
        await unit(second, second_output, 2)
        await first_output.aclose()
        await second_output.aclose()
        log = []
        while not events.empty():
            log.append(events.get(timeout=1))
        first_owners = {
            event[1] for event in log if event[0] == "append" and event[2] == "first"
        }
        second_owners = {
            event[1] for event in log if event[0] == "append" and event[2] == "second"
        }
        assert first_owners == {"source", "sink@r0"}
        assert second_owners == {"source", "sink@r1"}


@pytest.mark.asyncio(loop_scope="session")
async def test_accepted_input_snapshots_mutable_payload(linear_pair, monkeypatch):
    coordinator, _, _ = linear_pair
    submitted = []
    original = coordinator.control_plane.submit_to_stage

    async def submit(stage, endpoint, message):
        session_operation = find_session_operation(message.data.request.metadata)
        if session_operation is not None and session_operation.operation == "append":
            submitted.append(session_operation.chunk.payload)
        return await original(stage, endpoint, message)

    monkeypatch.setattr(coordinator.control_plane, "submit_to_stage", submit)
    session_identity = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"]
    )
    outputs = coordinator.session_outputs(session_identity)
    payload = {"values": [1]}
    try:
        await coordinator.append_session(
            session_identity, TimedChunk("audio", 0, 20, 0, payload, eos=True)
        )
        payload["values"].append(2)

        async def read_until_done():
            async for output in outputs:
                if output.kind == "input_done":
                    break

        await asyncio.wait_for(read_until_done(), 5)
        assert submitted and all(value == {"values": [1]} for value in submitted)
    finally:
        await outputs.aclose()
        await coordinator.close_session(session_identity)


@pytest.mark.asyncio(loop_scope="session")
async def test_next_input_is_accepted_while_one_unit_is_in_flight(
    linear_pair, monkeypatch
) -> None:
    coordinator, _, _ = linear_pair
    session_identity = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"]
    )
    entered, release = asyncio.Event(), asyncio.Event()
    append_seqs: list[int] = []
    original = coordinator.session_operation

    async def hold_first_append(
        session: Session,
        operation: Literal["open", "append", "close"],
        *,
        owner: str | None = None,
        chunk: TimedChunk | None = None,
    ) -> None:
        if operation == "append" and chunk is not None:
            append_seqs.append(chunk.seq)
            if chunk.seq == 0:
                entered.set()
                await release.wait()
        return await original(session, operation, owner=owner, chunk=chunk)

    monkeypatch.setattr(coordinator, "session_operation", hold_first_append)
    outputs = coordinator.session_outputs(session_identity)
    try:
        assert await coordinator.append_session(session_identity, chunk(0)) == 0
        await asyncio.wait_for(entered.wait(), STAGE_REPLY_TIMEOUT_S)
        accepted_seq = await asyncio.wait_for(
            coordinator.append_session(session_identity, chunk(1)),
            IN_FLIGHT_ACCEPT_TIMEOUT_S,
        )
        assert accepted_seq == 1
        await asyncio.sleep(0)
        assert append_seqs == [0]
    finally:
        release.set()
        await outputs.aclose()


@pytest.mark.asyncio(loop_scope="session")
async def test_append_visits_owners_in_route_order(linear_triple) -> None:
    coordinator, events, _ = linear_triple
    session_identity = await coordinator.open_session(
        OmniRequest(None), stages=["source", "middle", "sink"]
    )
    outputs = coordinator.session_outputs(session_identity)
    await coordinator.append_session(session_identity, chunk(0, eos=True))
    output_chunk = await asyncio.wait_for(anext(outputs), STAGE_REPLY_TIMEOUT_S)
    assert output_chunk.kind == "data"
    await outputs.aclose()
    assert append_owners(events) == ["source", "middle", "sink"]


@pytest.mark.asyncio(loop_scope="session")
async def test_mismatched_route_fails_before_leaving_the_session_route(
    linear_triple,
) -> None:
    coordinator, events, _ = linear_triple
    session_identity = await coordinator.open_session(
        OmniRequest(None), stages=["source", "sink"]
    )
    outputs = coordinator.session_outputs(session_identity)
    await coordinator.append_session(session_identity, chunk(0, eos=True))
    with pytest.raises(RuntimeError, match="session route"):
        await asyncio.wait_for(anext(outputs), STAGE_REPLY_TIMEOUT_S)
    assert append_owners(events) == ["source"]


@pytest.mark.asyncio(loop_scope="session")
async def test_later_operations_do_not_carry_the_opening_inputs(
    linear_pair, monkeypatch
) -> None:
    coordinator, _, _ = linear_pair
    submitted: list[
        tuple[Literal["open", "append", "close"], object | None, object | None]
    ] = []
    original = coordinator.control_plane.submit_to_stage

    async def submit(stage, endpoint, message):
        session_operation = find_session_operation(message.data.request.metadata)
        if session_operation is not None and session_operation.operation != "open":
            submitted.append(
                (
                    session_operation.operation,
                    message.data.request.inputs,
                    message.data.data["raw_inputs"],
                )
            )
        return await original(stage, endpoint, message)

    monkeypatch.setattr(coordinator.control_plane, "submit_to_stage", submit)
    request = OmniRequest(inputs={"media": b"secret-audio"})
    session_identity = await coordinator.open_session(
        request, stages=["source", "sink"]
    )
    outputs = coordinator.session_outputs(session_identity)
    try:
        await coordinator.append_session(session_identity, chunk(0, eos=True))
        await asyncio.wait_for(anext(outputs), STAGE_REPLY_TIMEOUT_S)
    finally:
        await outputs.aclose()
    operation_names = {name for name, _, _ in submitted}
    assert operation_names >= {"append", "close"}
    assert all(
        request_inputs is None and raw_inputs is None
        for _, request_inputs, raw_inputs in submitted
    )
