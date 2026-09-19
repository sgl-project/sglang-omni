# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import TimedChunk, find_session_command
from tests.unit_test.fixtures.session_pipeline import chunk, pipeline


@pytest.mark.asyncio
async def test_interleaved_cadences_input_during_output_eos_and_disconnect(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, processes):
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
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "middle", "sink"]
        )
        output = coordinator.session_outputs(ref)
        await coordinator.append_session(ref, chunk(0, eos=True))
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

        async def unit(ref, output, seq):
            await coordinator.append_session(ref, chunk(seq))
            data = await asyncio.wait_for(anext(output), 5)
            receipt = await asyncio.wait_for(anext(output), 5)
            assert data.kind == "data" and receipt.kind == "input_done"
            assert receipt.input_seq == seq

        await unit(first, first_output, 0)
        await unit(second, second_output, 0)
        first = await coordinator.abort_session(first)
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
        assert ("abort", "sink@r0", "first") in log
        assert ("abort", "sink@r1", "first") not in log


@pytest.mark.asyncio
async def test_accepted_input_snapshots_mutable_payload(tmp_path, monkeypatch):
    async with pipeline(tmp_path) as (coordinator, _, _):
        submitted = []
        original = coordinator.control_plane.submit_to_stage

        async def submit(stage, endpoint, message):
            command = find_session_command(message.data.request.metadata)
            if command is not None and command.op == "append":
                submitted.append(command.chunk.payload)
            return await original(stage, endpoint, message)

        monkeypatch.setattr(coordinator.control_plane, "submit_to_stage", submit)
        ref = await coordinator.open_session(
            OmniRequest(None), stages=["source", "sink"]
        )
        outputs = coordinator.session_outputs(ref)
        payload = {"values": [1]}
        try:
            await coordinator.append_session(
                ref, TimedChunk("audio", 0, 20, 0, payload, eos=True)
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
            await coordinator.close_session(ref)
