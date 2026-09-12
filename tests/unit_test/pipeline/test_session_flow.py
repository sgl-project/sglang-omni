# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import TimedChunk
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
        assert first.payload == [1, 0]
        await coordinator.append_session(a, chunk(1, eos=True))
        await coordinator.append_session(b, chunk(0))
        receipt = await asyncio.wait_for(anext(output_b), 5)
        assert receipt.kind == "input_done"
        await coordinator.append_session(b, chunk(1, eos=True))
        other = await asyncio.wait_for(anext(output_b), 5)
        assert other.payload == [2, 0] and other.eos
        rest = [await asyncio.wait_for(anext(output_a), 5) for _ in range(7)]
        assert [item.seq for item in [first, *rest]] == list(range(8))
        assert [item.input_seq for item in rest] == [0, 0, 0, 1, 1, 1, 1]
        assert rest[-1].eos
        with pytest.raises(ValueError, match="EOS"):
            await coordinator.append_session(a, chunk(2))
        await output_a.aclose()
        await output_b.aclose()
        assert not coordinator._sessions
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
        assert data.kind == "data" and data.payload == [1, 0]
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
            command = message.data.request.metadata.get("omni_session", {})
            if command.get("op") == "append":
                submitted.append(command["chunk"]["payload"])
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
            async with asyncio.timeout(5):
                async for output in outputs:
                    if output.kind == "input_done":
                        break
            assert submitted and all(value == {"values": [1]} for value in submitted)
        finally:
            await outputs.aclose()
            await coordinator.close_session(ref)


@pytest.mark.asyncio
async def test_open_inputs_are_not_relayed_by_later_commands(tmp_path, monkeypatch):
    from dataclasses import asdict

    import msgpack

    from sglang_omni.admission import QueueFullError
    from sglang_omni.proto.session import SESSION_METADATA_KEY, SessionLimits

    async with pipeline(tmp_path) as (coordinator, events, processes):
        submitted = []
        submit = coordinator._submit_request

        async def record(request_id, request, **kwargs):
            submitted.append(
                (request.metadata[SESSION_METADATA_KEY]["op"], request.inputs)
            )
            return await submit(request_id, request, **kwargs)

        monkeypatch.setattr(coordinator, "_submit_request", record)
        request = OmniRequest(b"initial audio")
        unit = TimedChunk("audio", 0, 80, 0, b"unit")
        limit = len(msgpack.packb(asdict(unit), use_bin_type=True))
        ref = await coordinator.open_session(
            request,
            stages=["source", "sink"],
            limits=SessionLimits(max_chunk_bytes=limit),
        )
        output = coordinator.session_outputs(ref)
        with pytest.raises(QueueFullError):
            await coordinator.append_session(
                ref, TimedChunk("audio", 0, 80, 0, b"units")
            )
        await coordinator.append_session(ref, unit)
        assert (await asyncio.wait_for(anext(output), 5)).kind == "data"
        assert (await asyncio.wait_for(anext(output), 5)).kind == "input_done"
        ref = await coordinator.abort_session(ref)
        await output.aclose()
        assert [value for op, value in submitted if op == "open"] == [
            request.inputs
        ] * 2
        later = [(op, value) for op, value in submitted if op != "open"]
        assert {op for op, _ in later} == {"append", "abort", "close"}
        assert all(value is None for _, value in later)
        assert request.inputs == b"initial audio"
