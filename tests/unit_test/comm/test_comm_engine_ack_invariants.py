# SPDX-License-Identifier: Apache-2.0
"""ACK state-machine invariants for CommEngine.

These tests pin down the engine's pending-transfer lifecycle against
adversarial ACK orderings: out-of-order, late-after-timeout, NACK isolation,
duplicates, and ACKs racing the failure-drain path. Slot-release semantics are
modeled by AckedOp: the release callback fires only on the successful
completion path, mirroring CudaIpcPutOperation._release_cb.

All assertions are event-driven (wait_until / future state); no wall-clock
duration assertions. Timeout cases use a tiny ack_timeout_s and only rely on
"the timeout has fired", never on how long it took.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.proto import DataAckMessage
from sglang_omni.relay.cuda_ipc import _ContiguousSlotAllocator
from tests.unit_test.fixtures.pipeline_fakes import (
    AckedOp,
    AckedRelay,
    RecordingStageControlPlane,
    make_stage_payload,
    wait_until,
)
from tests.unit_test.fixtures.trace_capture import capture_comm_trace


def _make_engine(
    relay: AckedRelay, *, ack_timeout_s: float = 30.0
) -> tuple[CommEngine, RecordingStageControlPlane]:
    control_plane = RecordingStageControlPlane()
    engine = CommEngine(
        CommRouter(
            stage_name="sender",
            gpu_id=None,
            same_process_targets=set(),
            gpu_stage_names=set(),
            comm_config={"ack_timeout_s": ack_timeout_s},
            injected_relay=relay,
        )
    )
    return engine, control_plane


async def _send(
    engine: CommEngine,
    relay: AckedRelay,
    control_plane: RecordingStageControlPlane,
    request_id: str,
) -> str:
    data_ref = await engine.send_payload(
        relay=relay,
        control_plane=control_plane,
        request_id=request_id,
        payload=make_stage_payload(request_id=request_id, data={"x": torch.ones(2)}),
        transport=TransportKind.SHM,
        from_stage="sender",
        to_stage="receiver",
        target_endpoint="inproc://receiver",
    )
    return data_ref.object_id


def _ack(
    object_id: str, *, success: bool = True, error: str | None = None
) -> DataAckMessage:
    return DataAckMessage(
        request_id="req-any",
        from_stage="receiver",
        to_stage="sender",
        object_id=object_id,
        success=success,
        error=error,
    )


def _capture_unhandled(unhandled: list) -> None:
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, ctx: unhandled.append(ctx)
    )


@pytest.mark.comm_invariant
def test_out_of_order_acks_release_matching_transfers() -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        relay = AckedRelay()
        engine, control_plane = _make_engine(relay)

        object_ids = [
            await _send(engine, relay, control_plane, f"req-{i}") for i in range(3)
        ]
        assert len(relay.ops) == 3
        assert set(object_ids) == set(engine._pending)

        for ack_index in (2, 0, 1):
            engine.ack_transfer(_ack(object_ids[ack_index]))
            await wait_until(lambda: relay.ops[ack_index].waited)
            assert relay.ops[ack_index].acked
            assert relay.ops[ack_index].release_calls == 1
            for other, op in enumerate(relay.ops):
                if other != ack_index and not op.waited:
                    assert not op.acked
                    assert op.release_calls == 0

        await wait_until(lambda: not engine._pending)
        for op in relay.ops:
            assert op.waited and op.acked and op.failed is None
            assert op.release_calls == 1
        assert not unhandled

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_late_ack_after_timeout_hits_stale_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        relay = AckedRelay()
        engine, control_plane = _make_engine(relay, ack_timeout_s=0.05)

        object_id = await _send(engine, relay, control_plane, "req-timeout")
        await wait_until(lambda: object_id not in engine._pending)

        op = relay.ops[0]
        assert isinstance(op.failed, TimeoutError)
        assert op.waited
        assert not op.acked
        assert op.release_calls == 0

        with caplog.at_level(logging.DEBUG, logger="sglang_omni.comm.engine"):
            engine.ack_transfer(_ack(object_id))
        assert "Ignoring stale data_ack" in caplog.text
        assert not op.acked
        assert op.release_calls == 0
        assert not unhandled

    asyncio.run(_run())


@pytest.mark.comm_characterization
def test_timeout_marks_ops_failed_without_slot_release() -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        relay = AckedRelay()
        engine, _ = _make_engine(relay, ack_timeout_s=0.05)

        # A stream-chunk transfer can register several ops under one pending
        # entry; drive that shape directly to pin the for-each drain semantics.
        ops = [AckedOp({"key": f"op-{i}"}) for i in range(2)]
        engine._register_pending("obj-multi", list(ops))
        engine._arm_pending("obj-multi")

        await wait_until(lambda: "obj-multi" not in engine._pending)
        for op in ops:
            assert isinstance(op.failed, TimeoutError)
            assert op.waited
            assert op.release_calls == 0
        # Bookkeeping is cleared but no slot was released: the failed transfer
        # intentionally parks its resources for the relay-failure path.
        assert not unhandled

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_nack_fails_only_target_transfer() -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        relay = AckedRelay()
        engine, control_plane = _make_engine(relay)

        nacked = await _send(engine, relay, control_plane, "req-nack")
        healthy = await _send(engine, relay, control_plane, "req-healthy")

        engine.ack_transfer(_ack(nacked, success=False, error="receiver OOM"))
        await wait_until(lambda: nacked not in engine._pending)

        assert isinstance(relay.ops[0].failed, RuntimeError)
        assert "receiver OOM" in str(relay.ops[0].failed)
        assert relay.ops[0].release_calls == 0

        assert healthy in engine._pending
        assert relay.ops[1].failed is None
        assert not relay.ops[1].acked

        engine.ack_transfer(_ack(healthy))
        await wait_until(lambda: relay.ops[1].waited)
        assert relay.ops[1].acked
        assert relay.ops[1].release_calls == 1
        assert not unhandled

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_duplicate_and_mixed_ack_nack_are_idempotent() -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        relay = AckedRelay()
        engine, control_plane = _make_engine(relay)

        # NACK after a completed ACK hits the stale path and cannot flip state.
        done = await _send(engine, relay, control_plane, "req-done")
        engine.ack_transfer(_ack(done))
        await wait_until(lambda: done not in engine._pending)
        engine.ack_transfer(_ack(done, success=False, error="too late"))
        assert relay.ops[0].failed is None
        assert relay.ops[0].release_calls == 1

        # NACK followed by ACK in the same tick: first writer wins, the ACK is
        # absorbed by the pending.ack.done() guard.
        raced = await _send(engine, relay, control_plane, "req-race")
        engine.ack_transfer(_ack(raced, success=False, error="lost race"))
        engine.ack_transfer(_ack(raced))
        await wait_until(lambda: raced not in engine._pending)
        assert isinstance(relay.ops[1].failed, RuntimeError)
        assert not relay.ops[1].acked
        assert relay.ops[1].release_calls == 0

        # A failed ack without an error is a contract violation.
        pending = await _send(engine, relay, control_plane, "req-invalid")
        with pytest.raises(ValueError, match="missing error"):
            engine.ack_transfer(_ack(pending, success=False, error=None))
        engine.ack_transfer(_ack(pending))
        await wait_until(lambda: pending not in engine._pending)
        assert relay.ops[2].release_calls == 1
        assert not unhandled

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_ack_release_unblocks_pending_producer() -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        allocator = _ContiguousSlotAllocator(slot_count=4, slot_size=64)
        full = await allocator.acquire_async(4)

        relay = AckedRelay(on_release=lambda op: allocator.release(full.offset, 4))
        engine, control_plane = _make_engine(relay)
        object_id = await _send(engine, relay, control_plane, "req-pool")

        waiter = asyncio.create_task(allocator.acquire_async(2))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not waiter.done()

        engine.ack_transfer(_ack(object_id))
        allocation = await asyncio.wait_for(waiter, timeout=1.0)
        assert allocation.wait_rounds >= 1
        assert relay.ops[0].release_calls == 1
        allocator.release(allocation.offset, 2)
        assert not unhandled

    asyncio.run(_run())


# Offline tools read these events by name and by field. Adding an event is safe.
# Renaming one, or dropping a field, breaks every tool that reads the trace.
_REQUIRED_FIELDS = {
    "comm_send_enqueue": {"request_id", "from_stage", "to_stage", "transport", "kind"},
    "comm_payload_send": {
        "request_id",
        "from_stage",
        "to_stage",
        "transport",
        "write_ms",
        "control_send_ms",
        "elapsed_ms",
    },
}


@pytest.mark.comm_invariant
def test_trace_events_carry_their_documented_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> str:
        allocator = _ContiguousSlotAllocator(slot_count=4, slot_size=64)
        full = await allocator.acquire_async(4)
        relay = AckedRelay(on_release=lambda op: allocator.release(full.offset, 4))
        engine, control_plane = _make_engine(relay)

        object_id = await _send(engine, relay, control_plane, "req-trace")
        engine.ack_transfer(_ack(object_id))
        await wait_until(lambda: object_id not in engine._pending)
        return object_id

    with capture_comm_trace(monkeypatch) as events:
        asyncio.run(_run())

    assert events, "a completed transfer emitted no trace events"
    for event in events:
        assert "event" in event and "ts_ns" in event
        required = _REQUIRED_FIELDS.get(event["event"])
        if required:
            missing = required - set(event)
            assert not missing, f"{event['event']} dropped {sorted(missing)}"

    # A send enqueues before it writes. Assert the order as a subsequence, so
    # that adding events later does not fail this test.
    names = [event["event"] for event in events]
    assert names.index("comm_send_enqueue") < names.index("comm_payload_send")


@pytest.mark.comm_invariant
def test_a_completed_transfer_emits_nothing_when_the_gate_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        allocator = _ContiguousSlotAllocator(slot_count=4, slot_size=64)
        full = await allocator.acquire_async(4)
        relay = AckedRelay(on_release=lambda op: allocator.release(full.offset, 4))
        engine, control_plane = _make_engine(relay)
        object_id = await _send(engine, relay, control_plane, "req-quiet")
        engine.ack_transfer(_ack(object_id))
        await wait_until(lambda: object_id not in engine._pending)

    with capture_comm_trace(monkeypatch, enable=False) as events:
        asyncio.run(_run())

    assert events == []


@pytest.mark.comm_invariant
def test_ack_during_failure_drain_is_ignored() -> None:
    async def _run() -> None:
        unhandled: list = []
        _capture_unhandled(unhandled)
        gate = asyncio.Event()
        relay = AckedRelay(completion_gate=gate)
        engine, control_plane = _make_engine(relay, ack_timeout_s=0.05)

        object_id = await _send(engine, relay, control_plane, "req-drain")
        op = relay.ops[0]

        # Timeout fires: mark_receiver_failed runs, then the drain blocks on
        # the gated wait_for_completion with the pending entry still
        # registered and its future cancelled by wait_for.
        await wait_until(lambda: op.failed is not None)
        assert object_id in engine._pending
        pending = engine._pending[object_id]
        assert pending.ack.done()

        # ACK and NACK arriving mid-drain must be absorbed by the done() guard
        # without raising InvalidStateError.
        engine.ack_transfer(_ack(object_id))
        engine.ack_transfer(_ack(object_id, success=False, error="mid drain"))
        assert not op.acked

        gate.set()
        await wait_until(lambda: object_id not in engine._pending)
        assert op.waited
        assert not op.acked
        assert op.release_calls == 0
        assert not unhandled

    asyncio.run(_run())
