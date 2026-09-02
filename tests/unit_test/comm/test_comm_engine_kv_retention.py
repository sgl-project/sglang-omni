# SPDX-License-Identifier: Apache-2.0
"""Invariants for the KV-transfer retention path added in #1315.

Ordinary transfers drop their pending entry on failure and mark their ops
failed. KV transfers opt out via ``retain_pending_on_failure``: a local failure
is not proof the peer stopped reading, so the pending entry (and with it the
source ops) is moved to ``_retained_pending_kv_transfers`` instead of being
torn down, and ``pending.ack`` is shielded so a cancel upstream does not kill
the future the peer may still resolve.

These tests pin that contract, and the control cases pin that the ordinary
path is unchanged by it. They deliberately drive ``_register_pending`` /
``_arm_pending`` directly rather than going through ``put_kv_pages``: the
retention decision lives entirely in ``_watch_pending`` and needs no GPU, so
the semantics can be tested on CPU.

One test (``test_retained_transfer_is_never_reclaimed``) documents current
behaviour rather than desired behaviour -- see its docstring.
"""

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.proto import DataAckMessage
from tests.unit_test.fixtures.pipeline_fakes import AckedOp, wait_until


def _make_engine(*, ack_timeout_s: float = 30.0) -> CommEngine:
    return CommEngine(
        CommRouter(
            stage_name="sender",
            gpu_id=None,
            same_process_targets=set(),
            gpu_stage_names=set(),
            comm_config={"ack_timeout_s": ack_timeout_s},
        )
    )


def _ack(object_id: str, *, success: bool = True) -> DataAckMessage:
    return DataAckMessage(
        request_id="req-kv",
        from_stage="receiver",
        to_stage="sender",
        object_id=object_id,
        success=success,
        error=None if success else "injected",
    )


@pytest.mark.comm_invariant
def test_local_cancel_retains_kv_transfer_without_failing_ops() -> None:
    """A cancelled watcher must not tell the ops the transfer failed: the peer
    may still be copying out of those slots."""

    async def _run() -> None:
        engine = _make_engine()
        op = AckedOp({"key": "kv-0"})
        engine._register_pending("kv-0", [op], retain_pending_on_failure=True)
        task = engine._arm_pending("kv-0")
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "kv-0" not in engine._pending
        assert len(engine._retained_pending_kv_transfers) == 1
        assert engine._retained_pending_kv_transfers[0].ops == [op]
        # The distinguishing property: ops are neither failed nor released.
        assert op.failed is None
        assert not op.acked
        assert op.release_calls == 0

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_ordinary_transfer_still_fails_fast_on_cancel() -> None:
    """Control: without the flag the pre-#1315 behaviour must be unchanged."""

    async def _run() -> None:
        engine = _make_engine()
        op = AckedOp({"key": "plain-0"})
        engine._register_pending("plain-0", [op])
        task = engine._arm_pending("plain-0")
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert "plain-0" not in engine._pending
        assert engine._retained_pending_kv_transfers == []
        assert op.release_calls == 0

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_receiver_terminal_ack_prevents_retention() -> None:
    """Once the peer has spoken, a later local failure has nothing to protect,
    so the transfer must not be retained."""

    async def _run() -> None:
        engine = _make_engine()
        op = AckedOp({"key": "kv-done"})
        engine._register_pending("kv-done", [op], retain_pending_on_failure=True)
        task = engine._arm_pending("kv-done")
        await asyncio.sleep(0)

        engine.ack_transfer(_ack("kv-done"))
        await wait_until(lambda: "kv-done" not in engine._pending)

        assert engine._retained_pending_kv_transfers == []
        assert op.acked
        assert op.release_calls == 1

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_shielded_ack_survives_watcher_cancel() -> None:
    """``asyncio.shield`` means a cancelled watcher leaves ``pending.ack``
    usable, so an ACK arriving afterwards still resolves rather than hitting a
    cancelled future."""

    async def _run() -> None:
        engine = _make_engine()
        op = AckedOp({"key": "kv-shield"})
        engine._register_pending("kv-shield", [op], retain_pending_on_failure=True)
        pending = engine._pending["kv-shield"]
        task = engine._arm_pending("kv-shield")
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Ordinary transfers reach here with a cancelled (done) future; the
        # shielded one is still pending and can be completed.
        assert not pending.ack.done()
        pending.ack.set_result(None)
        assert pending.ack.done() and pending.ack.exception() is None

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_late_ack_after_retention_is_ignored_without_error() -> None:
    """The retained entry is out of ``_pending``, so a late ACK for it takes
    the stale path and must not raise."""

    async def _run() -> None:
        engine = _make_engine()
        op = AckedOp({"key": "kv-late"})
        engine._register_pending("kv-late", [op], retain_pending_on_failure=True)
        task = engine._arm_pending("kv-late")
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        engine.ack_transfer(_ack("kv-late"))
        engine.ack_transfer(_ack("kv-late", success=False))

        assert op.release_calls == 0
        assert len(engine._retained_pending_kv_transfers) == 1

    asyncio.run(_run())


@pytest.mark.comm_invariant
def test_timeout_retains_kv_transfer() -> None:
    """The timeout path (as opposed to an external cancel) must retain too --
    ``wait_for`` raises TimeoutError, which the generic ``except Exception``
    branch routes to retention when the flag is set."""

    async def _run() -> None:
        engine = _make_engine(ack_timeout_s=0.05)
        op = AckedOp({"key": "kv-timeout"})
        engine._register_pending("kv-timeout", [op], retain_pending_on_failure=True)
        task = engine._arm_pending("kv-timeout")

        with pytest.raises(Exception):
            await task

        assert "kv-timeout" not in engine._pending
        assert len(engine._retained_pending_kv_transfers) == 1
        assert op.failed is None
        assert op.release_calls == 0

    asyncio.run(_run())


@pytest.mark.comm_characterization
def test_retained_transfer_is_never_reclaimed() -> None:
    """CURRENT BEHAVIOUR, not an endorsement.

    ``_retained_pending_kv_transfers`` is append-only: nothing reads it, and
    ``close()`` -- which drains ``_pending``, ``_inbound_kv``, ``_kv_ready``
    and the outbound/aborted KV request maps -- leaves it untouched. Each
    retained entry therefore keeps its ops (and, on the CUDA-IPC path, the
    slots and source tensors they hold) alive for the process lifetime, and
    the list itself is unbounded.

    Retention is deliberate: releasing while the peer may still be reading
    would be worse. But retention with no terminating condition is a leak, so
    this test exists to fail loudly once a reclaim path (timeout, peer
    liveness check, bounded cap, or explicit drain in ``close()``) is added --
    at which point it should be rewritten to assert the new contract.
    """

    async def _run() -> None:
        engine = _make_engine()
        ops = []
        for index in range(3):
            op = AckedOp({"key": f"kv-leak-{index}"})
            ops.append(op)
            engine._register_pending(
                f"kv-leak-{index}", [op], retain_pending_on_failure=True
            )
            task = engine._arm_pending(f"kv-leak-{index}")
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert len(engine._retained_pending_kv_transfers) == 3

        await engine.close()

        assert (
            len(engine._retained_pending_kv_transfers) == 3
        ), "close() does not drain retained KV transfers"
        assert all(op.release_calls == 0 for op in ops)

    asyncio.run(_run())
