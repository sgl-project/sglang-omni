# SPDX-License-Identifier: Apache-2.0
"""Tests for the fatal-forward fail-all-active plumbing.

Locks RFC v2's [Fatal forward failures require coordinator fail-all
plumbing] section: when a TP child exits non-zero,
``MultiProcessPipelineRunner._monitor_children`` must fail every
active Coordinator future / stream queue with a non-empty error
*before* tearing down the rest, otherwise outstanding HTTP requests
hang forever after the TP group is killed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import (
    CompleteMessage,
    RequestInfo,
    RequestState,
    StreamMessage,
)


@pytest.fixture
def coordinator():
    """Build a minimal Coordinator without starting its ZMQ control plane."""
    coord = Coordinator(
        completion_endpoint="ipc:///tmp/test_completion.sock",
        abort_endpoint="ipc:///tmp/test_abort.sock",
        entry_stage="x",
    )
    return coord


@pytest.mark.asyncio
async def test_fail_all_active_fails_completion_futures(coordinator):
    loop = asyncio.get_running_loop()
    futures = {}
    for rid in ("r0", "r1", "r2"):
        f = loop.create_future()
        coordinator._completion_futures[rid] = f
        coordinator._requests[rid] = RequestInfo(
            request_id=rid, state=RequestState.RUNNING, current_stage="x"
        )
        futures[rid] = f

    coordinator.fail_all_active("stage group 'thinker' died: rank0 exit=1")

    for rid, f in futures.items():
        assert f.done(), f"future for {rid} not resolved"
        with pytest.raises(RuntimeError, match="stage group 'thinker' died"):
            f.result()


@pytest.mark.asyncio
async def test_fail_all_active_pushes_complete_to_stream_queues(coordinator):
    loop = asyncio.get_running_loop()
    queues = {}
    for rid in ("s0", "s1"):
        q: asyncio.Queue = asyncio.Queue()
        coordinator._stream_queues[rid] = q
        coordinator._requests[rid] = RequestInfo(
            request_id=rid, state=RequestState.RUNNING, current_stage="x"
        )
        queues[rid] = q

    coordinator.fail_all_active("stage group died")

    for rid, q in queues.items():
        msg = q.get_nowait()
        assert isinstance(msg, CompleteMessage)
        assert msg.request_id == rid
        assert msg.success is False
        assert msg.error  # non-empty


@pytest.mark.asyncio
async def test_fail_all_active_uses_default_error_when_empty(coordinator):
    loop = asyncio.get_running_loop()
    f = loop.create_future()
    coordinator._completion_futures["r0"] = f
    coordinator._requests["r0"] = RequestInfo(
        request_id="r0", state=RequestState.RUNNING, current_stage="x"
    )

    # Empty error string must fall back to a non-empty default.
    coordinator.fail_all_active("")

    with pytest.raises(RuntimeError) as exc_info:
        f.result()
    assert str(exc_info.value)  # not empty


@pytest.mark.asyncio
async def test_fail_all_active_marks_request_state_failed(coordinator):
    loop = asyncio.get_running_loop()
    f = loop.create_future()
    coordinator._completion_futures["r0"] = f
    coordinator._requests["r0"] = RequestInfo(
        request_id="r0", state=RequestState.RUNNING, current_stage="x"
    )
    # Pre-completed request should NOT be re-marked.
    coordinator._requests["r1"] = RequestInfo(
        request_id="r1", state=RequestState.COMPLETED, current_stage="x"
    )

    coordinator.fail_all_active("die")

    assert coordinator._requests["r0"].state == RequestState.FAILED
    assert coordinator._requests["r1"].state == RequestState.COMPLETED


@pytest.mark.asyncio
async def test_fail_all_active_idempotent_for_already_done_future(coordinator):
    """Calling fail_all_active twice must not raise."""
    loop = asyncio.get_running_loop()
    f = loop.create_future()
    coordinator._completion_futures["r0"] = f
    coordinator._requests["r0"] = RequestInfo(
        request_id="r0", state=RequestState.RUNNING, current_stage="x"
    )
    coordinator.fail_all_active("first")
    # Second call: future has been popped; the call should still be safe.
    coordinator.fail_all_active("second")  # must not raise


@pytest.mark.asyncio
async def test_mp_runner_monitor_calls_fail_all_active_on_dead_child():
    """Locks the wiring in MultiProcessPipelineRunner._monitor_children.

    We swap out _coordinator with a stub that records fail_all_active
    calls, fake one StageGroup to be `any_dead`, and let the monitor
    loop fire once.
    """
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    runner = MultiProcessPipelineRunner.__new__(MultiProcessPipelineRunner)
    runner._started = True

    fake_coord = MagicMock()
    fake_coord.shutdown_stages = AsyncMock()
    runner._coordinator = fake_coord

    dead_group = SimpleNamespace(
        stage_name="image_encoder",
        any_dead=lambda: True,
        dead_summary=lambda: "rank0 exit=1",
        tp_size=2,
    )
    runner._groups = [dead_group]
    runner._monitor_task = None
    runner._completion_task = None

    # stop() awaits coordinator.stop() and shutdowns groups; stub them.
    async def _noop_stop():
        runner._started = False

    runner.stop = _noop_stop  # type: ignore[assignment]

    await runner._monitor_children()

    fake_coord.fail_all_active.assert_called_once()
    msg = fake_coord.fail_all_active.call_args.args[0]
    assert "image_encoder" in msg
    assert "rank0 exit=1" in msg
