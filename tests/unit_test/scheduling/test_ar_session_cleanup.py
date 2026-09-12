# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest

from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures import ar_session
from tests.unit_test.fixtures.ar_session import TestAdapter, payload

bridge_env = ar_session.bridge_env


def test_queued_cancel_detaches_request_without_releasing_session_kv(bridge_env):
    h = bridge_env
    h.bridge.command(payload("open"))
    d = h.append()
    slot = h.retain_slot(d, row=1)
    h.enqueue(d, queued=True)

    def abort(rid):
        assert not d.req.kv.holds_kv
        assert d.req.kv is not slot.kv

    h.scheduler.abort = abort
    h.bridge.cancel("r")
    assert slot.kv.holds_kv
    assert slot.kv.kv_allocated_len == 2
    assert not h.bridge.requests


def test_cancel_does_not_abort_active_core_request(bridge_env):
    h = bridge_env
    h.bridge.command(payload("open"))
    h.append()
    with pytest.raises(RuntimeError, match="unit completion"):
        h.bridge.command(payload("abort", epoch=1))
    assert h.inflight()
    assert h.bridge.owners["s"].ref.epoch == 0


def test_cancel_at_boundary_preserves_native_session(bridge_env):
    h = bridge_env
    h.bridge.command(payload("open"))
    native = h.native()
    h.bridge.command(payload("abort", epoch=1))
    assert h.native() is native
    assert h.bridge.owners["s"].ref.epoch == 1
    assert h.events == []


def test_idle_close_drains_completed_lookahead_before_slot_release(bridge_env):
    h = bridge_env
    h.bridge.command(payload("open"))
    h.trace_drain()
    h.bridge.command(payload("close"))
    assert h.events == ["drain", "slot released"]


def test_adapter_state_follows_core_session_lifetime(bridge_env):
    h = bridge_env
    b = h.bridge
    b.adapter = TestAdapter(
        open=lambda ref, request: h.events.append(("open", ref.session_id)),
        close=lambda ref: h.events.append(("close", ref.session_id)),
    )
    b.command(payload("open"))
    native = h.native()
    assert native is not None and native.timeout is None
    assert h.events == [("open", "s")]
    h.events.clear()
    b.command(payload("abort", epoch=1))
    assert h.native() is native
    assert b.owners["s"].ref.epoch == 1
    assert h.events == []
    b.command(payload("close", epoch=1))
    b.command(payload("close", epoch=1))
    assert h.events == ["slot released", ("close", "s")]


def test_request_abort_fences_output_before_async_drain(bridge_env):
    h = bridge_env
    h.bridge.command(payload("open"))
    d = h.append()
    h.enqueue(d)

    def abort(rid, **kw):
        h.scheduler._aborted_request_ids.add(rid)
        d.req.to_finish = object()
        h.events.append(kw)

    h.scheduler.abort = abort

    def drain():
        assert "r" in h.scheduler._aborted_request_ids
        assert d.req.to_finish is not None
        h.events.append("drain")

    h.trace_drain(drain)
    h.bridge.cancel("r")
    assert h.events == [{}, "drain", "release"]
    assert not h.bridge.requests


def test_cancelled_close_retains_cleanup_intent(bridge_env):
    h = bridge_env
    h.scheduler.process_input_requests([payload("open")])
    h.scheduler.outbox.get_nowait()
    s = h.scheduler
    s._aborted_request_ids.add("close")
    s.inbox.put(IncomingMessage("close", "new_request", payload("close", "close")))
    s._drain_inbox_for_request("close")
    s.process_input_requests(s.recv_requests())
    assert not h.bridge.owners
    assert h.scheduler.outbox.empty()


@dataclass(frozen=True)
class WaitCase:
    failing_step: str | None
    entry: str
    collect_error: bool = False
    launch_first: bool = False
    shutdown: bool = False


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(WaitCase("current", "close"), id="gpu_wait"),
        pytest.param(WaitCase("current", "resolve"), id="regular_async"),
        pytest.param(
            WaitCase("current", "shutdown", shutdown=True), id="regular_async_shutdown"
        ),
        pytest.param(
            WaitCase("previous", "launch", launch_first=True),
            id="launch_first_previous",
        ),
        pytest.param(
            WaitCase("current", "launch", launch_first=True), id="launch_first_current"
        ),
        pytest.param(
            WaitCase(None, "resolve", collect_error=True), id="post_wait_collect"
        ),
    ],
)
def test_failed_device_wait_retains_ownership_and_retries(bridge_env, case):
    h = bridge_env
    h.async_steps(case)
    if case.collect_error:
        h.run_pending()
        assert h.events == ["current waited", "current", "cleanup"]
        assert h.pending == (None, None)
    else:
        with pytest.raises(RuntimeError, match="device wait failed"):
            h.run_pending()
        assert "release" not in h.events and "process" not in h.events
        assert "previous" not in h.events and "current" not in h.events
        pending, previous = h.pending
        assert pending[2] is h.current[2]
        if case.launch_first:
            assert previous is h.previous
        else:
            assert pending is h.current
        assert h.bridge.requests["r"].active == "r"
        assert h.bridge.requests["r"].native_owned and h.inflight()
        assert len(h.scheduler.session_controller.sessions) == 1
        if case.shutdown:
            assert h.events.count("resources") == 1
        if case.launch_first:
            h.retry_pending()
        else:
            h.blocked = False
            assert h.bridge.command(payload("close", "retry")).data == {"closed": True}
        if case.launch_first:
            assert h.events.index("previous") < h.events.index("current")
            assert h.events.count("previous") == 1
        assert h.events.count("current") == 1
        assert h.pending == (None, None)
    assert h.bridge.command(payload("close", "retry")).data == {"closed": True}
    assert not h.bridge.owners and not h.bridge.requests
    assert not h.scheduler.session_controller.sessions
    assert h.events.count("release") == 1
    assert h.events.count("slot released") == 1
    before = list(h.events)
    h.retry_pending()
    h.bridge.command(payload("close", "repeat"))
    assert h.events == before


def test_shutdown_callback_runs_even_if_native_cleanup_fails(bridge_env):
    h = bridge_env

    def fail():
        raise RuntimeError("native failure")

    h.bridge.shutdown = fail
    with pytest.raises(RuntimeError, match="native failure"):
        h.scheduler.start()
    assert h.events == ["resources"]
