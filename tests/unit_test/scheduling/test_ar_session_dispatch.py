# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.proto.session import TimedChunk
from sglang_omni.scheduling.omni_scheduler import _Upstream
from tests.unit_test.fixtures import ar_session
from tests.unit_test.fixtures.ar_session import TestAdapter, data, payload

bridge_env = ar_session.bridge_env


def test_lifecycle_bypasses_builder_and_queue_capacity(bridge_env):
    h = bridge_env
    h.scheduler._request_builder = lambda _: pytest.fail(
        "lifecycle reached model builder"
    )
    h.scheduler._waiting_queue_is_full = lambda: True
    h.scheduler.process_input_requests([payload("open")])
    assert h.scheduler.outbox.get_nowait().data.data == {"opened": True}
    h.scheduler.process_input_requests([payload("close", "close")])
    assert h.scheduler.outbox.get_nowait().data.data == {"closed": True}
    assert not h.bridge.owners


def test_output_budget_includes_flush_and_terminal_only(bridge_env, monkeypatch):
    import sglang_omni.scheduling.sglang_backend.ar_session as module

    b = bridge_env.bridge
    chunk = TimedChunk("text", 0, 0, 0, "hello")
    b.adapter = TestAdapter(
        stream=lambda *args: [chunk], flush=lambda *args: [chunk, chunk]
    )
    b.command(payload("open"))
    b.accept(payload("append"))
    monkeypatch.setattr(module, "get_active_stage", lambda: "encoder")
    assert list(b.messages("r", data(), object())) == []
    monkeypatch.setattr(module, "get_active_stage", lambda: "ar")
    assert len(list(b.messages("r", data(), object()))) == 1
    with pytest.raises(QueueFullError):
        list(b.messages("r", data(), flush=True))


def test_ordinary_embeddings_and_sidecars_unchanged_with_bridge(bridge_env):
    h = bridge_env
    p = payload("append")
    p.request.metadata.clear()
    d = data()
    original_req = d.req
    marker = object()
    d.prefill_input_embeds = marker
    d.decode_input_embeds = [marker]
    d.model_inputs = {"marker": marker}
    d.capture_model_output_keys = ("hidden",)
    h.scheduler._request_builder = lambda _: d
    h.scheduler._request_kv_capacity_error = lambda req: None
    h.scheduler.process_input_requests([p])
    assert h.scheduler.waiting_queue == [original_req]
    assert d.prefill_input_embeds is marker
    assert d.decode_input_embeds == [marker]
    assert d.model_inputs == {"marker": marker}
    assert d.capture_model_output_keys == ("hidden",)


def test_stream_conversion_without_ordinary_builder(bridge_env, monkeypatch):
    import sglang_omni.scheduling.sglang_backend.ar_session as module

    h = bridge_env
    h.bridge.command(payload("open"))
    h.bridge.accept(payload("append"))
    chunk = TimedChunk("text", 0, 0, 0, "converted")
    monkeypatch.setattr(module, "get_active_stage", lambda: "ar")
    h.bridge.adapter = TestAdapter(stream=lambda *args: [chunk])
    sched = SimpleNamespace(requests=[SimpleNamespace(request_id="r", data=data())])
    output = SimpleNamespace(outputs={"r": object()})
    h.scheduler._emit_stream_output(sched, output)
    assert h.scheduler.outbox.get_nowait().data == asdict(chunk)
    h.scheduler._aborted_request_ids.add("r")
    h.scheduler._emit_stream_output(sched, output)
    assert h.scheduler.outbox.empty()


def test_retained_sessions_block_direct_cache_flush_and_weight_reset(
    bridge_env, monkeypatch
):
    h = bridge_env
    h.bridge.command(payload("open"))
    monkeypatch.setattr(_Upstream, "flush_cache", lambda *args, **kwargs: True)
    assert h.scheduler.flush_cache() is False
    result = h.scheduler._run_weight_update_with_lifecycle(
        {}, lambda _: pytest.fail("weight reset ran"), {}
    )
    assert result["success"] is False


@pytest.mark.parametrize(
    "content, relay, bypass",
    [(b"", True, True), (b"", False, False), ("text", True, False)],
    ids=["relayed", "unhandled", "nonempty"],
)
def test_empty_eof_is_relayed_or_bypassed(bridge_env, content, relay, bypass):
    h = bridge_env
    b = h.bridge
    b.command(payload("open"))
    native = h.native()
    marker = object()
    h.scheduler.tree_cache.slots["s"] = marker
    if relay:
        b.adapter.finish_input = lambda ref, value: value
    p = payload("append")
    p.request.metadata["omni_session"]["chunk"] = asdict(
        TimedChunk(
            "audio" if isinstance(content, bytes) else "text",
            0,
            0,
            0,
            content,
            eos=True,
        )
    )
    queued = []

    def stage(values):
        queued.extend(values)
        return [], []

    h.scheduler._stage_request_build_payloads = stage

    def unexpected_error(rid, exc):
        raise AssertionError(str(exc)) from exc

    h.scheduler._emit_request_error = unexpected_error
    h.scheduler.process_input_requests([p])
    assert queued == ([] if bypass else [p])
    assert h.native() is native
    assert h.scheduler.tree_cache.slots["s"] is marker
    assert not h.inflight()
    if bypass:
        assert h.scheduler.outbox.get_nowait().data is p
        assert not b.requests
        assert b.owners["s"].active is None
    else:
        assert p.request_id in b.requests
        assert b.owners["s"].active == p.request_id
        assert h.scheduler.outbox.empty()
