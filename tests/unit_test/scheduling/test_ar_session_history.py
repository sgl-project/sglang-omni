# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from sglang.srt.runtime_context import get_context

from tests.unit_test.fixtures import ar_session
from tests.unit_test.fixtures.ar_session import data, payload

bridge_env = ar_session.bridge_env


def test_materialization_preserves_sidecars_and_drains_previous_lookahead(bridge_env):
    h = bridge_env
    b = h.bridge
    b.command(payload("open"))
    p = payload("append")
    b.accept(p)
    d = data()
    sidecar = object()
    d.model_inputs = {"marker": sidecar}
    h.trace_drain()
    b.materialize(p, d)
    assert h.events == ["drain"]
    assert d.req.session is h.native()
    assert d.model_inputs["marker"] is sidecar
    assert h.inflight()
    b.rollback("r")
    assert not h.inflight()


@pytest.mark.parametrize(
    "rejection", ["native", "length", "input_length", "priority", "queue"]
)
def test_append_rejection_rolls_back_native_history(bridge_env, rejection):
    h = bridge_env
    b, s = h.bridge, h.scheduler
    b.command(payload("open"))
    h.finish(h.append("first"))
    retained_slot = object()
    s.tree_cache.slots["s"] = retained_slot
    if rejection == "native":
        failed = h.append("bad", [9])
        assert list(failed.req.origin_input_ids) == [1, 2, 3, 4, 9]
        b.rollback("bad")
    else:
        h.reject_append(rejection)
        with get_context().override_server_args(weight_version="test"):
            s.process_input_requests([payload("append")])
        assert s.outbox.get_nowait().type == "error"
    assert not h.inflight()
    assert s.tree_cache.slots["s"] is retained_slot
    following = h.append("next", [8])
    assert list(following.req.origin_input_ids) == [1, 2, 3, 4, 8]
    b.rollback("next")
    s.tree_cache.slots.clear()
    assert b.command(payload("close", "close")).data == {"closed": True}


def test_rejection_of_one_inflight_owner_leaves_others_intact(bridge_env, monkeypatch):
    import sglang.srt.managers.schedule_batch as native_batch

    h = bridge_env
    b = h.bridge
    b.command(payload("open"))
    h.append()
    with pytest.raises(ValueError, match="active"):
        b.accept(payload("append", "other"))
    assert h.inflight()
    b.rollback("r")
    b.accept(payload("append"))
    h.occupy_native()
    monkeypatch.setattr(
        native_batch, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )
    with pytest.raises(ValueError, match="native session rejected"):
        b.materialize(payload("append"), data())
    b.rollback("r")
    assert h.inflight()


def test_session_embeddings_explicitly_rejected(bridge_env):
    h = bridge_env
    b = h.bridge
    b.command(payload("open"))
    b.accept(payload("append"))
    d = data()
    d.prefill_input_embeds = object()
    with pytest.raises(ValueError, match="embedding"):
        b.materialize(payload("append"), d)
    assert not h.inflight()


@pytest.mark.parametrize(
    "case, rows, tokens, error",
    [
        ("retained_kv_available", 2, 2, None),
        ("retained_kv_exhausted", 2, 1, "session KV capacity exhausted"),
        (
            "reserve_row_exhausted",
            2,
            8,
            "session request slot capacity exhausted (one admission slot reserved)",
        ),
        ("reserve_row_available", 3, 8, None),
    ],
    ids=[
        "retained_kv_available",
        "retained_kv_exhausted",
        "reserve_row_exhausted",
        "reserve_row_available",
    ],
)
def test_admission_accounts_for_native_slots(bridge_env, case, rows, tokens, error):
    h = bridge_env
    h.bridge.command(payload("open"))
    d = h.append()
    if case.startswith("retained_kv"):
        h.retain_slot(d)
    else:
        opened = payload("open", "open-other")
        opened.request.metadata["omni_session"]["ref"]["session_id"] = "other"
        h.bridge.command(opened)
        h.append("other-r", sid="other")
    h.capacity(rows, tokens)
    assert h.bridge.capacity_error("r") == error
