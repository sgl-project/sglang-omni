# SPDX-License-Identifier: Apache-2.0
"""Streaming-session support in the scheduling layer.

Covers the cache factory wrap condition, the admin open/close lifecycle,
and session attach at request admission (Session.create_req concatenating
the previous turn onto the new input).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

sglang_session = pytest.importorskip("sglang.srt.session.session_controller")


class _ReleaseSpyCache:
    """Minimal BasePrefixCache stand-in for SessionController tests."""

    def __init__(self) -> None:
        self.released: list[str] = []

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.released.append(session_id)


def _make_controller():
    return sglang_session.SessionController(_ReleaseSpyCache())


class _SessionStubScheduler:
    _admin_open_session = OmniScheduler._admin_open_session
    _admin_close_session = OmniScheduler._admin_close_session
    _maybe_attach_session = OmniScheduler._maybe_attach_session
    _SESSION_REQ_CARRYOVER_ATTRS = OmniScheduler._SESSION_REQ_CARRYOVER_ATTRS

    def __init__(self, *, sessions_enabled: bool = True) -> None:
        self._sessions_enabled = sessions_enabled
        if sessions_enabled:
            self.session_controller = _make_controller()
        else:
            self.session_controller = SimpleNamespace(sessions={})


# ---------------------------------------------------------------------------
# Cache factory
# ---------------------------------------------------------------------------


def test_create_tree_cache_wraps_only_when_flag_set(monkeypatch) -> None:
    from sglang_omni.scheduling.sglang_backend import cache as cache_mod

    class _FakeInner:
        def __init__(self, params) -> None:
            self.params = params

        def supports_streaming_session(self) -> bool:
            return False

    monkeypatch.setattr(cache_mod, "RadixCache", _FakeInner)

    def _args(**kwargs):
        return SimpleNamespace(
            disable_radix_cache=False,
            chunked_prefill_size=None,
            **kwargs,
        )

    plain = cache_mod.create_tree_cache(
        _args(enable_streaming_session=False), None, None, 1
    )
    assert isinstance(plain, _FakeInner)

    wrapped = cache_mod.create_tree_cache(
        _args(enable_streaming_session=True), None, None, 1
    )
    from sglang.srt.session.streaming_session import StreamingSession

    assert isinstance(wrapped, StreamingSession)
    assert isinstance(wrapped.inner, _FakeInner)


# ---------------------------------------------------------------------------
# Admin open / close
# ---------------------------------------------------------------------------


def test_open_and_close_session_via_admin() -> None:
    scheduler = _SessionStubScheduler()

    opened = scheduler._admin_open_session({})
    assert opened["success"]
    session_id = opened["data"]["session_id"]
    assert session_id in scheduler.session_controller.sessions

    closed = scheduler._admin_close_session({"session_id": session_id})
    assert closed["success"]
    assert session_id not in scheduler.session_controller.sessions
    assert scheduler.session_controller.tree_cache.released == [session_id]


def test_session_admin_rejected_when_flag_off() -> None:
    scheduler = _SessionStubScheduler(sessions_enabled=False)

    opened = scheduler._admin_open_session({})
    assert not opened["success"]
    assert "disabled" in opened["error"]

    closed = scheduler._admin_close_session({"session_id": "s1"})
    assert not closed["success"]


# ---------------------------------------------------------------------------
# Session attach at admission
# ---------------------------------------------------------------------------


def _build_req(rid: str, input_ids: list[int]):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    sampling_params = SamplingParams(max_new_tokens=8, temperature=0.0)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(input_ids),
        sampling_params=sampling_params,
        vocab_size=1000,
    )
    req.tokenizer = None
    req.omni_model_inputs = {"marker": rid}
    return req


def _payload_and_data(req, session_id: str | None):
    params = {}
    if session_id is not None:
        params["session_params"] = {"session_id": session_id}
    payload = SimpleNamespace(request=SimpleNamespace(params=params))
    req_data = SimpleNamespace(req=req, output_ids=req.output_ids)
    return payload, req_data


def test_request_without_session_params_passes_through() -> None:
    scheduler = _SessionStubScheduler()
    req = _build_req("r1", [1, 2, 3])
    payload, req_data = _payload_and_data(req, None)

    assert scheduler._maybe_attach_session(payload, req_data) is None
    assert req_data.req is req


def test_session_attach_concatenates_previous_turn() -> None:
    scheduler = _SessionStubScheduler()
    session_id = scheduler._admin_open_session({})["data"]["session_id"]
    session = scheduler.session_controller.get(session_id)

    first = _build_req("t1", [1, 2, 3])
    payload, req_data = _payload_and_data(first, session_id)
    assert scheduler._maybe_attach_session(payload, req_data) is None
    turn1 = req_data.req
    assert turn1.session is session
    assert list(turn1.origin_input_ids) == [1, 2, 3]
    # Omni-side builder attributes carry over onto the session-built Req.
    assert turn1.omni_model_inputs == {"marker": "t1"}

    # Finish turn 1 with two generated tokens, then extend with new input.
    turn1.output_ids.extend([10, 11])
    session.finish_req(turn1)

    second = _build_req("t2", [4, 5])
    payload, req_data = _payload_and_data(second, session_id)
    assert scheduler._maybe_attach_session(payload, req_data) is None
    turn2 = req_data.req
    assert list(turn2.origin_input_ids) == [1, 2, 3, 10, 11, 4, 5]
    assert req_data.output_ids is turn2.output_ids


def test_session_attach_rejects_unknown_session() -> None:
    scheduler = _SessionStubScheduler()
    req = _build_req("r1", [1, 2, 3])
    payload, req_data = _payload_and_data(req, "missing")

    error = scheduler._maybe_attach_session(payload, req_data)
    assert error is not None
    assert "missing" in error
    assert req_data.req is req


def test_session_attach_rejects_when_flag_off() -> None:
    scheduler = _SessionStubScheduler(sessions_enabled=False)
    req = _build_req("r1", [1, 2, 3])
    payload, req_data = _payload_and_data(req, "s1")

    error = scheduler._maybe_attach_session(payload, req_data)
    assert error is not None
    assert "disabled" in error


def test_session_attach_rejects_second_inflight_turn(monkeypatch) -> None:
    # set_finish_with_abort logs on TP rank 0; no TP group exists in unit tests.
    from sglang.srt.managers import schedule_batch

    monkeypatch.setattr(
        schedule_batch, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )

    scheduler = _SessionStubScheduler()
    session_id = scheduler._admin_open_session({})["data"]["session_id"]

    first = _build_req("t1", [1, 2, 3])
    payload, req_data = _payload_and_data(first, session_id)
    assert scheduler._maybe_attach_session(payload, req_data) is None

    second = _build_req("t2", [4, 5])
    payload, req_data = _payload_and_data(second, session_id)
    error = scheduler._maybe_attach_session(payload, req_data)
    assert error is not None
