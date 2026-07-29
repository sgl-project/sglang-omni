# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import deque
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

# SGLang imports CUDA-only modules which decorate functions with
# ``torch.compile`` at import time.  On the macOS CPU test host that causes
# TorchInductor to import an unavailable Triton runtime before these unit tests
# can install their fakes.  The bridge itself does not exercise compilation.
_original_torch_compile = torch.compile


def _identity_compile(model=None, *args, **kwargs):
    del args, kwargs
    if model is None:
        return lambda fn: fn
    return model


torch.compile = _identity_compile
try:
    from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import cache as cache_module
finally:
    torch.compile = _original_torch_compile


class _FakeReq:
    def __init__(
        self,
        *,
        rid: str = "turn-1",
        finished_reason: str | None = None,
    ) -> None:
        self.rid = rid
        self.output_ids: list[int] = []
        self.finished_reason = finished_reason

    def finished(self) -> bool:
        return self.finished_reason is not None


class _FakeCacheParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeRadixCache:
    def __init__(self, params, *, supports_streaming: bool = False) -> None:
        self.params = params
        self._supports_streaming = supports_streaming

    def supports_streaming_session(self) -> bool:
        return self._supports_streaming


class _FakeStreamingSession:
    def __init__(self, inner) -> None:
        self.inner = inner


def _server_args(**overrides):
    values = {
        "disable_radix_cache": False,
        "chunked_prefill_size": 1024,
        "enable_streaming_session": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_create_tree_cache_wraps_cache_for_streaming_sessions(monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "CacheInitParams", _FakeCacheParams)
    monkeypatch.setattr(cache_module, "RadixCache", _FakeRadixCache)
    monkeypatch.setattr(cache_module, "StreamingSession", _FakeStreamingSession)

    tree_cache = cache_module.create_tree_cache(
        _server_args(enable_streaming_session=True),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        page_size=1,
    )

    assert isinstance(tree_cache, _FakeStreamingSession)
    assert isinstance(tree_cache.inner, _FakeRadixCache)


def test_create_tree_cache_does_not_double_wrap_supported_cache(monkeypatch) -> None:
    class StreamingCapableRadixCache(_FakeRadixCache):
        def __init__(self, params) -> None:
            super().__init__(params, supports_streaming=True)

    monkeypatch.setattr(cache_module, "CacheInitParams", _FakeCacheParams)
    monkeypatch.setattr(cache_module, "RadixCache", StreamingCapableRadixCache)
    monkeypatch.setattr(cache_module, "StreamingSession", _FakeStreamingSession)

    tree_cache = cache_module.create_tree_cache(
        _server_args(enable_streaming_session=True),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        page_size=1,
    )

    assert isinstance(tree_cache, StreamingCapableRadixCache)


def test_omni_scheduler_uses_real_controller_when_enabled(monkeypatch) -> None:
    created_with = []

    class FakeSessionController:
        def __init__(self, tree_cache) -> None:
            created_with.append(tree_cache)
            self.sessions = {}

    monkeypatch.setattr(
        omni_scheduler_module, "SessionController", FakeSessionController
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.tree_cache = object()

    scheduler._init_session_controller(SimpleNamespace(enable_streaming_session=True))

    assert isinstance(scheduler.session_controller, FakeSessionController)
    assert created_with == [scheduler.tree_cache]


def test_omni_scheduler_uses_noop_controller_when_disabled() -> None:
    scheduler = object.__new__(OmniScheduler)

    scheduler._init_session_controller(SimpleNamespace(enable_streaming_session=False))

    assert scheduler.session_controller.sessions == {}
    assert scheduler.session_controller.maybe_reap(123.0) is None


def test_process_input_requests_reaps_sessions_once_per_cycle(monkeypatch) -> None:
    reap_calls = []
    scheduler = object.__new__(OmniScheduler)
    scheduler.session_controller = SimpleNamespace(
        maybe_reap=lambda now: reap_calls.append(now)
    )
    scheduler._drain_request_build_results = lambda: None
    scheduler._stage_request_build_payloads = lambda recv_reqs: (recv_reqs, [])
    monkeypatch.setattr(omni_scheduler_module.time, "monotonic", lambda: 123.0)

    scheduler.process_input_requests([])

    assert reap_calls == [123.0]


def test_session_request_is_materialized_through_upstream_session_api() -> None:
    create_calls = []
    final_req = _FakeReq()

    class FakeSession:
        close_on_finish = False

        def create_req(self, recv_req, tokenizer, vocab_size, eos_token_ids=None):
            create_calls.append((recv_req, tokenizer, vocab_size, eos_token_ids))
            return final_req

    session = FakeSession()
    scheduler = object.__new__(OmniScheduler)
    scheduler.session_controller = SimpleNamespace(
        get=lambda session_id: session if session_id == "session-1" else None
    )
    scheduler.model_config = SimpleNamespace(
        vocab_size=32000,
        hf_eos_token_id=2,
    )
    recv_req = SimpleNamespace(session_params=SimpleNamespace(id="session-1"))
    setup_calls = []
    req_data = SimpleNamespace(
        req=None,
        output_ids=None,
        tokenized_session_req=recv_req,
        session_tokenizer=object(),
        session_req_setup=lambda req: setup_calls.append(req),
    )

    result = scheduler._materialize_streaming_session_req(req_data)

    assert result is final_req
    assert req_data.req is final_req
    assert req_data.output_ids is final_req.output_ids
    assert setup_calls == [final_req]
    assert create_calls == [
        (recv_req, req_data.session_tokenizer, 32000, 2),
    ]


def test_ordinary_request_materialization_is_identity() -> None:
    scheduler = object.__new__(OmniScheduler)
    req = object()
    req_data = SimpleNamespace(req=req, tokenized_session_req=None)

    assert scheduler._materialize_streaming_session_req(req_data) is req


def test_session_request_rejects_missing_session() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.session_controller = SimpleNamespace(get=lambda _session_id: None)
    scheduler.model_config = SimpleNamespace(vocab_size=32000)
    req_data = SimpleNamespace(
        req=None,
        tokenized_session_req=SimpleNamespace(
            session_params=SimpleNamespace(id="missing")
        ),
        session_tokenizer=None,
    )

    with pytest.raises(ValueError, match="does not exist"):
        scheduler._materialize_streaming_session_req(req_data)


def test_inflight_rejection_does_not_abort_the_existing_request() -> None:
    """A rejected second turn must not clear the first turn's inflight flag."""
    rejected_req = _FakeReq(finished_reason="active request")

    class FakeInflightSession:
        close_on_finish = False
        _inflight = True

        def __init__(self) -> None:
            self.abort_calls = 0

        def create_req(self, *_args, **_kwargs):
            return rejected_req

        def abort_req(self):
            self.abort_calls += 1

    session = FakeInflightSession()

    scheduler = object.__new__(OmniScheduler)
    scheduler.session_controller = SimpleNamespace(get=lambda _session_id: session)
    scheduler.model_config = SimpleNamespace(vocab_size=32000)
    req_data = SimpleNamespace(
        req=None,
        output_ids=None,
        tokenized_session_req=SimpleNamespace(
            session_params=SimpleNamespace(id="busy")
        ),
        session_tokenizer=None,
        session_req_setup=None,
    )

    with pytest.raises(ValueError, match="rejected request"):
        scheduler._materialize_streaming_session_req(req_data)

    assert session._inflight is True
    assert session.abort_calls == 0


def test_pre_enqueue_rejection_rolls_back_setup_and_owned_inflight() -> None:
    ledger: list[str] = []

    class FakeSession:
        close_on_finish = False

        def __init__(self) -> None:
            self._inflight = False
            self.abort_calls = 0

        def create_req(self, *_args, **_kwargs):
            self._inflight = True
            return _FakeReq()

        def abort_req(self):
            self.abort_calls += 1
            self._inflight = False

    session = FakeSession()
    scheduler = object.__new__(OmniScheduler)
    scheduler.session_controller = SimpleNamespace(get=lambda _session_id: session)
    scheduler.model_config = SimpleNamespace(vocab_size=32000, hf_eos_token_id=2)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.waiting_queue = []
    scheduler._pending_stream_done = set()
    scheduler._pending_stream_chunks = {}
    scheduler._deferred_request_payloads = {}
    scheduler._aborted_request_ids = set()
    scheduler._prepare_request_limits = lambda _req_data: "request is too long"
    scheduler._request_kv_capacity_error = lambda _req: None
    aborted: list[str] = []
    scheduler.abort = lambda request_id: aborted.append(request_id)

    def setup(_req):
        ledger.append("embedding-span")

        def rollback():
            ledger.pop()

        return rollback

    req_data = SimpleNamespace(
        req=None,
        output_ids=None,
        tokenized_session_req=SimpleNamespace(
            session_params=SimpleNamespace(id="session-1")
        ),
        session_tokenizer=None,
        session_req_setup=setup,
        enforce_request_limits=True,
    )

    scheduler._enqueue_built_request(
        SimpleNamespace(request_id="turn-1"),
        pending_stream_done=False,
        req_data=req_data,
    )

    output = scheduler.outbox.get_nowait()
    assert output.type == "error"
    assert ledger == []
    assert session._inflight is False
    assert session.abort_calls == 1
    assert scheduler.waiting_queue == []
    assert aborted == ["turn-1"]


def test_missing_session_failure_emits_error_without_enqueue() -> None:
    scheduler = object.__new__(OmniScheduler)
    scheduler.session_controller = SimpleNamespace(get=lambda _session_id: None)
    scheduler.model_config = SimpleNamespace(vocab_size=32000)
    scheduler.outbox = Queue()
    scheduler.is_entry_rank = True
    scheduler.waiting_queue = []
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    aborted = []
    scheduler.abort = lambda request_id: aborted.append(request_id)
    req_data = SimpleNamespace(
        req=None,
        tokenized_session_req=SimpleNamespace(
            session_params=SimpleNamespace(id="missing")
        ),
        session_tokenizer=None,
    )

    scheduler._enqueue_built_request(
        SimpleNamespace(request_id="turn-1"),
        pending_stream_done=False,
        req_data=req_data,
    )

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "turn-1"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert scheduler.waiting_queue == []
    assert aborted == ["turn-1"]


def test_abort_waiting_streaming_session_releases_before_callback() -> None:
    ledger: list[str] = []

    class FakeSession:
        streaming = True

        def __init__(self) -> None:
            self._inflight = True
            self.abort_calls = 0

        def abort_req(self) -> None:
            self.abort_calls += 1
            self._inflight = False
            ledger.append("session_abort")

    session = FakeSession()
    req = SimpleNamespace(rid="turn-1", session=session)
    scheduler = object.__new__(OmniScheduler)
    scheduler._mark_running_request_aborted = lambda _request_id: False
    scheduler._request_admission_lock = threading.RLock()
    scheduler._aborted_request_ids = set()
    scheduler._aborted_request_id_order = deque()
    scheduler._pending_request_builds = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler.waiting_queue = [req]
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler.running_batch = None
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending_batch = lambda: None
    scheduler._release_immediate_request_resources = lambda _request_id: ledger.append(
        "release"
    )
    scheduler._abort_callback = lambda _request_id: ledger.append("callback")
    scheduler._drain_inbox_for_request = lambda _request_id: None

    scheduler.abort("turn-1")

    assert session.abort_calls == 1
    assert session._inflight is False
    assert req.session is None
    assert scheduler.waiting_queue == []
    assert ledger == ["session_abort", "release", "callback"]
