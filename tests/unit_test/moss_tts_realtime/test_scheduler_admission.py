# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.session.session_controller import Session
from sglang.srt.session.streaming_session import SessionSlot

from sglang_omni.models.moss_tts_realtime import scheduler as scheduler_module
from sglang_omni.models.moss_tts_realtime.request_builders import (
    MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY,
    build_moss_tts_realtime_prefill_rows,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeSessionState,
    MossTTSRealtimeTurnPhase,
)
from sglang_omni.models.moss_tts_realtime.scheduler import MossTTSRealtimeScheduler
from sglang_omni.proto.messages import InputUpdateMessage
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG
from tests.unit_test.moss_tts_realtime.scheduler_test_utils import (
    _payload,
    _request_data,
    _scheduler,
    _seed_successful_session_turn,
    _set_limits,
    _wire_update,
)


def test_scheduler_reads_realtime_fields_from_hf_config(monkeypatch) -> None:
    captured_kwargs: list[dict[str, Any]] = []

    def fake_omni_init(self: Any, *args: Any, **kwargs: Any) -> None:
        del args
        captured_kwargs.append(dict(kwargs))
        self.server_args = SimpleNamespace(context_length=4096)
        self.max_total_num_tokens = 8192
        self.enable_overlap = False
        self.enable_async_decode = False

    monkeypatch.setattr(scheduler_module.OmniScheduler, "__init__", fake_omni_init)
    hf_config = SimpleNamespace(delay_tokens_len=12)
    wrapped_config = SimpleNamespace(hf_config=hf_config)

    wrapped = MossTTSRealtimeScheduler(
        model_config=wrapped_config,
        max_pending_text_tokens=12,
        max_pending_text_bytes=24,
        max_input_updates=3,
        terminal_tombstone_limit=4,
    )
    direct = MossTTSRealtimeScheduler(model_config=hf_config)

    assert wrapped._moss_tts_realtime_model_config is hf_config
    assert direct._moss_tts_realtime_model_config is hf_config
    assert wrapped._moss_tts_realtime_limits.max_pending_text_tokens == 12
    assert wrapped._moss_tts_realtime_limits.max_pending_text_bytes == 24
    assert wrapped._moss_tts_realtime_limits.max_input_updates == 3
    assert wrapped._terminal_tombstone_limit == 4
    assert captured_kwargs == [
        {"model_config": wrapped_config},
        {"model_config": hf_config},
    ]


@pytest.mark.parametrize("token_count", range(12))
def test_open_input_defers_until_twelve_tokens(token_count: int) -> None:
    scheduler = _scheduler()

    assert not scheduler._is_request_build_ready(
        _payload(tuple(range(token_count))),
        pending_stream_done=False,
    )


def test_twelve_tokens_and_short_done_are_build_ready() -> None:
    scheduler = _scheduler()

    assert scheduler._is_request_build_ready(
        _payload(tuple(range(12))),
        pending_stream_done=False,
    )
    assert scheduler._is_request_build_ready(
        _payload(tuple(range(5)), input_done=True),
        pending_stream_done=False,
    )
    assert scheduler._is_request_build_ready(
        _payload(input_done=True),
        pending_stream_done=False,
    )


def test_prepared_tokenization_metadata_controls_prefill_readiness() -> None:
    scheduler = _scheduler()
    payload = _payload()
    payload.data["initial_text"] = "tokenized outside the scheduler"
    payload.data[MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY] = list(range(12))

    assert scheduler._is_request_build_ready(
        payload,
        pending_stream_done=False,
    )


def test_input_update_wakes_deferred_payload_only_when_readiness_can_change() -> None:
    scheduler = _scheduler()
    payload = _payload()
    scheduler._deferred_request_payloads[payload.request_id] = payload

    scheduler._on_input_update(
        payload.request_id,
        _wire_update(seq_no=0, token_ids=tuple(range(12))),
    )

    assert payload.request_id in scheduler._dirty_deferred_request_ids
    assert scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_prefill_gate_ready_event_is_emitted_once_for_the_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scheduler_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(scheduler_module, "realtime_events_active", lambda: True)
    payload = _payload()
    payload.data.update(
        session_id="session-1",
        turn_id="turn-1",
        turn_index=0,
    )
    scheduler._on_input_update(
        payload.request_id,
        _wire_update(seq_no=0, token_ids=tuple(range(12))),
    )

    assert scheduler._is_request_build_ready(payload, pending_stream_done=False)
    assert scheduler._is_request_build_ready(payload, pending_stream_done=False)

    gate_events = [
        event
        for event in events
        if event["event_name"] == "moss_tts_realtime_prefill_gate_ready"
    ]
    assert len(gate_events) == 1
    assert gate_events[0]["metadata"] == {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "turn_index": 0,
        "seq_no": 0,
        "new_stable_token_count": 12,
        "stable_token_count": 12,
        "pending_bytes": 0,
        "input_done": False,
        "required_stable_token_count": 12,
        "short_input_done": False,
    }
    assert payload.request_id in scheduler._prefill_gate_ready_event_ids
    scheduler._mark_input_update_terminal(payload.request_id)
    assert payload.request_id not in scheduler._prefill_gate_ready_event_ids


def test_request_build_canonical_rows_and_queue_events_are_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scheduler_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(scheduler_module, "realtime_events_active", lambda: True)
    payload = _payload(tuple(range(12)))
    payload.data.update(
        session_id="session-1",
        turn_id="turn-1",
        turn_index=0,
    )
    scheduler._request_builder = lambda value: value.request_id

    assert scheduler._run_request_builder(payload, "tts_engine") == "request-1"
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(payload, False, data)

    def fake_enqueue(
        owner: Any,
        queued_payload: Any,
        pending_stream_done: bool,
        queued_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        del queued_payload, pending_stream_done, request_admission_lock_held
        queued_data.req._omni_data = queued_data
        owner.waiting_queue.append(queued_data.req)

    monkeypatch.setattr(
        scheduler_module.OmniScheduler,
        "_enqueue_built_request",
        fake_enqueue,
    )
    scheduler._enqueue_built_request(payload, False, data)

    critical = [
        event
        for event in events
        if event["event_name"]
        in {
            "moss_tts_realtime_request_build_start",
            "moss_tts_realtime_canonical_rows_ready",
            "moss_tts_realtime_scheduler_queue_enter",
        }
    ]
    assert [event["event_name"] for event in critical] == [
        "moss_tts_realtime_request_build_start",
        "moss_tts_realtime_canonical_rows_ready",
        "moss_tts_realtime_scheduler_queue_enter",
    ]
    assert critical[0]["stage"] == "tts_engine"
    assert critical[1]["metadata"]["prefill_token_count"] == 12
    assert critical[1]["metadata"]["canonical_prompt_rows"] == 13
    assert critical[2]["metadata"]["queue_depth"] == 1
    assert critical[2]["metadata"]["cached_rows"] == 0


def test_finalizer_replays_buffered_updates_before_freezing_prefill() -> None:
    scheduler = _scheduler()
    initial = (1, 2, 3)
    update = _wire_update(seq_no=0, token_ids=tuple(range(4, 13)))
    scheduler._on_input_update("request-1", update)
    data = _request_data(initial, input_done=False)

    finalized = scheduler._finalize_built_request(
        _payload(initial),
        False,
        data,
    )

    assert finalized.req is not None
    assert finalized.turn_state is not None
    assert finalized.turn_state.prefill_token_ids == tuple(range(1, 13))
    assert finalized.turn_state.pending_input.next_seq_no == 1
    assert len(finalized.turn_state.pending_input) == 0
    assert finalized.turn_state.phase is MossTTSRealtimeTurnPhase.RUNNING
    assert "request-1" not in scheduler._buffered_input_updates

    expected_prefill = build_moss_tts_realtime_prefill_rows(
        tuple(range(1, 13)),
        model_config=finalized.model_config,
    )
    assert torch.equal(finalized.prompt_rows[-12:], expected_prefill)
    assert list(finalized.req.origin_input_ids) == finalized.input_ids.tolist()
    assert len(finalized.turn_state.ledger.rows) == int(finalized.prompt_rows.shape[0])
    assert finalized.generation_row_start == len(finalized.turn_state.ledger.rows)


@pytest.mark.parametrize(
    ("limit_overrides", "updates", "match"),
    [
        (
            {"max_pending_text_tokens": 1},
            [_wire_update(seq_no=0, token_ids=(7, 8))],
            "token limit",
        ),
        (
            {"max_pending_text_bytes": 1},
            [_wire_update(seq_no=0, token_ids=(7,), byte_count=2)],
            "byte limit",
        ),
        (
            {"max_input_updates": 1},
            [
                _wire_update(seq_no=0, token_ids=(7,)),
                _wire_update(seq_no=1, token_ids=(8,)),
            ],
            "update limit",
        ),
    ],
)
def test_pre_payload_input_update_buffer_is_bounded(
    limit_overrides: dict[str, int],
    updates: list[InputUpdateMessage],
    match: str,
) -> None:
    scheduler = _scheduler()
    _set_limits(scheduler, **limit_overrides)

    for update in updates:
        scheduler._on_input_update("request-1", update)

    output = scheduler.outbox.get_nowait()
    assert output.request_id == "request-1"
    assert match in str(output.data)
    assert scheduler.is_input_update_terminal("request-1")
    assert "request-1" not in scheduler._buffered_input_updates


def test_pre_payload_protocol_validation_is_authoritative_at_replay() -> None:
    scheduler = _scheduler()
    scheduler._on_input_update(
        "request-1",
        _wire_update(seq_no=1, token_ids=tuple(range(12))),
    )

    with pytest.raises(ValueError, match="expected 0, got 1"):
        scheduler._finalize_built_request(
            _payload(),
            False,
            _request_data((), input_done=False),
        )


def test_live_input_update_uses_turn_state_for_duplicate_detection() -> None:
    scheduler = _scheduler()
    data = scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        _request_data(tuple(range(12)), input_done=False),
    )
    update = _wire_update(seq_no=0, token_ids=(99,))

    scheduler._on_input_update("request-1", update)
    scheduler._on_input_update("request-1", update)

    assert data.turn_state is not None
    assert len(data.turn_state.pending_input) == 1
    assert scheduler._resource_totals["input_update_accepted_total"] == 1
    assert scheduler._resource_totals["input_update_duplicate_total"] == 1


def test_terminal_tombstones_drop_late_updates_and_evict_oldest() -> None:
    scheduler = _scheduler()
    scheduler._terminal_tombstone_limit = 2
    scheduler._mark_input_update_terminal("request-1")
    scheduler._mark_input_update_terminal("request-2")
    scheduler._mark_input_update_terminal("request-3")

    assert scheduler._input_update_terminal_ids == {"request-2", "request-3"}
    scheduler._on_input_update(
        "request-3",
        _wire_update(seq_no=0, token_ids=(7,), request_id="request-3"),
    )
    assert "request-3" not in scheduler._buffered_input_updates
    assert scheduler.outbox.empty()


def test_short_done_prefill_and_empty_done_rejection_happen_before_req_creation() -> (
    None
):
    scheduler = _scheduler()
    short = _request_data((7, 8, 9), input_done=True)

    scheduler._finalize_built_request(
        _payload((7, 8, 9), input_done=True),
        False,
        short,
    )

    assert short.req is not None
    assert short.turn_state.prefill_token_ids == (7, 8, 9)
    assert short.turn_state.pending_input.input_done

    scheduler = _scheduler()
    empty = _request_data((), input_done=True)
    with pytest.raises(ValueError, match="empty realtime turn"):
        scheduler._finalize_built_request(
            _payload(input_done=True),
            False,
            empty,
        )
    assert empty.req is None
    assert empty.turn_state is not None
    assert empty.turn_state.model_state_slot_id is None


def test_max_sessions_rejects_new_identity_without_ownership_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    _set_limits(
        scheduler,
        max_sessions=1,
        max_held_sessions=1,
        max_active_turns=1,
    )
    scheduler._max_session_rows = 64
    scheduler._max_held_kv_tokens = 64
    scheduler._codec_slots = 1
    existing = MossTTSRealtimeSessionState(
        session_id="existing-session",
        model_config=MODEL_CONFIG,
    )
    scheduler._moss_tts_realtime_sessions[existing.session_id] = existing
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scheduler_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    rejected = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id="new-session",
    )

    with pytest.raises(RuntimeError, match="session limit exceeded"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12))),
            False,
            rejected,
        )

    assert scheduler._moss_tts_realtime_sessions == {existing.session_id: existing}
    assert scheduler._moss_tts_realtime_requests == {}
    assert scheduler.session_controller.sessions == {}
    assert scheduler.tree_cache.slots == {}
    assert rejected.req is None
    assert rejected.session_state is None
    assert rejected.turn_state is None
    assert rejected.context_reservation_rows == 0
    assert scheduler._admission_rejection_totals == {"session_limit": 1}
    assert events[-1]["event_name"] == "moss_tts_realtime_admission_rejected"
    assert events[-1]["metadata"]["reason"] == "session_limit"

    admitted = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id=existing.session_id,
    )
    scheduler._enforce_resource_admission("existing-request", admitted)
    assert admitted.context_reservation_rows == 45


def test_active_first_turn_reserves_held_session_before_physical_kv() -> None:
    scheduler = _scheduler()
    _set_limits(
        scheduler,
        max_sessions=2,
        max_held_sessions=1,
        max_active_turns=2,
    )
    scheduler._max_session_rows = 64
    scheduler._max_held_kv_tokens = 128
    scheduler._codec_slots = 2
    first = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id="session-a",
        turn_id="turn-a",
    )
    scheduler._finalize_built_request(
        _payload(tuple(range(12)), request_id="request-a"),
        False,
        first,
    )
    assert scheduler._logical_reservation_snapshot()["held_session_reservations"] == 1

    second = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id="session-b",
        turn_id="turn-b",
    )
    with pytest.raises(RuntimeError, match="held-session limit exceeded"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12)), request_id="request-b"),
            False,
            second,
        )

    assert set(scheduler._moss_tts_realtime_sessions) == {"session-a"}
    assert set(scheduler.session_controller.sessions) == {"session-a"}
    assert second.req is None
    assert second.session_state is None
    assert scheduler._admission_rejection_totals == {"held_session_limit": 1}


def test_large_turn_is_admitted_when_context_and_kv_capacity_allow() -> None:
    scheduler = _scheduler()
    scheduler._max_session_rows = 6000
    scheduler._max_held_kv_tokens = 6000
    data = _request_data(
        tuple(range(12)),
        input_done=False,
        max_new_tokens=5000,
    )

    scheduler._enforce_resource_admission("request-1", data)

    assert data.context_reservation_rows == 5013
    assert not getattr(scheduler, "_admission_rejection_totals", {})


def test_context_limit_rejects_before_session_create_req(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    scheduler._max_session_rows = 44
    scheduler._max_held_kv_tokens = 44
    create_calls = 0

    def unexpected_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal create_calls
        del args, kwargs
        create_calls += 1
        raise AssertionError(
            "Session.create_req must not run after admission rejection"
        )

    monkeypatch.setattr(Session, "create_req", unexpected_create)
    data = _request_data(tuple(range(12)), input_done=False)

    with pytest.raises(ValueError, match="session context limit exceeded"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12))),
            False,
            data,
        )

    assert create_calls == 0
    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler.session_controller.sessions == {}
    assert data.context_reservation_rows == 0
    assert scheduler._admission_rejection_totals == {"session_context_limit": 1}


def test_global_held_kv_limit_rejects_before_second_session_create_req(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    _set_limits(
        scheduler,
        max_sessions=2,
        max_held_sessions=2,
        max_active_turns=2,
    )
    scheduler._max_session_rows = 64
    scheduler._max_held_kv_tokens = 70
    scheduler._codec_slots = 2
    first = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id="session-a",
        turn_id="turn-a",
    )
    scheduler._finalize_built_request(
        _payload(tuple(range(12)), request_id="request-a"),
        False,
        first,
    )
    create_calls = 0

    def unexpected_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal create_calls
        del args, kwargs
        create_calls += 1
        raise AssertionError(
            "Session.create_req must not run after admission rejection"
        )

    monkeypatch.setattr(Session, "create_req", unexpected_create)
    second = _request_data(
        tuple(range(12)),
        input_done=False,
        session_id="session-b",
        turn_id="turn-b",
    )

    with pytest.raises(RuntimeError, match="held-KV reservation limit exceeded"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12)), request_id="request-b"),
            False,
            second,
        )

    assert create_calls == 0
    assert set(scheduler._moss_tts_realtime_sessions) == {"session-a"}
    assert set(scheduler.session_controller.sessions) == {"session-a"}
    assert second.session_state is None
    assert scheduler._admission_rejection_totals == {"held_kv_limit": 1}


def test_warm_session_admission_replaces_old_kv_reservation() -> None:
    scheduler = _scheduler()
    session_state, _, committed_rows, _ = _seed_successful_session_turn(scheduler)
    required_rows = len(committed_rows) + 1 + 12 + 32
    _set_limits(
        scheduler,
        max_sessions=1,
        max_held_sessions=1,
        max_active_turns=1,
    )
    scheduler._max_session_rows = required_rows
    scheduler._max_held_kv_tokens = required_rows
    scheduler._codec_slots = 1
    before = scheduler._logical_reservation_snapshot()
    assert before["held_session_reservations"] == 1
    assert before["held_kv_token_reservations"] == len(committed_rows)
    second = _request_data(
        tuple(range(100, 112)),
        input_done=False,
        turn_id="turn-2",
        turn_index=1,
    )

    scheduler._enforce_resource_admission("request-2", second)

    assert second.context_reservation_rows == required_rows
    assert scheduler._logical_reservation_snapshot() == before
    assert session_state.warm_kv_length == len(committed_rows)


def test_parked_req_pool_slot_is_excluded_from_physical_held_kv() -> None:
    scheduler = _scheduler()
    scheduler._parked_input["parked"] = SimpleNamespace(
        req=SimpleNamespace(req_pool_idx=3)
    )
    scheduler.tree_cache.slots["active-session"] = SessionSlot(
        req_pool_idx=3,
        kv_committed_len=5,
        kv=ReqKvInfo(kv_allocated_len=6, swa_evicted_seqlen=0),
    )
    scheduler.tree_cache.slots["idle-session"] = SessionSlot(
        req_pool_idx=4,
        kv_committed_len=7,
        kv=ReqKvInfo(kv_allocated_len=8, swa_evicted_seqlen=0),
    )

    snapshot = scheduler._physical_session_snapshot()

    assert snapshot == {
        "physical_held_session_count": 1,
        "physical_held_kv_rows": 7,
        "physical_held_kv_tokens": scheduler.tree_cache.session_held_tokens({3}),
    }
