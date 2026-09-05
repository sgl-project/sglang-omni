# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import threading
from array import array
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.session.session_controller import Session

from sglang_omni.models.moss_tts_realtime.request_builders import (
    build_moss_tts_realtime_row_cache_key_ids,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeRequestData,
    MossTTSRealtimeTurnPhase,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    REFERENCE_AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.scheduler_test_utils import (
    _AlignedDecodeBatch,
    _payload,
    _request_data,
    _scheduler,
    _seed_successful_session_turn,
)


def test_first_turn_lazily_opens_real_streaming_session() -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)

    assert scheduler.session_controller.get("session-1") is None

    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )

    session_state = scheduler._moss_tts_realtime_sessions["session-1"]
    session = scheduler.session_controller.get("session-1")
    assert session is not None
    assert session.streaming is True
    assert session._inflight is True
    assert data.session_state is session_state
    assert data.turn_state is not None
    assert session_state.active_turn_id == data.turn_state.turn_id
    assert data.req.session is session
    assert scheduler.session_controller.tree_cache is scheduler.tree_cache


def test_session_create_req_runs_on_scheduler_thread_with_append_only_params(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    observed: list[tuple[int, Any]] = []
    original = Session.create_req

    def checked_create_req(
        session: Session,
        tokenized_req: Any,
        tokenizer: Any,
        vocab_size: int,
        eos_token_ids: Any = None,
    ) -> Any:
        observed.append((threading.get_ident(), tokenized_req.session_params))
        return original(
            session,
            tokenized_req,
            tokenizer,
            vocab_size,
            eos_token_ids=eos_token_ids,
        )

    monkeypatch.setattr(Session, "create_req", checked_create_req)
    data = _request_data(tuple(range(12)), input_done=False)

    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )

    assert len(observed) == 1
    thread_id, params = observed[0]
    assert thread_id == scheduler._scheduler_thread_id
    assert params.id == "session-1"
    assert params.rid is None
    assert params.offset == 0
    assert params.replace is False
    assert params.drop_previous_output is False


def test_session_requests_use_stable_session_scoped_radix_namespace(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    observed: list[tuple[str, str | None, str | None]] = []
    original = Session.create_req

    def capture_create_req(
        session: Session,
        tokenized_req: Any,
        tokenizer: Any,
        vocab_size: int,
        eos_token_ids: Any = None,
    ) -> Any:
        req = original(
            session,
            tokenized_req,
            tokenizer,
            vocab_size,
            eos_token_ids=eos_token_ids,
        )
        observed.append((session.session_id, tokenized_req.extra_key, req.extra_key))
        return req

    monkeypatch.setattr(Session, "create_req", capture_create_req)
    _seed_successful_session_turn(scheduler)

    second = _request_data(
        tuple(range(100, 112)),
        input_done=False,
        turn_id="turn-2",
        turn_index=1,
    )
    scheduler._finalize_built_request(
        _payload(tuple(range(100, 112)), request_id="request-2"),
        False,
        second,
    )

    other_session = _request_data(
        tuple(range(200, 212)),
        input_done=False,
        session_id="session-2",
        turn_id="turn-1",
    )
    scheduler._finalize_built_request(
        _payload(tuple(range(200, 212)), request_id="request-3"),
        False,
        other_session,
    )

    session_1_key = (
        "moss-tts-realtime-session:" + hashlib.sha256(b"session-1").hexdigest()
    )
    session_2_key = (
        "moss-tts-realtime-session:" + hashlib.sha256(b"session-2").hexdigest()
    )
    assert observed == [
        ("session-1", session_1_key, session_1_key),
        ("session-1", session_1_key, session_1_key),
        ("session-2", session_2_key, session_2_key),
    ]
    assert session_1_key != session_2_key
    assert second.req.extra_key == session_1_key
    assert other_session.req.extra_key == session_2_key


def test_inflight_physical_session_close_is_recorded_as_deferred() -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    session = data.req.session
    assert session is not None
    assert session._inflight is True

    closed = scheduler._close_sglang_session_id(
        session.session_id,
        abort_inflight=False,
        reason="test_deferred_close",
        request_id=data.req.rid,
    )

    assert closed is False
    assert scheduler.session_controller.get(session.session_id) is session
    assert session.close_on_finish is True
    assert scheduler._resource_totals["physical_session_close_deferred_total"] == 1
    assert scheduler._resource_totals["physical_session_close_error_total"] == 0


def test_concurrent_turn_is_rejected_without_clearing_existing_inflight() -> None:
    scheduler = _scheduler()
    first = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        first,
    )
    session_state = first.session_state
    session = first.req.session
    assert session_state is not None
    assert session is not None
    assert session._inflight is True

    second = _request_data(
        tuple(range(12, 24)),
        input_done=False,
        turn_id="turn-2",
    )
    with pytest.raises(RuntimeError, match="already has active turn"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12, 24)), request_id="request-2"),
            False,
            second,
        )

    assert scheduler._moss_tts_realtime_sessions["session-1"] is session_state
    assert session._inflight is True
    assert second.req is None
    assert second.session_state is None
    assert second.turn_state is None


def test_turn_index_mismatch_does_not_publish_host_or_sglang_session() -> None:
    scheduler = _scheduler()
    data = _request_data(
        tuple(range(12)),
        input_done=False,
        turn_index=1,
    )

    with pytest.raises(ValueError, match="turn_index does not match"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12))),
            False,
            data,
        )

    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler.session_controller.get("session-1") is None
    assert data.req is None
    assert data.session_state is None
    assert data.turn_state is None


def test_first_turn_post_create_failure_rolls_back_new_session_state(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    created_sessions: list[Session] = []
    original = Session.create_req

    def create_corrupt_req(
        session: Session,
        tokenized_req: Any,
        tokenizer: Any,
        vocab_size: int,
        eos_token_ids: Any = None,
    ) -> Any:
        req = original(
            session,
            tokenized_req,
            tokenizer,
            vocab_size,
            eos_token_ids=eos_token_ids,
        )
        created_sessions.append(session)
        req.origin_input_ids[0] += 1
        return req

    monkeypatch.setattr(Session, "create_req", create_corrupt_req)
    data = _request_data(tuple(range(12)), input_done=False)

    with pytest.raises(RuntimeError, match="does not match canonical row hashes"):
        scheduler._finalize_built_request(
            _payload(tuple(range(12))),
            False,
            data,
        )

    assert len(created_sessions) == 1
    assert created_sessions[0]._inflight is False
    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler.session_controller.get("session-1") is None
    assert data.req is None
    assert data.session_state is None
    assert data.turn_state is not None


def test_next_turn_reuses_host_session_and_removes_prior_audio_eos(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    session_state, session, committed_rows, _ = _seed_successful_session_turn(scheduler)
    submitted_input_ids: list[list[int]] = []
    original = Session.create_req

    def capture_create_req(
        owner: Session,
        tokenized_req: Any,
        tokenizer: Any,
        vocab_size: int,
        eos_token_ids: Any = None,
    ) -> Any:
        submitted_input_ids.append(list(tokenized_req.input_ids))
        return original(
            owner,
            tokenized_req,
            tokenizer,
            vocab_size,
            eos_token_ids=eos_token_ids,
        )

    monkeypatch.setattr(Session, "create_req", capture_create_req)
    second_rows = torch.tensor(
        [[88, *([MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID] * 16)]],
        dtype=torch.long,
    )
    second = _request_data(
        tuple(range(100, 112)),
        input_done=False,
        turn_id="turn-2",
        turn_index=1,
        prompt_rows=second_rows,
    )

    scheduler._finalize_built_request(
        _payload(tuple(range(100, 112)), request_id="request-2"),
        False,
        second,
    )

    expected_rows = torch.tensor(second.turn_state.ledger.rows, dtype=torch.long)
    expected_ids = build_moss_tts_realtime_row_cache_key_ids(expected_rows)
    assert scheduler._moss_tts_realtime_sessions["session-1"] is session_state
    assert second.session_state is session_state
    assert second.req.session is session
    assert list(second.req.origin_input_ids) == expected_ids
    assert list(second.req.origin_input_ids_unpadded) == expected_ids
    assert second.input_ids.tolist() == expected_ids
    assert torch.equal(second.prompt_rows, expected_rows)
    assert tuple(second.turn_state.ledger.committed_prefix) == committed_rows
    assert MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID not in expected_ids
    assert session._inflight is True
    expected_suffix_ids = build_moss_tts_realtime_row_cache_key_ids(
        torch.tensor(second.turn_state.ledger.appended_rows, dtype=torch.long)
    )
    assert submitted_input_ids == [expected_suffix_ids]
    assert len(expected_suffix_ids) < len(expected_ids)


@pytest.mark.parametrize(
    "warm_damage",
    ("missing_slot", "committed_length_mismatch", "allocated_length_mismatch"),
)
def test_invalid_warm_slot_replays_full_committed_ledger(
    monkeypatch,
    warm_damage: str,
) -> None:
    scheduler = _scheduler()
    session_state, old_session, committed_rows, _ = _seed_successful_session_turn(
        scheduler
    )
    old_slot = scheduler.tree_cache.slots[old_session.session_id]
    if warm_damage == "missing_slot":
        scheduler.tree_cache.slots.pop(old_session.session_id)
    elif warm_damage == "committed_length_mismatch":
        old_slot.kv_committed_len -= 1
    else:
        old_slot.kv.kv_allocated_len = old_slot.kv_committed_len - 1
    submitted_input_ids: list[list[int]] = []
    original = Session.create_req

    def capture_create_req(
        owner: Session,
        tokenized_req: Any,
        tokenizer: Any,
        vocab_size: int,
        eos_token_ids: Any = None,
    ) -> Any:
        submitted_input_ids.append(list(tokenized_req.input_ids))
        return original(
            owner,
            tokenized_req,
            tokenizer,
            vocab_size,
            eos_token_ids=eos_token_ids,
        )

    monkeypatch.setattr(Session, "create_req", capture_create_req)
    second = _request_data(
        tuple(range(100, 112)),
        input_done=False,
        turn_id="turn-2",
        turn_index=1,
    )

    scheduler._finalize_built_request(
        _payload(tuple(range(100, 112)), request_id="request-2"),
        False,
        second,
    )

    expected_ids = build_moss_tts_realtime_row_cache_key_ids(second.prompt_rows)
    assert second.ledger_replay is True
    assert submitted_input_ids == [expected_ids]
    assert tuple(second.turn_state.ledger.committed_prefix) == committed_rows
    assert list(second.req.origin_input_ids) == expected_ids
    assert second.req.session is not old_session
    assert scheduler.session_controller.get("session-1") is second.req.session
    assert session_state.warm_session_id is None


def test_post_finalizer_admission_failure_cleans_live_session_state() -> None:
    scheduler = _scheduler()
    scheduler._prepare_request_limits = lambda data: "forced admission rejection"
    payload = _payload(tuple(range(12)))
    data = _request_data(tuple(range(12)), input_done=False)

    scheduler._enqueue_built_request(payload, False, data)

    assert data.req is not None
    assert data.req.session._inflight is False
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert data.session_state.active_turn_id is None
    assert data.session_state.warm_session_id is None
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    assert scheduler._moss_tts_realtime_requests == {}
    assert scheduler.waiting_queue == []
    output = scheduler.outbox.get_nowait()
    assert output.type == "error"
    assert "forced admission rejection" in str(output.data)


@pytest.mark.parametrize(
    ("prior_outputs", "error_match"),
    (
        ("missing", "missing prior audio EOS"),
        ("duplicate", "duplicate audio EOS"),
        ("hash_drift", "does not match canonical row hashes"),
    ),
)
def test_next_turn_terminal_history_corruption_fails_without_poisoning_inflight(
    prior_outputs: str,
    error_match: str,
) -> None:
    scheduler = _scheduler()
    session_state, session, _, generated_key = _seed_successful_session_turn(scheduler)
    if prior_outputs == "missing":
        session.req_nodes[next(iter(session.req_nodes))].req.output_ids[:] = array(
            "q",
            [generated_key, generated_key + 1],
        )
    elif prior_outputs == "duplicate":
        session.req_nodes[next(iter(session.req_nodes))].req.output_ids[:] = array(
            "q",
            [
                MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
                MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
            ],
        )
    else:
        session.req_nodes[next(iter(session.req_nodes))].req.output_ids[:] = array(
            "q",
            [generated_key + 1, MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID],
        )

    second = _request_data(
        tuple(range(100, 112)),
        input_done=False,
        turn_id="turn-2",
        turn_index=1,
    )
    with pytest.raises(RuntimeError, match=error_match):
        scheduler._finalize_built_request(
            _payload(tuple(range(100, 112)), request_id="request-2"),
            False,
            second,
        )

    assert scheduler._moss_tts_realtime_sessions["session-1"] is session_state
    assert session_state.active_turn_id is None
    assert session._inflight is False
    assert len(session.req_nodes) == 1
    assert second.req is None


def test_idle_realtime_session_close_is_idempotent() -> None:
    scheduler = _scheduler()
    session_state, _, _, _ = _seed_successful_session_turn(scheduler)

    first = scheduler._admin_close_realtime_session({"session_id": "session-1"})
    second = scheduler._admin_close_realtime_session({"session_id": "session-1"})

    assert first["success"] is True
    assert first["data"] == {
        "session_id": "session-1",
        "closed": True,
        "deferred": False,
        "existed": True,
    }
    assert second["success"] is True
    assert second["data"] == {
        "session_id": "session-1",
        "closed": True,
        "deferred": False,
        "existed": False,
    }
    assert session_state.closed is True
    assert session_state.close_requested is True
    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots


def test_active_realtime_session_close_defers_until_finish_abort(monkeypatch) -> None:
    from sglang.srt.managers import schedule_batch as schedule_batch_module

    monkeypatch.setattr(
        schedule_batch_module,
        "get_serving",
        lambda: SimpleNamespace(strip_thinking_cache=False),
    )
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    session_state = data.session_state
    assert req is not None
    assert turn is not None
    assert session_state is not None
    req._omni_data = data
    turn.observe_audio_frame(tuple(range(1, 17)), generation_step=0)
    req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    req.req_pool_idx = 4
    req.kv_committed_len = len(turn.ledger.rows)
    req.kv = ReqKvInfo(
        kv_allocated_len=len(turn.ledger.rows),
        swa_evicted_seqlen=0,
    )
    scheduler.running_batch = _AlignedDecodeBatch([req])

    def mark_running_aborted(request_id: str) -> bool:
        assert request_id == req.rid
        req.to_finish = FINISH_ABORT()
        return True

    scheduler._mark_running_request_aborted = mark_running_aborted

    first = scheduler._admin_close_realtime_session({"session_id": "session-1"})
    second = scheduler._admin_close_realtime_session({"session_id": "session-1"})

    assert first["data"]["closed"] is False
    assert first["data"]["deferred"] is True
    assert second["data"]["closed"] is False
    assert second["data"]["deferred"] is True
    assert session_state.close_requested is True
    assert session_state.active_turn_id == "turn-1"
    assert scheduler.session_controller.get("session-1") is req.session
    assert req.session._inflight is True
    assert scheduler._moss_tts_realtime_requests == {"request-1": data}

    req.finished_reason = req.to_finish
    req.to_finish = None
    scheduler.tree_cache.cache_finished_req(req)
    scheduler.stream_output([req])

    assert turn.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert turn.terminal_reason == "aborted"
    assert data.lifecycle_finalized is True
    assert scheduler._moss_tts_realtime_requests == {}
    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    repeated = scheduler._admin_close_realtime_session({"session_id": "session-1"})
    assert repeated["data"]["existed"] is False


def test_idle_ttl_reaps_held_session_but_skips_active_turn() -> None:
    idle_scheduler = _scheduler()
    idle_scheduler._session_idle_ttl_s = 5.0
    idle_state, _, _, _ = _seed_successful_session_turn(idle_scheduler)
    idle_state.last_active_at = 100.0

    assert idle_scheduler._reap_idle_realtime_sessions(105.0, force=True) == 1
    assert idle_scheduler._moss_tts_realtime_sessions == {}
    assert idle_scheduler.session_controller.get("session-1") is None
    assert "session-1" not in idle_scheduler.tree_cache.slots

    active_scheduler = _scheduler()
    active_scheduler._session_idle_ttl_s = 5.0
    active = _request_data(tuple(range(12)), input_done=False)
    active_scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        active,
    )
    active.session_state.last_active_at = 100.0

    assert active_scheduler._reap_idle_realtime_sessions(1000.0, force=True) == 0
    assert active.session_state.active_turn_id == "turn-1"
    assert active_scheduler._moss_tts_realtime_sessions == {
        "session-1": active.session_state
    }
    assert active_scheduler.session_controller.get("session-1") is active.req.session


def test_stop_cleans_waiting_running_parked_and_held_sessions_once(monkeypatch) -> None:
    from sglang.srt.managers import schedule_batch as schedule_batch_module

    monkeypatch.setattr(
        schedule_batch_module,
        "get_serving",
        lambda: SimpleNamespace(strip_thinking_cache=False),
    )
    scheduler = _scheduler()
    _seed_successful_session_turn(
        scheduler,
        request_id="request-held",
        session_id="session-held",
        turn_id="turn-held",
        req_pool_idx=1,
    )
    _, waiting_old_session, _, _ = _seed_successful_session_turn(
        scheduler,
        request_id="request-waiting-old",
        session_id="session-waiting",
        turn_id="turn-waiting-old",
        req_pool_idx=4,
    )
    scheduler.tree_cache.slots.pop(waiting_old_session.session_id)

    def finalize_live(
        *,
        request_id: str,
        session_id: str,
        turn_id: str,
        turn_index: int = 0,
    ) -> MossTTSRealtimeRequestData:
        data = _request_data(
            tuple(range(12)),
            input_done=False,
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
        )
        scheduler._finalize_built_request(
            _payload(tuple(range(12)), request_id=request_id),
            False,
            data,
        )
        data.req._omni_data = data
        return data

    waiting = finalize_live(
        request_id="request-waiting",
        session_id="session-waiting",
        turn_id="turn-waiting",
        turn_index=1,
    )
    assert waiting.ledger_replay is True
    scheduler.waiting_queue.append(waiting.req)

    running = finalize_live(
        request_id="request-running",
        session_id="session-running",
        turn_id="turn-running",
    )
    running.turn_state.observe_audio_frame(tuple(range(1, 17)), generation_step=0)
    running.req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    running.req.req_pool_idx = 2
    running.req.kv_committed_len = len(running.turn_state.ledger.rows)
    running.req.kv = ReqKvInfo(
        kv_allocated_len=len(running.turn_state.ledger.rows),
        swa_evicted_seqlen=0,
    )

    parked = finalize_live(
        request_id="request-parked",
        session_id="session-parked",
        turn_id="turn-parked",
    )
    parked.turn_state.observe_audio_frame(tuple(range(17, 33)), generation_step=0)
    parked.req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    parked.req.req_pool_idx = 3
    parked.req.kv_committed_len = len(parked.turn_state.ledger.rows)
    parked.req.kv = ReqKvInfo(
        kv_allocated_len=len(parked.turn_state.ledger.rows),
        swa_evicted_seqlen=0,
    )
    parked_batch = _AlignedDecodeBatch([parked.req])
    assert scheduler._park_starved_requests(parked_batch) == 1
    assert parked_batch.reqs == []

    scheduler.running_batch = _AlignedDecodeBatch([running.req])
    scheduler._running = True

    scheduler.stop()

    assert scheduler._running is False
    assert waiting.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert running.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert parked.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert scheduler.waiting_queue == []
    assert scheduler.running_batch.reqs == []
    assert scheduler._parked_input == {}
    assert scheduler._moss_tts_realtime_requests == {}
    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler.session_controller.sessions == {}
    assert scheduler.tree_cache.slots == {}
    assert sorted(scheduler.req_to_token_pool.free_slots) == [1, 2, 3]
