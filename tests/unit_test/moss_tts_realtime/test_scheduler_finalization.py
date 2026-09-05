# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from array import array
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    ReqKvInfo,
)
from sglang.srt.session.session_controller import Session
from sglang.srt.session.streaming_session import SessionSlot

from sglang_omni.models.moss_tts_realtime import scheduler as scheduler_module
from sglang_omni.models.moss_tts_realtime.request_builders import (
    build_moss_tts_realtime_row_cache_key,
    build_moss_tts_realtime_row_cache_key_ids,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeRequestData,
    MossTTSRealtimeSessionState,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeTurnState,
)
from sglang_omni.models.moss_tts_realtime.scheduler import MossTTSRealtimeScheduler
from sglang_omni.proto.messages import InputUpdateMessage
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
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
    _set_limits,
)


def _cache_finished_realtime_request(
    scheduler: MossTTSRealtimeScheduler,
    data: MossTTSRealtimeRequestData,
    *,
    audio_eos: bool,
    req_pool_idx: int = 2,
) -> Any:
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    if audio_eos:
        frame = (MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID, *([1] * 15))
        output_id = MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID
        finished_reason = FINISH_MATCHED_TOKEN(output_id)
    else:
        frame = tuple(range(1, 17))
        output_id = MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID
        finished_reason = FINISH_LENGTH(1)
    turn.observe_audio_frame(frame, generation_step=0)
    req.output_ids[:] = array("q", [output_id])
    req.finished_reason = finished_reason
    req._omni_data = data
    req._omni_terminal_claimed = False
    session = req.session
    assert session is not None
    slot = scheduler.tree_cache.slots.get(session.session_id)
    is_first = slot is None
    if slot is None:
        slot = SessionSlot()
        scheduler.tree_cache.slots[session.session_id] = slot
    elif slot.req_pool_idx is not None:
        req_pool_idx = int(slot.req_pool_idx)
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = len(turn.ledger.rows)
    req.kv = ReqKvInfo(
        kv_allocated_len=len(turn.ledger.rows),
        swa_evicted_seqlen=0,
    )
    slot.save_from_req(req, is_first=is_first)
    session.finish_req(req)
    return req


def _terminal_request(
    scheduler: MossTTSRealtimeScheduler,
    *,
    audio_eos: bool,
) -> tuple[Any, MossTTSRealtimeRequestData]:
    data = _request_data(tuple(range(12)), input_done=True)
    scheduler._finalize_built_request(
        _payload(tuple(range(12)), input_done=True),
        False,
        data,
    )
    req = _cache_finished_realtime_request(
        scheduler,
        data,
        audio_eos=audio_eos,
    )
    return req, data


def test_audio_eos_completes_turn_without_materializing_terminal_frame() -> None:
    scheduler = _scheduler()
    req, data = _terminal_request(scheduler, audio_eos=True)

    scheduler.stream_output([req])

    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.COMPLETED
    assert data.turn_state.last_materialized_row is None
    assert data.turn_state.ledger.rows[-1][0] == 11
    assert data.session_state.committed_rows == data.turn_state.ledger.rows
    assert data.session_state.ledger_revision == 1
    assert data.session_state.successful_turns == 1
    assert data.session_state.warm_session_id == "session-1"
    assert data.session_state.warm_kv_length == len(data.turn_state.ledger.rows)
    slot = scheduler.tree_cache.slots["session-1"]
    assert slot.is_holding_kv
    assert slot.kv_committed_len == len(data.turn_state.ledger.rows)
    assert data.lifecycle_finalized is True
    assert scheduler._moss_tts_realtime_requests == {}
    output = scheduler.outbox.get_nowait()
    assert output.type == "result"
    assert output.data == {"phase": "completed"}


def test_ephemeral_offline_turns_release_sessions_after_success() -> None:
    scheduler = _scheduler()

    for index in range(65):
        request_id = f"request-{index}"
        session_id = f"offline:{request_id}"
        data = _request_data(
            tuple(range(12)),
            input_done=True,
            keep_session=False,
            session_id=session_id,
            turn_id=request_id,
        )
        scheduler._finalize_built_request(
            _payload(
                tuple(range(12)),
                input_done=True,
                request_id=request_id,
            ),
            False,
            data,
        )
        req = _cache_finished_realtime_request(
            scheduler,
            data,
            audio_eos=True,
        )

        scheduler.stream_output([req])

        assert data.lifecycle_finalized is True
        assert data.session_state.closed is True
        assert scheduler._moss_tts_realtime_sessions == {}
        assert scheduler.session_controller.sessions == {}
        assert scheduler.tree_cache.slots == {}
        # The session-close marker rides the turn's stream edge ahead of the
        # terminal result so the vocoder releases the codec slot after the
        # final PCM flush.
        marker = scheduler.outbox.get_nowait()
        assert marker.type == "stream"
        assert marker.request_id == request_id
        assert marker.metadata["session_control"] == "close"
        assert marker.metadata["session_id"] == session_id
        output = scheduler.outbox.get_nowait()
        assert output.type == "result"
        assert output.data == {"phase": "completed"}

    assert scheduler._resource_totals["session_ephemeral_close_total"] == 65


def test_terminal_commit_includes_materialized_rows() -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=True)
    scheduler._finalize_built_request(
        _payload(tuple(range(12)), input_done=True),
        False,
        data,
    )
    turn = data.turn_state
    assert turn is not None
    frame = tuple(range(1, 17))
    turn.observe_audio_frame(frame, generation_step=0)
    next_text_token = turn.next_text_token()
    assert next_text_token is not None
    generated_row = (next_text_token, *frame)
    turn.materialize_provisional(
        next_text_token=next_text_token,
        cache_key=build_moss_tts_realtime_row_cache_key(generated_row),
    )
    req = _cache_finished_realtime_request(
        scheduler,
        data,
        audio_eos=True,
    )

    scheduler.stream_output([req])

    assert data.session_state.committed_rows[-1] == generated_row
    assert data.session_state.warm_kv_length == len(data.session_state.committed_rows)


def test_max_length_without_audio_eos_is_error_not_success() -> None:
    scheduler = _scheduler()
    req, data = _terminal_request(scheduler, audio_eos=False)

    scheduler.stream_output([req])

    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED
    assert data.turn_state.terminal_reason == "max_length_without_audio_eos"
    assert data.turn_state.provisional_frame is None
    assert data.session_state.committed_rows == ()
    assert data.session_state.successful_turns == 0
    assert data.session_state.warm_session_id is None
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    output = scheduler.outbox.get_nowait()
    assert output.type == "error"
    assert "without audio EOS" in str(output.data)
    assert scheduler.outbox.empty()


def test_later_turn_max_length_preserves_prior_committed_ledger() -> None:
    scheduler = _scheduler()
    session_state, _, committed_rows, _ = _seed_successful_session_turn(scheduler)
    second = _request_data(
        tuple(range(100, 112)),
        input_done=True,
        turn_id="turn-2",
        turn_index=1,
    )
    scheduler._finalize_built_request(
        _payload(
            tuple(range(100, 112)),
            input_done=True,
            request_id="request-2",
        ),
        False,
        second,
    )
    req = _cache_finished_realtime_request(
        scheduler,
        second,
        audio_eos=False,
    )

    scheduler.stream_output([req])

    assert second.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED
    assert second.turn_state.terminal_reason == "max_length_without_audio_eos"
    assert session_state.committed_rows == committed_rows
    assert session_state.ledger_revision == 1
    assert session_state.successful_turns == 1
    assert session_state.active_turn_id is None
    assert session_state.warm_session_id is None
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    assert scheduler._moss_tts_realtime_requests == {}


def test_model_state_release_failure_still_invalidates_turn_and_session() -> None:
    scheduler = _scheduler()
    req, data = _terminal_request(scheduler, audio_eos=True)
    data.turn_state.assign_model_state_slot(7)
    release_calls = 0

    def release_request(request: Any) -> int:
        nonlocal release_calls
        assert request.request_id == req.rid
        release_calls += 1
        if release_calls == 1:
            raise RuntimeError("injected model-state release failure")
        return data.turn_state.release_model_state_slot(expected_slot_id=7)

    scheduler._model_runner = SimpleNamespace(release_request=release_request)

    scheduler.stream_output([req])

    assert release_calls == 2
    assert data.turn_state.model_state_slot_id is None
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED
    assert data.turn_state.terminal_reason == "model_state_release_failed"
    assert data.lifecycle_finalized is True
    assert scheduler._moss_tts_realtime_requests == {}
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["cleanup_success_total"] == 0
    output = scheduler.outbox.get_nowait()
    assert output.type == "error"
    assert "model-state release failure" in str(output.data)


def test_concurrent_abort_releases_model_state_once() -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    req._omni_data = data
    turn.assign_model_state_slot(7)

    release_started = threading.Event()
    allow_release = threading.Event()
    release_calls = 0

    def release_request(request: Any) -> int:
        nonlocal release_calls
        assert request.request_id == req.rid
        release_calls += 1
        release_started.set()
        assert allow_release.wait(timeout=2.0)
        return turn.release_model_state_slot(expected_slot_id=7)

    scheduler._model_runner = SimpleNamespace(release_request=release_request)
    errors: list[BaseException] = []

    def abort() -> None:
        try:
            scheduler.abort(req.rid, defer_running_cleanup=False)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=abort)
    second = threading.Thread(target=abort)
    first.start()
    assert release_started.wait(timeout=2.0)
    second.start()
    allow_release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert release_calls == 1
    assert turn.model_state_slot_id is None
    assert turn.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert turn.terminal_reason == "aborted"
    assert data.lifecycle_finalized is True
    assert data.observability_finalized is True
    assert data.cleanup_observability_finalized is True
    assert scheduler._resource_totals["cleanup_success_total"] == 1
    assert scheduler._resource_totals["cleanup_error_total"] == 0


def test_failure_finalization_keeps_data_after_backend_detaches_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    assert req is not None
    req._omni_data = data
    original_invalidate = scheduler._invalidate_request_backend

    def invalidate_and_detach(*args: Any, **kwargs: Any) -> None:
        original_invalidate(*args, **kwargs)
        req._omni_data = None

    monkeypatch.setattr(
        scheduler,
        "_invalidate_request_backend",
        invalidate_and_detach,
    )

    scheduler._terminate_live_turn(
        req,
        reason="decode_failed",
        cancelled=False,
    )

    assert req._omni_data is None
    assert data.lifecycle_finalized is True
    assert data.observability_finalized is True
    assert data.cleanup_observability_finalized is True
    assert data.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED


def test_partial_host_commit_failure_restores_old_ledger_and_releases_kv(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    session_state, _, committed_rows, _ = _seed_successful_session_turn(scheduler)
    second = _request_data(
        tuple(range(100, 112)),
        input_done=True,
        turn_id="turn-2",
        turn_index=1,
    )
    scheduler._finalize_built_request(
        _payload(
            tuple(range(100, 112)),
            input_done=True,
            request_id="request-2",
        ),
        False,
        second,
    )
    req = _cache_finished_realtime_request(
        scheduler,
        second,
        audio_eos=True,
    )

    def partial_commit_then_fail(
        owner: MossTTSRealtimeSessionState,
        turn: MossTTSRealtimeTurnState,
        *,
        warm_session_id: str,
    ) -> None:
        owner.committed_rows = turn.ledger.rows
        owner.ledger_revision += 1
        owner.successful_turns += 1
        owner.active_turn_id = None
        owner.warm_session_id = warm_session_id
        owner.warm_kv_length = len(turn.ledger.rows)
        raise RuntimeError("injected partial host commit failure")

    monkeypatch.setattr(
        MossTTSRealtimeSessionState,
        "commit_turn",
        partial_commit_then_fail,
    )

    scheduler.stream_output([req])

    assert session_state.committed_rows == committed_rows
    assert session_state.ledger_revision == 1
    assert session_state.successful_turns == 1
    assert session_state.active_turn_id is None
    assert session_state.warm_session_id is None
    assert session_state.warm_kv_length == 0
    assert second.turn_state.phase is MossTTSRealtimeTurnPhase.FAILED
    assert second.turn_state.terminal_reason == "terminal_commit_failed"
    assert second.lifecycle_finalized is True
    assert second.backend_session_invalidated is True
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    assert scheduler._moss_tts_realtime_requests == {}
    output = scheduler.outbox.get_nowait()
    assert output.type == "error"
    assert "partial host commit failure" in str(output.data)


def test_later_turn_abort_preserves_ledger_and_forces_next_turn_replay(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    session_state, _, committed_rows, _ = _seed_successful_session_turn(scheduler)
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

    scheduler.abort("request-2", defer_running_cleanup=False)

    assert second.turn_state.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert session_state.committed_rows == committed_rows
    assert session_state.ledger_revision == 1
    assert session_state.successful_turns == 1
    assert session_state.active_turn_id is None
    assert session_state.warm_session_id is None
    assert scheduler.session_controller.get("session-1") is None
    assert "session-1" not in scheduler.tree_cache.slots
    assert scheduler._moss_tts_realtime_requests == {}

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
    third = _request_data(
        tuple(range(200, 212)),
        input_done=False,
        turn_id="turn-3",
        turn_index=1,
    )
    scheduler._finalize_built_request(
        _payload(tuple(range(200, 212)), request_id="request-3"),
        False,
        third,
    )

    expected_ids = build_moss_tts_realtime_row_cache_key_ids(third.prompt_rows)
    assert third.ledger_replay is True
    assert submitted_input_ids == [expected_ids]
    assert tuple(third.turn_state.ledger.committed_prefix) == committed_rows


@pytest.mark.parametrize("location", ["waiting", "runnable", "parked"])
def test_turn_timeout_cleans_every_scheduler_owned_location_once(
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    from sglang.srt.managers import schedule_batch as schedule_batch_module

    monkeypatch.setattr(
        schedule_batch_module,
        "get_serving",
        lambda: SimpleNamespace(strip_thinking_cache=False),
    )
    scheduler = _scheduler()
    _set_limits(scheduler, turn_timeout_s=1.0)
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    req._omni_data = data
    if location == "waiting":
        scheduler.waiting_queue.append(req)
    else:
        turn.observe_audio_frame(tuple(range(1, 17)), generation_step=0)
        req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
        req.req_pool_idx = 3
        req.kv_committed_len = len(turn.ledger.rows)
        req.kv = ReqKvInfo(
            kv_allocated_len=len(turn.ledger.rows),
            swa_evicted_seqlen=0,
        )
        batch = _AlignedDecodeBatch([req])
        if location == "runnable":
            scheduler.running_batch = batch
        else:
            assert scheduler._park_starved_requests(batch) == 1
            assert batch.reqs == []
    turn.started_at = 100.0

    assert scheduler._expire_realtime_turns(now=101.0) == 1

    assert turn.phase is MossTTSRealtimeTurnPhase.FAILED
    assert turn.terminal_reason == "turn_timeout"
    assert data.lifecycle_finalized is True
    assert scheduler._moss_tts_realtime_requests == {}
    assert scheduler._parked_input == {}
    assert scheduler.waiting_queue == []
    assert scheduler.running_batch.reqs == []
    assert scheduler.session_controller.sessions == {}
    assert scheduler.tree_cache.slots == {}
    logical = scheduler._logical_reservation_snapshot()
    assert logical["active_session_count"] == 0
    assert logical["held_session_reservations"] == 0
    assert logical["held_kv_token_reservations"] == 0
    totals = scheduler._resource_totals
    assert totals["turn_timeout_total"] == 1
    assert totals["terminal_total"] == 1
    assert totals["cleanup_success_total"] == 1
    assert scheduler._terminal_reason_totals == {"turn_timeout": 1}

    assert scheduler._expire_realtime_turns(now=1000.0) == 0
    assert totals["turn_timeout_total"] == 1
    assert totals["terminal_total"] == 1
    assert totals["cleanup_success_total"] == 1


def test_admin_model_info_reports_queue_session_counter_and_model_state_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scheduler_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    scheduler._model_runner = SimpleNamespace(
        resource_snapshot=lambda: {
            "model_state_capacity": 16,
            "model_state_active_rows": 1,
            "model_state_free_rows": 15,
            "model_state_max_active_rows_observed": 3,
        }
    )
    scheduler._on_input_update(
        "request-1",
        InputUpdateMessage(
            request_id="request-1",
            session_id="session-1",
            turn_id="turn-1",
            seq_no=0,
            token_ids=tuple(range(12)),
            byte_count=7,
        ),
    )
    data = _request_data((), input_done=False)
    scheduler._finalize_built_request(_payload(), False, data)
    data.req._omni_data = data

    active = scheduler._admin_model_info()["data"]
    assert active["active_turn_count"] == 1
    assert active["max_session_rows"] == 4096
    assert active["max_held_kv_tokens"] == 64 * 4096
    assert active["codec_slots"] == 16
    assert active["session_registry_count"] == 1
    assert active["held_session_reservations"] == 1
    assert active["held_kv_token_reservations"] == 45
    assert active["queued_input_tokens"] == 0
    assert active["queued_input_bytes"] == 0
    assert active["queued_input_tokens_high_water"] == 12
    assert active["queued_input_bytes_high_water"] == 7
    assert active["model_state_capacity"] == 16
    assert active["model_state_active_rows"] == 1
    assert active["model_state_free_rows"] == 15
    assert active["model_state_max_active_rows_observed"] == 3
    assert active["resource_totals"]["input_update_accepted_total"] == 1
    assert active["resource_totals"]["host_session_open_total"] == 1
    assert active["resource_totals"]["physical_session_open_total"] == 1
    assert active["resource_totals"]["turn_admitted_total"] == 1

    scheduler.abort("request-1", defer_running_cleanup=False)
    scheduler._admin_close_realtime_session({"session_id": "session-1"})
    terminal = scheduler._admin_model_info()["data"]
    assert terminal["active_turn_count"] == 0
    assert terminal["session_registry_count"] == 0
    assert terminal["physical_held_session_count"] == 0
    assert terminal["physical_held_kv_tokens"] == 0
    assert terminal["terminal_reason_totals"] == {"aborted": 1}
    assert terminal["cleanup_success_total"] == 1
    assert terminal["cleanup_error_total"] == 0
    assert terminal["session_close_total"] == 1
    assert {event["event_name"] for event in events} >= {
        "moss_tts_realtime_input_update_accepted",
        "moss_tts_realtime_physical_session_open",
        "moss_tts_realtime_turn_admitted",
        "moss_tts_realtime_turn_terminal",
        "moss_tts_realtime_host_session_close",
    }


def test_parked_cleanup_failure_records_error_without_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    _set_limits(scheduler, turn_timeout_s=1.0)
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    req._omni_data = data
    turn.observe_audio_frame(tuple(range(1, 17)), generation_step=0)
    req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    req.req_pool_idx = 3
    req.kv_committed_len = len(turn.ledger.rows)
    req.kv = ReqKvInfo(
        kv_allocated_len=len(turn.ledger.rows),
        swa_evicted_seqlen=0,
    )
    batch = _AlignedDecodeBatch([req])
    assert scheduler._park_starved_requests(batch) == 1
    turn.started_at = 100.0
    release_calls = 0

    def fail_release(released_req: Any) -> None:
        nonlocal release_calls
        assert released_req is req
        release_calls += 1
        raise RuntimeError("injected parked KV release failure")

    monkeypatch.setattr(scheduler, "_release_request_kv_cache", fail_release)

    assert scheduler._expire_realtime_turns(now=101.0) == 1

    assert turn.phase is MossTTSRealtimeTurnPhase.FAILED
    assert turn.terminal_reason == "turn_timeout"
    assert data.observability_finalized is True
    assert data.cleanup_observability_finalized is True
    assert scheduler._resource_totals["terminal_total"] == 1
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["cleanup_success_total"] == 0
    assert scheduler._resource_totals["kv_release_retry_total"] == 1
    assert scheduler._resource_totals["kv_release_retry_error_total"] == 1
    assert scheduler._resource_totals["kv_release_retry_success_total"] == 0
    assert release_calls == 2
    assert req.req_pool_idx == 3
    assert scheduler._expire_realtime_turns(now=1000.0) == 0
    assert scheduler._resource_totals["cleanup_error_total"] == 1


def test_parked_cleanup_recovers_after_transient_kv_release_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.managers import schedule_batch as schedule_batch_module

    monkeypatch.setattr(
        schedule_batch_module,
        "get_serving",
        lambda: SimpleNamespace(strip_thinking_cache=False),
    )
    scheduler = _scheduler()
    _set_limits(scheduler, turn_timeout_s=1.0)
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    req._omni_data = data
    turn.observe_audio_frame(tuple(range(1, 17)), generation_step=0)
    req.output_ids.append(MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID)
    req.req_pool_idx = 3
    req.kv_committed_len = len(turn.ledger.rows)
    req.kv = ReqKvInfo(
        kv_allocated_len=len(turn.ledger.rows),
        swa_evicted_seqlen=0,
    )
    batch = _AlignedDecodeBatch([req])
    assert scheduler._park_starved_requests(batch) == 1
    turn.started_at = 100.0
    release_calls = 0
    original_release = scheduler._release_request_kv_cache

    def release_once_then_succeed(released_req: Any) -> None:
        nonlocal release_calls
        assert released_req is req
        release_calls += 1
        if release_calls == 1:
            raise RuntimeError("injected transient parked KV release failure")
        original_release(released_req)

    monkeypatch.setattr(
        scheduler,
        "_release_request_kv_cache",
        release_once_then_succeed,
    )

    assert scheduler._expire_realtime_turns(now=101.0) == 1

    assert release_calls == 2
    assert req.req_pool_idx is None
    assert scheduler.session_controller.sessions == {}
    assert scheduler.tree_cache.slots == {}
    assert set(scheduler._moss_tts_realtime_sessions) == {"session-1"}
    close_result = scheduler._admin_close_realtime_session({"session_id": "session-1"})
    assert close_result["data"]["closed"] is True
    assert scheduler._moss_tts_realtime_sessions == {}
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["cleanup_success_total"] == 0
    assert scheduler._resource_totals["kv_release_retry_total"] == 1
    assert scheduler._resource_totals["kv_release_retry_success_total"] == 1
    assert scheduler._resource_totals["kv_release_retry_error_total"] == 0


def test_immediate_abort_kv_release_failure_records_error_without_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    req._omni_data = data
    req.req_pool_idx = 3
    release_calls = 0

    def fail_release(released_req: Any) -> None:
        nonlocal release_calls
        assert released_req is req
        release_calls += 1
        raise RuntimeError("injected immediate KV release failure")

    monkeypatch.setattr(scheduler, "_release_request_kv_cache", fail_release)

    scheduler.abort(req.rid, defer_running_cleanup=False)

    assert turn.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert turn.terminal_reason == "aborted"
    assert data.lifecycle_finalized is True
    assert data.observability_finalized is True
    assert data.cleanup_observability_finalized is True
    assert scheduler._resource_totals["terminal_total"] == 1
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["cleanup_success_total"] == 0
    assert scheduler._resource_totals["kv_release_retry_total"] == 1
    assert scheduler._resource_totals["kv_release_retry_error_total"] == 1
    assert scheduler._resource_totals["kv_release_retry_success_total"] == 0
    assert release_calls == 2


def test_immediate_abort_reap_failure_records_error_without_false_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    data = _request_data(tuple(range(12)), input_done=False)
    scheduler._finalize_built_request(
        _payload(tuple(range(12))),
        False,
        data,
    )
    req = data.req
    turn = data.turn_state
    assert req is not None
    assert turn is not None
    req._omni_data = data
    reap_calls = 0

    def fail_reap(*args: Any, **kwargs: Any) -> Any:
        nonlocal reap_calls
        reap_calls += 1
        raise RuntimeError("injected immediate session reap failure")

    monkeypatch.setattr(scheduler.session_controller, "maybe_reap", fail_reap)

    scheduler.abort(req.rid, defer_running_cleanup=False)

    assert reap_calls == 1
    assert turn.phase is MossTTSRealtimeTurnPhase.CANCELLED
    assert turn.terminal_reason == "aborted"
    assert data.lifecycle_finalized is True
    assert data.observability_finalized is True
    assert data.cleanup_observability_finalized is True
    assert scheduler._resource_totals["terminal_total"] == 1
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["cleanup_success_total"] == 0


def test_periodic_session_reap_failure_is_contained_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler()
    reap_calls = 0
    original_reap = scheduler.session_controller.maybe_reap

    def fail_once_then_reap(*args: Any, **kwargs: Any) -> Any:
        nonlocal reap_calls
        reap_calls += 1
        if reap_calls == 1:
            raise RuntimeError("injected periodic session reap failure")
        return original_reap(*args, **kwargs)

    monkeypatch.setattr(
        scheduler.session_controller,
        "maybe_reap",
        fail_once_then_reap,
    )

    scheduler.process_input_requests([])

    assert reap_calls == 1
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["physical_session_reap_error_total"] == 1
    assert scheduler._resource_totals["physical_session_reap_recovery_total"] == 0

    scheduler.process_input_requests([])

    assert reap_calls == 2
    assert scheduler._resource_totals["cleanup_error_total"] == 1
    assert scheduler._resource_totals["physical_session_reap_error_total"] == 1
    assert scheduler._resource_totals["physical_session_reap_recovery_total"] == 1
