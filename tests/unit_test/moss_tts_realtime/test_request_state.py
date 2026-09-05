# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeInputUpdate,
    MossTTSRealtimeLedgerDisposition,
    MossTTSRealtimePendingInput,
    MossTTSRealtimeProvisionalFrame,
    MossTTSRealtimeRequestData,
    MossTTSRealtimeSessionConfig,
    MossTTSRealtimeSessionState,
    MossTTSRealtimeTurnLedger,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeUpdateDisposition,
    MossTTSRealtimeVoiceReference,
    apply_moss_tts_realtime_input_update,
    normalize_moss_tts_realtime_row,
)
from sglang_omni.proto.messages import InputUpdateMessage
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG
from tests.unit_test.moss_tts_realtime.runtime_config import (
    TEXT_PAD_TOKEN_ID as MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID,
)


def _frame(first_code: int = 7) -> tuple[int, ...]:
    return (first_code, *range(1, 16))


def _row(text_token: int = 1, first_code: int = 7) -> tuple[int, ...]:
    return (text_token, *_frame(first_code))


def test_pending_input_is_ordered_bounded_and_retry_safe() -> None:
    pending = MossTTSRealtimePendingInput(
        max_tokens=3,
        max_bytes=10,
        max_updates=2,
    )
    first = MossTTSRealtimeInputUpdate(
        seq_no=0,
        token_ids=(10, 11),
        byte_count=4,
    )

    assert pending.append(first) is MossTTSRealtimeUpdateDisposition.ACCEPTED
    assert pending.append(first) is MossTTSRealtimeUpdateDisposition.DUPLICATE
    assert len(pending) == 2
    assert pending.pending_bytes == 4

    with pytest.raises(ValueError, match="different content"):
        pending.append(
            MossTTSRealtimeInputUpdate(seq_no=0, token_ids=(99,), byte_count=4)
        )
    with pytest.raises(ValueError, match="expected 1, got 2"):
        pending.append(MossTTSRealtimeInputUpdate(seq_no=2, token_ids=(12,)))
    with pytest.raises(ValueError, match="pending token limit"):
        pending.append(MossTTSRealtimeInputUpdate(seq_no=1, token_ids=(12, 13)))

    final = MossTTSRealtimeInputUpdate(
        seq_no=1,
        token_ids=(12,),
        byte_count=2,
        input_done=True,
    )
    assert pending.append(final) is MossTTSRealtimeUpdateDisposition.ACCEPTED
    assert pending.next_seq_no == 2
    assert pending.input_done is True
    assert pending.max_pending_tokens_observed == 3
    assert pending.max_pending_bytes_observed == 6
    assert pending.pop_tokens(2) == (10, 11)
    assert pending.pending_bytes == 2
    assert pending.popleft() == 12
    assert pending.pending_bytes == 0

    assert pending.append(final) is MossTTSRealtimeUpdateDisposition.DUPLICATE
    with pytest.raises(RuntimeError, match="after input_done"):
        pending.append(MossTTSRealtimeInputUpdate(seq_no=2, token_ids=(13,)))


def test_turn_records_monotonic_start_time_at_session_admission() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )

    before = time.monotonic()
    turn = session.begin_turn(turn_id="turn", request_id="request")
    after = time.monotonic()

    assert before <= turn.started_at <= after


def test_pending_input_accepts_ordered_tokenless_text_delta_noop() -> None:
    pending = MossTTSRealtimePendingInput(
        max_tokens=3,
        max_bytes=10,
        max_updates=2,
    )
    noop = MossTTSRealtimeInputUpdate(seq_no=0)

    assert pending.append(noop) is MossTTSRealtimeUpdateDisposition.ACCEPTED
    assert pending.append(noop) is MossTTSRealtimeUpdateDisposition.DUPLICATE
    assert pending.next_seq_no == 1
    assert pending.accepted_update_count == 1
    assert len(pending) == 0
    assert pending.pending_bytes == 0


def test_pending_input_rejects_empty_done_at_turn_boundary() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )
    turn = session.begin_turn(turn_id="turn", request_id="request")

    with pytest.raises(ValueError, match="empty realtime turn"):
        turn.append_input_update(MossTTSRealtimeInputUpdate(seq_no=0, input_done=True))

    assert turn.pending_input.next_seq_no == 0
    assert turn.pending_input.input_done is False


def test_wire_input_update_adapter_validates_identity_and_updates_turn() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )
    turn = session.begin_turn(turn_id="turn", request_id="request")
    req_data = MossTTSRealtimeRequestData(turn_state=turn)
    message = InputUpdateMessage(
        request_id="request",
        session_id="session",
        turn_id="turn",
        seq_no=0,
        token_ids=(10, 11),
        byte_count=4,
    )

    assert (
        apply_moss_tts_realtime_input_update(req_data, message)
        is MossTTSRealtimeUpdateDisposition.ACCEPTED
    )
    assert (
        apply_moss_tts_realtime_input_update(req_data, message)
        is MossTTSRealtimeUpdateDisposition.DUPLICATE
    )
    assert turn.pending_input.pop_tokens(2) == (10, 11)

    with pytest.raises(ValueError, match="session identity mismatch"):
        apply_moss_tts_realtime_input_update(
            req_data,
            InputUpdateMessage(
                request_id="request",
                session_id="other-session",
                turn_id="turn",
                seq_no=1,
                token_ids=(12,),
            ),
        )


def test_wire_input_update_adapter_requires_live_moss_request_data() -> None:
    message = InputUpdateMessage(
        request_id="request",
        session_id="session",
        turn_id="turn",
        seq_no=0,
        token_ids=(10,),
    )

    with pytest.raises(TypeError, match="MossTTSRealtimeRequestData"):
        apply_moss_tts_realtime_input_update(object(), message)
    with pytest.raises(RuntimeError, match="no live turn state"):
        apply_moss_tts_realtime_input_update(
            MossTTSRealtimeRequestData(),
            message,
        )


def test_turn_waits_for_twelve_tokens_then_parks_and_wakes_same_request() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )
    turn = session.begin_turn(turn_id="turn", request_id="request")

    turn.append_input_update(
        MossTTSRealtimeInputUpdate(seq_no=0, token_ids=tuple(range(11)))
    )
    assert turn.ready_for_prefill is False
    assert turn.phase is MossTTSRealtimeTurnPhase.WAITING_PREFILL

    turn.append_input_update(MossTTSRealtimeInputUpdate(seq_no=1, token_ids=(11,)))
    assert turn.ready_for_prefill is True
    assert turn.take_prefill_tokens() == tuple(range(12))
    assert turn.phase is MossTTSRealtimeTurnPhase.RUNNING

    turn.observe_audio_frame(_frame(), generation_step=0)
    assert turn.next_text_token() is None
    assert turn.phase is MossTTSRealtimeTurnPhase.PARKED_INPUT

    turn.append_input_update(MossTTSRealtimeInputUpdate(seq_no=2, token_ids=(42,)))
    assert turn.phase is MossTTSRealtimeTurnPhase.RUNNING
    next_text = turn.next_text_token()
    assert next_text == 42

    materialized = turn.materialize_provisional(
        next_text_token=next_text,
        cache_key=123,
    )
    assert materialized.row == (42, *_frame())
    assert turn.ledger.rows == (materialized.row,)


def test_terminal_audio_eos_is_not_materialized_or_committed_as_input() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )
    turn = session.begin_turn(turn_id="turn", request_id="request")
    turn.append_input_update(
        MossTTSRealtimeInputUpdate(
            seq_no=0,
            token_ids=tuple(range(12)),
            input_done=True,
        )
    )
    turn.take_prefill_tokens()
    turn.assign_model_state_slot(3)
    turn.assign_codec_slot(5)

    turn.observe_audio_frame(_frame(), generation_step=0)
    next_text = turn.next_text_token()
    assert next_text == MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID
    assert turn.phase is MossTTSRealtimeTurnPhase.DRAINING
    materialized = turn.materialize_provisional(
        next_text_token=next_text,
        cache_key=456,
    )

    eos_frame = (MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID, *([0] * 15))
    terminal = turn.observe_audio_frame(eos_frame, generation_step=1)
    assert terminal.is_audio_eos is True
    assert turn.provisional_frame is None

    with pytest.raises(RuntimeError, match="slots must be released"):
        turn.complete(committed_kv_length=1)

    assert turn.release_model_state_slot(expected_slot_id=3) == 3
    assert turn.release_codec_slot(expected_slot_id=5) == 5
    with pytest.raises(ValueError, match="ledger/KV length mismatch"):
        turn.complete(committed_kv_length=2)

    turn.complete(committed_kv_length=1)
    session.commit_turn(
        turn,
        warm_session_id="warm-session",
    )

    assert turn.phase is MossTTSRealtimeTurnPhase.COMPLETED
    assert turn.ledger.disposition is MossTTSRealtimeLedgerDisposition.COMMITTED
    assert session.committed_rows == (materialized.row,)
    assert all(
        row[1] != MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID for row in session.committed_rows
    )
    assert session.warm_kv_length == 1
    assert session.needs_ledger_replay is False

    session.release_warm_session()
    assert session.needs_ledger_replay is True


def test_cancel_rolls_back_working_suffix_and_invalidates_warm_session() -> None:
    committed = (_row(),)
    session = MossTTSRealtimeSessionState(
        session_id="session",
        model_config=MODEL_CONFIG,
        committed_rows=committed,
        ledger_revision=1,
        successful_turns=1,
        warm_session_id="warm-session",
        warm_kv_length=1,
    )
    turn = session.begin_turn(turn_id="turn-2", request_id="request-2")
    turn.ledger.append_row(_row(text_token=2, first_code=8))

    turn.cancel("client_cancelled")
    session.abort_turn(turn)

    assert turn.ledger.disposition is MossTTSRealtimeLedgerDisposition.ROLLED_BACK
    assert turn.ledger.rows == committed
    assert session.committed_rows == committed
    assert session.active_turn_id is None
    assert session.warm_session_id is None
    assert session.needs_ledger_replay is True


def test_model_eos_before_input_done_closes_turn_without_synthetic_done() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )
    turn = session.begin_turn(turn_id="turn", request_id="request")
    turn.append_input_update(
        MossTTSRealtimeInputUpdate(seq_no=0, token_ids=tuple(range(12)))
    )
    turn.take_prefill_tokens()
    turn.observe_audio_frame(
        (MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID, *([0] * 15)),
        generation_step=0,
    )

    turn.complete(committed_kv_length=0)

    assert turn.phase is MossTTSRealtimeTurnPhase.COMPLETED
    assert turn.terminal_reason == "model_eos"
    assert turn.pending_input.input_done is False
    assert turn.pending_input.closed is True
    with pytest.raises(RuntimeError, match="terminal turn"):
        turn.append_input_update(MossTTSRealtimeInputUpdate(seq_no=1, token_ids=(99,)))


def test_voice_config_is_frozen_and_cannot_change_after_success() -> None:
    config = MossTTSRealtimeSessionConfig(
        voice=MossTTSRealtimeVoiceReference(
            voice="speaker-a",
            reference_audio="ref.wav",
            reference_text="reference",
        )
    )
    restored = MossTTSRealtimeSessionConfig.model_validate(config.model_dump())
    assert restored == config

    with pytest.raises(ValidationError, match="frozen"):
        config.voice.voice = "speaker-b"

    session = MossTTSRealtimeSessionState(
        session_id="session",
        model_config=MODEL_CONFIG,
        config=config,
        successful_turns=1,
    )
    with pytest.raises(RuntimeError, match="immutable"):
        session.reconfigure(MossTTSRealtimeSessionConfig())


def test_row_and_provisional_frame_validation() -> None:
    assert normalize_moss_tts_realtime_row(_row(), model_config=MODEL_CONFIG) == _row()
    with pytest.raises(ValueError, match="17 columns"):
        normalize_moss_tts_realtime_row((1, 2), model_config=MODEL_CONFIG)
    with pytest.raises(ValueError, match="audio vocabulary"):
        normalize_moss_tts_realtime_row((1, *([1027] * 16)), model_config=MODEL_CONFIG)

    terminal = MossTTSRealtimeProvisionalFrame(
        model_config=MODEL_CONFIG,
        audio_codes=(MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID, *([0] * 15)),
        generation_step=1,
    )
    with pytest.raises(ValueError, match="must remain unmaterialized"):
        terminal.materialize(next_text_token=1, cache_key=1)


def test_slot_ownership_and_mutable_request_defaults_are_strict() -> None:
    session = MossTTSRealtimeSessionState(
        session_id="session", model_config=MODEL_CONFIG
    )
    turn = session.begin_turn(turn_id="turn", request_id="request")
    turn.assign_codec_slot(0)
    assert turn.release_codec_slot() == 0
    with pytest.raises(RuntimeError, match="does not own a codec slot"):
        turn.release_codec_slot()

    first = MossTTSRealtimeRequestData()
    second = MossTTSRealtimeRequestData()
    first.output_ids.append(1)
    first.state.generation_kwargs["seed"] = 7

    assert second.output_ids == []
    assert second.state.generation_kwargs == {}


def test_ledger_extend_is_atomic_on_invalid_row() -> None:
    ledger = MossTTSRealtimeTurnLedger(model_config=MODEL_CONFIG)

    with pytest.raises(ValueError, match="17 columns"):
        ledger.extend_rows([_row(), (1, 2)])

    assert ledger.rows == ()
