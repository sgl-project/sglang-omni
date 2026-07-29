# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from collections import OrderedDict
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Importing SGLang on a macOS CPU host otherwise initializes TorchInductor's
# Triton decorators.  These state-machine tests do not compile any kernels.
_original_torch_compile = torch.compile


def _identity_compile(model=None, *args, **kwargs):
    del args, kwargs
    if model is None:
        return lambda fn: fn
    return model


torch.compile = _identity_compile
try:
    from sglang_omni.models.minicpmo_4_5 import scheduler as scheduler_module
    from sglang_omni.models.minicpmo_4_5.protocol import SessionCommand
    from sglang_omni.models.minicpmo_4_5.scheduler import (
        MiniCPMO45Scheduler,
        _BoundedSessionInbox,
    )
    from sglang_omni.models.minicpmo_4_5.state import (
        MiniCPMOSessionState,
        MiniCPMOSpecialTokens,
        MiniCPMOUnitRequestData,
    )
    from sglang_omni.proto import OmniRequest, StagePayload
    from sglang_omni.scheduling.messages import IncomingMessage
finally:
    torch.compile = _original_torch_compile


def _tokens() -> MiniCPMOSpecialTokens:
    return MiniCPMOSpecialTokens(
        unit_start=1,
        unit_end=2,
        image_start=3,
        image_end=4,
        slice_start=5,
        slice_end=6,
        listen=7,
        speak=8,
        tts_bos=9,
        tts_eos=10,
        tts_pad=11,
        chunk_eos=12,
        chunk_tts_eos=13,
        turn_eos=14,
        media_placeholder=0,
    )


def _state() -> MiniCPMOSessionState:
    return MiniCPMOSessionState(
        request_id="outer-1",
        session_id="session-1",
        generation=1,
        response_epoch=0,
        next_input_seq=1,
        system_prompt="Streaming Omni Conversation.",
    )


def _bare_scheduler(state: MiniCPMOSessionState) -> MiniCPMO45Scheduler:
    scheduler = object.__new__(MiniCPMO45Scheduler)
    scheduler._state_lock = threading.RLock()
    scheduler._state = state
    scheduler._duplex_sampling = {
        "force_listen_count": 0,
        "generate_audio": False,
        "text_repetition_window_size": 512,
    }
    scheduler._session_ttl_s = 300.0
    scheduler._unit_by_rid = {}
    scheduler._failed_outer_requests = OrderedDict()
    scheduler._poisoned_error = None
    scheduler._external_abort_callback = None
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._aborted_request_ids = set()
    scheduler.outbox = Queue()
    scheduler.inbox = _BoundedSessionInbox(4, 16)
    scheduler.special_tokens = _tokens()
    scheduler._tts_runtime = None
    scheduler._perception = SimpleNamespace(close_session=lambda _session_id: None)
    scheduler._tokenizer = SimpleNamespace(decode=lambda _tokens, **_kwargs: "")
    return scheduler


class _RecordingTTS:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.interrupt_calls: list[tuple[str, bool]] = []
        self.close_calls: list[str] = []
        self.close_error = close_error

    def interrupt_session(self, session_id: str, *, flush: bool):
        self.interrupt_calls.append((session_id, flush))
        return np.empty(0, dtype=np.float32)

    def close_session(self, session_id: str) -> None:
        self.close_calls.append(session_id)
        if self.close_error is not None:
            raise self.close_error


class _BlockingTTS(_RecordingTTS):
    def __init__(self) -> None:
        super().__init__()
        self.synthesize_entered = threading.Event()
        self.release_synthesize = threading.Event()
        self.closed = threading.Event()

    def synthesize(
        self,
        session_id: str,
        token_ids: list[int],
        hidden,
        *,
        end_of_turn: bool,
    ):
        del session_id, token_ids, hidden, end_of_turn
        self.synthesize_entered.set()
        assert self.release_synthesize.wait(timeout=2.0)
        return SimpleNamespace(
            waveform=np.empty(0, dtype=np.float32),
            sample_rate=24_000,
        )

    def close_session(self, session_id: str) -> None:
        assert self.release_synthesize.is_set()
        super().close_session(session_id)
        self.closed.set()


class _SessionController:
    def __init__(self, session_id: str, *, retain_after_close: bool = False) -> None:
        self.live = {session_id: object()}
        self.close_calls: list[str] = []
        self.retain_after_close = retain_after_close

    def close(self, request) -> None:
        self.close_calls.append(request.session_id)
        if not self.retain_after_close:
            self.live.pop(request.session_id, None)

    def get(self, session_id: str):
        return self.live.get(session_id)


def test_overload_signal_does_not_overtake_session_open() -> None:
    inbox = _BoundedSessionInbox(max_pending_units=1, max_pending_commands=4)
    opened = IncomingMessage("outer-1", "new_request", data=None)

    def append(input_seq: int) -> IncomingMessage:
        return IncomingMessage(
            "outer-1",
            "session_command",
            data={
                "session_id": "session-1",
                "generation": 7,
                "input_seq": input_seq,
                "response_epoch": 0,
                "command": "append",
                "data": {},
            },
        )

    inbox.put(opened)
    inbox.put(append(1))
    inbox.put(append(2))

    assert inbox.get_nowait() is opened
    overload = inbox.get_nowait()
    assert isinstance(overload, scheduler_module._OverloadSignal)
    assert overload.generation == 7
    assert overload.limit_name == "max_pending_units"


def test_malformed_open_still_preserves_raw_generation_for_terminal() -> None:
    payload = StagePayload(
        request_id="outer-1",
        request=OmniRequest(
            inputs={},
            metadata={
                "duplex_session": {
                    "session_id": "outer-1",
                    "generation": 9,
                    "config": "not-a-mapping",
                }
            },
        ),
        data=None,
    )
    message = IncomingMessage("outer-1", "new_request", data=payload)

    with pytest.raises(scheduler_module.DuplexProtocolError):
        scheduler_module.extract_open_session(payload)
    assert scheduler_module._message_generation(message) == 9


def test_force_listen_hard_cuts_tts_before_building_next_unit() -> None:
    state = _state()
    state.current_turn_ended = False
    state.generated_unit_count = 10
    scheduler = _bare_scheduler(state)
    tts = _RecordingTTS()
    scheduler._tts_runtime = tts
    command = SessionCommand(
        session_id=state.session_id,
        generation=state.generation,
        input_seq=1,
        response_epoch=0,
        command="append",
        data={
            "audio_pcm16": b"\x00\x00" * 16_000,
            "sample_rate": 16_000,
            "force_listen": True,
        },
    )

    payload = scheduler._accept_append(state, command)

    assert tts.interrupt_calls == [(state.session_id, False)]
    assert payload.data.forced_listen is True
    assert payload.data.close_speaking_turn is True
    assert state.inflight_rid == "session-1:g1:u1"
    assert state.next_input_seq == 2


def test_invalid_unit_terminator_is_not_committed() -> None:
    state = _state()
    state.inflight_rid = "session-1:g1:u1"
    scheduler = _bare_scheduler(state)
    data = MiniCPMOUnitRequestData(
        session_state=state,
        input_seq=1,
        response_epoch=0,
        output_ids=[999],
        finish_reason="length",
    )
    data.req = SimpleNamespace(rid=state.inflight_rid)

    with pytest.raises(RuntimeError, match="valid chunk terminator"):
        scheduler._complete_unit(state, data)

    assert state.inflight_rid == "session-1:g1:u1"
    assert state.generated_unit_count == 0
    assert state.unit_journal == []


def test_completed_speaking_turn_emits_ordered_response_boundary() -> None:
    state = _state()
    state.inflight_rid = "session-1:g1:u1"
    state.inflight_input_seq = 1
    state.inflight_response_epoch = 0
    scheduler = _bare_scheduler(state)
    data = MiniCPMOUnitRequestData(
        session_state=state,
        input_seq=1,
        response_epoch=0,
        output_ids=[scheduler.special_tokens.chunk_eos],
        generated_unit_ids=[42],
        tts_pairs=[(scheduler.special_tokens.turn_eos, torch.zeros(4), True)],
        input_mode="audio",
    )
    data.req = SimpleNamespace(rid=state.inflight_rid)

    scheduler._complete_unit(state, data)

    turn_done = scheduler.outbox.get_nowait()
    input_processed = scheduler.outbox.get_nowait()
    assert turn_done.data["type"] == "response.output.done"
    assert turn_done.data["session_id"] == state.session_id
    assert turn_done.data["generation"] == state.generation
    assert turn_done.data["input_seq"] == 1
    assert turn_done.data["response_epoch"] == 0
    assert isinstance(turn_done.data["output_seq"], int)
    assert turn_done.data["output_seq"] > 0
    assert turn_done.data["reason"] == "model_turn_end"
    assert input_processed.data["type"] == "session.input_processed"
    assert turn_done.data["output_seq"] < input_processed.data["output_seq"]


def test_aborted_running_unit_cleans_session_only_at_terminal_callback() -> None:
    state = _state()
    state.inflight_rid = "session-1:g1:u1"
    state.inflight_input_seq = 1
    state.inflight_response_epoch = 0
    state.aborted = True
    scheduler = _bare_scheduler(state)
    cleaned: list[MiniCPMOSessionState] = []
    scheduler._cleanup_state = lambda target: cleaned.append(target) or []

    assert cleaned == []
    scheduler._on_internal_abort_terminal(state.inflight_rid)

    assert cleaned == [state]
    assert state.inflight_rid is None
    assert state.inflight_input_seq is None
    assert state.inflight_response_epoch is None


def test_failure_terminal_waits_until_running_unit_cleanup(monkeypatch) -> None:
    state = _state()
    state.inflight_rid = "session-1:g1:u1"
    state.inflight_input_seq = 1
    state.inflight_response_epoch = 0
    state.next_input_seq = 2
    scheduler = _bare_scheduler(state)
    scheduler.session_controller = _SessionController(state.session_id)
    aborts: list[str] = []

    def record_abort(_scheduler, request_id: str, **_kwargs) -> None:
        aborts.append(request_id)

    monkeypatch.setattr(scheduler_module.OmniScheduler, "abort", record_abort)
    scheduler._fail_session(state.request_id, RuntimeError("generate failed"))

    assert aborts == [state.inflight_rid]
    assert scheduler.outbox.empty()

    scheduler._on_internal_abort_terminal(state.inflight_rid)
    terminal = scheduler.outbox.get_nowait()
    assert terminal.type == "error"
    assert terminal.metadata["terminal_event"]["error"] == "generate failed"
    assert terminal.metadata["terminal_event"]["input_seq"] == 1
    assert scheduler.outbox.empty()
    assert scheduler._state is None


def test_idle_ttl_never_closes_an_inflight_unit(monkeypatch) -> None:
    state = _state()
    state.last_activity = 10.0
    state.inflight_rid = "session-1:g1:u1"
    scheduler = _bare_scheduler(state)
    scheduler._session_ttl_s = 5.0
    closed: list[int] = []
    scheduler._finish_close = lambda _state, *, input_seq: closed.append(input_seq)
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: 20.0)

    scheduler._expire_local_session()
    assert closed == []
    assert state.closing is False

    state.inflight_rid = None
    scheduler._expire_local_session()
    assert closed == [0]
    assert state.close_reason == "session_ttl"


def test_outer_abort_waits_for_unit_completion_side_model_use() -> None:
    state = _state()
    state.inflight_rid = "session-1:g1:u1"
    state.inflight_input_seq = 1
    state.inflight_response_epoch = 0
    state.next_input_seq = 2
    scheduler = _bare_scheduler(state)
    scheduler._duplex_sampling["generate_audio"] = True
    scheduler.session_controller = _SessionController(state.session_id)
    tts = _BlockingTTS()
    scheduler._tts_runtime = tts
    data = MiniCPMOUnitRequestData(
        session_state=state,
        input_seq=1,
        response_epoch=0,
        output_ids=[scheduler.special_tokens.chunk_eos],
    )
    data.req = SimpleNamespace(rid=state.inflight_rid)
    errors: list[BaseException] = []

    def complete() -> None:
        try:
            scheduler._complete_unit(state, data)
        except BaseException as exc:
            errors.append(exc)

    abort_started = threading.Event()

    def abort() -> None:
        abort_started.set()
        try:
            scheduler.abort(state.request_id)
        except BaseException as exc:
            errors.append(exc)

    complete_thread = threading.Thread(target=complete)
    complete_thread.start()
    assert tts.synthesize_entered.wait(timeout=2.0)
    abort_thread = threading.Thread(target=abort)
    abort_thread.start()
    assert abort_started.wait(timeout=2.0)
    assert not tts.closed.is_set()

    tts.release_synthesize.set()
    complete_thread.join(timeout=2.0)
    abort_thread.join(timeout=2.0)

    assert not complete_thread.is_alive()
    assert not abort_thread.is_alive()
    assert errors == []
    assert tts.closed.is_set()
    assert scheduler._state is None


def test_cleanup_failure_emits_session_error_and_poisons_actor() -> None:
    state = _state()
    state.closing = True
    state.close_reason = "client_close"
    scheduler = _bare_scheduler(state)
    scheduler._tts_runtime = _RecordingTTS(close_error=RuntimeError("tts close"))
    perception_calls: list[str] = []
    scheduler._perception = SimpleNamespace(
        close_session=lambda session_id: perception_calls.append(session_id)
    )
    controller = _SessionController(state.session_id)
    scheduler.session_controller = controller

    scheduler._finish_close(state, input_seq=1)

    terminal = scheduler.outbox.get_nowait()
    assert terminal.type == "error"
    assert "tts close" in str(terminal.data)
    assert terminal.metadata["terminal_event"]["type"] == "session.error"
    assert scheduler._poisoned_error is not None
    assert scheduler._state is None
    assert perception_calls == [state.session_id]
    assert controller.close_calls == [state.session_id]


def test_rejected_second_session_uses_its_generation_and_one_terminal() -> None:
    state = _state()
    scheduler = _bare_scheduler(state)

    scheduler._fail_session("second", RuntimeError("busy"), generation=9)

    terminal = scheduler.outbox.get_nowait()
    assert terminal.type == "error"
    assert terminal.metadata["generation"] == 9
    assert terminal.metadata["terminal_event"]["generation"] == 9
    assert scheduler.outbox.empty()
    assert scheduler._state is state


def test_stale_generation_failure_does_not_abort_replacement_session() -> None:
    state = _state()
    state.generation = 2
    scheduler = _bare_scheduler(state)

    scheduler._fail_session(
        state.request_id,
        RuntimeError("late generation one command"),
        generation=1,
    )

    assert scheduler._state is state
    assert state.aborted is False
    assert scheduler._failed_outer_requests == OrderedDict()
    assert scheduler.outbox.empty()


def test_failed_request_tombstones_are_bounded() -> None:
    state = _state()
    scheduler = _bare_scheduler(state)
    limit = scheduler_module._FAILED_REQUEST_ID_LIMIT
    retained = scheduler_module._FAILED_REQUEST_ID_RETAINED
    scheduler._failed_outer_requests.update(
        (f"old-{index}", None) for index in range(limit - 1)
    )

    scheduler._fail_session("rejected", RuntimeError("invalid"), generation=7)

    assert len(scheduler._failed_outer_requests) == retained
    assert "rejected" in scheduler._failed_outer_requests


def test_sglang_session_must_be_gone_after_cleanup() -> None:
    state = _state()
    scheduler = _bare_scheduler(state)
    controller = _SessionController(state.session_id, retain_after_close=True)
    scheduler.session_controller = controller

    errors = scheduler._cleanup_state(state)

    assert len(errors) == 1
    assert "remained live" in str(errors[0])
    assert scheduler._poisoned_error is not None
    assert scheduler._state is None


def test_temp_reference_cleanup_failure_is_retained_and_poisons_stage(
    monkeypatch,
) -> None:
    state = _state()
    state.temp_paths = ["/tmp/minicpmo-voice.wav"]
    scheduler = _bare_scheduler(state)
    scheduler.session_controller = _SessionController(state.session_id)

    def fail_unlink(_path: str) -> None:
        raise OSError("unlink failed")

    monkeypatch.setattr(scheduler_module.os, "unlink", fail_unlink)
    errors = scheduler._cleanup_state(state)

    assert len(errors) == 1
    assert "unlink failed" in str(errors[0])
    assert state.temp_paths == ["/tmp/minicpmo-voice.wav"]
    assert scheduler._poisoned_error is not None
