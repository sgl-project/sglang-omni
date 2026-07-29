# SPDX-License-Identifier: Apache-2.0
"""Native SGLang scheduler for MiniCPM-o 4.5 full-duplex sessions."""

from __future__ import annotations

import base64
import logging
import math
import os
import queue as _queue_mod
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sglang.srt.managers.io_struct import CloseSessionReqInput, OpenSessionReqInput

from sglang_omni.models.minicpmo_4_5.protocol import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    DuplexProtocolError,
    SessionCommand,
    extract_open_session,
    extract_session_command,
    make_envelope,
    normalize_append_data,
)
from sglang_omni.models.minicpmo_4_5.request_builders import (
    MiniCPMOUnitBuild,
    build_unit_request_data,
    prepare_session_prefix,
)
from sglang_omni.models.minicpmo_4_5.state import (
    MiniCPMOSessionState,
    MiniCPMOSpecialTokens,
    MiniCPMOUnitRequestData,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)

_FAILED_REQUEST_ID_LIMIT = 10_000
_FAILED_REQUEST_ID_RETAINED = 5_000


@dataclass(frozen=True)
class _OverloadSignal:
    request_id: str
    limit_name: str
    generation: int | None = None


class _BoundedSessionInbox(_queue_mod.Queue[Any]):
    """Bound queued duplex commands per session before model preprocessing."""

    def __init__(self, max_pending_units: int, max_pending_commands: int):
        super().__init__()
        self._max_pending_units = int(max_pending_units)
        self._max_pending_commands = int(max_pending_commands)
        self._pending_units: dict[str, int] = {}
        self._pending_commands: dict[str, int] = {}
        self._overloaded: set[str] = set()
        self._pending_lock = threading.Lock()

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        if _is_session_command(item):
            request_id = item.request_id
            with self._pending_lock:
                if request_id in self._overloaded:
                    return
                commands = self._pending_commands.get(request_id, 0)
                units = self._pending_units.get(request_id, 0)
                unit_overflow = (
                    _is_append_message(item) and units >= self._max_pending_units
                )
                if commands >= self._max_pending_commands or unit_overflow:
                    self._overloaded.add(request_id)
                    name = (
                        "max_pending_commands"
                        if commands >= self._max_pending_commands
                        else "max_pending_units"
                    )
                    with self.mutex:
                        signal = _OverloadSignal(
                            request_id,
                            name,
                            generation=_message_generation(item),
                        )
                        # Prioritize overload over queued commands, but never
                        # overtake the new_request that registers this
                        # generation.  Otherwise one drain can emit an error,
                        # then open an orphaned backend session for the same
                        # generation after Stage has already cleared it.
                        insert_at = 0
                        for index, queued in enumerate(self.queue):
                            if _is_new_request(queued, request_id):
                                insert_at = index + 1
                                break
                        self.queue.insert(insert_at, signal)
                        self.unfinished_tasks += 1
                        self.not_empty.notify()
                    return
                self._pending_commands[request_id] = commands + 1
                if _is_append_message(item):
                    self._pending_units[request_id] = units + 1
        super().put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        item = super().get(block=block, timeout=timeout)
        if _is_session_command(item):
            with self._pending_lock:
                self._decrement(self._pending_commands, item.request_id)
                if _is_append_message(item):
                    self._decrement(self._pending_units, item.request_id)
        elif isinstance(item, _OverloadSignal):
            with self._pending_lock:
                self._pending_commands.pop(item.request_id, None)
                self._pending_units.pop(item.request_id, None)
                self._overloaded.discard(item.request_id)
        return item

    def discard_request(self, request_id: str) -> None:
        with self._pending_lock, self.mutex:
            kept = type(self.queue)(
                item
                for item in self.queue
                if not (
                    getattr(item, "request_id", None) == request_id
                    and (_is_session_command(item) or isinstance(item, _OverloadSignal))
                )
            )
            removed = len(self.queue) - len(kept)
            self.queue.clear()
            self.queue.extend(kept)
            self.unfinished_tasks = max(0, self.unfinished_tasks - removed)
            self._pending_commands.pop(request_id, None)
            self._pending_units.pop(request_id, None)
            self._overloaded.discard(request_id)
            if removed:
                self.not_full.notify_all()

    @staticmethod
    def _decrement(mapping: dict[str, int], key: str) -> None:
        value = mapping.get(key, 0) - 1
        if value > 0:
            mapping[key] = value
        else:
            mapping.pop(key, None)


class MiniCPMO45Scheduler(OmniScheduler):
    """Compose the duplex state machine with the normal ``OmniScheduler``.

    The main decoder request is never delegated to another process.  Each
    append becomes one SGLang request in an upstream streaming session, while
    perception and TTS keep only their model-specific per-session side state.
    """

    requires_tp_work_fanout = False

    def __init__(
        self,
        *,
        perception: Any,
        tokenizer: Any,
        tts_runtime: Any,
        ref_audio_path: str | None = None,
        prompt_wav_path: str | None = None,
        duplex_sampling: dict[str, Any] | None = None,
        max_sessions: int = 1,
        max_pending_units: int = 4,
        max_pending_commands: int = 16,
        session_ttl_s: float = 300.0,
        **omni_kwargs: Any,
    ) -> None:
        if max_sessions != 1:
            raise ValueError(
                "MiniCPM-o native duplex currently requires max_sessions=1"
            )
        if max_pending_units < 1:
            raise ValueError("max_pending_units must be >= 1")
        if max_pending_commands < max_pending_units:
            raise ValueError("max_pending_commands must be >= max_pending_units")
        if session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive")
        server_args = omni_kwargs.get("server_args")
        if server_args is None or int(server_args.tp_size) != 1:
            raise ValueError(
                "MiniCPM-o hybrid duplex stage currently supports TP=1 only"
            )
        if not bool(getattr(server_args, "enable_streaming_session", False)):
            raise ValueError("MiniCPM-o requires enable_streaming_session=True")

        self._perception = perception
        self._tokenizer = tokenizer
        self._tts_runtime = tts_runtime
        self._ref_audio_path = ref_audio_path
        self._prompt_wav_path = prompt_wav_path
        self._duplex_sampling = dict(duplex_sampling or {})
        if self._duplex_sampling.get("ls_mode", "explicit") != "explicit":
            raise ValueError(
                "MiniCPM-o native integration currently supports ls_mode='explicit' only"
            )
        self._session_ttl_s = float(session_ttl_s)
        self._state: MiniCPMOSessionState | None = None
        self._state_lock = threading.RLock()
        self._unit_by_rid: dict[str, MiniCPMOUnitRequestData] = {}
        self._failed_outer_requests: OrderedDict[str, None] = OrderedDict()
        self._poisoned_error: str | None = None
        self._external_abort_callback = omni_kwargs.pop("abort_callback", None)

        super().__init__(
            request_builder=self._build_unit_request,
            result_adapter=lambda data: data,
            stream_output_builder=None,
            request_build_max_workers=1,
            request_build_max_pending=max_pending_units,
            abort_callback=self._on_internal_abort_terminal,
            **omni_kwargs,
        )
        self.inbox = _BoundedSessionInbox(max_pending_units, max_pending_commands)
        self.special_tokens = MiniCPMOSpecialTokens.from_tokenizer(tokenizer)

    # ------------------------------------------------------------------
    # Command admission
    # ------------------------------------------------------------------

    def recv_requests(self) -> list[StagePayload]:
        ready: list[StagePayload] = []
        for msg in self._recv_scheduler_messages():
            if isinstance(msg, _OverloadSignal):
                self._fail_session(
                    msg.request_id,
                    RuntimeError(
                        f"MiniCPM-o duplex input queue exceeded {msg.limit_name}"
                    ),
                    generation=msg.generation,
                )
                continue
            try:
                if msg.type == "new_request":
                    self._open_session(msg)
                elif msg.type == "session_command":
                    payload = self._handle_session_command(msg)
                    if payload is not None:
                        ready.append(payload)
            except Exception as exc:
                logger.exception(
                    "MiniCPM-o duplex command failed for %s", msg.request_id
                )
                self._fail_session(
                    msg.request_id,
                    exc,
                    generation=_message_generation(msg),
                )
        return ready

    def process_input_requests(self, recv_reqs: list[Any]) -> None:
        self._expire_local_session()
        super().process_input_requests(recv_reqs)

    def _open_session(self, msg: IncomingMessage) -> None:
        with self._state_lock:
            self._open_session_locked(msg)

    def _open_session_locked(self, msg: IncomingMessage) -> None:
        if self._poisoned_error is not None:
            raise RuntimeError(
                "MiniCPM-o stage is poisoned after an incomplete cleanup; "
                f"restart it before opening another session: {self._poisoned_error}"
            )
        opened = extract_open_session(msg.data)
        self._failed_outer_requests.pop(msg.request_id, None)
        with self._state_lock:
            if self._state is not None:
                raise DuplexProtocolError(
                    "MiniCPM-o duplex stage already owns an active session"
                )
            state = MiniCPMOSessionState(
                request_id=msg.request_id,
                session_id=opened.session_id,
                generation=opened.generation,
                response_epoch=opened.response_epoch,
                next_input_seq=opened.next_input_seq,
                system_prompt=opened.system_prompt,
            )
            self._state = state

        session_opened = False
        perception_opened = False
        tts_opened = False
        try:
            result = self.session_controller.open(
                OpenSessionReqInput(
                    capacity_of_str_len=int(self.server_args.context_length),
                    session_id=state.session_id,
                    streaming=True,
                    # Local state reaps first and closes all side components.
                    timeout=None,
                )
            )
            if not result.success:
                raise RuntimeError(
                    f"SGLang session {state.session_id!r} already exists"
                )
            session_opened = True
            self._perception.open_session(state.session_id)
            perception_opened = True

            reference = self._resolve_reference_waveform(opened.voice)
            reference_embedding = (
                self._perception.prepare_reference_audio(state.session_id, reference)
                if reference is not None
                else None
            )
            prepare_session_prefix(
                state,
                tokenizer=self._tokenizer,
                special_tokens=self.special_tokens,
                reference_embedding=reference_embedding,
            )

            if self._tts_runtime is not None:
                prompt_path = self._resolve_prompt_wav(state, opened.voice)
                state.tts = self._tts_runtime.open_session(
                    state.session_id,
                    prompt_wav_path=prompt_path,
                )
                tts_opened = True
        except Exception:
            # Opening spans three independent state owners.  Roll every owner
            # back even if an earlier close itself fails, and preserve the
            # original open error for the client.
            rollback_errors: list[BaseException] = []
            if tts_opened and self._tts_runtime is not None:
                try:
                    self._tts_runtime.close_session(state.session_id)
                except Exception as exc:
                    rollback_errors.append(exc)
                    logger.exception("MiniCPM-o TTS open rollback failed")
            if perception_opened:
                try:
                    self._perception.close_session(state.session_id)
                except Exception as exc:
                    rollback_errors.append(exc)
                    logger.exception("MiniCPM-o perception open rollback failed")
            if session_opened:
                try:
                    self.session_controller.close(
                        CloseSessionReqInput(session_id=state.session_id)
                    )
                    if self.session_controller.get(state.session_id) is not None:
                        raise RuntimeError(
                            "SGLang session remained live after open rollback"
                        )
                except Exception as exc:
                    rollback_errors.append(exc)
                    logger.exception("MiniCPM-o SGLang open rollback failed")
            rollback_errors.extend(self._remove_temp_paths(state))
            with self._state_lock:
                if self._state is state:
                    self._state = None
            if rollback_errors:
                self._poisoned_error = _format_cleanup_errors(rollback_errors)
            raise

        self._emit_control(state, "session.created", input_seq=0, backend="sglang")

    def _handle_session_command(self, msg: IncomingMessage) -> StagePayload | None:
        with self._state_lock:
            return self._handle_session_command_locked(msg)

    def _handle_session_command_locked(
        self, msg: IncomingMessage
    ) -> StagePayload | None:
        command = extract_session_command(msg.data)
        state = self._require_state(msg.request_id, command)
        if command.command == "append":
            return self._accept_append(state, command)
        if command.command == "interrupt":
            self._accept_interrupt(state, command)
        elif command.command == "playback_ack":
            self._accept_playback_ack(state, command)
        elif command.command == "close":
            self._accept_close(state, command)
        return None

    def _accept_append(
        self, state: MiniCPMOSessionState, command: SessionCommand
    ) -> StagePayload:
        self._validate_current_command(state, command)
        if state.closing:
            raise DuplexProtocolError("duplex session is closing")
        if state.inflight_rid is not None:
            raise DuplexProtocolError(
                "duplex session already has an in-flight unit; wait for "
                "session.input_processed"
            )
        normalized = normalize_append_data(command.data)
        forced = bool(normalized.get("force_listen")) or state.force_listen_next
        forced = forced or state.generated_unit_count < int(
            self._duplex_sampling.get("force_listen_count", 3)
        )
        close_speaking_turn = forced and not state.current_turn_ended
        if close_speaking_turn and self._tts_runtime is not None:
            # A forced listen is a hard speech cut, even when it arrives as a
            # normal audio unit rather than an explicit interrupt event.  Do
            # not carry TTS KV or Token2wav lookahead into the next turn.
            self._tts_runtime.interrupt_session(state.session_id, flush=False)
        internal_rid = f"{state.session_id}:g{state.generation}:u{command.input_seq}"
        state.inflight_rid = internal_rid
        state.inflight_input_seq = command.input_seq
        state.inflight_response_epoch = command.response_epoch
        state.next_input_seq += 1
        state.last_activity = time.monotonic()
        state.force_listen_next = False

        return StagePayload(
            request_id=internal_rid,
            request=OmniRequest(inputs={}, params={}, metadata={}),
            data=MiniCPMOUnitBuild(
                internal_request_id=internal_rid,
                state=state,
                prepared_unit=normalized,
                forced_listen=forced,
                close_speaking_turn=close_speaking_turn,
                sampling=self._duplex_sampling,
            ),
        )

    def _accept_interrupt(
        self, state: MiniCPMOSessionState, command: SessionCommand
    ) -> None:
        self._validate_epoch_command(state, command, "interrupt")
        state.response_epoch = command.response_epoch
        state.force_listen_next = True
        state.next_input_seq += 1
        state.last_activity = time.monotonic()
        # Flush before reset.  Its old-epoch waveform is intentionally dropped:
        # interrupt fences audio the client has not committed to playback.
        if self._tts_runtime is not None:
            self._tts_runtime.interrupt_session(state.session_id, flush=False)

    def _accept_playback_ack(
        self, state: MiniCPMOSessionState, command: SessionCommand
    ) -> None:
        self._validate_current_command(state, command)
        audio_end_ms = command.data.get("audio_end_ms")
        if (
            isinstance(audio_end_ms, bool)
            or not isinstance(audio_end_ms, (int, float))
            or not math.isfinite(float(audio_end_ms))
            or float(audio_end_ms) < state.playback_audio_end_ms
        ):
            raise DuplexProtocolError("playback audio_end_ms must be monotonic")
        if float(audio_end_ms) > state.emitted_audio_end_ms + 1e-6:
            raise DuplexProtocolError("playback audio_end_ms exceeds emitted audio")
        state.playback_audio_end_ms = float(audio_end_ms)
        state.next_input_seq += 1
        state.last_activity = time.monotonic()

    def _accept_close(
        self, state: MiniCPMOSessionState, command: SessionCommand
    ) -> None:
        self._validate_epoch_command(state, command, "close")
        reason = command.data.get("reason", "client_close")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 128:
            raise DuplexProtocolError(
                "close reason must be a non-empty string of at most 128 characters"
            )
        state.response_epoch = command.response_epoch
        state.next_input_seq += 1
        state.closing = True
        state.close_reason = reason.strip()
        state.last_activity = time.monotonic()
        if state.inflight_rid is None:
            self._finish_close(state, input_seq=command.input_seq)

    # ------------------------------------------------------------------
    # Unit request build / result
    # ------------------------------------------------------------------

    def _build_unit_request(self, payload: StagePayload) -> MiniCPMOUnitRequestData:
        with self._state_lock:
            return self._build_unit_request_locked(payload)

    def _build_unit_request_locked(
        self, payload: StagePayload
    ) -> MiniCPMOUnitRequestData:
        build = payload.data
        if not isinstance(build, MiniCPMOUnitBuild):
            raise TypeError("MiniCPM-o scheduler received a non-unit payload")
        prepared = self._perception.prepare_unit(
            build.state.session_id,
            build.prepared_unit,
        )
        actual = MiniCPMOUnitBuild(
            internal_request_id=build.internal_request_id,
            state=build.state,
            prepared_unit=prepared,
            forced_listen=build.forced_listen,
            close_speaking_turn=build.close_speaking_turn,
            sampling=build.sampling,
        )
        data = build_unit_request_data(
            actual,
            tokenizer=self._tokenizer,
            vocab_size=int(self.model_config.vocab_size),
            special_tokens=self.special_tokens,
        )
        self._unit_by_rid[build.internal_request_id] = data
        return data

    def stream_output(
        self, reqs: list[Any], return_logprob: bool = False, skip_req=None
    ):
        del return_logprob
        for req in reqs:
            if skip_req is not None and req is skip_req:
                continue
            if not req.finished():
                continue
            rid = req.rid
            if rid in self._aborted_request_ids:
                # OmniScheduler normally performs this callback in its own
                # stream_output implementation. This model overrides that
                # method to run TTS, so preserve the same deferred-abort
                # terminal contract explicitly.
                self._on_internal_abort_terminal(rid)
                self._first_emit_done.discard(rid)
                self._prefill_start_done.discard(rid)
                continue
            data = getattr(req, "_omni_data", None)
            if not isinstance(data, MiniCPMOUnitRequestData):
                continue
            state = data.session_state
            if state is None:
                continue

            try:
                data.output_ids = list(req.output_ids)
                data.weight_version = self.server_args.weight_version
                reason = req.finished_reason
                data.finish_reason = (
                    reason.to_json().get("type") if reason is not None else None
                )
                self._complete_unit(state, data)
            except Exception as exc:
                logger.exception("MiniCPM-o unit completion failed for %s", rid)
                self._fail_session(state.request_id, exc)
            finally:
                self._unit_by_rid.pop(rid, None)
                self._first_emit_done.discard(rid)
                self._prefill_start_done.discard(rid)
                data.prefill_input_embeds = None
                data.decode_input_embeds = []

    def _complete_unit(
        self, state: MiniCPMOSessionState, data: MiniCPMOUnitRequestData
    ) -> None:
        # Stage aborts are delivered by a listener thread while unit
        # completion runs on the scheduler thread.  Serialize all side-model
        # use and teardown so TTS/perception/session close cannot race a
        # finished unit that is still being committed.
        with self._state_lock:
            self._complete_unit_locked(state, data)

    def _complete_unit_locked(
        self, state: MiniCPMOSessionState, data: MiniCPMOUnitRequestData
    ) -> None:
        if self._state is not state or state.aborted:
            return
        if state.inflight_rid != data.req.rid:
            raise RuntimeError("completed MiniCPM-o unit does not own the session")

        output_ids = list(data.output_ids)
        terminator = output_ids[-1] if output_ids else None
        if terminator not in self.special_tokens.chunk_terminators:
            raise RuntimeError(
                "MiniCPM-o unit finished without a valid chunk terminator: "
                f"finish_reason={data.finish_reason!r}, last_token={terminator!r}"
            )
        is_listen = terminator == self.special_tokens.listen
        end_of_turn = any(pair[2] for pair in data.tts_pairs)
        text = self._tokenizer.decode(
            data.generated_unit_ids,
            skip_special_tokens=True,
        )
        response_is_current = data.response_epoch == state.response_epoch

        if response_is_current:
            if is_listen:
                self._emit_delta(
                    state,
                    input_seq=data.input_seq,
                    response_epoch=data.response_epoch,
                    kind="listen",
                    delta={"is_listen": True},
                )
            else:
                if text:
                    self._emit_delta(
                        state,
                        input_seq=data.input_seq,
                        response_epoch=data.response_epoch,
                        kind="text",
                        delta={"text": text},
                    )
                if bool(self._duplex_sampling.get("generate_audio", True)):
                    if self._tts_runtime is None:
                        raise RuntimeError(
                            "MiniCPM-o audio output is enabled but TTS was not loaded"
                        )
                    token_ids = [pair[0] for pair in data.tts_pairs]
                    hidden = (
                        torch.stack([pair[1] for pair in data.tts_pairs], dim=0)
                        if data.tts_pairs
                        else None
                    )
                    chunk = self._tts_runtime.synthesize(
                        state.session_id,
                        token_ids,
                        hidden,
                        end_of_turn=end_of_turn,
                    )
                    if chunk.waveform is not None and len(chunk.waveform):
                        self._emit_audio(
                            state,
                            input_seq=data.input_seq,
                            response_epoch=data.response_epoch,
                            waveform=chunk.waveform,
                            sample_rate=int(chunk.sample_rate),
                        )
                if end_of_turn:
                    # This is an ordered response boundary, not a session
                    # terminal. Realtime uses it to emit text/audio done and
                    # response.done even when no following listen unit arrives.
                    self._emit_control(
                        state,
                        "response.output.done",
                        input_seq=data.input_seq,
                        response_epoch=data.response_epoch,
                        reason="model_turn_end",
                    )

        state.unit_journal.append(
            {
                "input_seq": data.input_seq,
                "response_epoch": data.response_epoch,
                "mode": data.input_mode,
                "output_ids": output_ids,
                "generated_text": text,
                "is_listen": is_listen,
                "end_of_turn": end_of_turn,
            }
        )
        if len(state.unit_journal) > 256:
            del state.unit_journal[:-256]
        repetition_window = int(
            self._duplex_sampling.get("text_repetition_window_size", 512)
        )
        if repetition_window > 0 and len(state.generated_text_ids) > repetition_window:
            del state.generated_text_ids[:-repetition_window]
        state.generated_unit_count += 1
        state.inflight_rid = None
        state.inflight_input_seq = None
        state.inflight_response_epoch = None
        state.last_activity = time.monotonic()
        _offload_embedding_ledger(state, data)

        self._emit_control(
            state,
            "session.input_processed",
            input_seq=data.input_seq,
            response_epoch=data.response_epoch,
            metrics={
                "mode": data.input_mode,
                "llm_tokens": len(output_ids),
                "is_listen": is_listen,
            },
        )
        if state.closing:
            self._finish_close(state, input_seq=state.next_input_seq - 1)

    # ------------------------------------------------------------------
    # Lifecycle / error handling
    # ------------------------------------------------------------------

    def _on_internal_abort_terminal(self, request_id: str) -> None:
        """Release side state only after an aborted SGLang unit is quiescent."""

        with self._state_lock:
            self._on_internal_abort_terminal_locked(request_id)

    def _on_internal_abort_terminal_locked(self, request_id: str) -> None:
        """Abort callback body; caller owns ``_state_lock``."""

        data = self._unit_by_rid.pop(request_id, None)
        if data is not None:
            data.prefill_input_embeds = None
            data.decode_input_embeds = []
        self._first_emit_done.discard(request_id)
        self._prefill_start_done.discard(request_id)

        callback = self._external_abort_callback
        if callback is not None:
            try:
                callback(request_id)
            except Exception:
                logger.exception(
                    "external MiniCPM-o abort callback failed for %s", request_id
                )

        with self._state_lock:
            state = self._state
        if state is None or state.inflight_rid != request_id:
            return
        state.inflight_rid = None
        state.inflight_input_seq = None
        state.inflight_response_epoch = None
        if state.aborted:
            errors = self._cleanup_state(state)
            if errors:
                logger.error(
                    "MiniCPM-o abort cleanup failed: %s",
                    _format_cleanup_errors(errors),
                )
            if state.pending_failure is not None:
                self._emit_failure_terminal(
                    state,
                    state.pending_failure,
                    cleanup_errors=errors,
                )
        elif state.closing:
            self._finish_close(state, input_seq=state.next_input_seq - 1)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        with self._state_lock:
            self._abort_locked(
                request_id,
                defer_running_cleanup=defer_running_cleanup,
            )

    def _abort_locked(
        self,
        request_id: str,
        *,
        defer_running_cleanup: bool,
    ) -> None:
        with self._state_lock:
            state = self._state
        if state is None:
            super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
            return
        if request_id == state.request_id:
            state.aborted = True
            state.response_epoch += 1
            if state.inflight_rid is not None:
                OmniScheduler.abort(
                    self,
                    state.inflight_rid,
                    defer_running_cleanup=defer_running_cleanup,
                )
            else:
                errors = self._cleanup_state(state)
                if errors:
                    logger.error(
                        "MiniCPM-o abort cleanup failed: %s",
                        _format_cleanup_errors(errors),
                    )
            return
        if request_id == state.inflight_rid:
            super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
            return
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)

    def _emit_request_error(self, request_id: str, error: Exception) -> None:
        data = self._unit_by_rid.get(request_id)
        if data is not None and data.session_state is not None:
            self._fail_session(data.outer_request_id, error)
            return
        with self._state_lock:
            state = self._state
        if state is not None and request_id == state.inflight_rid:
            self._fail_session(state.request_id, error)
            return
        super()._emit_request_error(request_id, error)

    def _fail_session(
        self,
        request_id: str,
        exc: BaseException,
        *,
        generation: int | None = None,
    ) -> None:
        with self._state_lock:
            self._fail_session_locked(
                request_id,
                exc,
                generation=generation,
            )

    def _fail_session_locked(
        self,
        request_id: str,
        exc: BaseException,
        *,
        generation: int | None = None,
    ) -> None:
        if not request_id:
            return
        with self._state_lock:
            state = self._state
        if (
            generation is not None
            and state is not None
            and state.request_id == request_id
            and generation != state.generation
        ):
            # A delayed command from an older generation may reuse the same
            # outer request id.  Its failure must never abort the replacement
            # session that now owns that id.
            logger.warning(
                "Dropping stale MiniCPM-o failure for %s generation %s; "
                "active generation is %s",
                request_id,
                generation,
                state.generation,
            )
            return
        if request_id in self._failed_outer_requests:
            return
        self._failed_outer_requests[request_id] = None
        if len(self._failed_outer_requests) >= _FAILED_REQUEST_ID_LIMIT:
            while len(self._failed_outer_requests) > _FAILED_REQUEST_ID_RETAINED:
                self._failed_outer_requests.popitem(last=False)
        if state is None or state.request_id != request_id:
            metadata: dict[str, Any] = {}
            if generation is not None:
                event = make_envelope(
                    event_type="session.error",
                    session_id=request_id,
                    generation=generation,
                    input_seq=0,
                    response_epoch=0,
                    output_seq=1,
                    error=str(exc) or type(exc).__name__,
                )
                metadata = {
                    "generation": generation,
                    "terminal_event": event,
                }
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="error",
                    data=exc,
                    metadata=metadata,
                )
            )
            return

        state.aborted = True
        state.pending_failure = exc
        if state.inflight_rid is not None:
            # Pending/waiting aborts invoke the callback synchronously; running
            # aborts invoke it from stream_output after FINISH_ABORT is safe.
            OmniScheduler.abort(self, state.inflight_rid)
        else:
            errors = self._cleanup_state(state)
            if errors:
                logger.error(
                    "MiniCPM-o failure cleanup also failed: %s",
                    _format_cleanup_errors(errors),
                )
            self._emit_failure_terminal(state, exc, cleanup_errors=errors)

    def _emit_failure_terminal(
        self,
        state: MiniCPMOSessionState,
        exc: BaseException,
        *,
        cleanup_errors: list[BaseException],
    ) -> None:
        if state.failure_terminal_emitted:
            return
        state.failure_terminal_emitted = True
        terminal_error: BaseException = exc
        if cleanup_errors:
            cleanup_error = _combine_cleanup_errors(cleanup_errors)
            terminal_error = RuntimeError(
                f"{str(exc) or type(exc).__name__}; cleanup failed: {cleanup_error}"
            )
        event = self._next_envelope(
            state,
            event_type="session.error",
            input_seq=max(0, state.next_input_seq - 1),
            response_epoch=state.response_epoch,
            error=str(terminal_error) or type(terminal_error).__name__,
        )
        self.outbox.put(
            OutgoingMessage(
                request_id=state.request_id,
                type="error",
                data=terminal_error,
                metadata={"generation": state.generation, "terminal_event": event},
            )
        )

    def _finish_close(self, state: MiniCPMOSessionState, *, input_seq: int) -> None:
        with self._state_lock:
            self._finish_close_locked(state, input_seq=input_seq)

    def _finish_close_locked(
        self,
        state: MiniCPMOSessionState,
        *,
        input_seq: int,
    ) -> None:
        if self._state is not state:
            return
        cleanup_errors: list[BaseException] = []
        try:
            flushed = (
                self._tts_runtime.interrupt_session(state.session_id, flush=True)
                if self._tts_runtime is not None
                else None
            )
            if flushed is not None and len(flushed):
                self._emit_audio(
                    state,
                    input_seq=input_seq,
                    response_epoch=state.response_epoch,
                    waveform=flushed,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                )
        except BaseException as exc:
            cleanup_errors.append(exc)
            logger.exception("MiniCPM-o final TTS flush failed")

        cleanup_errors.extend(self._cleanup_state(state))
        cleanup_error = _combine_cleanup_errors(cleanup_errors)
        event = self._next_envelope(
            state,
            event_type="session.error" if cleanup_error else "session.closed",
            input_seq=input_seq,
            response_epoch=state.response_epoch,
            reason=state.close_reason or "client_close",
            cleanup_error=str(cleanup_error) if cleanup_error else None,
            **(
                {"error": f"MiniCPM-o session cleanup failed: {cleanup_error}"}
                if cleanup_error
                else {}
            ),
        )
        if cleanup_error:
            self.outbox.put(
                OutgoingMessage(
                    request_id=state.request_id,
                    type="error",
                    data=cleanup_error,
                    metadata={"generation": state.generation, "terminal_event": event},
                )
            )
        else:
            self.outbox.put(
                OutgoingMessage(
                    request_id=state.request_id,
                    type="result",
                    data=event,
                    metadata={"generation": state.generation},
                )
            )

    def _cleanup_state(self, state: MiniCPMOSessionState) -> list[BaseException]:
        with self._state_lock:
            return self._cleanup_state_locked(state)

    def _cleanup_state_locked(self, state: MiniCPMOSessionState) -> list[BaseException]:
        with self._state_lock:
            if self._state is not state:
                return []
        errors: list[BaseException] = []
        self.inbox.discard_request(state.request_id)
        try:
            if self._tts_runtime is not None:
                self._tts_runtime.close_session(state.session_id)
        except Exception as exc:
            errors.append(exc)
            logger.exception("MiniCPM-o TTS session close failed")
        try:
            self._perception.close_session(state.session_id)
        except Exception as exc:
            errors.append(exc)
            logger.exception("MiniCPM-o perception session close failed")
        try:
            self.session_controller.close(
                CloseSessionReqInput(session_id=state.session_id)
            )
            if self.session_controller.get(state.session_id) is not None:
                raise RuntimeError("SGLang streaming session remained live after close")
        except Exception as exc:
            errors.append(exc)
            logger.exception("MiniCPM-o SGLang session close failed")
        errors.extend(self._remove_temp_paths(state))
        state.embedding_spans.clear()
        with self._state_lock:
            if self._state is state:
                self._state = None
        if errors:
            self._poisoned_error = _format_cleanup_errors(errors)
        return errors

    def _expire_local_session(self) -> None:
        with self._state_lock:
            self._expire_local_session_locked()

    def _expire_local_session_locked(self) -> None:
        state = self._state
        if (
            state is not None
            and not state.aborted
            and state.inflight_rid is None
            and time.monotonic() - state.last_activity >= self._session_ttl_s
        ):
            state.closing = True
            state.close_reason = "session_ttl"
            if state.inflight_rid is None:
                self._finish_close(
                    state,
                    input_seq=max(0, state.next_input_seq - 1),
                )

    def start(self) -> None:
        try:
            super().start()
        finally:
            self._shutdown_duplex_state()

    def stop(self) -> None:
        # The scheduler thread owns final teardown.  If it was never started,
        # perform the same quiescent cleanup synchronously for tests/startup
        # failures.
        super().stop()
        if self._scheduler_thread_id is None:
            self._shutdown_duplex_state()

    def _shutdown_duplex_state(self) -> None:
        with self._state_lock:
            self._shutdown_duplex_state_locked()

    def _shutdown_duplex_state_locked(self) -> None:
        with self._state_lock:
            state = self._state
        if state is not None:
            state.aborted = True
            state.closing = True
            state.close_reason = "scheduler_stopped"
            if state.inflight_rid is not None:
                OmniScheduler.abort(
                    self,
                    state.inflight_rid,
                    defer_running_cleanup=False,
                )
            errors = self._cleanup_state(state)
            if errors:
                logger.error(
                    "MiniCPM-o shutdown cleanup failed: %s",
                    _format_cleanup_errors(errors),
                )
        close = getattr(self._tts_runtime, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # Protocol helpers / event emission
    # ------------------------------------------------------------------

    def _require_state(
        self, request_id: str, command: SessionCommand
    ) -> MiniCPMOSessionState:
        with self._state_lock:
            state = self._state
        if (
            state is None
            or state.request_id != request_id
            or state.aborted
            or command.session_id != state.session_id
            or command.generation != state.generation
        ):
            raise DuplexProtocolError("duplex session is not active")
        return state

    @staticmethod
    def _validate_current_command(
        state: MiniCPMOSessionState, command: SessionCommand
    ) -> None:
        if command.input_seq != state.next_input_seq:
            raise DuplexProtocolError(
                f"command input_seq must be {state.next_input_seq}, got {command.input_seq}"
            )
        if command.response_epoch != state.response_epoch:
            raise DuplexProtocolError(
                "command response_epoch does not match active response epoch"
            )

    @staticmethod
    def _validate_epoch_command(
        state: MiniCPMOSessionState,
        command: SessionCommand,
        name: str,
    ) -> None:
        if command.input_seq != state.next_input_seq:
            raise DuplexProtocolError(
                f"{name} input_seq must be {state.next_input_seq}"
            )
        if command.response_epoch != state.response_epoch + 1:
            raise DuplexProtocolError(
                f"{name} response_epoch must be {state.response_epoch + 1}"
            )

    def _emit_control(
        self,
        state: MiniCPMOSessionState,
        event_type: str,
        *,
        input_seq: int,
        response_epoch: int | None = None,
        **fields: Any,
    ) -> None:
        if self._state is not state or state.aborted:
            return
        event = self._next_envelope(
            state,
            event_type=event_type,
            input_seq=input_seq,
            response_epoch=(
                state.response_epoch if response_epoch is None else response_epoch
            ),
            **fields,
        )
        self.outbox.put(
            OutgoingMessage(
                request_id=state.request_id,
                type="stream",
                data=event,
                metadata={"generation": state.generation, "modality": "control"},
            )
        )

    def _emit_delta(
        self,
        state: MiniCPMOSessionState,
        *,
        input_seq: int,
        response_epoch: int,
        kind: str,
        delta: dict[str, Any],
    ) -> None:
        if (
            self._state is not state
            or state.aborted
            or response_epoch != state.response_epoch
        ):
            return
        event = self._next_envelope(
            state,
            event_type="response.output.delta",
            input_seq=input_seq,
            response_epoch=response_epoch,
            kind=kind,
            **delta,
        )
        self.outbox.put(
            OutgoingMessage(
                request_id=state.request_id,
                type="stream",
                data=event,
                metadata={"generation": state.generation, "modality": kind},
            )
        )

    def _emit_audio(
        self,
        state: MiniCPMOSessionState,
        *,
        input_seq: int,
        response_epoch: int,
        waveform: Any,
        sample_rate: int,
    ) -> None:
        audio = np.asarray(waveform, dtype="<f4").reshape(-1)
        if not np.isfinite(audio).all():
            raise RuntimeError("MiniCPM-o TTS returned non-finite waveform")
        start_ms = state.emitted_audio_end_ms
        end_ms = start_ms + len(audio) * 1000.0 / int(sample_rate)
        state.emitted_audio_end_ms = end_ms
        self._emit_delta(
            state,
            input_seq=input_seq,
            response_epoch=response_epoch,
            kind="audio",
            delta={
                "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
                "format": "f32le",
                "sample_rate": int(sample_rate),
                "audio_start_ms": start_ms,
                "audio_end_ms": end_ms,
            },
        )

    @staticmethod
    def _next_envelope(
        state: MiniCPMOSessionState,
        *,
        event_type: str,
        input_seq: int,
        response_epoch: int,
        **fields: Any,
    ) -> dict[str, Any]:
        output_seq = state.next_output_seq
        state.next_output_seq += 1
        return make_envelope(
            event_type=event_type,
            session_id=state.session_id,
            generation=state.generation,
            input_seq=input_seq,
            response_epoch=response_epoch,
            output_seq=output_seq,
            **fields,
        )

    def _resolve_reference_waveform(self, voice: dict[str, Any]) -> np.ndarray | None:
        encoded = voice.get("ref_audio_base64") or voice.get("tts_ref_audio_base64")
        if encoded:
            return np.frombuffer(base64.b64decode(encoded), dtype="<f4").copy()
        if self._ref_audio_path:
            return _load_audio_16k(self._ref_audio_path)
        return None

    def _resolve_prompt_wav(
        self, state: MiniCPMOSessionState, voice: dict[str, Any]
    ) -> str | None:
        encoded = voice.get("tts_ref_audio_base64") or voice.get("ref_audio_base64")
        if not encoded:
            return self._prompt_wav_path or self._ref_audio_path
        waveform = np.frombuffer(base64.b64decode(encoded), dtype="<f4").copy()
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "soundfile is required for inline TTS reference audio"
            ) from exc
        handle = tempfile.NamedTemporaryFile(
            prefix="sglang-omni-minicpmo-",
            suffix=".wav",
            delete=False,
        )
        handle.close()
        sf.write(handle.name, waveform, INPUT_SAMPLE_RATE)
        state.temp_paths.append(handle.name)
        return handle.name

    @staticmethod
    def _remove_temp_paths(state: MiniCPMOSessionState) -> list[BaseException]:
        errors: list[BaseException] = []
        remaining: list[str] = []
        for path in state.temp_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(exc)
                remaining.append(path)
                logger.warning(
                    "failed to remove MiniCPM-o temp file %s", path, exc_info=True
                )
        state.temp_paths[:] = remaining
        return errors


def _load_audio_16k(path: str) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is required to load MiniCPM-o reference audio"
        ) from exc
    waveform, _ = librosa.load(path, sr=INPUT_SAMPLE_RATE, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def _message_generation(message: IncomingMessage) -> int | None:
    """Best-effort generation extraction for an admission failure envelope."""

    raw_generation: Any = None
    if message.type == "session_command":
        value = message.data
        raw_generation = (
            value.get("generation")
            if isinstance(value, Mapping)
            else getattr(value, "generation", None)
        )
    elif message.type == "new_request" and isinstance(message.data, StagePayload):
        request = message.data.request
        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        inputs = request.inputs if isinstance(request.inputs, dict) else {}
        raw: dict[str, Any] = {}
        user_session = metadata.get("_duplex_session")
        if isinstance(user_session, Mapping):
            raw.update(user_session)
        for key in ("_duplex_session", "duplex_session"):
            input_session = inputs.get(key)
            if isinstance(input_session, Mapping):
                raw.update(input_session)
        coordinator_session = metadata.get("duplex_session")
        if isinstance(coordinator_session, Mapping):
            raw.update(coordinator_session)
        raw_generation = raw.get("generation")

    if type(raw_generation) is int and raw_generation > 0:
        return raw_generation
    return None


def _offload_embedding_ledger(
    state: MiniCPMOSessionState,
    data: MiniCPMOUnitRequestData,
) -> None:
    """Keep replay embeddings in host memory after a successful KV commit.

    StreamingSession normally restores the paged-KV slot directly.  A
    mid-request abort intentionally discards that slot, however, so every old
    media span must remain replayable until session close or a future context
    rebase.  Retaining those tensors on the GPU would make long sessions grow
    device memory linearly; the runner moves only overlapping slices back when
    a full re-prefill is actually required.
    """

    offloaded: list[Any] = []
    for span in state.embedding_spans:
        embedding = span.embedding.detach()
        if embedding.device.type != "cpu":
            embedding = embedding.to(device="cpu")
        offloaded.append(
            type(span)(
                start=span.start,
                end=span.end,
                embedding=embedding,
                modality=span.modality,
            )
        )
    state.embedding_spans = offloaded
    # Prefix framing is now present in the committed token arrays and in the
    # absolute ledger above; keeping a second copy serves no active path.
    state.prefix_embedding_spans.clear()
    if not state.prefix_pending:
        state.prefix_input_ids.clear()
    data.local_embedding_spans.clear()
    data.absolute_embedding_spans.clear()


def _format_cleanup_errors(errors: list[BaseException]) -> str:
    return "; ".join(f"{type(error).__name__}: {error}" for error in errors)


def _combine_cleanup_errors(
    errors: list[BaseException],
) -> RuntimeError | None:
    if not errors:
        return None
    return RuntimeError(_format_cleanup_errors(errors))


def _is_session_command(item: Any) -> bool:
    return isinstance(item, IncomingMessage) and item.type == "session_command"


def _is_new_request(item: Any, request_id: str) -> bool:
    return (
        isinstance(item, IncomingMessage)
        and item.type == "new_request"
        and item.request_id == request_id
    )


def _is_append_message(item: Any) -> bool:
    if not _is_session_command(item):
        return False
    data = item.data
    command = (
        data.get("command")
        if isinstance(data, dict)
        else getattr(data, "command", None)
    )
    return command == "append"


__all__ = ["MiniCPMO45Scheduler"]
