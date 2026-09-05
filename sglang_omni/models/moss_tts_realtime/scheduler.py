# SPDX-License-Identifier: Apache-2.0
"""Scheduler policy entry point for MOSS-TTS-Realtime."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from array import array
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.models.moss_tts_realtime.config import MossTTSRealtimeResourceLimits
from sglang_omni.models.moss_tts_realtime.observability import (
    emit_realtime_event as _emit_event,
)
from sglang_omni.models.moss_tts_realtime.observability import (
    realtime_events_active,
    realtime_identity_metadata,
)
from sglang_omni.models.moss_tts_realtime.request_builders import (
    MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY,
    build_moss_tts_realtime_prefill_rows,
    build_moss_tts_realtime_row_cache_key,
    build_moss_tts_realtime_row_cache_key_ids,
)
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimeRequestData,
    MossTTSRealtimeSessionState,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeTurnState,
    MossTTSRealtimeUpdateDisposition,
    apply_moss_tts_realtime_input_update,
)
from sglang_omni.proto.messages import InputUpdateMessage
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import SchedulerRequest

logger = logging.getLogger(__name__)

_DEFAULT_LIMITS = MossTTSRealtimeResourceLimits()
_CLOSE_REALTIME_SESSION_ACTION = "close_realtime_session"


def _streaming_slot_allocated_len(slot: Any, *, default: int) -> int:
    """Return the allocated KV length from SGLang's session-slot KV state."""

    kv = getattr(slot, "kv", None)
    return int(getattr(kv, "kv_allocated_len", default))


class _RealtimeMaterializationFailure(RuntimeError):
    def __init__(self, request_id: str, error: Exception) -> None:
        self.request_id = request_id
        self.error = error
        super().__init__(
            f"MOSS-TTS-Realtime row materialization failed for {request_id!r}: {error}"
        )


@dataclass(slots=True)
class _ParkedRealtimeRequest:
    req: Any
    req_pool_idx: int
    seq_len: int
    orig_seq_len: int
    provisional_output_id: int
    parked_at: float
    last_input_at: float
    park_sequence: int


@dataclass(slots=True)
class _BufferedInputUpdates:
    messages: deque[InputUpdateMessage] = field(default_factory=deque)
    token_count: int = 0
    byte_count: int = 0
    input_done: bool = False


@dataclass(frozen=True, slots=True)
class _SGLangSessionAdmission:
    session: Any
    opened: bool
    replay_required: bool


@dataclass(frozen=True, slots=True)
class _SessionHostSnapshot:
    committed_rows: tuple[tuple[int, ...], ...]
    ledger_revision: int
    successful_turns: int
    active_turn_id: str | None
    warm_session_id: str | None
    warm_kv_length: int
    close_requested: bool
    closed: bool
    last_active_at: float

    @classmethod
    def capture(
        cls,
        session_state: MossTTSRealtimeSessionState,
    ) -> _SessionHostSnapshot:
        return cls(
            committed_rows=session_state.committed_rows,
            ledger_revision=session_state.ledger_revision,
            successful_turns=session_state.successful_turns,
            active_turn_id=session_state.active_turn_id,
            warm_session_id=session_state.warm_session_id,
            warm_kv_length=session_state.warm_kv_length,
            close_requested=session_state.close_requested,
            closed=session_state.closed,
            last_active_at=session_state.last_active_at,
        )

    def restore(self, session_state: MossTTSRealtimeSessionState) -> None:
        session_state.committed_rows = self.committed_rows
        session_state.ledger_revision = self.ledger_revision
        session_state.successful_turns = self.successful_turns
        session_state.active_turn_id = self.active_turn_id
        session_state.warm_session_id = self.warm_session_id
        session_state.warm_kv_length = self.warm_kv_length
        session_state.close_requested = self.close_requested
        session_state.closed = self.closed
        session_state.last_active_at = self.last_active_at


class MossTTSRealtimeScheduler(OmniScheduler):
    """Omni scheduler with realtime admission and canonical row materialization."""

    def __init__(
        self,
        *args: Any,
        max_sessions: int = _DEFAULT_LIMITS.max_sessions,
        max_held_sessions: int = _DEFAULT_LIMITS.max_held_sessions,
        max_pending_text_tokens: int = _DEFAULT_LIMITS.max_pending_text_tokens,
        max_pending_text_bytes: int = _DEFAULT_LIMITS.max_pending_text_bytes,
        max_input_updates: int = _DEFAULT_LIMITS.max_input_updates,
        max_active_turns: int = _DEFAULT_LIMITS.max_active_turns,
        input_idle_timeout_s: float = _DEFAULT_LIMITS.input_idle_timeout_s,
        turn_timeout_s: float = _DEFAULT_LIMITS.turn_timeout_s,
        session_idle_ttl_s: float = _DEFAULT_LIMITS.session_idle_ttl_s,
        terminal_tombstone_limit: int = _DEFAULT_LIMITS.terminal_tombstone_limit,
        **kwargs: Any,
    ) -> None:
        limit_values = _DEFAULT_LIMITS.model_dump()
        limit_values.update(
            max_sessions=max_sessions,
            max_held_sessions=max_held_sessions,
            max_pending_text_tokens=max_pending_text_tokens,
            max_pending_text_bytes=max_pending_text_bytes,
            max_input_updates=max_input_updates,
            max_active_turns=max_active_turns,
            input_idle_timeout_s=input_idle_timeout_s,
            turn_timeout_s=turn_timeout_s,
            session_idle_ttl_s=session_idle_ttl_s,
            terminal_tombstone_limit=terminal_tombstone_limit,
        )
        self._moss_tts_realtime_limits = MossTTSRealtimeResourceLimits(**limit_values)
        scheduler_model_config = kwargs.get("model_config")
        if scheduler_model_config is None:
            raise ValueError("MOSS-TTS-Realtime scheduler requires model_config")
        self._moss_tts_realtime_model_config = getattr(
            scheduler_model_config,
            "hf_config",
            scheduler_model_config,
        )
        self._max_active_turns = self._moss_tts_realtime_limits.max_active_turns
        self._input_idle_timeout_s = self._moss_tts_realtime_limits.input_idle_timeout_s
        self._turn_timeout_s = self._moss_tts_realtime_limits.turn_timeout_s
        self._session_idle_ttl_s = self._moss_tts_realtime_limits.session_idle_ttl_s
        self._session_reap_interval_s = 1.0
        self._last_session_reap_at = 0.0
        self._parked_input: dict[str, _ParkedRealtimeRequest] = {}
        self._park_sequence = 0
        self._park_total = 0
        self._wake_total = 0
        self._park_timeout_total = 0
        self._moss_tts_realtime_sessions: dict[str, MossTTSRealtimeSessionState] = {}
        self._moss_tts_realtime_requests: dict[str, MossTTSRealtimeRequestData] = {}
        self._buffered_input_updates: dict[str, _BufferedInputUpdates] = {}
        self._input_update_terminal_ids: set[str] = set()
        self._input_update_terminal_order: deque[str] = deque()
        self._terminal_tombstone_limit = terminal_tombstone_limit
        self._ensure_observability_state()
        super().__init__(*args, **kwargs)
        self._max_session_rows = int(self.server_args.context_length)
        self._max_held_kv_tokens = int(self.max_total_num_tokens)
        self._codec_slots = self._max_active_turns
        if bool(getattr(self, "enable_overlap", False)):
            raise ValueError(
                "MOSS-TTS-Realtime parked scheduling requires overlap disabled"
            )
        if bool(getattr(self, "enable_async_decode", False)):
            raise ValueError(
                "MOSS-TTS-Realtime parked scheduling requires async decode disabled"
            )

    def _ensure_observability_state(self) -> None:
        if not hasattr(self, "_resource_totals"):
            self._resource_totals: Counter[str] = Counter()
        if not hasattr(self, "_terminal_reason_totals"):
            self._terminal_reason_totals: Counter[str] = Counter()
        if not hasattr(self, "_admission_rejection_totals"):
            self._admission_rejection_totals: Counter[str] = Counter()
        if not hasattr(self, "_queued_input_tokens_high_water"):
            self._queued_input_tokens_high_water = 0
        if not hasattr(self, "_queued_input_bytes_high_water"):
            self._queued_input_bytes_high_water = 0
        if not hasattr(self, "_active_turns_high_water"):
            self._active_turns_high_water = 0
        if not hasattr(self, "_session_count_high_water"):
            self._session_count_high_water = 0
        if not hasattr(self, "_held_sessions_high_water"):
            self._held_sessions_high_water = 0
        if not hasattr(self, "_held_kv_tokens_high_water"):
            self._held_kv_tokens_high_water = 0
        if not hasattr(self, "_prefill_gate_ready_event_ids"):
            self._prefill_gate_ready_event_ids: set[str] = set()

    def _bump_resource_total(self, name: str, amount: int = 1) -> None:
        self._ensure_observability_state()
        self._resource_totals[name] += int(amount)

    def _record_cleanup_error(
        self,
        request_id: str,
        *,
        operation: str,
        error: BaseException,
        data: MossTTSRealtimeRequestData | None = None,
    ) -> None:
        if data is not None and data.cleanup_observability_finalized:
            return
        self._bump_resource_total("cleanup_error_total")
        self._emit_realtime_event(
            request_id,
            "cleanup_error",
            metadata={"operation": operation, "error": str(error)},
        )
        if data is not None:
            data.cleanup_observability_finalized = True

    def _record_cleanup_success(
        self,
        req: Any,
        data: MossTTSRealtimeRequestData,
    ) -> None:
        if data.cleanup_observability_finalized:
            return
        if not data.lifecycle_finalized or not data.observability_finalized:
            raise RuntimeError(
                "cannot record cleanup success before terminal lifecycle finalization"
            )
        turn = data.turn_state
        if turn is None or not turn.is_terminal or not turn.terminal_reason:
            raise RuntimeError(
                "cannot record cleanup success before realtime terminal invariants"
            )
        self._bump_resource_total("cleanup_success_total")
        self._emit_realtime_event(
            req.rid,
            "cleanup_success",
            metadata=self._event_metadata(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                reason=turn.terminal_reason,
            ),
        )
        data.cleanup_observability_finalized = True
        self._refresh_resource_high_water()

    @staticmethod
    def _event_metadata(
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        **values: Any,
    ) -> dict[str, Any]:
        metadata = dict(values)
        if session_id is not None:
            metadata["session_id"] = session_id
        if turn_id is not None:
            metadata["turn_id"] = turn_id
        return metadata

    def _emit_realtime_event(
        self,
        request_id: str,
        event_name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        stage: str | None = None,
    ) -> None:
        _emit_event(
            request_id=str(request_id),
            stage=stage,
            event_name=f"moss_tts_realtime_{event_name}",
            metadata=metadata,
        )

    def _record_admission_rejection(
        self,
        request_id: str,
        *,
        reason: str,
        message: str,
        error_type: type[Exception] = RuntimeError,
    ) -> None:
        self._ensure_observability_state()
        self._resource_totals["admission_rejected_total"] += 1
        self._admission_rejection_totals[reason] += 1
        self._emit_realtime_event(
            request_id,
            "admission_rejected",
            metadata={"reason": reason, "error": message},
        )
        raise error_type(message)

    def _active_realtime_pool_indices(self) -> set[int]:
        pool_indices: set[int] = set()
        seen_batches: set[int] = set()
        async_pending_batch = None
        async_pending = getattr(self, "_async_pending", None)
        if async_pending is not None:
            async_pending_batch = getattr(async_pending, "batch", None)
        collections: list[Sequence[Any]] = [
            tuple(getattr(self, "waiting_queue", ()) or ()),
            tuple(record.req for record in getattr(self, "_parked_input", {}).values()),
        ]
        for batch in (
            getattr(self, "running_batch", None),
            getattr(self, "cur_batch", None),
            getattr(self, "last_batch", None),
            async_pending_batch,
        ):
            if batch is None or id(batch) in seen_batches:
                continue
            seen_batches.add(id(batch))
            collections.append(tuple(getattr(batch, "reqs", ()) or ()))
        live_requests = getattr(self, "_moss_tts_realtime_requests", {})
        collections.append(
            tuple(
                data.req
                for data in live_requests.values()
                if getattr(data, "req", None) is not None
            )
        )
        for reqs in collections:
            for req in reqs:
                pool_idx = getattr(req, "req_pool_idx", None)
                if isinstance(pool_idx, torch.Tensor):
                    if pool_idx.numel() != 1:
                        continue
                    pool_idx = pool_idx.item()
                if isinstance(pool_idx, int) and not isinstance(pool_idx, bool):
                    pool_indices.add(pool_idx)
        return pool_indices

    def _physical_session_snapshot(self) -> dict[str, int]:
        active_pool_indices = self._active_realtime_pool_indices()
        tree_cache = getattr(self, "tree_cache", None)
        slots = getattr(tree_cache, "slots", {})
        held_rows = 0
        if isinstance(slots, dict):
            for slot in slots.values():
                pool_idx = getattr(slot, "req_pool_idx", None)
                if pool_idx is None or pool_idx in active_pool_indices:
                    continue
                held_rows += max(int(getattr(slot, "kv_committed_len", 0)), 0)

        held_sessions = 0
        held_tokens = 0
        try:
            held_sessions_fn = getattr(tree_cache, "session_held_req_count")
            held_tokens_fn = getattr(tree_cache, "session_held_tokens")
            held_sessions = int(held_sessions_fn(active_pool_indices))
            held_tokens = int(held_tokens_fn(active_pool_indices))
        except Exception:
            self._bump_resource_total("observability_read_error_total")
            held_sessions = 0
            held_tokens = 0
            if isinstance(slots, dict):
                page_size = max(int(getattr(tree_cache, "page_size", 1) or 1), 1)
                for slot in slots.values():
                    pool_idx = getattr(slot, "req_pool_idx", None)
                    if pool_idx is None or pool_idx in active_pool_indices:
                        continue
                    held_sessions += 1
                    allocated = max(
                        _streaming_slot_allocated_len(slot, default=0),
                        0,
                    )
                    protected = max(int(getattr(slot, "cache_protected_len", 0)), 0)
                    allocated = ((allocated + page_size - 1) // page_size) * page_size
                    held_tokens += max(allocated - protected, 0)
        return {
            "physical_held_session_count": held_sessions,
            "physical_held_kv_rows": held_rows,
            "physical_held_kv_tokens": held_tokens,
        }

    def _session_reservation_rows(
        self,
        session_state: MossTTSRealtimeSessionState,
    ) -> int:
        if session_state.active_turn_id is not None:
            for data in getattr(self, "_moss_tts_realtime_requests", {}).values():
                if data.session_state is session_state:
                    reserved = int(getattr(data, "context_reservation_rows", 0) or 0)
                    if reserved > 0:
                        return reserved
            return max(
                len(session_state.committed_rows),
                int(session_state.warm_kv_length),
            )
        if session_state.warm_session_id is not None:
            return int(session_state.warm_kv_length)
        return 0

    def _logical_reservation_snapshot(self) -> dict[str, int]:
        sessions = getattr(self, "_moss_tts_realtime_sessions", {})
        held_sessions = 0
        held_kv_tokens = 0
        active_sessions = 0
        idle_sessions = 0
        replay_needed_sessions = 0
        for session_state in sessions.values():
            if session_state.active_turn_id is not None:
                active_sessions += 1
            else:
                idle_sessions += 1
            reserved_rows = self._session_reservation_rows(session_state)
            if reserved_rows:
                held_sessions += 1
                held_kv_tokens += reserved_rows
            if session_state.needs_ledger_replay:
                replay_needed_sessions += 1
        return {
            "session_registry_count": len(sessions),
            "active_session_count": active_sessions,
            "idle_session_count": idle_sessions,
            "replay_needed_session_count": replay_needed_sessions,
            "held_session_reservations": held_sessions,
            "held_kv_token_reservations": held_kv_tokens,
        }

    def _queued_input_totals(self) -> tuple[int, int]:
        tokens = 0
        byte_count = 0
        for buffered in getattr(self, "_buffered_input_updates", {}).values():
            tokens += int(buffered.token_count)
            byte_count += int(buffered.byte_count)
        for data in getattr(self, "_moss_tts_realtime_requests", {}).values():
            turn = data.turn_state
            if turn is None or turn.is_terminal:
                continue
            tokens += len(turn.pending_input)
            byte_count += turn.pending_input.pending_bytes
        return tokens, byte_count

    def _model_state_snapshot(self) -> dict[str, int]:
        empty = {
            "model_state_capacity": 0,
            "model_state_active_rows": 0,
            "model_state_free_rows": 0,
            "model_state_max_active_rows_observed": 0,
        }
        runner = getattr(self, "_model_runner", None)
        snapshot = getattr(runner, "resource_snapshot", None)
        if not callable(snapshot):
            return empty
        try:
            return {key: int(value) for key, value in snapshot().items()}
        except Exception:
            self._bump_resource_total("observability_read_error_total")
            return empty

    def _refresh_resource_high_water(self) -> None:
        self._ensure_observability_state()
        queued_tokens, queued_bytes = self._queued_input_totals()
        logical = self._logical_reservation_snapshot()
        physical = self._physical_session_snapshot()
        active_turns = len(self._live_turn_request_ids())
        self._queued_input_tokens_high_water = max(
            self._queued_input_tokens_high_water,
            queued_tokens,
        )
        self._queued_input_bytes_high_water = max(
            self._queued_input_bytes_high_water,
            queued_bytes,
        )
        self._active_turns_high_water = max(
            self._active_turns_high_water,
            active_turns,
        )
        self._session_count_high_water = max(
            self._session_count_high_water,
            logical["session_registry_count"],
        )
        self._held_sessions_high_water = max(
            self._held_sessions_high_water,
            physical["physical_held_session_count"],
        )
        self._held_kv_tokens_high_water = max(
            self._held_kv_tokens_high_water,
            physical["physical_held_kv_tokens"],
        )

    @staticmethod
    def _payload_initial_input(payload: Any) -> tuple[tuple[int, ...], bool]:
        data = getattr(payload, "data", None)
        if not isinstance(data, Mapping):
            return (), False
        raw_token_ids = (
            data.get(
                MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY,
                data.get("initial_token_ids"),
            )
            or ()
        )
        if not isinstance(raw_token_ids, Sequence) or isinstance(
            raw_token_ids, (str, bytes)
        ):
            raise TypeError("initial_token_ids must be a sequence of integers")
        token_ids = tuple(raw_token_ids)
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("initial_token_ids entries must be integers")
            if token_id < 0:
                raise ValueError("initial_token_ids entries must be non-negative")
        input_done = data.get("input_done", False)
        if not isinstance(input_done, bool):
            raise TypeError("input_done must be a boolean")
        return token_ids, input_done

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        del pending_stream_done
        initial_token_ids, initial_input_done = self._payload_initial_input(payload)
        buffered = self._buffered_input_updates.get(payload.request_id)
        pending_update_tokens = int(buffered.token_count) if buffered is not None else 0
        input_done = initial_input_done or bool(
            buffered is not None and buffered.input_done
        )
        token_count = len(initial_token_ids) + pending_update_tokens
        ready = (
            token_count >= int(self._moss_tts_realtime_model_config.delay_tokens_len)
            or input_done
        )
        request_id = str(payload.request_id)
        observe_events = ready and realtime_events_active()
        if observe_events:
            self._ensure_observability_state()
        if observe_events and request_id not in self._prefill_gate_ready_event_ids:
            self._prefill_gate_ready_event_ids.add(request_id)
            last_update = (
                buffered.messages[-1]
                if buffered is not None and buffered.messages
                else None
            )
            metadata = realtime_identity_metadata(getattr(payload, "data", None))
            metadata.update(
                {
                    "seq_no": getattr(last_update, "seq_no", None),
                    "new_stable_token_count": (
                        len(last_update.token_ids)
                        if last_update is not None
                        else len(initial_token_ids)
                    ),
                    "stable_token_count": token_count,
                    "pending_bytes": (
                        int(buffered.byte_count) if buffered is not None else 0
                    ),
                    "input_done": input_done,
                    "required_stable_token_count": int(
                        self._moss_tts_realtime_model_config.delay_tokens_len
                    ),
                    "short_input_done": bool(
                        input_done
                        and token_count
                        < int(self._moss_tts_realtime_model_config.delay_tokens_len)
                    ),
                }
            )
            self._emit_realtime_event(
                request_id,
                "prefill_gate_ready",
                metadata=metadata,
            )
        return ready

    def _run_request_builder(self, payload: Any, active_stage: str | None) -> Any:
        if realtime_events_active():
            initial_token_ids, initial_input_done = self._payload_initial_input(payload)
            metadata = realtime_identity_metadata(getattr(payload, "data", None))
            metadata.update(
                {
                    "initial_stable_token_count": len(initial_token_ids),
                    "initial_input_done": initial_input_done,
                }
            )
            self._emit_realtime_event(
                str(payload.request_id),
                "request_build_start",
                metadata=metadata,
                stage=active_stage,
            )
        return super()._run_request_builder(payload, active_stage)

    def _begin_session_turn(
        self,
        *,
        request_id: str,
        data: MossTTSRealtimeRequestData,
    ) -> tuple[MossTTSRealtimeSessionState, MossTTSRealtimeTurnState, bool]:
        state = data.state
        session_id = state.session_id
        turn_id = state.turn_id
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("MOSS-TTS-Realtime session_id must be non-empty")
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("MOSS-TTS-Realtime turn_id must be non-empty")
        turn_index = state.turn_index
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            raise TypeError("MOSS-TTS-Realtime turn_index must be an integer")
        if data.session_state is not None:
            raise RuntimeError(
                "MOSS-TTS-Realtime session state must be scheduler-owned"
            )
        if data.turn_state is not None:
            raise RuntimeError(
                "MOSS-TTS-Realtime turn state must be created during finalization"
            )

        sessions = self._moss_tts_realtime_sessions
        session_state = sessions.get(session_id)
        created = session_state is None
        if session_state is None:
            session_state = MossTTSRealtimeSessionState(
                session_id=session_id,
                model_config=data.model_config,
            )

        if turn_index != session_state.successful_turns:
            raise ValueError(
                "MOSS-TTS-Realtime turn_index does not match successful turns: "
                f"{turn_index} != {session_state.successful_turns}"
            )

        if created:
            sessions[session_id] = session_state
        try:
            turn = session_state.begin_turn(
                turn_id=turn_id,
                request_id=request_id,
                limits=self._moss_tts_realtime_limits,
            )
        except Exception:
            if created and sessions.get(session_id) is session_state:
                sessions.pop(session_id, None)
            raise

        data.session_state = session_state
        data.turn_state = turn
        return session_state, turn, created

    @staticmethod
    def _rollback_session_turn_claim(
        session_state: MossTTSRealtimeSessionState,
        turn: MossTTSRealtimeTurnState,
    ) -> None:
        if session_state.active_turn_id == turn.turn_id:
            session_state.active_turn_id = None

    def _streaming_session_slot(self, session_id: str) -> Any | None:
        slots = getattr(self.tree_cache, "slots", None)
        if not isinstance(slots, dict):
            raise TypeError(
                "MOSS-TTS-Realtime tree cache must expose streaming-session slots"
            )
        return slots.get(session_id)

    def _close_sglang_session_id(
        self,
        session_id: str,
        *,
        abort_inflight: bool = False,
        reason: str = "cleanup",
        request_id: str | None = None,
    ) -> bool:
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        controller = self.session_controller
        session = controller.get(session_id)
        slot_existed = self._streaming_session_slot(session_id) is not None
        existed = session is not None or slot_existed
        if (
            session is not None
            and abort_inflight
            and bool(getattr(session, "_inflight", False))
        ):
            session.abort_req()
        if session is not None:
            controller.close(CloseSessionReqInput(session_id=session_id))
            remaining = controller.get(session_id)
            if remaining is not None and not bool(
                getattr(remaining, "_inflight", False)
            ):
                controller.maybe_reap(time.monotonic(), interval=0.0)
        else:
            release_session = getattr(self.tree_cache, "release_session", None)
            if callable(release_session):
                release_session(session_id)
        remaining = controller.get(session_id)
        closed = remaining is None
        deferred = remaining is not None and bool(
            getattr(remaining, "close_on_finish", False)
        )
        if existed:
            if closed:
                self._bump_resource_total("physical_session_close_total")
                event_name = "physical_session_close"
            elif deferred:
                self._bump_resource_total("physical_session_close_deferred_total")
                event_name = "physical_session_close_deferred"
            else:
                self._bump_resource_total("physical_session_close_error_total")
                event_name = "physical_session_close_error"
            self._emit_realtime_event(
                request_id or session_id,
                event_name,
                metadata=self._event_metadata(
                    session_id=session_id,
                    reason=reason,
                    abort_inflight=abort_inflight,
                ),
            )
        return closed

    def _warm_session_reuse_error(
        self,
        session_state: MossTTSRealtimeSessionState,
        session: Any | None,
    ) -> str | None:
        if session is None:
            return "controller session is missing"
        if session.session_id != session_state.warm_session_id:
            return "controller session identity differs from host warm identity"
        if not bool(getattr(session, "streaming", False)):
            return "controller session is not streaming"
        if bool(getattr(session, "close_on_finish", False)):
            return "controller session is closing"
        if bool(getattr(session, "_inflight", False)):
            raise RuntimeError(
                "MOSS-TTS-Realtime warm session still has an inflight request"
            )
        req_nodes = getattr(session, "req_nodes", None)
        if not isinstance(req_nodes, dict) or len(req_nodes) != 1:
            return "controller session has no single last-successful request"
        cached_req = next(iter(req_nodes.values())).req
        if cached_req is None or not cached_req.finished():
            return "controller session last request is not successful"
        if getattr(cached_req, "session", None) is not session:
            return "controller session last request lost its owner"
        slot = self._streaming_session_slot(session.session_id)
        if slot is None:
            return "streaming-session KV slot is missing"
        if not bool(getattr(slot, "is_holding_kv", False)):
            return "streaming-session slot does not hold KV"
        expected_length = len(session_state.committed_rows)
        if session_state.warm_kv_length != expected_length:
            return "host warm length differs from committed ledger length"
        if int(getattr(slot, "kv_committed_len", -1)) != expected_length:
            return "streaming-session KV length differs from committed ledger"
        if _streaming_slot_allocated_len(slot, default=-1) < expected_length:
            return "streaming-session allocated KV is shorter than committed"
        return None

    def _get_or_open_sglang_session(
        self,
        session_state: MossTTSRealtimeSessionState,
        *,
        request_id: str,
        host_record_created: bool,
    ) -> _SGLangSessionAdmission:
        from sglang.srt.managers.io_struct import OpenSessionReqInput

        controller = self.session_controller
        if getattr(controller, "tree_cache", None) is not self.tree_cache:
            raise RuntimeError(
                "MOSS-TTS-Realtime session controller/cache identity changed"
            )

        replay_required = False
        warm_session_id = session_state.warm_session_id
        session = controller.get(warm_session_id) if warm_session_id else None
        if session_state.committed_rows:
            reuse_error = (
                self._warm_session_reuse_error(session_state, session)
                if warm_session_id is not None
                else "host warm identity is absent"
            )
            if reuse_error is None:
                self._bump_resource_total("session_cache_hit_total")
                self._emit_realtime_event(
                    request_id,
                    "session_cache_hit",
                    metadata=self._event_metadata(
                        session_id=session_state.session_id,
                        held_rows=len(session_state.committed_rows),
                    ),
                )
                return _SGLangSessionAdmission(
                    session=session,
                    opened=False,
                    replay_required=False,
                )
            logger.info(
                "MOSS-TTS-Realtime replaying session %s: %s",
                session_state.session_id,
                reuse_error,
            )
            self._bump_resource_total("session_cache_miss_total")
            self._bump_resource_total("ledger_replay_total")
            self._emit_realtime_event(
                request_id,
                "ledger_replay",
                metadata=self._event_metadata(
                    session_id=session_state.session_id,
                    reason=reuse_error,
                    replay_rows=len(session_state.committed_rows),
                ),
            )
            if warm_session_id is not None:
                if session is not None and bool(getattr(session, "_inflight", False)):
                    raise RuntimeError(
                        "MOSS-TTS-Realtime cannot replay while stale session is inflight"
                    )
                self._close_sglang_session_id(warm_session_id)
            session_state.release_warm_session()
            replay_required = True

        sglang_session_id = session_state.session_id
        session = controller.get(sglang_session_id)
        if host_record_created and session is not None:
            raise RuntimeError(
                "MOSS-TTS-Realtime SGLang session exists without a host record"
            )
        if session is not None:
            if bool(getattr(session, "_inflight", False)):
                raise RuntimeError(
                    "MOSS-TTS-Realtime stale physical session is still inflight"
                )
            self._close_sglang_session_id(sglang_session_id)
            session = controller.get(sglang_session_id)
            if session is not None:
                raise RuntimeError(
                    "MOSS-TTS-Realtime stale physical session cleanup is pending"
                )

        opened = False
        if session is None:
            capacity = getattr(self, "max_req_input_len", None)
            if isinstance(capacity, bool) or not isinstance(capacity, int):
                raise RuntimeError(
                    "MOSS-TTS-Realtime scheduler has no integer input capacity"
                )
            if capacity < 1:
                raise RuntimeError(
                    "MOSS-TTS-Realtime scheduler input capacity must be positive"
                )
            output = controller.open(
                OpenSessionReqInput(
                    capacity_of_str_len=capacity,
                    session_id=sglang_session_id,
                    streaming=True,
                )
            )
            if not output.success:
                raise RuntimeError(
                    f"failed to open SGLang session {sglang_session_id!r}"
                )
            session = controller.get(sglang_session_id)
            if session is None:
                raise RuntimeError(
                    "SGLang session controller reported success without a session"
                )
            opened = True
            self._bump_resource_total("physical_session_open_total")
            self._emit_realtime_event(
                request_id,
                "physical_session_open",
                metadata=self._event_metadata(
                    session_id=sglang_session_id,
                    replay_required=replay_required,
                ),
            )

        try:
            if session.session_id != sglang_session_id:
                raise RuntimeError("SGLang session identity does not match host state")
            if not bool(getattr(session, "streaming", False)):
                raise RuntimeError(
                    "MOSS-TTS-Realtime requires a streaming SGLang session"
                )
            if bool(getattr(session, "close_on_finish", False)):
                raise RuntimeError("MOSS-TTS-Realtime SGLang session is closing")
            if bool(getattr(session, "_inflight", False)):
                raise RuntimeError(
                    "MOSS-TTS-Realtime SGLang session is already inflight"
                )

            req_nodes = getattr(session, "req_nodes", None)
            if not isinstance(req_nodes, dict):
                raise TypeError("SGLang session must expose a request-node registry")
            if len(req_nodes) > 1:
                raise RuntimeError(
                    "streaming SGLang session has multiple request nodes"
                )
            if req_nodes:
                raise RuntimeError(
                    "MOSS-TTS-Realtime fresh or replay session has request history"
                )
        except Exception:
            if opened:
                self._close_new_sglang_session(
                    sglang_session_id,
                    request_id=request_id,
                )
            raise
        return _SGLangSessionAdmission(
            session=session,
            opened=opened,
            replay_required=replay_required,
        )

    def _live_turn_request_ids(self) -> set[str]:
        request_ids: set[str] = set()
        for req in getattr(self, "waiting_queue", ()):
            rid = getattr(req, "rid", None)
            if rid is not None and not req.finished():
                request_ids.add(str(rid))

        async_pending_batch = None
        async_pending = getattr(self, "_async_pending", None)
        if async_pending is not None:
            async_pending_batch = getattr(async_pending, "batch", None)
        for batch in (
            getattr(self, "running_batch", None),
            getattr(self, "cur_batch", None),
            getattr(self, "last_batch", None),
            async_pending_batch,
        ):
            if batch is None:
                continue
            for req in getattr(batch, "reqs", ()):
                rid = getattr(req, "rid", None)
                if rid is not None and not req.finished():
                    request_ids.add(str(rid))

        request_ids.update(getattr(self, "_parked_input", {}).keys())
        request_ids.update(
            request_id
            for request_id, data in getattr(
                self,
                "_moss_tts_realtime_requests",
                {},
            ).items()
            if not bool(getattr(data, "lifecycle_finalized", False))
        )
        return request_ids

    def _enforce_active_turn_quota(self, request_id: str) -> None:
        active_ids = self._live_turn_request_ids()
        active_ids.discard(request_id)
        max_active_turns = int(
            getattr(
                self,
                "_max_active_turns",
                self._moss_tts_realtime_limits.max_active_turns,
            )
        )
        if len(active_ids) >= max_active_turns:
            self._record_admission_rejection(
                request_id,
                reason="active_turn_limit",
                message=(
                    "MOSS-TTS-Realtime active-turn limit exceeded: "
                    f"{len(active_ids) + 1} > {max_active_turns}"
                ),
            )

    def _enforce_resource_admission(
        self,
        request_id: str,
        data: MossTTSRealtimeRequestData,
    ) -> None:
        self._enforce_active_turn_quota(request_id)
        limits = self._moss_tts_realtime_limits
        state = data.state
        session_id = state.session_id
        if not isinstance(session_id, str) or not session_id.strip():
            self._record_admission_rejection(
                request_id,
                reason="invalid_session_id",
                message="MOSS-TTS-Realtime session_id must be non-empty",
                error_type=ValueError,
            )

        sessions = self._moss_tts_realtime_sessions
        session_state = sessions.get(session_id)
        if session_state is not None:
            if session_state.closed or session_state.close_requested:
                self._record_admission_rejection(
                    request_id,
                    reason="session_closed",
                    message="cannot start a turn on a closed realtime session",
                )
            if session_state.active_turn_id is not None:
                self._record_admission_rejection(
                    request_id,
                    reason="session_active_turn",
                    message=(
                        "session already has active turn "
                        f"{session_state.active_turn_id!r}"
                    ),
                )
        elif len(sessions) >= limits.max_sessions:
            self._record_admission_rejection(
                request_id,
                reason="session_limit",
                message=(
                    "MOSS-TTS-Realtime session limit exceeded: "
                    f"{len(sessions) + 1} > {limits.max_sessions}"
                ),
            )

        max_new_tokens = data.max_new_tokens
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            self._record_admission_rejection(
                request_id,
                reason="invalid_max_new_tokens",
                message="MOSS-TTS-Realtime max_new_tokens must be an integer",
                error_type=TypeError,
            )
        if max_new_tokens < 1:
            self._record_admission_rejection(
                request_id,
                reason="invalid_max_new_tokens",
                message="MOSS-TTS-Realtime max_new_tokens must be positive",
                error_type=ValueError,
            )
        prompt_rows = data.prompt_rows
        if not isinstance(prompt_rows, torch.Tensor):
            self._record_admission_rejection(
                request_id,
                reason="invalid_prompt_rows",
                message="MOSS-TTS-Realtime prompt_rows must be a tensor",
                error_type=TypeError,
            )
        if (
            prompt_rows.ndim != 2
            or int(prompt_rows.shape[1]) != int(data.model_config.rvq) + 1
        ):
            self._record_admission_rejection(
                request_id,
                reason="invalid_prompt_rows",
                message=(
                    "MOSS-TTS-Realtime prompt_rows must have shape "
                    f"[T, {int(data.model_config.rvq) + 1}]"
                ),
                error_type=ValueError,
            )

        buffered = self._buffered_input_updates.get(request_id)
        buffered_tokens = int(buffered.token_count) if buffered is not None else 0
        available_prefill_tokens = len(data.initial_token_ids) + buffered_tokens
        prefill_rows = min(
            int(data.model_config.delay_tokens_len),
            available_prefill_tokens,
        )
        committed_rows = (
            len(session_state.committed_rows) if session_state is not None else 0
        )
        required_rows = (
            committed_rows + int(prompt_rows.shape[0]) + prefill_rows + max_new_tokens
        )
        if required_rows > self._max_session_rows:
            self._record_admission_rejection(
                request_id,
                reason="session_context_limit",
                message=(
                    "MOSS-TTS-Realtime session context limit exceeded: "
                    f"committed_rows={committed_rows}, "
                    f"turn_prompt_rows={int(prompt_rows.shape[0])}, "
                    f"prefill_rows={prefill_rows}, "
                    f"max_new_tokens={max_new_tokens}, "
                    f"required_rows={required_rows}, "
                    f"max_session_rows={self._max_session_rows}"
                ),
                error_type=ValueError,
            )

        logical = self._logical_reservation_snapshot()
        prior_reservation = (
            self._session_reservation_rows(session_state)
            if session_state is not None
            else 0
        )
        held_session_reservations = logical["held_session_reservations"]
        if prior_reservation == 0:
            held_session_reservations += 1
        if held_session_reservations > limits.max_held_sessions:
            self._record_admission_rejection(
                request_id,
                reason="held_session_limit",
                message=(
                    "MOSS-TTS-Realtime held-session limit exceeded: "
                    f"{held_session_reservations} > {limits.max_held_sessions}"
                ),
            )

        held_kv_reservation = (
            logical["held_kv_token_reservations"] - prior_reservation + required_rows
        )
        if held_kv_reservation > self._max_held_kv_tokens:
            self._record_admission_rejection(
                request_id,
                reason="held_kv_limit",
                message=(
                    "MOSS-TTS-Realtime held-KV reservation limit exceeded: "
                    f"{held_kv_reservation} > {self._max_held_kv_tokens}"
                ),
            )

        data.context_reservation_rows = required_rows

    @staticmethod
    def _build_sampling_params(
        data: MossTTSRealtimeRequestData,
    ) -> Any:
        from sglang.srt.sampling.sampling_params import SamplingParams

        model_config = data.model_config
        vocab_size = int(getattr(model_config, "vocab_size"))
        max_new_tokens = int(data.max_new_tokens or 0)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=[int(model_config.audio_eos_token)],
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)
        return sampling_params

    @classmethod
    def _build_tokenized_session_req(
        cls,
        request_id: str,
        data: MossTTSRealtimeRequestData,
        *,
        session_id: str,
        suffix_input_ids: Sequence[int],
    ) -> Any:
        from sglang.srt.managers.io_struct import (
            SessionParams,
            TokenizedGenerateReqInput,
        )

        return TokenizedGenerateReqInput(
            input_text="",
            input_ids=array("q", (int(token_id) for token_id in suffix_input_ids)),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=cls._build_sampling_params(data),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
            session_params=SessionParams(
                id=session_id,
                rid=None,
                offset=0,
                replace=False,
                drop_previous_output=False,
            ),
            rid=request_id,
            extra_key=(
                "moss-tts-realtime-session:"
                + hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            ),
        )

    @staticmethod
    def _normalize_session_origin_ids(
        req: Any,
        *,
        canonical_ids: Sequence[int],
        committed_row_count: int,
        has_prior_request: bool,
        audio_eos_token: int,
    ) -> None:
        expected = [int(token_id) for token_id in canonical_ids]
        normalized_fields: dict[str, array[int]] = {}
        for field_name in (
            "origin_input_ids",
            "origin_input_ids_unpadded",
        ):
            raw = getattr(req, field_name, None)
            if not isinstance(raw, (list, tuple, array)):
                raise TypeError(f"SGLang request {field_name} must be a sequence")
            values = [int(token_id) for token_id in raw]
            if has_prior_request:
                if len(values) != len(expected) + 1:
                    raise RuntimeError(
                        f"SGLang request {field_name} must contain exactly one "
                        "prior terminal id"
                    )
                if (
                    committed_row_count >= len(values)
                    or values[committed_row_count] != audio_eos_token
                ):
                    raise RuntimeError(
                        f"SGLang request {field_name} is missing prior audio EOS "
                        "at the committed-ledger boundary"
                    )
                if values.count(audio_eos_token) != 1:
                    raise RuntimeError(
                        f"SGLang request {field_name} contains duplicate audio EOS"
                    )
                values = (
                    values[:committed_row_count] + values[committed_row_count + 1 :]
                )
            if values != expected:
                raise RuntimeError(
                    f"SGLang request {field_name} does not match canonical row hashes"
                )
            normalized_fields[field_name] = array("q", values)

        req.origin_input_ids = normalized_fields["origin_input_ids"]
        req.origin_input_ids_unpadded = normalized_fields["origin_input_ids_unpadded"]

    def _close_new_sglang_session(
        self,
        session_id: str,
        *,
        request_id: str | None = None,
    ) -> None:
        try:
            self._close_sglang_session_id(
                session_id,
                abort_inflight=True,
                reason="admission_rollback",
                request_id=request_id,
            )
        except Exception:
            logger.exception(
                "Failed to close newly opened MOSS-TTS-Realtime session %s",
                session_id,
            )

    def _assert_scheduler_thread(self, operation: str) -> None:
        scheduler_thread_id = getattr(self, "_scheduler_thread_id", None)
        current_thread_id = threading.get_ident()
        if scheduler_thread_id is None:
            raise RuntimeError(
                f"{operation} requires an active scheduler thread, but the "
                "scheduler is not running"
            )
        if current_thread_id != scheduler_thread_id:
            raise RuntimeError(
                f"{operation} must run on scheduler thread {scheduler_thread_id}, "
                f"got thread {current_thread_id}"
            )

    def _replay_buffered_input_updates(
        self,
        request_id: str,
        req_data: MossTTSRealtimeRequestData,
    ) -> None:
        buffered = self._buffered_input_updates.get(request_id)
        if buffered is None:
            return
        while buffered.messages:
            message = buffered.messages[0]
            apply_moss_tts_realtime_input_update(req_data, message)
            buffered.messages.popleft()
            buffered.token_count -= len(message.token_ids)
            buffered.byte_count -= message.byte_count
        if buffered.token_count or buffered.byte_count:
            raise RuntimeError(
                f"input update replay accounting mismatch for {request_id!r}"
            )
        self._buffered_input_updates.pop(request_id, None)

    def _finalize_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
    ) -> MossTTSRealtimeRequestData:
        del pending_stream_done
        self._assert_scheduler_thread("MOSS-TTS-Realtime request finalization")
        if not isinstance(req_data, MossTTSRealtimeRequestData):
            raise TypeError(
                "MOSS-TTS-Realtime finalization requires MossTTSRealtimeRequestData"
            )
        if req_data.req is not None:
            raise RuntimeError("MOSS-TTS-Realtime request was finalized twice")

        request_id = str(payload.request_id)
        self._enforce_resource_admission(request_id, req_data)
        session_state, turn, host_record_created = self._begin_session_turn(
            request_id=request_id,
            data=req_data,
        )
        session = None
        session_opened = False
        inflight_before_create = False
        try:
            turn.seed_initial_input(
                req_data.initial_token_ids,
                input_done=bool(req_data.state.input_done),
            )
            self._replay_buffered_input_updates(request_id, req_data)
            self._ensure_observability_state()
            self._queued_input_tokens_high_water = max(
                self._queued_input_tokens_high_water,
                len(turn.pending_input),
            )
            self._queued_input_bytes_high_water = max(
                self._queued_input_bytes_high_water,
                turn.pending_input.pending_bytes,
            )
            if not turn.ready_for_prefill:
                raise RuntimeError(
                    "MOSS-TTS-Realtime request was finalized before prefill readiness"
                )

            prefill_token_ids = turn.take_prefill_tokens()
            prefill_rows = build_moss_tts_realtime_prefill_rows(
                prefill_token_ids,
                model_config=req_data.model_config,
            )
            turn_prompt_rows = req_data.prompt_rows
            if not isinstance(turn_prompt_rows, torch.Tensor):
                raise TypeError("MOSS-TTS-Realtime prompt_rows must be a tensor")
            if (
                turn_prompt_rows.ndim != 2
                or int(turn_prompt_rows.shape[1]) != int(req_data.model_config.rvq) + 1
            ):
                raise ValueError(
                    "MOSS-TTS-Realtime prompt_rows must have shape "
                    f"[T, {int(req_data.model_config.rvq) + 1}]"
                )
            turn_prompt_rows = torch.cat(
                [
                    turn_prompt_rows.to(device="cpu", dtype=torch.long),
                    prefill_rows,
                ],
                dim=0,
            )

            base_cache_ids = build_moss_tts_realtime_row_cache_key_ids(
                turn_prompt_rows[: -len(prefill_token_ids)],
                model_config=req_data.model_config,
            )
            if not isinstance(req_data.input_ids, torch.Tensor):
                raise TypeError("prepared MOSS-TTS-Realtime input_ids must be a tensor")
            if base_cache_ids != [int(value) for value in req_data.input_ids.tolist()]:
                raise RuntimeError(
                    "prepared prompt row cache ids do not match raw rows"
                )
            suffix_cache_ids = (
                base_cache_ids
                + build_moss_tts_realtime_row_cache_key_ids(
                    prefill_rows,
                    model_config=req_data.model_config,
                )
            )

            turn.ledger.extend_rows(turn_prompt_rows.tolist())
            canonical_rows = torch.tensor(turn.ledger.rows, dtype=torch.long)
            canonical_ids = build_moss_tts_realtime_row_cache_key_ids(
                canonical_rows,
                model_config=req_data.model_config,
            )
            if realtime_events_active():
                canonical_metadata = realtime_identity_metadata(req_data.state)
                canonical_metadata.update(
                    {
                        "seq_no": (
                            turn.pending_input.next_seq_no - 1
                            if turn.pending_input.next_seq_no
                            else None
                        ),
                        "stable_token_count": turn.pending_input.total_received_tokens,
                        "prefill_token_count": len(prefill_token_ids),
                        "remaining_stable_token_count": len(turn.pending_input),
                        "base_prompt_rows": len(base_cache_ids),
                        "canonical_prompt_rows": int(canonical_rows.shape[0]),
                        "committed_rows": len(session_state.committed_rows),
                        "canonical_cache_ids": len(canonical_ids),
                    }
                )
                self._emit_realtime_event(
                    request_id,
                    "canonical_rows_ready",
                    metadata=canonical_metadata,
                )
            req_data.generation_row_start = len(turn.ledger.rows)
            req_data.prompt_rows = canonical_rows
            req_data.state.prompt_rows = canonical_rows
            if req_data.provisional_output_id is None:
                req_data.provisional_output_id = int(
                    req_data.model_config.reference_audio_pad
                )
            provisional_output_id = req_data.provisional_output_id
            vocab_size = int(getattr(req_data.model_config, "vocab_size"))
            if (
                isinstance(provisional_output_id, bool)
                or not isinstance(provisional_output_id, int)
                or provisional_output_id < 0
                or provisional_output_id >= vocab_size
            ):
                raise ValueError(
                    "provisional output id must be inside the text vocabulary"
                )
            if provisional_output_id == int(req_data.model_config.audio_eos_token):
                raise ValueError("provisional output id must differ from audio EOS")

            admission = self._get_or_open_sglang_session(
                session_state,
                request_id=request_id,
                host_record_created=host_record_created,
            )
            session = admission.session
            session_opened = admission.opened
            has_prior_request = not admission.replay_required and bool(
                session.req_nodes
            )
            session_input_ids = (
                canonical_ids if admission.replay_required else suffix_cache_ids
            )
            tokenized_req = self._build_tokenized_session_req(
                request_id,
                req_data,
                session_id=session.session_id,
                suffix_input_ids=session_input_ids,
            )
            inflight_before_create = bool(session._inflight)
            req = session.create_req(
                tokenized_req,
                tokenizer=None,
                vocab_size=vocab_size,
                eos_token_ids={int(req_data.model_config.audio_eos_token)},
            )
            if getattr(req, "to_finish", None) is not None:
                raise RuntimeError("SGLang rejected the realtime session request")
            if req.session is not session:
                raise RuntimeError("SGLang request lost its streaming session owner")

            self._normalize_session_origin_ids(
                req,
                canonical_ids=canonical_ids,
                committed_row_count=len(session_state.committed_rows),
                has_prior_request=has_prior_request,
                audio_eos_token=int(req_data.model_config.audio_eos_token),
            )
            req._input_embeds_are_projected = True
            req._codec_suppress_tokens = None
            req_data.req = req
            req_data.output_ids = req.output_ids
            req_data.input_ids = torch.tensor(canonical_ids, dtype=torch.long)
            req_data.input_embeds_are_projected = True
            req_data.backend_session_id = session.session_id
            req_data.ledger_replay = admission.replay_required
            req_data.backend_session_invalidated = False
            req_data.lifecycle_finalized = False
            live_requests = self._moss_tts_realtime_requests
            existing = live_requests.get(request_id)
            if existing is not None and existing is not req_data:
                raise RuntimeError(
                    "MOSS-TTS-Realtime request id is already lifecycle-owned"
                )
            live_requests[request_id] = req_data
            if host_record_created:
                self._bump_resource_total("host_session_open_total")
            self._bump_resource_total("turn_admitted_total")
            self._emit_realtime_event(
                request_id,
                "turn_admitted",
                metadata=self._event_metadata(
                    session_id=session_state.session_id,
                    turn_id=turn.turn_id,
                    context_reservation_rows=req_data.context_reservation_rows,
                    ledger_replay=admission.replay_required,
                ),
            )
            self._refresh_resource_high_water()
            return req_data
        except Exception:
            self._moss_tts_realtime_requests.pop(request_id, None)
            if (
                session is not None
                and not inflight_before_create
                and bool(getattr(session, "_inflight", False))
            ):
                session.abort_req()
            if session_opened and session is not None:
                self._close_new_sglang_session(
                    session.session_id,
                    request_id=request_id,
                )
            req_data.backend_session_id = None
            req_data.ledger_replay = False
            req_data.backend_session_invalidated = False
            req_data.lifecycle_finalized = False
            req_data.context_reservation_rows = 0
            self._rollback_session_turn_claim(session_state, turn)
            if host_record_created:
                sessions = self._moss_tts_realtime_sessions
                if sessions.get(session_state.session_id) is session_state:
                    sessions.pop(session_state.session_id, None)
                if req_data.session_state is session_state:
                    req_data.session_state = None
            raise

    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        super()._enqueue_built_request(
            payload,
            pending_stream_done,
            req_data,
            request_admission_lock_held=request_admission_lock_held,
        )
        if not isinstance(req_data, MossTTSRealtimeRequestData):
            return
        req = req_data.req
        if (
            req is None
            or not realtime_events_active()
            or not any(item is req for item in self.waiting_queue)
        ):
            return
        turn = req_data.turn_state
        prefix_indices = getattr(req, "prefix_indices", None)
        metadata = realtime_identity_metadata(req_data.state)
        metadata.update(
            {
                "seq_no": (
                    turn.pending_input.next_seq_no - 1
                    if turn is not None and turn.pending_input.next_seq_no
                    else None
                ),
                "stable_token_count": (
                    turn.pending_input.total_received_tokens
                    if turn is not None
                    else len(req_data.initial_token_ids)
                ),
                "prefill_token_count": (
                    len(turn.prefill_token_ids) if turn is not None else 0
                ),
                "prompt_rows": (
                    int(req_data.prompt_rows.shape[0])
                    if isinstance(req_data.prompt_rows, torch.Tensor)
                    else None
                ),
                "cached_rows": (
                    len(prefix_indices) if prefix_indices is not None else 0
                ),
                "queue_depth": len(self.waiting_queue),
            }
        )
        self._emit_realtime_event(
            str(payload.request_id),
            "scheduler_queue_enter",
            metadata=metadata,
        )

    @staticmethod
    def _is_live_decode_req(req: Any) -> bool:
        return not req.finished() and not bool(getattr(req, "is_retracted", False))

    @staticmethod
    def _batch_aligned_int(batch: Any, field_name: str, index: int) -> int:
        values = getattr(batch, field_name, None)
        if values is None:
            raise RuntimeError(f"decode batch is missing {field_name}")
        try:
            value = values[index]
        except (IndexError, TypeError) as exc:
            raise RuntimeError(
                f"decode batch {field_name} is not request aligned"
            ) from exc
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RuntimeError(f"decode batch {field_name} entry is not scalar")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"decode batch {field_name} entries must be integers")
        return int(value)

    @staticmethod
    def _turn_has_decode_input(turn: MossTTSRealtimeTurnState) -> bool:
        return bool(
            turn.phase is MossTTSRealtimeTurnPhase.DRAINING
            or len(turn.pending_input)
            or turn.pending_input.input_done
        )

    def _make_parked_record(
        self,
        batch: Any,
        index: int,
        req: Any,
        *,
        now: float,
        park_sequence: int,
    ) -> _ParkedRealtimeRequest:
        data = req._omni_data
        if not isinstance(data, MossTTSRealtimeRequestData):
            raise TypeError("parked request has the wrong request-data type")
        turn = data.turn_state
        if turn is None or turn.provisional_frame is None:
            raise RuntimeError("only a provisional realtime frame may be parked")

        req_pool_idx = getattr(req, "req_pool_idx", None)
        if isinstance(req_pool_idx, torch.Tensor):
            if req_pool_idx.numel() != 1:
                raise RuntimeError("request req-pool index is not scalar")
            req_pool_idx = req_pool_idx.item()
        if isinstance(req_pool_idx, bool) or not isinstance(req_pool_idx, int):
            raise RuntimeError("parked request has no integer req-pool index")
        batch_req_pool_idx = self._batch_aligned_int(batch, "req_pool_indices", index)
        if req_pool_idx != batch_req_pool_idx:
            raise RuntimeError("request and batch req-pool indices do not match")

        seq_len = self._batch_aligned_int(batch, "seq_lens", index)
        seq_len_cpu = self._batch_aligned_int(batch, "seq_lens_cpu", index)
        if seq_len != seq_len_cpu:
            raise RuntimeError("device and CPU sequence lengths do not match")
        orig_seq_len = self._batch_aligned_int(batch, "orig_seq_lens", index)
        kv_committed_len = getattr(req, "kv_committed_len", None)
        if isinstance(kv_committed_len, bool) or not isinstance(kv_committed_len, int):
            raise RuntimeError("parked request has no committed KV length")
        if kv_committed_len != seq_len:
            raise RuntimeError(
                "request committed KV length does not match decode batch length"
            )

        output_id = self._batch_output_id(batch, index)
        provisional_id = data.provisional_output_id
        if isinstance(provisional_id, bool) or not isinstance(provisional_id, int):
            raise RuntimeError("parked request has no integer provisional id")
        if output_id != provisional_id:
            raise RuntimeError("batch output id is not the provisional id")
        if not req.output_ids or int(req.output_ids[-1]) != provisional_id:
            raise RuntimeError("request output history is not provisional")

        return _ParkedRealtimeRequest(
            req=req,
            req_pool_idx=req_pool_idx,
            seq_len=seq_len,
            orig_seq_len=orig_seq_len,
            provisional_output_id=provisional_id,
            parked_at=now,
            last_input_at=now,
            park_sequence=park_sequence,
        )

    def _park_starved_requests(self, batch: Any) -> int:
        now = time.monotonic()
        candidates: list[
            tuple[int, Any, MossTTSRealtimeTurnState, _ParkedRealtimeRequest]
        ] = []
        next_sequence = self._park_sequence
        for index, req in enumerate(batch.reqs):
            if not self._is_live_decode_req(req):
                continue
            data = req._omni_data
            if not isinstance(data, MossTTSRealtimeRequestData):
                raise _RealtimeMaterializationFailure(
                    req.rid,
                    TypeError("MOSS-TTS-Realtime decode has the wrong request data"),
                )
            turn = data.turn_state
            if turn is None:
                raise _RealtimeMaterializationFailure(
                    req.rid,
                    RuntimeError("decode request is missing turn state"),
                )
            if turn.provisional_frame is None or self._turn_has_decode_input(turn):
                continue
            try:
                next_sequence += 1
                record = self._make_parked_record(
                    batch,
                    index,
                    req,
                    now=now,
                    park_sequence=next_sequence,
                )
            except Exception as exc:
                raise _RealtimeMaterializationFailure(req.rid, exc) from exc
            candidates.append((index, req, turn, record))

        if not candidates:
            return 0

        parked_ids = {req.rid for _, req, _, _ in candidates}
        keep_indices = [
            index for index, req in enumerate(batch.reqs) if req.rid not in parked_ids
        ]
        with self._request_admission_lock:
            for _, req, _, _ in candidates:
                if req.rid in self._parked_input:
                    raise _RealtimeMaterializationFailure(
                        req.rid,
                        RuntimeError("request is already present in parked_input"),
                    )

            committed: list[tuple[Any, MossTTSRealtimeTurnState]] = []
            try:
                for _, req, turn, record in candidates:
                    if turn.next_text_token() is not None:
                        raise RuntimeError(
                            "starved request unexpectedly produced a text token"
                        )
                    self._parked_input[req.rid] = record
                    committed.append((req, turn))
                batch.filter_batch(keep_indices=keep_indices)
                if batch.reqs:
                    # filter_batch knows nothing about the scheduler-owned
                    # output-id tensor; realign it for the materialization step
                    # that follows parking.
                    self._resync_batch_output_ids(batch)
            except Exception as exc:
                for req, turn in committed:
                    self._parked_input.pop(req.rid, None)
                    if turn.phase is MossTTSRealtimeTurnPhase.PARKED_INPUT:
                        turn.transition_to(MossTTSRealtimeTurnPhase.RUNNING)
                request_id = candidates[0][1].rid
                raise _RealtimeMaterializationFailure(
                    request_id,
                    exc,
                ) from exc

            self._park_sequence = next_sequence
            self._park_total += len(candidates)
            self._bump_resource_total("park_total", len(candidates))
            for _, req, turn, record in candidates:
                self._emit_realtime_event(
                    req.rid,
                    "turn_parked",
                    metadata=self._event_metadata(
                        session_id=turn.session_id,
                        turn_id=turn.turn_id,
                        park_sequence=record.park_sequence,
                    ),
                )

        batch.batch_is_full = False
        return len(candidates)

    def _validate_parked_record(self, record: _ParkedRealtimeRequest) -> None:
        req = record.req
        if req.finished() or bool(getattr(req, "is_retracted", False)):
            raise RuntimeError("parked request is no longer a live decode request")
        data = req._omni_data
        if not isinstance(data, MossTTSRealtimeRequestData):
            raise TypeError("parked request has the wrong request-data type")
        turn = data.turn_state
        if turn is None or turn.is_terminal:
            raise RuntimeError("parked request has no live turn state")
        if turn.provisional_frame is None:
            raise RuntimeError("parked request lost its provisional audio frame")
        if not self._turn_has_decode_input(turn):
            raise RuntimeError("parked request is not ready to wake")
        if int(req.req_pool_idx) != record.req_pool_idx:
            raise RuntimeError("parked request req-pool ownership changed")
        if int(req.kv_committed_len) != record.seq_len:
            raise RuntimeError("parked request committed KV length changed")
        if not req.output_ids or int(req.output_ids[-1]) != (
            record.provisional_output_id
        ):
            raise RuntimeError("parked request provisional output id changed")
        if data.provisional_output_id != record.provisional_output_id:
            raise RuntimeError("parked request-data provisional id changed")

    def _build_parked_decode_batch(
        self,
        records: Sequence[_ParkedRealtimeRequest],
    ) -> ScheduleBatch:
        reqs = [record.req for record in records]
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        device = batch.device
        batch.req_pool_indices = torch.tensor(
            [record.req_pool_idx for record in records],
            dtype=torch.int64,
            device=device,
        )
        seq_lens = [record.seq_len for record in records]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            [record.orig_seq_len for record in records],
            dtype=torch.int32,
            device=device,
        )
        batch.seq_lens_sum = sum(seq_lens)
        batch.output_ids = torch.tensor(
            [record.provisional_output_id for record in records],
            dtype=torch.int64,
            device=device,
        )
        if batch.return_logprob:
            batch.top_logprobs_nums = [req.top_logprobs_num for req in reqs]
            batch.token_ids_logprobs = [list(req.origin_input_ids) for req in reqs]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            self.model_config.vocab_size,
        )
        return batch

    def _wake_parked_requests(self) -> int:
        with self._request_admission_lock:
            records: list[_ParkedRealtimeRequest] = []
            for record in self._parked_input.values():
                data = record.req._omni_data
                turn = data.turn_state
                if turn is not None and self._turn_has_decode_input(turn):
                    try:
                        self._validate_parked_record(record)
                    except Exception as exc:
                        raise _RealtimeMaterializationFailure(
                            record.req.rid,
                            exc,
                        ) from exc
                    records.append(record)
            if not records:
                return 0

            running_ids = {req.rid for req in getattr(self.running_batch, "reqs", ())}
            duplicate_ids = running_ids.intersection(
                record.req.rid for record in records
            )
            if duplicate_ids:
                request_id = sorted(duplicate_ids)[0]
                raise _RealtimeMaterializationFailure(
                    request_id,
                    RuntimeError("request exists in both runnable and parked state"),
                )

            try:
                wake_batch = self._build_parked_decode_batch(records)
                if self.running_batch is None or not getattr(
                    self.running_batch, "reqs", ()
                ):
                    self.running_batch = wake_batch
                else:
                    output_ids = getattr(self.running_batch, "output_ids", None)
                    if not isinstance(output_ids, torch.Tensor) or output_ids.ndim != 1:
                        raise RuntimeError(
                            "runnable batch has no aligned output ids for wake merge"
                        )
                    if int(output_ids.shape[0]) != len(self.running_batch.reqs):
                        raise RuntimeError(
                            "runnable batch output ids are misaligned before wake"
                        )
                    self.running_batch.merge_batch(wake_batch)
            except _RealtimeMaterializationFailure:
                raise
            except Exception as exc:
                raise _RealtimeMaterializationFailure(
                    records[0].req.rid,
                    exc,
                ) from exc

            for record in records:
                self._parked_input.pop(record.req.rid, None)
            self._wake_total += len(records)
            self._bump_resource_total("wake_total", len(records))
            for record in records:
                turn = record.req._omni_data.turn_state
                self._emit_realtime_event(
                    record.req.rid,
                    "turn_woken",
                    metadata=self._event_metadata(
                        session_id=(turn.session_id if turn is not None else None),
                        turn_id=(turn.turn_id if turn is not None else None),
                        park_sequence=record.park_sequence,
                    ),
                )
            self.running_batch.batch_is_full = False
            return len(records)

    def _buffer_input_update(
        self,
        request_id: str,
        message: InputUpdateMessage,
    ) -> _BufferedInputUpdates:
        buffered = self._buffered_input_updates.get(request_id)
        if buffered is None:
            buffered = _BufferedInputUpdates()

        pending_tokens = buffered.token_count + len(message.token_ids)
        pending_bytes = buffered.byte_count + message.byte_count
        update_count = len(buffered.messages) + 1
        limits = self._moss_tts_realtime_limits
        if pending_tokens > limits.max_pending_text_tokens:
            raise ValueError(
                "pending input-update token limit exceeded: "
                f"{pending_tokens} > {limits.max_pending_text_tokens}"
            )
        if pending_bytes > limits.max_pending_text_bytes:
            raise ValueError(
                "pending input-update byte limit exceeded: "
                f"{pending_bytes} > {limits.max_pending_text_bytes}"
            )
        if update_count > limits.max_input_updates:
            raise ValueError(
                "input update limit exceeded: "
                f"{update_count} > {limits.max_input_updates}"
            )

        buffered.messages.append(message)
        buffered.token_count = pending_tokens
        buffered.byte_count = pending_bytes
        buffered.input_done = buffered.input_done or message.input_done
        self._buffered_input_updates[request_id] = buffered
        return buffered

    def _fail_input_update_request(
        self,
        request_id: str,
        message: Any,
        error: Exception,
    ) -> None:
        logger.error(
            "MOSS-TTS-Realtime input update failed for request %s: %s",
            request_id,
            error,
        )
        self._bump_resource_total("input_update_rejected_total")
        self._emit_realtime_event(
            request_id,
            "input_update_rejected",
            metadata={
                "seq_no": getattr(message, "seq_no", None),
                "error": str(error),
            },
        )
        self._emit_request_error(request_id, error)
        self.abort(request_id)

    def _on_input_update(self, request_id: str, message: Any) -> None:
        try:
            with self._request_admission_lock:
                if self._is_input_update_terminal_locked(request_id):
                    return
                if request_id in self._aborted_request_ids:
                    return
                if not isinstance(message, InputUpdateMessage):
                    raise TypeError(
                        "input_update scheduler messages must carry an "
                        "InputUpdateMessage"
                    )
                if message.request_id != request_id:
                    raise ValueError(
                        "input update request identity mismatch: "
                        f"{message.request_id!r} != {request_id!r}"
                    )

                req_data = self._find_request_data(request_id)
                if req_data is None:
                    buffered = self._buffer_input_update(request_id, message)
                    disposition = MossTTSRealtimeUpdateDisposition.ACCEPTED
                    pending_tokens = buffered.token_count
                    pending_bytes = buffered.byte_count
                else:
                    disposition = apply_moss_tts_realtime_input_update(
                        req_data,
                        message,
                    )
                    turn = req_data.turn_state
                    pending_tokens = len(turn.pending_input)
                    pending_bytes = turn.pending_input.pending_bytes

                if request_id in self._deferred_request_payloads and (
                    message.token_ids or message.input_done
                ):
                    self._dirty_deferred_request_ids.add(request_id)

                if disposition is MossTTSRealtimeUpdateDisposition.DUPLICATE:
                    self._bump_resource_total("input_update_duplicate_total")
                    event_name = "input_update_duplicate"
                else:
                    self._bump_resource_total("input_update_accepted_total")
                    event_name = "input_update_accepted"
                    record = self._parked_input.get(request_id)
                    if record is not None:
                        record.last_input_at = time.monotonic()
                self._emit_realtime_event(
                    request_id,
                    event_name,
                    metadata={
                        "seq_no": message.seq_no,
                        "pending_tokens": pending_tokens,
                        "pending_bytes": pending_bytes,
                    },
                )
        except Exception as exc:
            self._fail_input_update_request(request_id, message, exc)
        self._refresh_resource_high_water()

    def is_input_update_terminal(self, request_id: str) -> bool:
        lock = getattr(self, "_request_admission_lock", None)
        if lock is None:
            return False
        with lock:
            return self._is_input_update_terminal_locked(request_id)

    def _is_input_update_terminal_locked(self, request_id: str) -> bool:
        terminal_ids = getattr(self, "_input_update_terminal_ids", None)
        return terminal_ids is not None and request_id in terminal_ids

    def _mark_input_update_terminal(self, request_id: str) -> None:
        lock = getattr(self, "_request_admission_lock", None)
        if lock is None:
            return
        with lock:
            self._buffered_input_updates.pop(request_id, None)
            gate_event_ids = getattr(self, "_prefill_gate_ready_event_ids", None)
            if gate_event_ids is not None:
                gate_event_ids.discard(request_id)
            terminal_ids = self._input_update_terminal_ids
            if request_id in terminal_ids:
                return
            terminal_order = self._input_update_terminal_order
            while len(terminal_ids) >= self._terminal_tombstone_limit:
                terminal_ids.discard(terminal_order.popleft())
            terminal_ids.add(request_id)
            terminal_order.append(request_id)

    def _remove_parked_request(
        self,
        request_id: str,
        *,
        reason: str,
        cancelled: bool,
    ) -> _ParkedRealtimeRequest | None:
        with self._request_admission_lock:
            record = self._parked_input.get(request_id)
            if record is not None:
                self._mark_input_update_terminal(request_id)
                self._parked_input.pop(request_id, None)
        if record is None:
            return None

        data = record.req._omni_data
        self._promote_immediate_abort_finish_reason(record.req)
        try:
            self._terminate_live_turn(
                record.req,
                reason=reason,
                cancelled=cancelled,
                cleanup_complete=False,
            )
        except Exception:
            logger.exception(
                "Failed to terminate parked MOSS-TTS-Realtime request %s",
                request_id,
            )
        self._release_request_kv_cache_resilient(
            record.req,
            request_id=request_id,
            operation="parked_kv_release",
            data=data,
        )
        try:
            self.session_controller.maybe_reap(time.monotonic(), interval=0.0)
            session_state = getattr(data, "session_state", None)
            if session_state is not None and session_state.close_requested:
                physical_session_ids = (
                    (data.backend_session_id,) if data.backend_session_id else ()
                )
                self._finalize_host_session_close(
                    session_state,
                    physical_session_ids=physical_session_ids,
                    reason="explicit",
                    request_id=request_id,
                )
        except Exception as exc:
            self._record_cleanup_error(
                request_id,
                operation="parked_session_cleanup",
                error=exc,
                data=data,
            )
            logger.exception(
                "Failed to finish parked MOSS-TTS-Realtime session cleanup for %s",
                request_id,
            )
        if data.lifecycle_finalized and data.observability_finalized:
            self._record_cleanup_success(record.req, data)
        return record

    def _release_request_kv_cache_resilient(
        self,
        req: Any,
        *,
        request_id: str,
        operation: str,
        data: MossTTSRealtimeRequestData,
    ) -> Exception | None:
        """Release terminal KV with one bounded retry after an observed error."""

        # Abort can arrive from the stage listener while the scheduler thread is
        # terminalizing the same request. Use the admission/terminal-claim lock
        # so request-pool ownership is released by only one cleanup path.
        with self._request_admission_lock:
            try:
                self._release_request_kv_cache(req)
                return None
            except Exception as exc:
                self._record_cleanup_error(
                    request_id,
                    operation=operation,
                    error=exc,
                    data=data,
                )
                self._bump_resource_total("kv_release_retry_total")
                logger.exception(
                    "Failed to release MOSS-TTS-Realtime KV for %s during %s; "
                    "retrying once",
                    request_id,
                    operation,
                )

            try:
                self._release_request_kv_cache(req)
            except Exception as retry_error:
                self._bump_resource_total("kv_release_retry_error_total")
                logger.exception(
                    "MOSS-TTS-Realtime KV release retry failed for %s during %s",
                    request_id,
                    operation,
                )
                return retry_error

            self._bump_resource_total("kv_release_retry_success_total")
            logger.warning(
                "MOSS-TTS-Realtime KV release recovered after retry for %s during %s",
                request_id,
                operation,
            )
            return None

    @staticmethod
    def _promote_immediate_abort_finish_reason(req: Any) -> None:
        """Make forced KV release take StreamingSession's abort branch."""

        if req.finished():
            return
        req.finished_reason = FINISH_ABORT()
        req.to_finish = None

    def _fail_parked_request(
        self,
        request_id: str,
        *,
        reason: str,
        error: Exception,
    ) -> bool:
        record = self._remove_parked_request(
            request_id,
            reason=reason,
            cancelled=False,
        )
        if record is None:
            return False
        self._emit_request_error(request_id, error)
        super().abort(request_id, defer_running_cleanup=False)
        return True

    def _expire_parked_requests(self) -> None:
        timeout_s = float(
            getattr(
                self,
                "_input_idle_timeout_s",
                self._moss_tts_realtime_limits.input_idle_timeout_s,
            )
        )
        now = time.monotonic()
        with self._request_admission_lock:
            expired = [
                request_id
                for request_id, record in self._parked_input.items()
                if not self._turn_has_decode_input(record.req._omni_data.turn_state)
                and now - record.last_input_at >= timeout_s
            ]
        for request_id in expired:
            error = TimeoutError(
                "MOSS-TTS-Realtime input idle timeout reached while parked"
            )
            if self._fail_parked_request(
                request_id,
                reason="input_idle_timeout",
                error=error,
            ):
                self._park_timeout_total += 1
                self._bump_resource_total("input_idle_timeout_total")

    def _fail_live_realtime_request(
        self,
        request_id: str,
        *,
        reason: str,
        error: Exception,
    ) -> bool:
        if request_id in self._parked_input:
            return self._fail_parked_request(
                request_id,
                reason=reason,
                error=error,
            )

        data = self._moss_tts_realtime_requests.get(request_id)
        if data is None or data.lifecycle_finalized or data.req is None:
            return False
        req = data.req
        if getattr(req, "_omni_data", None) is None:
            req._omni_data = data
        self._emit_request_error(request_id, error)
        self._promote_immediate_abort_finish_reason(req)
        cleanup_error: Exception | None = None
        try:
            if (
                getattr(req, "req_pool_idx", None) is not None
                or getattr(req, "mamba_pool_idx", None) is not None
            ):
                cleanup_error = self._release_request_kv_cache_resilient(
                    req,
                    request_id=request_id,
                    operation=f"{reason}_kv_release",
                    data=data,
                )
            self._terminate_live_turn(
                req,
                reason=reason,
                cancelled=False,
            )
        finally:
            super().abort(request_id, defer_running_cleanup=False)
        if cleanup_error is not None:
            raise cleanup_error
        return True

    def _expire_realtime_turns(self, now: float | None = None) -> int:
        timestamp = time.monotonic() if now is None else float(now)
        timeout_s = float(
            getattr(
                self,
                "_turn_timeout_s",
                self._moss_tts_realtime_limits.turn_timeout_s,
            )
        )
        expired = []
        for request_id, data in list(self._moss_tts_realtime_requests.items()):
            turn = data.turn_state
            if (
                turn is None
                or turn.is_terminal
                or data.lifecycle_finalized
                or timestamp - turn.started_at < timeout_s
            ):
                continue
            expired.append(request_id)

        expired_count = 0
        for request_id in expired:
            if self._fail_live_realtime_request(
                request_id,
                reason="turn_timeout",
                error=TimeoutError(
                    "MOSS-TTS-Realtime maximum turn duration was exceeded"
                ),
            ):
                expired_count += 1
                self._bump_resource_total("turn_timeout_total")
        return expired_count

    def _find_request_data(self, request_id: str) -> Any | None:
        live = getattr(self, "_moss_tts_realtime_requests", {}).get(request_id)
        if live is not None:
            return live
        data = super()._find_request_data(request_id)
        if data is not None:
            return data
        with self._request_admission_lock:
            record = self._parked_input.get(request_id)
            return record.req._omni_data if record is not None else None

    def _active_request_ids(self) -> list[str]:
        request_ids = set(super()._active_request_ids())
        with self._request_admission_lock:
            request_ids.update(self._parked_input)
            request_ids.update(self._moss_tts_realtime_requests)
        return sorted(request_ids)

    def get_num_allocatable_reqs(self, running_bs: int) -> int:
        with self._request_admission_lock:
            parked_count = len(self._parked_input)
        active_decode_count = int(running_bs) + parked_count
        upstream_capacity = _Upstream.get_num_allocatable_reqs(
            self,
            active_decode_count,
        )
        active_capacity = max(0, self._max_active_turns - active_decode_count)
        return min(int(upstream_capacity), active_capacity)

    def _admin_model_info(self) -> dict[str, Any]:
        result = super()._admin_model_info()
        self._refresh_resource_high_water()
        with self._request_admission_lock:
            parked_records = list(self._parked_input.values())
        parked_pending_tokens = 0
        parked_pending_bytes = 0
        for record in parked_records:
            turn = record.req._omni_data.turn_state
            if turn is not None:
                parked_pending_tokens += len(turn.pending_input)
                parked_pending_bytes += turn.pending_input.pending_bytes
        queued_input_tokens, queued_input_bytes = self._queued_input_totals()
        logical = self._logical_reservation_snapshot()
        physical = self._physical_session_snapshot()
        model_state = self._model_state_snapshot()
        limits = self._moss_tts_realtime_limits
        totals = dict(sorted(self._resource_totals.items()))
        result["data"].update(
            {
                "runnable_batch_size": len(
                    getattr(self.running_batch, "reqs", ()) or ()
                ),
                "parked_input_size": len(parked_records),
                "active_turn_count": len(self._live_turn_request_ids()),
                "active_turns_high_water": self._active_turns_high_water,
                "max_active_turns": self._max_active_turns,
                "parked_pending_tokens": parked_pending_tokens,
                "parked_pending_bytes": parked_pending_bytes,
                "queued_input_tokens": queued_input_tokens,
                "queued_input_bytes": queued_input_bytes,
                "queued_input_tokens_high_water": (
                    self._queued_input_tokens_high_water
                ),
                "queued_input_bytes_high_water": self._queued_input_bytes_high_water,
                "park_total": self._park_total,
                "wake_total": self._wake_total,
                "park_timeout_total": self._park_timeout_total,
                "session_count_high_water": self._session_count_high_water,
                "physical_held_sessions_high_water": (self._held_sessions_high_water),
                "physical_held_kv_tokens_high_water": (self._held_kv_tokens_high_water),
                "max_sessions": limits.max_sessions,
                "max_held_sessions": limits.max_held_sessions,
                "max_session_rows": self._max_session_rows,
                "max_held_kv_tokens": self._max_held_kv_tokens,
                "codec_slots": self._codec_slots,
                "input_idle_timeout_s": limits.input_idle_timeout_s,
                "turn_timeout_s": limits.turn_timeout_s,
                "session_idle_ttl_s": limits.session_idle_ttl_s,
                "resource_totals": totals,
                "terminal_reason_totals": dict(
                    sorted(self._terminal_reason_totals.items())
                ),
                "admission_rejection_totals": dict(
                    sorted(self._admission_rejection_totals.items())
                ),
                "session_open_total": totals.get("host_session_open_total", 0),
                "session_close_total": totals.get("host_session_close_total", 0),
                "session_ttl_close_total": totals.get(
                    "session_ttl_close_total",
                    0,
                ),
                "ledger_replay_total": totals.get("ledger_replay_total", 0),
                "session_cache_hit_total": totals.get(
                    "session_cache_hit_total",
                    0,
                ),
                "session_cache_miss_total": totals.get(
                    "session_cache_miss_total",
                    0,
                ),
                "cleanup_success_total": totals.get("cleanup_success_total", 0),
                "cleanup_error_total": totals.get("cleanup_error_total", 0),
                **logical,
                **physical,
                **model_state,
            }
        )
        return result

    def _admin_close_realtime_session(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        session_state = self._moss_tts_realtime_sessions.get(session_id)
        if session_state is None:
            self._close_sglang_session_id(
                session_id,
                abort_inflight=True,
                reason="explicit",
            )
            return {
                "success": True,
                "message": "realtime session was already closed",
                "data": {
                    "session_id": session_id,
                    "closed": True,
                    "deferred": False,
                    "existed": False,
                },
            }

        can_close_now = session_state.request_close()
        if not can_close_now:
            active_data = next(
                (
                    data
                    for data in self._moss_tts_realtime_requests.values()
                    if data.session_state is session_state
                    and data.turn_state is not None
                    and data.turn_state.turn_id == session_state.active_turn_id
                ),
                None,
            )
            if active_data is None or active_data.req is None:
                raise RuntimeError(
                    "active realtime session has no scheduler-owned request"
                )
            self.abort(active_data.req.rid)

        current = self._moss_tts_realtime_sessions.get(session_id)
        closed = current is None
        if current is not None and current.active_turn_id is None:
            physical_session_ids = tuple(
                data.backend_session_id
                for data in self._moss_tts_realtime_requests.values()
                if data.session_state is current and data.backend_session_id
            )
            closed = self._finalize_host_session_close(
                current,
                physical_session_ids=physical_session_ids,
                reason="explicit",
            )
        return {
            "success": True,
            "message": "realtime session close accepted",
            "data": {
                "session_id": session_id,
                "closed": closed,
                "deferred": not closed,
                "existed": True,
            },
        }

    def _run_admin_action(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action == _CLOSE_REALTIME_SESSION_ACTION:
            return self._admin_close_realtime_session(dict(payload or {}))
        return super()._run_admin_action(action, payload)

    def _reap_idle_realtime_sessions(
        self,
        now: float | None = None,
        *,
        force: bool = False,
    ) -> int:
        timestamp = time.monotonic() if now is None else float(now)
        interval = float(getattr(self, "_session_reap_interval_s", 1.0))
        last_reap = float(getattr(self, "_last_session_reap_at", 0.0))
        if not force and timestamp - last_reap < interval:
            return 0
        self._last_session_reap_at = timestamp
        ttl_s = float(
            getattr(
                self,
                "_session_idle_ttl_s",
                self._moss_tts_realtime_limits.session_idle_ttl_s,
            )
        )

        reaped = 0
        for session_state in list(self._moss_tts_realtime_sessions.values()):
            if session_state.active_turn_id is not None:
                continue
            should_close = session_state.close_requested or (
                timestamp - session_state.last_active_at >= ttl_s
            )
            if not should_close:
                continue
            close_reason = "requested" if session_state.close_requested else "ttl"
            if self._finalize_host_session_close(
                session_state,
                reason=close_reason,
            ):
                reaped += 1
        return reaped

    def _run_periodic_session_reapers(self, now: float) -> None:
        try:
            self.session_controller.maybe_reap(now)
        except Exception as exc:
            if not bool(getattr(self, "_physical_session_reap_failed", False)):
                self._record_cleanup_error(
                    "moss-tts-realtime-session-reaper",
                    operation="periodic_physical_session_reap",
                    error=exc,
                )
            self._physical_session_reap_failed = True
            self._bump_resource_total("physical_session_reap_error_total")
            logger.exception(
                "MOSS-TTS-Realtime physical session reap failed; "
                "the scheduler will retry on the next loop"
            )
        else:
            if bool(getattr(self, "_physical_session_reap_failed", False)):
                self._bump_resource_total("physical_session_reap_recovery_total")
            self._physical_session_reap_failed = False

        try:
            self._reap_idle_realtime_sessions(now)
        except Exception as exc:
            if not bool(getattr(self, "_host_session_reap_failed", False)):
                self._record_cleanup_error(
                    "moss-tts-realtime-session-reaper",
                    operation="periodic_host_session_reap",
                    error=exc,
                )
            self._host_session_reap_failed = True
            self._bump_resource_total("host_session_reap_error_total")
            logger.exception(
                "MOSS-TTS-Realtime host session reap failed; "
                "the scheduler will retry on the next loop"
            )
        else:
            if bool(getattr(self, "_host_session_reap_failed", False)):
                self._bump_resource_total("host_session_reap_recovery_total")
            self._host_session_reap_failed = False

    def process_input_requests(self, recv_reqs):
        now = time.monotonic()
        self._run_periodic_session_reapers(now)
        return super().process_input_requests(recv_reqs)

    def is_fully_idle(self, for_health_check: bool = False) -> bool:
        if not for_health_check:
            with self._request_admission_lock:
                if self._parked_input:
                    return False
        return bool(_Upstream.is_fully_idle(self, for_health_check=for_health_check))

    def self_check_during_idle(self) -> None:
        with self._request_admission_lock:
            if self._parked_input:
                return
        super().self_check_during_idle()

    def _request_is_running(self, request_id: str) -> bool:
        seen: set[int] = set()
        async_pending_batch = None
        async_pending = getattr(self, "_async_pending", None)
        if async_pending is not None:
            async_pending_batch = getattr(async_pending, "batch", None)
        for batch in (
            getattr(self, "running_batch", None),
            getattr(self, "cur_batch", None),
            getattr(self, "last_batch", None),
            async_pending_batch,
        ):
            if batch is None or id(batch) in seen:
                continue
            seen.add(id(batch))
            for req in getattr(batch, "reqs", ()):
                if req.rid == request_id and not req.finished():
                    return True
        return False

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        self._mark_input_update_terminal(request_id)
        data = getattr(self, "_moss_tts_realtime_requests", {}).get(request_id)
        parked = self._remove_parked_request(
            request_id,
            reason="aborted",
            cancelled=True,
        )
        running = parked is None and self._request_is_running(request_id)
        immediate_cleanup = (
            parked is None
            and data is not None
            and (not running or not defer_running_cleanup)
        )
        if immediate_cleanup:
            req = data.req
            if req is not None:
                if getattr(req, "_omni_data", None) is None:
                    req._omni_data = data
                self._promote_immediate_abort_finish_reason(req)
                if (
                    getattr(req, "req_pool_idx", None) is not None
                    or getattr(req, "mamba_pool_idx", None) is not None
                ):
                    self._release_request_kv_cache_resilient(
                        req,
                        request_id=request_id,
                        operation="immediate_kv_release",
                        data=data,
                    )
                try:
                    self._terminate_live_turn(
                        req,
                        reason="aborted",
                        cancelled=True,
                        cleanup_complete=False,
                    )
                except Exception as exc:
                    self._record_cleanup_error(
                        request_id,
                        operation="immediate_turn_termination",
                        error=exc,
                        data=data,
                    )
                    logger.exception(
                        "Failed to terminate immediate MOSS-TTS-Realtime request %s",
                        request_id,
                    )
                try:
                    self.session_controller.maybe_reap(
                        time.monotonic(),
                        interval=0.0,
                    )
                except Exception as exc:
                    self._record_cleanup_error(
                        request_id,
                        operation="immediate_session_reap",
                        error=exc,
                        data=data,
                    )
                    logger.exception(
                        "Failed to reap immediate MOSS-TTS-Realtime request %s",
                        request_id,
                    )
                if data.lifecycle_finalized and data.observability_finalized:
                    self._record_cleanup_success(req, data)
        super().abort(
            request_id,
            defer_running_cleanup=(
                defer_running_cleanup
                if parked is None and not immediate_cleanup
                else False
            ),
        )

    def stop(self) -> None:
        live_request_ids = list(getattr(self, "_moss_tts_realtime_requests", {}).keys())
        for request_id in live_request_ids:
            try:
                self.abort(request_id, defer_running_cleanup=False)
            except Exception:
                logger.exception(
                    "Failed to clean up realtime request %s during stop",
                    request_id,
                )
        with self._request_admission_lock:
            parked_ids = list(self._parked_input)
        for request_id in parked_ids:
            try:
                self.abort(request_id, defer_running_cleanup=False)
            except Exception:
                logger.exception(
                    "Failed to clean up parked request %s during stop",
                    request_id,
                )
        for session_state in list(
            getattr(self, "_moss_tts_realtime_sessions", {}).values()
        ):
            try:
                if session_state.active_turn_id is not None:
                    logger.error(
                        "MOSS-TTS-Realtime session %s remained active during stop",
                        session_state.session_id,
                    )
                    continue
                self._finalize_host_session_close(
                    session_state,
                    reason="shutdown",
                )
            except Exception:
                logger.exception(
                    "Failed to close MOSS-TTS-Realtime session %s during stop",
                    session_state.session_id,
                )
        super().stop()

    @staticmethod
    def _batch_output_id(batch: Any, index: int) -> int:
        output_ids = batch.output_ids
        if not isinstance(output_ids, torch.Tensor):
            raise TypeError("decode batch output_ids must be a tensor")
        if output_ids.ndim != 1 or index >= int(output_ids.shape[0]):
            raise RuntimeError("decode batch output_ids are not request aligned")
        return int(output_ids[index].item())

    def _validate_existing_materialized_row(
        self,
        batch: Any,
        index: int,
        req: Any,
        turn: MossTTSRealtimeTurnState,
    ) -> None:
        materialized = turn.last_materialized_row
        if materialized is None:
            if turn.audio_eos_seen:
                raise RuntimeError("audio-EOS request remained runnable")
            raise RuntimeError(
                "runnable request has no provisional or materialized row"
            )
        if not req.output_ids or int(req.output_ids[-1]) != materialized.cache_key:
            raise RuntimeError("request output id does not match materialized row hash")
        if self._batch_output_id(batch, index) != materialized.cache_key:
            raise RuntimeError("batch output id does not match materialized row hash")

    def _materialize_realtime_rows(self, batch: Any) -> None:
        pending: list[tuple[int, Any, MossTTSRealtimeRequestData]] = []
        for index, req in enumerate(batch.reqs):
            if not self._is_live_decode_req(req):
                continue
            try:
                data = req._omni_data
                if not isinstance(data, MossTTSRealtimeRequestData):
                    raise TypeError(
                        "MOSS-TTS-Realtime decode requires its request-data type"
                    )
                turn = data.turn_state
                if turn is None:
                    raise RuntimeError("decode request is missing turn state")
                if turn.provisional_frame is None:
                    self._validate_existing_materialized_row(batch, index, req, turn)
                    continue
                if (
                    turn.phase is not MossTTSRealtimeTurnPhase.DRAINING
                    and not len(turn.pending_input)
                    and not turn.pending_input.input_done
                ):
                    raise RuntimeError(
                        "input-starved request remained in the runnable batch"
                    )
                pending.append((index, req, data))
            except Exception as exc:
                raise _RealtimeMaterializationFailure(req.rid, exc) from exc

        for index, req, data in pending:
            try:
                turn = data.turn_state
                assert turn is not None
                provisional_id = data.provisional_output_id
                if isinstance(provisional_id, bool) or not isinstance(
                    provisional_id, int
                ):
                    raise RuntimeError("request has no integer provisional id")
                if not req.output_ids or int(req.output_ids[-1]) != provisional_id:
                    raise RuntimeError(
                        "request output history does not end in its provisional id"
                    )
                if self._batch_output_id(batch, index) != provisional_id:
                    raise RuntimeError(
                        "decode batch does not contain the request provisional id"
                    )

                next_text_token = turn.next_text_token()
                if next_text_token is None:
                    raise RuntimeError(
                        "ready request became starved during materialization"
                    )
                frame = turn.provisional_frame
                if frame is None:
                    raise RuntimeError(
                        "provisional frame disappeared before materialization"
                    )
                row = (next_text_token, *frame.audio_codes)
                cache_key = build_moss_tts_realtime_row_cache_key(
                    row,
                    model_config=data.model_config,
                )
                materialized = turn.materialize_provisional(
                    next_text_token=next_text_token,
                    cache_key=cache_key,
                )

                prompt_rows = data.prompt_rows
                if not isinstance(prompt_rows, torch.Tensor):
                    raise TypeError("request prompt_rows must be a tensor")
                row_t = torch.tensor([materialized.row], dtype=torch.long)
                data.prompt_rows = torch.cat(
                    [prompt_rows.to(device="cpu", dtype=torch.long), row_t],
                    dim=0,
                )
                data.state.prompt_rows = data.prompt_rows
                req.output_ids[-1] = cache_key
                batch.output_ids[index] = cache_key
                input_ids = getattr(batch, "input_ids", None)
                if isinstance(input_ids, torch.Tensor):
                    if input_ids.ndim != 1 or index >= int(input_ids.shape[0]):
                        raise RuntimeError(
                            "decode batch input_ids are not request aligned"
                        )
                    input_ids[index] = cache_key

                req_pool_indices = batch.req_pool_indices
                if (
                    not isinstance(req_pool_indices, torch.Tensor)
                    or req_pool_indices.ndim != 1
                    or index >= int(req_pool_indices.shape[0])
                ):
                    raise RuntimeError(
                        "decode batch req_pool_indices are not request aligned"
                    )
                future_indices = req_pool_indices[index : index + 1]
                relay_tokens = torch.tensor(
                    [cache_key],
                    dtype=torch.long,
                    device=future_indices.device,
                )
                self.future_map.stash(
                    future_indices,
                    RelayPayload(bonus_tokens=relay_tokens),
                )

                if self._model_runner is not None:
                    hook = getattr(
                        self._model_runner,
                        "on_realtime_row_materialized",
                        None,
                    )
                    if not callable(hook):
                        raise RuntimeError(
                            "MOSS-TTS-Realtime model runner lacks the "
                            "row-materialization hook"
                        )
                    hook(
                        SchedulerRequest(request_id=req.rid, data=data),
                        materialized,
                    )
            except Exception as exc:
                raise _RealtimeMaterializationFailure(req.rid, exc) from exc

    @staticmethod
    def _resync_batch_output_ids(batch: Any) -> None:
        # ``output_ids`` is the scheduler-owned per-row token tensor the model
        # runner rewrites after each decode step. Upstream merge_batch /
        # filter_batch realign reqs/req_pool_indices/seq_lens but know nothing
        # about this custom attribute, so rows merged in after a prefill batch
        # left it shorter than batch.reqs and row materialization read
        # misaligned ids. Rebuild it from each request's last output id -- the
        # value every row invariant below already asserts against.
        batch.output_ids = torch.tensor(
            [int(req.output_ids[-1]) for req in batch.reqs],
            dtype=torch.int64,
            device=batch.req_pool_indices.device,
        )

    def update_running_batch(self, batch: Any) -> Any:
        initial_bs = len(batch.reqs)
        batch.filter_batch()
        if len(batch.reqs) < initial_bs:
            batch.batch_is_full = False
        if not batch.reqs:
            return batch
        self._resync_batch_output_ids(batch)
        try:
            self._park_starved_requests(batch)
        except _RealtimeMaterializationFailure:
            raise
        except Exception as exc:
            request_id = batch.reqs[0].rid if batch.reqs else "unknown"
            raise _RealtimeMaterializationFailure(request_id, exc) from exc
        if not batch.reqs:
            return batch
        self._materialize_realtime_rows(batch)
        return _Upstream.update_running_batch(self, batch)

    def _detach_failed_runnable_request(self, request_id: str) -> bool:
        removed = False
        released = False
        seen_batches: set[int] = set()
        for batch in (
            getattr(self, "running_batch", None),
            getattr(self, "cur_batch", None),
            getattr(self, "last_batch", None),
        ):
            if batch is None or id(batch) in seen_batches:
                continue
            seen_batches.add(id(batch))
            matching = [
                (index, req)
                for index, req in enumerate(getattr(batch, "reqs", ()))
                if req.rid == request_id
            ]
            if not matching:
                continue
            if not released:
                req = matching[0][1]
                self._promote_immediate_abort_finish_reason(req)
                data = req._omni_data
                cleanup_error = self._release_request_kv_cache_resilient(
                    req,
                    request_id=request_id,
                    operation="materialization_kv_release",
                    data=data,
                )
                if cleanup_error is not None:
                    raise cleanup_error
                released = True
            keep_indices = [
                index for index, req in enumerate(batch.reqs) if req.rid != request_id
            ]
            batch.filter_batch(keep_indices=keep_indices)
            if batch.reqs:
                # Keep the scheduler-owned output-id tensor aligned for the next
                # wake merge: upstream filtering does not touch it.
                self._resync_batch_output_ids(batch)
            batch.batch_is_full = False
            removed = True
        return removed

    def get_next_batch_to_run(self) -> Any | None:
        try:
            self._expire_realtime_turns()
            self._expire_parked_requests()
            self._wake_parked_requests()
            return super().get_next_batch_to_run()
        except _RealtimeMaterializationFailure as exc:
            logger.error("%s", exc)
            data = self._find_request_data(exc.request_id)
            was_parked = exc.request_id in self._parked_input
            if isinstance(data, MossTTSRealtimeRequestData) and data.req is not None:
                try:
                    self._terminate_live_turn(
                        data.req,
                        reason="row_materialization_failed",
                        cancelled=False,
                        cleanup_complete=False,
                    )
                except Exception:
                    logger.exception(
                        "Failed to clean up materialization error for %s",
                        exc.request_id,
                    )
            self._emit_request_error(exc.request_id, exc.error)
            if was_parked:
                self.abort(exc.request_id, defer_running_cleanup=False)
            else:
                try:
                    self._detach_failed_runnable_request(exc.request_id)
                except Exception as cleanup_exc:
                    if isinstance(data, MossTTSRealtimeRequestData):
                        self._record_cleanup_error(
                            exc.request_id,
                            operation="materialization_kv_release",
                            error=cleanup_exc,
                            data=data,
                        )
                    logger.exception(
                        "Failed to detach materialization error request %s",
                        exc.request_id,
                    )
                super().abort(exc.request_id, defer_running_cleanup=False)
            if (
                isinstance(data, MossTTSRealtimeRequestData)
                and data.req is not None
                and data.lifecycle_finalized
                and data.observability_finalized
            ):
                self._record_cleanup_success(data.req, data)
            return None

    def _release_turn_model_state(self, req: Any) -> None:
        data = req._omni_data
        turn = data.turn_state
        if turn is None or turn.model_state_slot_id is None:
            return
        if self._model_runner is None:
            raise RuntimeError("cannot release realtime model state without a runner")
        release_request = getattr(self._model_runner, "release_request", None)
        if not callable(release_request):
            raise RuntimeError("MOSS-TTS-Realtime model runner cannot release requests")
        release_request(SchedulerRequest(request_id=req.rid, data=data))

    def _invalidate_request_backend(
        self,
        data: MossTTSRealtimeRequestData,
        req: Any,
        *,
        released_warm_session_id: str | None = None,
    ) -> None:
        if data.backend_session_invalidated:
            return
        session_ids: set[str] = set()
        for candidate in (
            data.backend_session_id,
            released_warm_session_id,
            getattr(getattr(req, "session", None), "session_id", None),
        ):
            if isinstance(candidate, str) and candidate:
                session_ids.add(candidate)

        abort_inflight = getattr(req, "req_pool_idx", None) is None
        for session_id in session_ids:
            self._close_sglang_session_id(
                session_id,
                abort_inflight=abort_inflight,
                reason="turn_invalidated",
                request_id=req.rid,
            )
        data.backend_session_invalidated = True

    def _finalize_host_session_close(
        self,
        session_state: MossTTSRealtimeSessionState,
        *,
        physical_session_ids: Sequence[str] = (),
        reason: str = "requested",
        request_id: str | None = None,
    ) -> bool:
        if session_state.active_turn_id is not None:
            return False
        session_state.request_close()
        session_ids = {
            session_id
            for session_id in (
                session_state.session_id,
                session_state.warm_session_id,
                *physical_session_ids,
            )
            if isinstance(session_id, str) and session_id
        }
        all_closed = True
        for session_id in session_ids:
            all_closed = (
                self._close_sglang_session_id(
                    session_id,
                    reason=reason,
                    request_id=request_id,
                )
                and all_closed
            )
        if not all_closed:
            self._bump_resource_total("host_session_close_error_total")
            return False
        session_state.close()
        sessions = self._moss_tts_realtime_sessions
        if sessions.get(session_state.session_id) is session_state:
            sessions.pop(session_state.session_id, None)
        self._bump_resource_total("host_session_close_total")
        if reason == "ttl":
            self._bump_resource_total("session_ttl_close_total")
        elif reason == "explicit":
            self._bump_resource_total("session_explicit_close_total")
        elif reason == "shutdown":
            self._bump_resource_total("session_shutdown_close_total")
        elif reason == "ephemeral":
            self._bump_resource_total("session_ephemeral_close_total")
        self._emit_realtime_event(
            request_id or session_state.session_id,
            "host_session_close",
            metadata=self._event_metadata(
                session_id=session_state.session_id,
                reason=reason,
            ),
        )
        if request_id is not None:
            # The vocoder keys its codec slot lease to the session, but the
            # request's stream edge is still open here (the terminal result
            # follows in this outbox), so ride it with a session-close marker.
            # The vocoder drains the turn's pending PCM at stream_done and only
            # then releases the session slot.
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=torch.empty(0, dtype=torch.long),
                    metadata={
                        "stream": True,
                        "modality": "audio_codes",
                        "session_control": "close",
                        "session_id": session_state.session_id,
                    },
                )
            )
        self._refresh_resource_high_water()
        return True

    def _record_terminal_lifecycle(
        self,
        req: Any,
        data: MossTTSRealtimeRequestData,
    ) -> None:
        if data.observability_finalized:
            return
        turn = data.turn_state
        if turn is None or not turn.is_terminal or not turn.terminal_reason:
            raise RuntimeError(
                "cannot record observability before realtime terminal invariants"
            )
        self._ensure_observability_state()
        self._resource_totals["terminal_total"] += 1
        self._terminal_reason_totals[turn.terminal_reason] += 1
        self._emit_realtime_event(
            req.rid,
            "turn_terminal",
            metadata=self._event_metadata(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                phase=turn.phase.value,
                reason=turn.terminal_reason,
                sampled_frames=turn.sampled_frame_count,
            ),
        )
        data.observability_finalized = True
        self._refresh_resource_high_water()

    def _finalize_unsuccessful_turn(
        self,
        req: Any,
        *,
        reason: str,
        cancelled: bool,
        cleanup_complete: bool = True,
    ) -> None:
        # The same request can be failed by the scheduler and then aborted by
        # its WebSocket error handler. Keep model-state release, host lifecycle
        # finalization, and request-data detachment under one ownership lock.
        with self._request_admission_lock:
            data = getattr(req, "_omni_data", None)
            if data is None:
                data = self._moss_tts_realtime_requests.get(req.rid)
                if data is None:
                    return
                req._omni_data = data
            if not isinstance(data, MossTTSRealtimeRequestData):
                raise TypeError(
                    "terminal realtime request has the wrong request-data type"
                )
            if data.lifecycle_finalized:
                return

            try:
                self._finalize_unsuccessful_turn_impl(
                    req,
                    reason=reason,
                    cancelled=cancelled,
                )
            except Exception as exc:
                self._record_cleanup_error(
                    getattr(req, "rid", "unknown"),
                    operation="terminal_rollback",
                    error=exc,
                    data=data,
                )
                raise
            self._record_terminal_lifecycle(req, data)
            if cleanup_complete:
                self._record_cleanup_success(req, data)

    def _finalize_unsuccessful_turn_impl(
        self,
        req: Any,
        *,
        reason: str,
        cancelled: bool,
    ) -> None:
        data = req._omni_data
        if not isinstance(data, MossTTSRealtimeRequestData):
            raise TypeError("terminal realtime request has the wrong request-data type")
        turn = data.turn_state
        if turn is None:
            raise RuntimeError("terminal realtime request is missing turn state")
        if not req.finished():
            req.to_finish = FINISH_ABORT()

        if not turn.is_terminal:
            self._release_turn_model_state(req)
            if cancelled:
                turn.cancel(reason)
            else:
                turn.fail(reason)
        elif turn.phase is MossTTSRealtimeTurnPhase.COMPLETED:
            turn.invalidate_completion(reason)

        released_warm_session_id = None
        session_state = data.session_state
        if session_state is not None and session_state.active_turn_id == turn.turn_id:
            released_warm_session_id = session_state.abort_turn(turn)
        self._invalidate_request_backend(
            data,
            req,
            released_warm_session_id=released_warm_session_id,
        )
        data.lifecycle_finalized = True
        self._moss_tts_realtime_requests.pop(req.rid, None)

        if session_state is not None and session_state.close_requested:
            physical_session_ids = (
                (data.backend_session_id,) if data.backend_session_id else ()
            )
            self._finalize_host_session_close(
                session_state,
                physical_session_ids=physical_session_ids,
                reason="explicit",
                request_id=req.rid,
            )

    def _terminate_live_turn(
        self,
        req: Any,
        *,
        reason: str,
        cancelled: bool,
        cleanup_complete: bool = True,
    ) -> None:
        with self._request_admission_lock:
            data = getattr(req, "_omni_data", None)
            if data is None:
                data = self._moss_tts_realtime_requests.get(req.rid)
                if data is not None:
                    req._omni_data = data
            if not isinstance(data, MossTTSRealtimeRequestData):
                return
            turn = data.turn_state
            if turn is None or data.lifecycle_finalized:
                return
            self._finalize_unsuccessful_turn(
                req,
                reason=reason,
                cancelled=cancelled,
                cleanup_complete=cleanup_complete,
            )

    @staticmethod
    def _finished_reason_type(req: Any) -> str:
        reason = getattr(req, "finished_reason", None)
        if reason is None:
            return "unknown"
        payload = reason.to_json()
        if not isinstance(payload, Mapping):
            return "unknown"
        return str(payload.get("type") or "unknown")

    def _validate_cached_finished_session(
        self,
        req: Any,
        data: MossTTSRealtimeRequestData,
    ) -> tuple[str, int]:
        session = getattr(req, "session", None)
        if session is None:
            raise RuntimeError("finished realtime request lost its SGLang session")
        session_id = data.backend_session_id
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("finished realtime request has no physical session id")
        if session.session_id != session_id:
            raise RuntimeError("finished request session identity changed")
        if self.session_controller.get(session_id) is not session:
            raise RuntimeError("session controller lost the finished request owner")
        if bool(getattr(session, "_inflight", False)):
            raise RuntimeError("finished SGLang session is still inflight")
        if bool(getattr(session, "close_on_finish", False)):
            raise RuntimeError("finished SGLang session is pending close")
        req_nodes = getattr(session, "req_nodes", None)
        if not isinstance(req_nodes, dict) or len(req_nodes) != 1:
            raise RuntimeError(
                "finished SGLang session has no single successful request node"
            )
        node = next(iter(req_nodes.values()))
        if getattr(node, "req", None) is not req:
            raise RuntimeError("finished SGLang session cached a different request")
        if getattr(req, "req_pool_idx", None) is not None:
            raise RuntimeError("finished request still owns its request-pool slot")

        slot = self._streaming_session_slot(session_id)
        if slot is None or not bool(getattr(slot, "is_holding_kv", False)):
            raise RuntimeError("finished request has no held streaming-session KV slot")
        committed_kv_length = int(getattr(slot, "kv_committed_len", -1))
        expected_length = len(data.turn_state.ledger.rows)
        if committed_kv_length != expected_length:
            raise RuntimeError(
                "cached streaming-session KV length does not match canonical ledger: "
                f"{committed_kv_length} != {expected_length}"
            )
        allocated_length = _streaming_slot_allocated_len(slot, default=-1)
        if allocated_length < committed_kv_length:
            raise RuntimeError(
                "streaming-session allocated KV is shorter than committed"
            )
        return session_id, committed_kv_length

    def _complete_finished_turn(self, req: Any) -> None:
        with self._request_admission_lock:
            self._complete_finished_turn_impl(req)

    def _complete_finished_turn_impl(self, req: Any) -> None:
        data = req._omni_data
        if not isinstance(data, MossTTSRealtimeRequestData):
            raise TypeError("terminal realtime request has the wrong request-data type")
        turn = data.turn_state
        if turn is None:
            raise RuntimeError("terminal realtime request is missing turn state")

        if data.lifecycle_finalized:
            if turn.phase is not MossTTSRealtimeTurnPhase.COMPLETED:
                raise RuntimeError(
                    "finished request lifecycle was already unsuccessful"
                )
            return

        try:
            self._release_turn_model_state(req)
        except Exception as exc:
            self._record_cleanup_error(
                req.rid,
                operation="model_state_release",
                error=exc,
                data=data,
            )
            try:
                self._finalize_unsuccessful_turn(
                    req,
                    reason="model_state_release_failed",
                    cancelled=False,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up realtime request %s after model-state "
                    "release failure",
                    req.rid,
                )
            raise
        if not turn.audio_eos_seen:
            reason_type = self._finished_reason_type(req)
            failure_reason = (
                "max_length_without_audio_eos"
                if reason_type == "length"
                else f"{reason_type}_without_audio_eos"
            )
            self._finalize_unsuccessful_turn(
                req,
                reason=failure_reason,
                cancelled=False,
            )
            raise RuntimeError(
                "MOSS-TTS-Realtime terminated without audio EOS "
                f"(finish_reason={reason_type})"
            )

        if not req.output_ids or int(req.output_ids[-1]) != int(
            data.model_config.audio_eos_token
        ):
            self._finalize_unsuccessful_turn(
                req,
                reason="audio_eos_output_id_mismatch",
                cancelled=False,
            )
            raise RuntimeError("audio EOS state does not match the terminal output id")

        session_state = data.session_state
        if session_state is None:
            self._finalize_unsuccessful_turn(
                req,
                reason="missing_session_state_at_commit",
                cancelled=False,
            )
            raise RuntimeError(
                "terminal realtime request is missing host session state"
            )
        snapshot = _SessionHostSnapshot.capture(session_state)
        try:
            session_id, committed_kv_length = self._validate_cached_finished_session(
                req,
                data,
            )
            turn.complete(committed_kv_length=committed_kv_length)
            session_state.commit_turn(
                turn,
                warm_session_id=session_id,
            )
            if not data.state.keep_session and not self._finalize_host_session_close(
                session_state,
                physical_session_ids=(session_id,),
                reason="ephemeral",
                request_id=req.rid,
            ):
                raise RuntimeError(
                    "failed to close ephemeral MOSS-TTS-Realtime session"
                )
        except Exception:
            snapshot.restore(session_state)
            self._finalize_unsuccessful_turn(
                req,
                reason="terminal_commit_failed",
                cancelled=False,
            )
            raise

        data.lifecycle_finalized = True
        self._moss_tts_realtime_requests.pop(req.rid, None)
        self._bump_resource_total("ledger_commit_total")
        self._record_terminal_lifecycle(req, data)
        self._record_cleanup_success(req, data)

    def stream_output(self, reqs, return_logprob=False, skip_req=None):
        successful: list[Any] = []
        for req in reqs:
            if req is skip_req or not req.finished():
                successful.append(req)
                continue
            self._mark_input_update_terminal(req.rid)
            if req.rid in self._aborted_request_ids:
                try:
                    self._terminate_live_turn(
                        req,
                        reason="aborted",
                        cancelled=True,
                    )
                except Exception:
                    logger.exception(
                        "Failed to clean up aborted realtime turn %s",
                        req.rid,
                    )
                successful.append(req)
                continue
            try:
                self._complete_finished_turn(req)
            except Exception as exc:
                logger.error(
                    "MOSS-TTS-Realtime terminal validation failed for %s: %s",
                    req.rid,
                    exc,
                )
                self._first_emit_done.discard(req.rid)
                self._prefill_start_done.discard(req.rid)
                self._emit_request_error(req.rid, exc)
                continue
            successful.append(req)
        return super().stream_output(
            successful,
            return_logprob=return_logprob,
            skip_req=skip_req,
        )
