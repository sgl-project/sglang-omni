# SPDX-License-Identifier: Apache-2.0
"""Session, turn, ledger, and scheduler-owned state for MOSS-TTS-Realtime."""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.moss_tts_realtime.config import MossTTSRealtimeResourceLimits
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.proto.messages import InputUpdateMessage
from sglang_omni.scheduling.types import ARRequestData

_MAX_WIRE_INT = (1 << 63) - 1


def _require_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > _MAX_WIRE_INT:
        raise ValueError(f"{name} exceeds the signed 64-bit wire range")
    return value


class MossTTSRealtimeVoiceReference(BaseModel):
    """Immutable voice identity and optional reference prompt metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    voice: str = "default"
    reference_audio: str | None = None
    reference_text: str | None = None
    language: str | None = None
    instructions: str | None = None

    def model_post_init(self, __context: Any = None) -> None:
        for field_name in (
            "voice",
            "reference_audio",
            "reference_text",
            "language",
            "instructions",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty")


class MossTTSRealtimeSamplingConfig(BaseModel):
    """Pinned HF sampling defaults with optional deterministic seed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = 0.8
    top_p: float = 0.6
    top_k: int = 30
    repetition_penalty: float = 1.1
    repetition_window: int = 50
    seed: int | None = None

    def model_post_init(self, __context: Any = None) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if isinstance(self.repetition_window, bool) or self.repetition_window < 1:
            raise ValueError("repetition_window must be a positive integer")
        if self.seed is not None:
            _require_non_negative_int(self.seed, "seed")


class MossTTSRealtimeSessionConfig(BaseModel):
    """Immutable configuration shared by every turn in one API session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_format: Literal["pcm"] = "pcm"
    sample_rate: Literal[24000] = 24000
    voice: MossTTSRealtimeVoiceReference = Field(
        default_factory=MossTTSRealtimeVoiceReference
    )
    sampling: MossTTSRealtimeSamplingConfig = Field(
        default_factory=MossTTSRealtimeSamplingConfig
    )


@dataclass(frozen=True, slots=True)
class MossTTSRealtimeInputUpdate:
    """Internal ordered token update; this is not a control-plane message.

    A tokenless, non-terminal update is an ordered no-op used when a raw-text
    delta produces no stable BPE token yet. Public ``input.tokens`` messages
    remain non-empty; only the API delta-tokenizer path creates these no-ops.
    """

    seq_no: int
    token_ids: tuple[int, ...] = ()
    byte_count: int = 0
    input_done: bool = False

    def __post_init__(self) -> None:
        _require_non_negative_int(self.seq_no, "seq_no")
        _require_non_negative_int(self.byte_count, "byte_count")
        token_ids = tuple(self.token_ids)
        for token_id in token_ids:
            _require_non_negative_int(token_id, "token_id")
        object.__setattr__(self, "token_ids", token_ids)
        if not token_ids and self.byte_count:
            raise ValueError("a tokenless input update cannot retain queued bytes")

    @property
    def fingerprint(self) -> str:
        digest = hashlib.blake2b(digest_size=16, person=b"moss-tts-rt-v1")
        for value in (self.seq_no, self.byte_count, int(self.input_done)):
            digest.update(value.to_bytes(8, byteorder="little", signed=False))
        digest.update(len(self.token_ids).to_bytes(8, "little", signed=False))
        for token_id in self.token_ids:
            digest.update(token_id.to_bytes(8, byteorder="little", signed=False))
        return digest.hexdigest()


class MossTTSRealtimeUpdateDisposition(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


@dataclass(slots=True)
class _PendingUpdateChunk:
    token_ids: tuple[int, ...]
    byte_count: int
    cursor: int = 0

    @property
    def remaining(self) -> int:
        return len(self.token_ids) - self.cursor

    def popleft(self) -> int:
        token_id = self.token_ids[self.cursor]
        self.cursor += 1
        return token_id


@dataclass(slots=True)
class MossTTSRealtimePendingInput:
    """Bounded FIFO plus compact retry records for one active turn."""

    max_tokens: int
    max_bytes: int
    max_updates: int
    next_seq_no: int = 0
    input_done: bool = False
    closed: bool = False
    initial_seeded: bool = False
    total_received_tokens: int = 0
    max_pending_tokens_observed: int = 0
    max_pending_bytes_observed: int = 0
    _pending_tokens: int = 0
    _pending_bytes: int = 0
    _chunks: deque[_PendingUpdateChunk] = field(default_factory=deque, repr=False)
    _accepted_fingerprints: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_bytes", "max_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        _require_non_negative_int(self.next_seq_no, "next_seq_no")

    @classmethod
    def from_limits(
        cls, limits: MossTTSRealtimeResourceLimits
    ) -> MossTTSRealtimePendingInput:
        return cls(
            max_tokens=limits.max_pending_text_tokens,
            max_bytes=limits.max_pending_text_bytes,
            max_updates=limits.max_input_updates,
        )

    def __len__(self) -> int:
        return self._pending_tokens

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    @property
    def accepted_update_count(self) -> int:
        return len(self._accepted_fingerprints)

    def append(
        self, update: MossTTSRealtimeInputUpdate
    ) -> MossTTSRealtimeUpdateDisposition:
        previous_fingerprint = self._accepted_fingerprints.get(update.seq_no)
        if previous_fingerprint is not None:
            if previous_fingerprint != update.fingerprint:
                raise ValueError(
                    f"input update seq_no {update.seq_no} was retried with "
                    "different content"
                )
            return MossTTSRealtimeUpdateDisposition.DUPLICATE

        if self.closed:
            raise RuntimeError("pending input is closed")
        if self.input_done:
            raise RuntimeError("cannot append input after input_done")
        if update.seq_no != self.next_seq_no:
            raise ValueError(
                f"input update sequence gap: expected {self.next_seq_no}, "
                f"got {update.seq_no}"
            )

        pending_tokens = self._pending_tokens + len(update.token_ids)
        pending_bytes = self._pending_bytes + update.byte_count
        update_count = len(self._accepted_fingerprints) + 1
        if pending_tokens > self.max_tokens:
            raise ValueError(
                f"pending token limit exceeded: {pending_tokens} > {self.max_tokens}"
            )
        if pending_bytes > self.max_bytes:
            raise ValueError(
                f"pending byte limit exceeded: {pending_bytes} > {self.max_bytes}"
            )
        if update_count > self.max_updates:
            raise ValueError(
                f"input update limit exceeded: {update_count} > {self.max_updates}"
            )

        if update.token_ids:
            self._chunks.append(
                _PendingUpdateChunk(update.token_ids, update.byte_count)
            )
        self._pending_tokens = pending_tokens
        self._pending_bytes = pending_bytes
        self.max_pending_tokens_observed = max(
            self.max_pending_tokens_observed,
            pending_tokens,
        )
        self.max_pending_bytes_observed = max(
            self.max_pending_bytes_observed,
            pending_bytes,
        )
        self.total_received_tokens += len(update.token_ids)
        self._accepted_fingerprints[update.seq_no] = update.fingerprint
        self.next_seq_no += 1
        self.input_done = update.input_done
        return MossTTSRealtimeUpdateDisposition.ACCEPTED

    def seed_initial(
        self,
        token_ids: Sequence[int],
        *,
        input_done: bool,
    ) -> None:
        """Seed payload-native input without consuming the wire sequence."""

        if not isinstance(input_done, bool):
            raise TypeError("input_done must be a boolean")
        if (
            self.initial_seeded
            or self.closed
            or self.input_done
            or self.next_seq_no
            or self.total_received_tokens
            or self._pending_tokens
            or self._pending_bytes
            or self._chunks
            or self._accepted_fingerprints
        ):
            raise RuntimeError("initial input must be seeded before wire updates")

        normalized = tuple(token_ids)
        for token_id in normalized:
            _require_non_negative_int(token_id, "token_id")
        if len(normalized) > self.max_tokens:
            raise ValueError(
                f"pending token limit exceeded: {len(normalized)} > {self.max_tokens}"
            )
        if normalized:
            self._chunks.append(_PendingUpdateChunk(normalized, 0))
        self._pending_tokens = len(normalized)
        self.max_pending_tokens_observed = max(
            self.max_pending_tokens_observed,
            len(normalized),
        )
        self.total_received_tokens = len(normalized)
        self.input_done = input_done
        self.initial_seeded = True

    def popleft(self) -> int:
        if not self._chunks:
            raise IndexError("pending input is empty")
        chunk = self._chunks[0]
        token_id = chunk.popleft()
        self._pending_tokens -= 1
        if chunk.remaining == 0:
            self._chunks.popleft()
            self._pending_bytes -= chunk.byte_count
        return token_id

    def pop_tokens(self, count: int) -> tuple[int, ...]:
        _require_non_negative_int(count, "count")
        if count > self._pending_tokens:
            raise IndexError(
                f"cannot consume {count} tokens from {self._pending_tokens} pending"
            )
        return tuple(self.popleft() for _ in range(count))

    def close(self, *, discard_pending: bool) -> None:
        if discard_pending:
            self._chunks.clear()
            self._pending_tokens = 0
            self._pending_bytes = 0
        elif self._pending_tokens:
            raise RuntimeError("cannot close pending input while tokens remain")
        self.closed = True


MossTTSRealtimeRow = tuple[int, ...]
MossTTSRealtimeAudioFrame = tuple[int, ...]


def normalize_moss_tts_realtime_audio_frame(
    audio_codes: Sequence[int],
    *,
    model_config: Any,
) -> MossTTSRealtimeAudioFrame:
    codes = tuple(audio_codes)
    num_codebooks = int(model_config.rvq)
    if len(codes) != num_codebooks:
        raise ValueError(
            "MOSS-TTS-Realtime audio frames must contain exactly "
            f"{num_codebooks} codebooks"
        )
    for code in codes:
        _require_non_negative_int(code, "audio code")
        if code >= int(model_config.audio_vocab_size):
            raise ValueError(
                "audio code must be below the MOSS realtime audio vocabulary "
                f"size {model_config.audio_vocab_size}"
            )
    return codes


def normalize_moss_tts_realtime_row(
    row: Sequence[int],
    *,
    model_config: Any,
) -> MossTTSRealtimeRow:
    values = tuple(row)
    row_width = int(model_config.rvq) + 1
    if len(values) != row_width:
        raise ValueError(
            "MOSS-TTS-Realtime rows must contain exactly " f"{row_width} columns"
        )
    _require_non_negative_int(values[0], "text token")
    normalize_moss_tts_realtime_audio_frame(values[1:], model_config=model_config)
    return values


@dataclass(frozen=True, slots=True)
class MossTTSRealtimeMaterializedRow:
    """Canonical 17-column input row and its scheduler/cache identity."""

    row: MossTTSRealtimeRow
    cache_key: int
    generation_step: int
    model_config: Any = field(repr=False)

    def __post_init__(self) -> None:
        row = normalize_moss_tts_realtime_row(self.row, model_config=self.model_config)
        if row[1] == int(self.model_config.audio_eos_token):
            raise ValueError("an audio-EOS frame cannot become a backbone input row")
        object.__setattr__(self, "row", row)
        _require_non_negative_int(self.cache_key, "cache_key")
        _require_non_negative_int(self.generation_step, "generation_step")


@dataclass(frozen=True, slots=True)
class MossTTSRealtimeProvisionalFrame:
    """Sampled audio frame not yet paired with its next text token."""

    audio_codes: MossTTSRealtimeAudioFrame
    generation_step: int
    model_config: Any = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audio_codes",
            normalize_moss_tts_realtime_audio_frame(
                self.audio_codes, model_config=self.model_config
            ),
        )
        _require_non_negative_int(self.generation_step, "generation_step")

    @property
    def is_audio_eos(self) -> bool:
        return self.audio_codes[0] == int(self.model_config.audio_eos_token)

    def materialize(
        self, *, next_text_token: int, cache_key: int
    ) -> MossTTSRealtimeMaterializedRow:
        if self.is_audio_eos:
            raise ValueError("the terminal audio-EOS frame must remain unmaterialized")
        _require_non_negative_int(next_text_token, "next_text_token")
        return MossTTSRealtimeMaterializedRow(
            row=(next_text_token, *self.audio_codes),
            cache_key=cache_key,
            generation_step=self.generation_step,
            model_config=self.model_config,
        )


class MossTTSRealtimeLedgerDisposition(str, Enum):
    OPEN = "open"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(slots=True)
class MossTTSRealtimeTurnLedger:
    """Transactional working suffix over an immutable committed prefix."""

    model_config: Any = field(repr=False)
    committed_prefix: tuple[MossTTSRealtimeRow, ...] = ()
    base_revision: int = 0
    appended_rows: list[MossTTSRealtimeRow] = field(default_factory=list)
    disposition: MossTTSRealtimeLedgerDisposition = (
        MossTTSRealtimeLedgerDisposition.OPEN
    )

    def __post_init__(self) -> None:
        self.committed_prefix = tuple(
            normalize_moss_tts_realtime_row(row, model_config=self.model_config)
            for row in self.committed_prefix
        )
        self.appended_rows = [
            normalize_moss_tts_realtime_row(row, model_config=self.model_config)
            for row in self.appended_rows
        ]
        _require_non_negative_int(self.base_revision, "base_revision")

    @property
    def rows(self) -> tuple[MossTTSRealtimeRow, ...]:
        return self.committed_prefix + tuple(self.appended_rows)

    def _require_open(self) -> None:
        if self.disposition is not MossTTSRealtimeLedgerDisposition.OPEN:
            raise RuntimeError(f"turn ledger is already {self.disposition.value}")

    def append_row(self, row: Sequence[int]) -> MossTTSRealtimeRow:
        self._require_open()
        normalized = normalize_moss_tts_realtime_row(
            row, model_config=self.model_config
        )
        self.appended_rows.append(normalized)
        return normalized

    def append_materialized(
        self, materialized: MossTTSRealtimeMaterializedRow
    ) -> MossTTSRealtimeRow:
        return self.append_row(materialized.row)

    def extend_rows(self, rows: Iterable[Sequence[int]]) -> None:
        normalized = [
            normalize_moss_tts_realtime_row(row, model_config=self.model_config)
            for row in rows
        ]
        self._require_open()
        self.appended_rows.extend(normalized)

    def assert_kv_length(self, kv_length: int) -> None:
        _require_non_negative_int(kv_length, "kv_length")
        if kv_length != len(self.rows):
            raise ValueError(
                f"ledger/KV length mismatch: {len(self.rows)} rows != "
                f"{kv_length} KV positions"
            )

    def commit(self) -> tuple[MossTTSRealtimeRow, ...]:
        if self.disposition is MossTTSRealtimeLedgerDisposition.COMMITTED:
            return self.rows
        self._require_open()
        self.disposition = MossTTSRealtimeLedgerDisposition.COMMITTED
        return self.rows

    def rollback(self) -> tuple[MossTTSRealtimeRow, ...]:
        if self.disposition is MossTTSRealtimeLedgerDisposition.ROLLED_BACK:
            return self.committed_prefix
        self._require_open()
        self.appended_rows.clear()
        self.disposition = MossTTSRealtimeLedgerDisposition.ROLLED_BACK
        return self.committed_prefix

    def invalidate_commit(self) -> tuple[MossTTSRealtimeRow, ...]:
        """Discard a locally completed suffix after host commit fails."""

        if self.disposition is MossTTSRealtimeLedgerDisposition.ROLLED_BACK:
            return self.committed_prefix
        if self.disposition is MossTTSRealtimeLedgerDisposition.OPEN:
            return self.rollback()
        self.appended_rows.clear()
        self.disposition = MossTTSRealtimeLedgerDisposition.ROLLED_BACK
        return self.committed_prefix


class MossTTSRealtimeTurnPhase(str, Enum):
    WAITING_PREFILL = "waiting_prefill"
    RUNNING = "running"
    PARKED_INPUT = "parked_input"
    DRAINING = "draining"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            MossTTSRealtimeTurnPhase.COMPLETED,
            MossTTSRealtimeTurnPhase.CANCELLED,
            MossTTSRealtimeTurnPhase.FAILED,
        }


_ALLOWED_TRANSITIONS: dict[
    MossTTSRealtimeTurnPhase, frozenset[MossTTSRealtimeTurnPhase]
] = {
    MossTTSRealtimeTurnPhase.WAITING_PREFILL: frozenset(
        {MossTTSRealtimeTurnPhase.RUNNING}
    ),
    MossTTSRealtimeTurnPhase.RUNNING: frozenset(
        {
            MossTTSRealtimeTurnPhase.PARKED_INPUT,
            MossTTSRealtimeTurnPhase.DRAINING,
        }
    ),
    MossTTSRealtimeTurnPhase.PARKED_INPUT: frozenset(
        {
            MossTTSRealtimeTurnPhase.RUNNING,
            MossTTSRealtimeTurnPhase.DRAINING,
        }
    ),
    MossTTSRealtimeTurnPhase.DRAINING: frozenset(),
    MossTTSRealtimeTurnPhase.COMPLETED: frozenset(),
    MossTTSRealtimeTurnPhase.CANCELLED: frozenset(),
    MossTTSRealtimeTurnPhase.FAILED: frozenset(),
}


@dataclass(slots=True)
class MossTTSRealtimeTurnState:
    """Host/scheduler lifecycle for one live realtime TTS request."""

    session_id: str
    turn_id: str
    request_id: str
    pending_input: MossTTSRealtimePendingInput
    ledger: MossTTSRealtimeTurnLedger
    phase: MossTTSRealtimeTurnPhase = MossTTSRealtimeTurnPhase.WAITING_PREFILL
    prefill_token_ids: tuple[int, ...] = ()
    provisional_frame: MossTTSRealtimeProvisionalFrame | None = None
    last_materialized_row: MossTTSRealtimeMaterializedRow | None = None
    model_state_slot_id: int | None = None
    codec_slot_id: int | None = None
    sampled_frame_count: int = 0
    audio_eos_seen: bool = False
    terminal_reason: str | None = None
    committed_kv_length: int | None = None
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        _require_identifier(self.session_id, "session_id")
        _require_identifier(self.turn_id, "turn_id")
        _require_identifier(self.request_id, "request_id")
        if isinstance(self.started_at, bool) or not isinstance(
            self.started_at, (int, float)
        ):
            raise TypeError("turn started_at must be a number")
        if not math.isfinite(float(self.started_at)) or self.started_at < 0:
            raise ValueError("turn started_at must be finite and non-negative")
        self.started_at = float(self.started_at)

    @property
    def is_terminal(self) -> bool:
        return self.phase.is_terminal

    @property
    def ready_for_prefill(self) -> bool:
        if self.phase is not MossTTSRealtimeTurnPhase.WAITING_PREFILL:
            return False
        pending = len(self.pending_input)
        return pending >= int(self.ledger.model_config.delay_tokens_len) or (
            self.pending_input.input_done and pending > 0
        )

    def transition_to(self, phase: MossTTSRealtimeTurnPhase) -> None:
        if phase.is_terminal:
            raise ValueError("use complete(), cancel(), or fail() for terminal phases")
        if phase is self.phase:
            return
        if phase not in _ALLOWED_TRANSITIONS[self.phase]:
            raise ValueError(
                f"invalid realtime turn transition: {self.phase.value} -> {phase.value}"
            )
        self.phase = phase

    def append_input_update(
        self, update: MossTTSRealtimeInputUpdate
    ) -> MossTTSRealtimeUpdateDisposition:
        if self.is_terminal:
            raise RuntimeError("cannot update a terminal turn")
        if (
            update.input_done
            and not update.token_ids
            and self.pending_input.total_received_tokens == 0
            and len(self.pending_input) == 0
        ):
            raise ValueError("input.done cannot complete an empty realtime turn")

        disposition = self.pending_input.append(update)
        if disposition is MossTTSRealtimeUpdateDisposition.DUPLICATE:
            return disposition

        if self.phase is MossTTSRealtimeTurnPhase.PARKED_INPUT:
            if len(self.pending_input):
                self.transition_to(MossTTSRealtimeTurnPhase.RUNNING)
            elif self.pending_input.input_done:
                self.transition_to(MossTTSRealtimeTurnPhase.DRAINING)
        return disposition

    def seed_initial_input(
        self,
        token_ids: Sequence[int],
        *,
        input_done: bool,
    ) -> None:
        """Install request-payload tokens before ordered live updates replay."""

        normalized = tuple(token_ids)
        if input_done and not normalized:
            raise ValueError("input.done cannot complete an empty realtime turn")
        self.pending_input.seed_initial(normalized, input_done=input_done)

    def take_prefill_tokens(self) -> tuple[int, ...]:
        if not self.ready_for_prefill:
            raise RuntimeError("turn is not ready for realtime prefill")
        count = min(
            int(self.ledger.model_config.delay_tokens_len), len(self.pending_input)
        )
        token_ids = self.pending_input.pop_tokens(count)
        if not token_ids:
            raise RuntimeError("realtime prefill requires at least one text token")
        self.prefill_token_ids = token_ids
        self.transition_to(MossTTSRealtimeTurnPhase.RUNNING)
        return token_ids

    def next_text_token(self) -> int | None:
        if self.phase is MossTTSRealtimeTurnPhase.WAITING_PREFILL:
            raise RuntimeError("prefill must run before incremental decode")
        if self.is_terminal:
            raise RuntimeError("terminal turns cannot produce another text token")
        if self.phase is MossTTSRealtimeTurnPhase.DRAINING:
            return int(self.ledger.model_config.text_pad)
        if len(self.pending_input):
            if self.phase is MossTTSRealtimeTurnPhase.PARKED_INPUT:
                self.transition_to(MossTTSRealtimeTurnPhase.RUNNING)
            return self.pending_input.popleft()
        if self.pending_input.input_done:
            self.transition_to(MossTTSRealtimeTurnPhase.DRAINING)
            return int(self.ledger.model_config.text_pad)
        if self.phase is MossTTSRealtimeTurnPhase.RUNNING:
            self.transition_to(MossTTSRealtimeTurnPhase.PARKED_INPUT)
        return None

    def observe_audio_frame(
        self, audio_codes: Sequence[int], *, generation_step: int
    ) -> MossTTSRealtimeProvisionalFrame:
        if self.phase not in {
            MossTTSRealtimeTurnPhase.RUNNING,
            MossTTSRealtimeTurnPhase.DRAINING,
        }:
            raise RuntimeError(
                f"cannot observe an audio frame while turn is {self.phase.value}"
            )
        if self.provisional_frame is not None:
            raise RuntimeError(
                "the previous audio frame must be materialized before sampling "
                "another frame"
            )
        frame = MossTTSRealtimeProvisionalFrame(
            audio_codes=tuple(audio_codes),
            generation_step=generation_step,
            model_config=self.ledger.model_config,
        )
        self.sampled_frame_count += 1
        if frame.is_audio_eos:
            self.audio_eos_seen = True
        else:
            self.provisional_frame = frame
        return frame

    def materialize_provisional(
        self, *, next_text_token: int, cache_key: int
    ) -> MossTTSRealtimeMaterializedRow:
        frame = self.provisional_frame
        if frame is None:
            raise RuntimeError("no provisional audio frame is available")
        materialized = frame.materialize(
            next_text_token=next_text_token,
            cache_key=cache_key,
        )
        self.ledger.append_materialized(materialized)
        self.provisional_frame = None
        self.last_materialized_row = materialized
        return materialized

    def assign_model_state_slot(self, slot_id: int) -> None:
        _require_non_negative_int(slot_id, "model_state_slot_id")
        if self.model_state_slot_id is not None:
            raise RuntimeError("turn already owns a model-state slot")
        self.model_state_slot_id = slot_id

    def release_model_state_slot(self, *, expected_slot_id: int | None = None) -> int:
        slot_id = self.model_state_slot_id
        if slot_id is None:
            raise RuntimeError("turn does not own a model-state slot")
        if expected_slot_id is not None and slot_id != expected_slot_id:
            raise RuntimeError(
                f"model-state slot ownership mismatch: {slot_id} != {expected_slot_id}"
            )
        self.model_state_slot_id = None
        return slot_id

    def assign_codec_slot(self, slot_id: int) -> None:
        _require_non_negative_int(slot_id, "codec_slot_id")
        if self.codec_slot_id is not None:
            raise RuntimeError("turn already owns a codec slot")
        self.codec_slot_id = slot_id

    def release_codec_slot(self, *, expected_slot_id: int | None = None) -> int:
        slot_id = self.codec_slot_id
        if slot_id is None:
            raise RuntimeError("turn does not own a codec slot")
        if expected_slot_id is not None and slot_id != expected_slot_id:
            raise RuntimeError(
                f"codec slot ownership mismatch: {slot_id} != {expected_slot_id}"
            )
        self.codec_slot_id = None
        return slot_id

    def _require_released_turn_slots(self) -> None:
        if self.model_state_slot_id is not None or self.codec_slot_id is not None:
            raise RuntimeError(
                "turn-owned model and codec slots must be released before terminal "
                "state"
            )

    def complete(self, *, committed_kv_length: int) -> None:
        if self.phase is MossTTSRealtimeTurnPhase.COMPLETED:
            self.assert_terminal_invariants()
            return
        if self.is_terminal:
            raise RuntimeError(f"cannot complete a {self.phase.value} turn")
        if self.phase not in {
            MossTTSRealtimeTurnPhase.RUNNING,
            MossTTSRealtimeTurnPhase.DRAINING,
        }:
            raise RuntimeError(f"cannot complete a turn while it is {self.phase.value}")
        if not self.audio_eos_seen:
            raise RuntimeError("successful completion requires audio EOS")
        if len(self.pending_input):
            raise RuntimeError("successful completion requires an empty token queue")
        if self.provisional_frame is not None:
            raise RuntimeError("a provisional frame remains at turn completion")
        self._require_released_turn_slots()
        self.ledger.assert_kv_length(committed_kv_length)
        self.ledger.commit()
        self.pending_input.close(discard_pending=False)
        self.committed_kv_length = committed_kv_length
        self.terminal_reason = (
            "audio_eos" if self.pending_input.input_done else "model_eos"
        )
        self.phase = MossTTSRealtimeTurnPhase.COMPLETED
        self.assert_terminal_invariants()

    def cancel(self, reason: str = "cancelled") -> None:
        self._terminate(MossTTSRealtimeTurnPhase.CANCELLED, reason)

    def fail(self, reason: str) -> None:
        self._terminate(MossTTSRealtimeTurnPhase.FAILED, reason)

    def invalidate_completion(self, reason: str) -> None:
        """Turn a locally completed turn into a failed rolled-back turn."""

        _require_identifier(reason, "terminal reason")
        if self.phase is MossTTSRealtimeTurnPhase.FAILED:
            self.assert_terminal_invariants()
            return
        if self.phase is not MossTTSRealtimeTurnPhase.COMPLETED:
            raise RuntimeError(
                "only a completed turn may invalidate its successful completion"
            )
        self.ledger.invalidate_commit()
        self.committed_kv_length = None
        self.terminal_reason = reason
        self.phase = MossTTSRealtimeTurnPhase.FAILED
        self.assert_terminal_invariants()

    def _terminate(self, phase: MossTTSRealtimeTurnPhase, reason: str) -> None:
        _require_identifier(reason, "terminal reason")
        if self.phase is phase:
            self.assert_terminal_invariants()
            return
        if self.is_terminal:
            raise RuntimeError(
                f"cannot change terminal turn from {self.phase.value} to {phase.value}"
            )
        self._require_released_turn_slots()
        self.provisional_frame = None
        self.pending_input.close(discard_pending=True)
        self.ledger.rollback()
        self.terminal_reason = reason
        self.phase = phase
        self.assert_terminal_invariants()

    def assert_terminal_invariants(self) -> None:
        if not self.is_terminal:
            raise RuntimeError("turn is not terminal")
        if not self.terminal_reason:
            raise RuntimeError("terminal turn is missing a reason")
        if self.provisional_frame is not None:
            raise RuntimeError("terminal turn retains a provisional frame")
        if len(self.pending_input) or self.pending_input.pending_bytes:
            raise RuntimeError("terminal turn retains pending input")
        self._require_released_turn_slots()

        if self.phase is MossTTSRealtimeTurnPhase.COMPLETED:
            if not self.audio_eos_seen:
                raise RuntimeError("completed turn is missing audio EOS")
            if (
                self.ledger.disposition
                is not MossTTSRealtimeLedgerDisposition.COMMITTED
            ):
                raise RuntimeError("completed turn ledger is not committed")
            if self.committed_kv_length is None:
                raise RuntimeError("completed turn is missing committed KV length")
            self.ledger.assert_kv_length(self.committed_kv_length)
        elif (
            self.ledger.disposition is not MossTTSRealtimeLedgerDisposition.ROLLED_BACK
        ):
            raise RuntimeError("unsuccessful terminal turn ledger is not rolled back")


@dataclass(slots=True)
class MossTTSRealtimeSessionState:
    """Authoritative host record for committed dialogue and warm KV identity."""

    session_id: str
    model_config: Any = field(repr=False)
    config: MossTTSRealtimeSessionConfig = field(
        default_factory=MossTTSRealtimeSessionConfig
    )
    committed_rows: tuple[MossTTSRealtimeRow, ...] = ()
    ledger_revision: int = 0
    successful_turns: int = 0
    active_turn_id: str | None = None
    warm_session_id: str | None = None
    warm_kv_length: int = 0
    close_requested: bool = False
    closed: bool = False
    last_active_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        _require_identifier(self.session_id, "session_id")
        self.committed_rows = tuple(
            normalize_moss_tts_realtime_row(row, model_config=self.model_config)
            for row in self.committed_rows
        )
        _require_non_negative_int(self.ledger_revision, "ledger_revision")
        _require_non_negative_int(self.successful_turns, "successful_turns")
        _require_non_negative_int(self.warm_kv_length, "warm_kv_length")
        if self.warm_session_id is not None:
            _require_identifier(self.warm_session_id, "warm_session_id")
            if self.warm_kv_length != len(self.committed_rows):
                raise ValueError(
                    "warm KV length must equal the canonical committed ledger length"
                )
        elif self.warm_kv_length:
            raise ValueError("warm_kv_length requires warm_session_id")
        if not isinstance(self.close_requested, bool):
            raise TypeError("close_requested must be a boolean")
        if not isinstance(self.closed, bool):
            raise TypeError("closed must be a boolean")
        if self.closed and not self.close_requested:
            raise ValueError("closed sessions must have close_requested set")
        if isinstance(self.last_active_at, bool) or not isinstance(
            self.last_active_at, (int, float)
        ):
            raise TypeError("last_active_at must be a number")
        if not math.isfinite(float(self.last_active_at)) or self.last_active_at < 0:
            raise ValueError("last_active_at must be finite and non-negative")
        self.last_active_at = float(self.last_active_at)

    @property
    def needs_ledger_replay(self) -> bool:
        return bool(self.committed_rows) and self.warm_session_id is None

    def touch(self, now: float | None = None) -> float:
        timestamp = time.monotonic() if now is None else now
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError("session activity time must be a number")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("session activity time must be finite and non-negative")
        self.last_active_at = timestamp
        return timestamp

    def reconfigure(self, config: MossTTSRealtimeSessionConfig) -> None:
        if self.closed or self.close_requested:
            raise RuntimeError("cannot configure a closed realtime session")
        if self.active_turn_id is not None:
            raise RuntimeError("cannot configure a session with an active turn")
        if self.successful_turns:
            raise RuntimeError(
                "voice/reference configuration is immutable after a successful turn"
            )
        self.config = config

    def begin_turn(
        self,
        *,
        turn_id: str,
        request_id: str,
        limits: MossTTSRealtimeResourceLimits | None = None,
    ) -> MossTTSRealtimeTurnState:
        if self.closed or self.close_requested:
            raise RuntimeError("cannot start a turn on a closed realtime session")
        if self.active_turn_id is not None:
            raise RuntimeError(
                f"session already has active turn {self.active_turn_id!r}"
            )
        _require_identifier(turn_id, "turn_id")
        _require_identifier(request_id, "request_id")
        limits = limits or MossTTSRealtimeResourceLimits()
        turn = MossTTSRealtimeTurnState(
            session_id=self.session_id,
            turn_id=turn_id,
            request_id=request_id,
            pending_input=MossTTSRealtimePendingInput.from_limits(limits),
            ledger=MossTTSRealtimeTurnLedger(
                model_config=self.model_config,
                committed_prefix=self.committed_rows,
                base_revision=self.ledger_revision,
            ),
        )
        self.active_turn_id = turn_id
        self.touch()
        return turn

    def _require_active_turn(self, turn: MossTTSRealtimeTurnState) -> None:
        if turn.session_id != self.session_id:
            raise ValueError("turn belongs to a different realtime session")
        if self.active_turn_id != turn.turn_id:
            raise RuntimeError(f"turn {turn.turn_id!r} is not the active session turn")
        if turn.ledger.base_revision != self.ledger_revision:
            raise RuntimeError("turn ledger was built from a stale session revision")

    def commit_turn(
        self,
        turn: MossTTSRealtimeTurnState,
        *,
        warm_session_id: str,
    ) -> None:
        self._require_active_turn(turn)
        turn.assert_terminal_invariants()
        if turn.phase is not MossTTSRealtimeTurnPhase.COMPLETED:
            raise RuntimeError("only a successfully completed turn may commit")
        if self.close_requested:
            raise RuntimeError("cannot commit a turn after session close was requested")
        _require_identifier(warm_session_id, "warm_session_id")
        committed_rows = turn.ledger.rows
        if turn.committed_kv_length != len(committed_rows):
            raise RuntimeError("completed turn KV length differs from its ledger")
        if len(committed_rows) < len(self.committed_rows):
            raise RuntimeError("completed turn removed committed session rows")

        self.committed_rows = committed_rows
        self.ledger_revision += 1
        self.successful_turns += 1
        self.active_turn_id = None
        self.warm_session_id = warm_session_id
        self.warm_kv_length = len(committed_rows)
        self.touch()

    def abort_turn(self, turn: MossTTSRealtimeTurnState) -> str | None:
        self._require_active_turn(turn)
        turn.assert_terminal_invariants()
        if turn.phase not in {
            MossTTSRealtimeTurnPhase.CANCELLED,
            MossTTSRealtimeTurnPhase.FAILED,
        }:
            raise RuntimeError("abort_turn requires a cancelled or failed turn")
        return self.invalidate_turn(turn)

    def invalidate_turn(self, turn: MossTTSRealtimeTurnState) -> str | None:
        """Clear one active claim and invalidate its warm physical session."""

        self._require_active_turn(turn)
        self.active_turn_id = None
        released = self.release_warm_session()
        self.touch()
        return released

    def release_warm_session(self) -> str | None:
        warm_session_id = self.warm_session_id
        self.warm_session_id = None
        self.warm_kv_length = 0
        return warm_session_id

    def close(self) -> str | None:
        if self.active_turn_id is not None:
            raise RuntimeError("active turn must be terminated before session.close")
        if self.closed:
            return None
        released = self.release_warm_session()
        self.close_requested = True
        self.closed = True
        self.touch()
        return released

    def request_close(self) -> bool:
        """Prevent new turns and report whether final close can run now."""

        if self.closed:
            return True
        self.close_requested = True
        self.touch()
        return self.active_turn_id is None


@dataclass
class MossTTSRealtimeRequestData(ARRequestData):
    """Backend-neutral scheduler data for the dedicated realtime AR path."""

    enforce_request_limits: bool = True
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    sampling_steps: int | None = None
    input_embeds_are_projected: bool = False
    stage_payload: Any = None
    state: MossTTSRealtimeState = field(default_factory=MossTTSRealtimeState)
    session_state: MossTTSRealtimeSessionState | None = None
    turn_state: MossTTSRealtimeTurnState | None = None
    model_config: Any = None
    prompt_rows: Any = None
    initial_token_ids: tuple[int, ...] = ()
    generation_row_start: int = 0
    provisional_output_id: int | None = None
    sampling_seed: int | None = None
    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None
    backend_session_id: str | None = None
    ledger_replay: bool = False
    backend_session_invalidated: bool = False
    lifecycle_finalized: bool = False
    context_reservation_rows: int = 0
    observability_finalized: bool = False
    cleanup_observability_finalized: bool = False


def apply_moss_tts_realtime_input_update(
    req_data: Any,
    message: InputUpdateMessage,
) -> MossTTSRealtimeUpdateDisposition:
    """Apply one scheduler-admitted wire update to its authoritative turn."""

    if not isinstance(req_data, MossTTSRealtimeRequestData):
        raise TypeError(
            "MOSS-TTS-Realtime input updates require MossTTSRealtimeRequestData"
        )
    if not isinstance(message, InputUpdateMessage):
        raise TypeError("message must be an InputUpdateMessage")
    turn = req_data.turn_state
    if turn is None:
        raise RuntimeError("MOSS-TTS-Realtime request data has no live turn state")
    if message.request_id != turn.request_id:
        raise ValueError(
            "input update request identity mismatch: "
            f"{message.request_id!r} != {turn.request_id!r}"
        )
    if message.session_id != turn.session_id:
        raise ValueError(
            "input update session identity mismatch: "
            f"{message.session_id!r} != {turn.session_id!r}"
        )
    if message.turn_id != turn.turn_id:
        raise ValueError(
            "input update turn identity mismatch: "
            f"{message.turn_id!r} != {turn.turn_id!r}"
        )
    return turn.append_input_update(
        MossTTSRealtimeInputUpdate(
            seq_no=message.seq_no,
            token_ids=message.token_ids,
            byte_count=message.byte_count,
            input_done=message.input_done,
        )
    )
