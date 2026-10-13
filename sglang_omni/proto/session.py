# SPDX-License-Identifier: Apache-2.0
"""Model-independent, bounded session operation and output contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import msgpack
import msgspec

SESSION_METADATA_KEY: Literal["omni_session"] = "omni_session"
# Note (Junnan Li): msgspec encodes bytes as base64 text by default; keep them native on both sides.
BUILTIN_TYPES = (bytes,)
ChunkPayload = bytes | dict[str, object] | None
DEFAULT_MAX_MODALITIES = 8
DEFAULT_MAX_PENDING_CHUNKS = 16
DEFAULT_MAX_PENDING_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_OUTPUT_CHUNKS = 64
DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_CHUNK_BYTES = 1024 * 1024
DEFAULT_OPERATION_TIMEOUT_S = 30.0
DEFAULT_IDLE_TIMEOUT_S = 300.0
# Note (Junnan Li): Msgpack bin headers grow by 1 byte at 256 bytes and 3 bytes at 65536.
MSGPACK_BIN8_LIMIT = 256
MSGPACK_BIN16_LIMIT = 65536
MSGPACK_BIN16_HEADER_GROWTH = 1
MSGPACK_BIN32_HEADER_GROWTH = 3


class SessionIdentityDict(TypedDict):
    id: str
    open_index: int


class TimedChunkDict(TypedDict):
    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: ChunkPayload
    format: str | None
    eos: bool


class OutputChunkDict(TypedDict):
    session_identity: SessionIdentityDict
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: ChunkPayload
    format: str | None
    eos: bool
    kind: Literal["data", "input_done"]


class SessionOperationDict(TypedDict):
    operation: Literal["open", "append", "close"]
    session_identity: SessionIdentityDict
    stages: list[str]
    chunk: TimedChunkDict | None


@dataclass(frozen=True)
class SessionIdentity:
    """One open of an id. open_index is the number issued for that open."""

    id: str
    open_index: int = 1

    def to_dict(self) -> SessionIdentityDict:
        return {"id": self.id, "open_index": self.open_index}


@dataclass(frozen=True)
class TimedChunk:
    """Input seq is global across modalities within one open_index."""

    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: ChunkPayload
    format: str | None = None
    eos: bool = False

    def to_dict(self) -> TimedChunkDict:
        return {
            "modality": self.modality,
            "t_start_ms": self.t_start_ms,
            "duration_ms": self.duration_ms,
            "seq": self.seq,
            "payload": self.payload,
            "format": self.format,
            "eos": self.eos,
        }

    @classmethod
    def from_dict(cls, data: object) -> TimedChunk:
        if not isinstance(data, dict):
            raise ValueError("timed chunk must be an object")
        else:
            return msgspec.convert(
                data, type=cls, strict=True, builtin_types=BUILTIN_TYPES
            )


@dataclass(frozen=True)
class OutputChunk:
    """input_seq identifies the originating pipeline input, not a stream seq."""

    session_identity: SessionIdentity
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: ChunkPayload
    format: str | None = None
    eos: bool = False
    kind: Literal["data", "input_done"] = "data"

    def to_dict(self) -> OutputChunkDict:
        return {
            "session_identity": self.session_identity.to_dict(),
            "seq": self.seq,
            "input_seq": self.input_seq,
            "modality": self.modality,
            "t_start_ms": self.t_start_ms,
            "duration_ms": self.duration_ms,
            "payload": self.payload,
            "format": self.format,
            "eos": self.eos,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: object) -> OutputChunk:
        if not isinstance(data, dict):
            raise ValueError("output chunk must be an object")
        else:
            return msgspec.convert(
                data, type=cls, strict=True, builtin_types=BUILTIN_TYPES
            )


@dataclass(frozen=True)
class ResourceUsage:
    kv_tokens: int = 0
    slots: dict[str, int] = field(default_factory=dict)
    bytes: int = 0


@dataclass(frozen=True)
class SessionLimits:
    max_modalities: int = DEFAULT_MAX_MODALITIES
    max_pending_chunks: int = DEFAULT_MAX_PENDING_CHUNKS
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES
    max_output_chunks: int = DEFAULT_MAX_OUTPUT_CHUNKS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES
    operation_timeout_s: float = DEFAULT_OPERATION_TIMEOUT_S
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S


@dataclass(frozen=True)
class SessionOperation:
    """Coordinator-to-stage session operation, carried in request metadata."""

    operation: Literal["open", "append", "close"]
    session_identity: SessionIdentity
    stages: tuple[str, ...]
    chunk: TimedChunk | None = None

    def to_dict(self) -> SessionOperationDict:
        return {
            "operation": self.operation,
            "session_identity": self.session_identity.to_dict(),
            "stages": list(self.stages),
            "chunk": None if self.chunk is None else self.chunk.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> SessionOperation:
        if not isinstance(data, dict):
            raise ValueError("session operation must be an object")
        else:
            return msgspec.convert(
                data, type=cls, strict=True, builtin_types=BUILTIN_TYPES
            )


def find_session_operation(
    metadata: Mapping[str, object],
) -> SessionOperation | None:
    """Return the operation in request metadata, or None for an ordinary request."""
    operation_fields = metadata.get(SESSION_METADATA_KEY)
    if operation_fields is None:
        return None
    else:
        return SessionOperation.from_dict(operation_fields)


def wire_size(chunk_fields: TimedChunkDict | OutputChunkDict) -> int:
    """Return the msgpack wire size of a chunk dict without copying a binary payload."""
    payload = chunk_fields["payload"]
    if isinstance(payload, bytes):
        payload_size = len(payload)
        if payload_size < MSGPACK_BIN8_LIMIT:
            header_growth = 0
        elif payload_size < MSGPACK_BIN16_LIMIT:
            header_growth = MSGPACK_BIN16_HEADER_GROWTH
        else:
            header_growth = MSGPACK_BIN32_HEADER_GROWTH
        packed_without_payload = msgpack.packb(
            {**chunk_fields, "payload": b""}, use_bin_type=True
        )
        return len(packed_without_payload) + payload_size + header_growth
    else:
        return len(msgpack.packb(chunk_fields, use_bin_type=True))
