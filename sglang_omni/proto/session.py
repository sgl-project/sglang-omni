# SPDX-License-Identifier: Apache-2.0
"""Model-independent, bounded session command and output contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import msgpack

SESSION_METADATA_KEY = "omni_session"


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    incarnation: int = 1
    epoch: int = 0


@dataclass(frozen=True)
class TimedChunk:
    """Input seq is global across modalities within a session incarnation."""

    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: bytes | dict[str, Any] | None
    format: str | None = None
    eos: bool = False


@dataclass(frozen=True)
class OutputChunk:
    """input_seq identifies the originating pipeline input, not a stream seq."""

    ref: SessionRef
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: bytes | dict[str, Any] | None
    format: str | None = None
    eos: bool = False
    kind: Literal["data", "input_done"] = "data"


@dataclass(frozen=True)
class ResourceUsage:
    kv_tokens: int = 0
    slots: dict[str, int] = field(default_factory=dict)
    bytes: int = 0


@dataclass(frozen=True)
class SessionLimits:
    max_modalities: int = 8
    max_pending_chunks: int = 16
    max_pending_bytes: int = 4 * 1024 * 1024
    max_output_chunks: int = 64
    max_output_bytes: int = 4 * 1024 * 1024
    max_chunk_bytes: int = 1024 * 1024
    command_timeout_s: float = 30.0
    idle_timeout_s: float = 300.0


def wire_size(value: Any) -> int:
    """Return the msgpack wire size without copying a binary payload."""
    if isinstance(value, dict) and isinstance(value.get("payload"), bytes):
        size = len(value["payload"])
        # Note (Junnan Li): Msgpack bin headers grow by 1 byte at 256 bytes and 3 bytes at 65536.
        header_growth = 0 if size < 256 else 1 if size < 65536 else 3
        return (
            len(msgpack.packb({**value, "payload": b""}, use_bin_type=True))
            + size
            + header_growth
        )
    return len(msgpack.packb(value, use_bin_type=True))
