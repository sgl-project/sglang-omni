# SPDX-License-Identifier: Apache-2.0
"""Model-independent, bounded session command and output contracts."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import msgpack

SESSION_METADATA_KEY = "omni_session"


@dataclass(frozen=True)
class SessionRef:
    session_id: str
    incarnation: int = 1
    epoch: int = 0

    def __post_init__(self):
        if not self.session_id or not isinstance(self.session_id, str):
            raise ValueError("session_id must be a nonempty string")
        if type(self.incarnation) is not int or self.incarnation < 1:
            raise ValueError("incarnation must be a positive integer")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")


@dataclass(frozen=True)
class TimedChunk:
    """Input seq is global across modalities within a session incarnation."""

    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: Any
    format: str | None = None
    eos: bool = False

    def __post_init__(self):
        if not self.modality or type(self.seq) is not int or self.seq < 0:
            raise ValueError("modality and nonnegative seq are required")
        if any(
            not math.isfinite(v) or v < 0 for v in (self.t_start_ms, self.duration_ms)
        ):
            raise ValueError("chunk timing must be finite and nonnegative")


@dataclass(frozen=True)
class OutputChunk:
    """input_seq identifies the originating pipeline input, not a stream seq."""

    ref: SessionRef
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: Any
    format: str | None = None
    eos: bool = False
    kind: Literal["data", "input_done"] = "data"


@dataclass(frozen=True)
class ResourceUsage:
    kv_tokens: int = 0
    slots: dict[str, int] = field(default_factory=dict)
    bytes: int = 0

    def __post_init__(self):
        values = [self.kv_tokens, self.bytes, *self.slots.values()]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("resource usage cannot be negative")


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

    def __post_init__(self):
        values = asdict(self)
        if any(
            type(value) is not int
            for key, value in values.items()
            if key.startswith("max_")
        ):
            raise ValueError("queue limits must be integers")
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("session limits must be finite and positive")


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
