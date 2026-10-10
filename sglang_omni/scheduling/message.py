# SPDX-License-Identifier: Apache-2.0
"""Lightweight scheduler message types shared across scheduling backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

IncomingMessageType = Literal[
    "new_request", "stream_chunk", "stream_chunk_batch", "stream_done", "abort"
]


@dataclass
class IncomingMessage:
    request_id: str
    type: IncomingMessageType
    # ``stream_chunk_batch`` carries a StreamItemBatch and reaches only
    # schedulers that set ``accepts_stream_chunk_batch``; ``request_id`` is
    # then the first row's owner.
    data: Any = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error", "kv_transfer", "admitted"]
    data: Any = None
    target: str | None = None
    metadata: dict[str, Any] | None = None
    # Set on a row-batched stream: row ``i`` of ``data`` belongs to
    # ``request_ids[i]``, and ``request_id`` is ``request_ids[0]``.
    request_ids: tuple[str, ...] | None = None
