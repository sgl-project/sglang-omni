# SPDX-License-Identifier: Apache-2.0
"""Lightweight scheduler message types shared across scheduling backends."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Literal, Protocol


@dataclass
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: object = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error", "kv_transfer", "admitted"]
    data: object = None
    target: str | None = None
    metadata: dict[str, object] | None = None


class StageScheduler(Protocol):
    """Scheduler lifecycle and message queues consumed by a pipeline stage."""

    @property
    def inbox(self) -> Queue[IncomingMessage]: ...

    @property
    def outbox(self) -> Queue[OutgoingMessage]: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def abort(self, request_id: str) -> None: ...
