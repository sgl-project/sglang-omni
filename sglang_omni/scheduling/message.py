# SPDX-License-Identifier: Apache-2.0
"""Lightweight scheduler message types shared across scheduling backends."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Literal, Protocol, runtime_checkable

# A consumer that stops puts a wakeup into its own queue to release a blocked get.
IncomingMessageType = Literal[
    "new_request", "stream_chunk", "stream_done", "abort", "admin", "wakeup"
]


@dataclass
class IncomingMessage:
    request_id: str
    type: IncomingMessageType
    data: object = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error", "kv_transfer", "admitted", "wakeup"]
    data: object = None
    target: str | None = None
    metadata: dict[str, object] | None = None


def put_messages(
    outbox: Queue[OutgoingMessage], messages: list[OutgoingMessage]
) -> None:
    """Add messages to an unbounded outbox in order under one lock hold.

    A put per message wakes the consumer at the first one, and the consumer then
    competes with this thread for the interpreter lock while the rest are added.
    """
    if messages:
        with outbox.not_full:
            outbox.queue.extend(messages)
            outbox.unfinished_tasks += len(messages)
            outbox.not_empty.notify(len(messages))
    else:
        pass


class StageScheduler(Protocol):
    """Scheduler lifecycle and message queues consumed by a pipeline stage."""

    @property
    def inbox(self) -> Queue[IncomingMessage]: ...

    @property
    def outbox(self) -> Queue[OutgoingMessage]: ...

    def warm_up_serving_thread(self) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def abort(self, request_id: str) -> None: ...


@runtime_checkable
class ResultOwningScheduler(StageScheduler, Protocol):
    """A scheduler whose results hold resources until routing settles.

    The stage claims each result before it signals stream completion or routes
    it, and skips a result whose claim fails. Once routing settles, routed,
    failed, cancelled or dropped, the stage releases the result, with
    ``delivered`` telling whether a transport accepted it. Schedulers that do
    not implement both methods own nothing and are routed unchanged.
    """

    def claim_result(self, result: object, *, terminal: bool) -> bool: ...

    def release_result(self, result: object, *, delivered: bool) -> None: ...
