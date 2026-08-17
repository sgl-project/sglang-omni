# SPDX-License-Identifier: Apache-2.0
"""Queue-full rejects. Stage IPC stringifies exceptions; use matches()."""

from __future__ import annotations


class QueueFullError(RuntimeError):
    """A serving queue is at capacity (HTTP 503)."""

    MESSAGE = "The request queue is full."

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)

    @classmethod
    def matches(cls, exc: BaseException | str | None) -> bool:
        return isinstance(exc, cls) or (exc is not None and cls.MESSAGE in str(exc))

    @classmethod
    def from_message(cls, message: str | None) -> Exception:
        if cls.matches(message):
            return cls()
        return RuntimeError(message or "Unknown error")
