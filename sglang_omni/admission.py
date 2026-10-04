# SPDX-License-Identifier: Apache-2.0
"""Admission rejects. Stage IPC stringifies exceptions; use matches()."""

from __future__ import annotations

REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV = 1


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
        else:
            pass
        return RuntimeError(message or "Unknown error")


class ContextExhaustedError(ValueError):
    """A session unit that no longer fits the thinker context."""

    CODE = "context_exhausted"

    @classmethod
    def matches(cls, exc: BaseException) -> bool:
        return isinstance(exc, cls) or str(exc).startswith(f"{cls.CODE}:")
