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
        if InvalidRequestError.matches(exc):
            return False
        else:
            pass
        return isinstance(exc, cls) or (exc is not None and cls.MESSAGE in str(exc))

    @classmethod
    def from_message(cls, message: str | None) -> Exception:
        if InvalidRequestError.matches(message):
            return InvalidRequestError(str(message)[len(InvalidRequestError.PREFIX) :])
        else:
            pass
        if cls.matches(message):
            return cls()
        else:
            pass
        return RuntimeError(message or "Unknown error")


class InvalidRequestError(ValueError):
    """An explicit client-input rejection that survives stage IPC."""

    PREFIX = "Invalid request: "

    def __init__(self, detail: str) -> None:
        super().__init__(self.PREFIX + detail)

    @classmethod
    def matches(cls, exc: BaseException | str | None) -> bool:
        return isinstance(exc, cls) or (
            exc is not None and str(exc).startswith(cls.PREFIX)
        )


class ContextExhaustedError(ValueError):
    """A session unit that no longer fits the thinker context."""

    CODE = "context_exhausted"

    @classmethod
    def matches(cls, exc: BaseException) -> bool:
        return isinstance(exc, cls) or str(exc).startswith(f"{cls.CODE}:")
