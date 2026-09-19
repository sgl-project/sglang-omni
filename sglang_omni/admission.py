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
        if InvalidRequestError.matches(exc):
            return False
        return isinstance(exc, cls) or (exc is not None and cls.MESSAGE in str(exc))

    @classmethod
    def from_message(cls, message: str | None) -> Exception:
        if InvalidRequestError.matches(message):
            return InvalidRequestError(str(message)[len(InvalidRequestError.PREFIX) :])
        if cls.matches(message):
            return cls()
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
