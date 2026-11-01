# SPDX-License-Identifier: Apache-2.0
"""Coordinator session failures that the realtime layer maps to client error codes."""

from __future__ import annotations


class SessionIdleTimeoutError(TimeoutError):
    """A session received no input within its idle timeout."""

    def __init__(self, idle_timeout_s: float) -> None:
        super().__init__(f"no session input for {idle_timeout_s:g} s")


class SessionOutputBudgetError(RuntimeError):
    """The session output queue is at capacity."""

    MESSAGE = "session output budget exhausted"

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message)
