# SPDX-License-Identifier: Apache-2.0
"""Coordinator session failures that carry their own client error codes."""

from __future__ import annotations


class SessionIdleTimeoutError(TimeoutError):
    """A session received no input within its idle timeout."""

    CODE = "session_idle_timeout"

    def __init__(self, idle_timeout_s: float) -> None:
        super().__init__(f"no session input for {idle_timeout_s:g} s")


class SessionOutputBudgetError(RuntimeError):
    """The session output queue is at capacity."""

    CODE = "output_budget_exhausted"
    MESSAGE = "session output budget exhausted"

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)
