# SPDX-License-Identifier: Apache-2.0
"""Executor interface for pipeline workers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sglang_omni.proto import StagePayload


class Executor(ABC):
    """Worker-facing executor interface.

    Uses StagePayload for both input and output.
    """

    async def start(self) -> None:
        """Optional lifecycle hook, called before processing."""
        return None

    async def stop(self) -> None:
        """Optional lifecycle hook, called on shutdown."""
        return None

    @abstractmethod
    async def add_request(self, payload: StagePayload) -> None:
        """Accept a request payload for processing."""
        ...

    @abstractmethod
    async def get_result(self) -> StagePayload:
        """Return the next completed payload."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request if possible."""
        ...

    def set_stream_fn(self, fn) -> None:
        """Set the streaming output callback. Sync, non-blocking."""
        pass  # Default no-op; streaming executors override
