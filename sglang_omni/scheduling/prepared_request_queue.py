# SPDX-License-Identifier: Apache-2.0
"""Shared preprocessing -> AR-engine handoff queue (RFC #661, Template 7).

A process-wide tri-state registry — ``prepared`` (published, awaiting the AR
scheduler), ``inflight`` (currently preprocessing), ``aborted`` (in-flight ids
aborted before publish, so the pending insert is dropped) — plus the opaque
per-model preprocessing ``context``. ``moss_tts`` and ``moss_tts_local`` used to
duplicate these transitions line for line; only the context and payload type
differ, so the registry is generic over both.

It is correctness-critical: wrong abort-ordering either strands a GPU slot
forever (ghost request) or publishes a handoff for an already-gone request.
Every method holds ``lock`` for the whole transition. Pure-Python (no torch).
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

CtxT = TypeVar("CtxT")
PrepT = TypeVar("PrepT")


class PreparedRequestQueue(Generic[CtxT, PrepT]):
    """Thread-safe tri-state handoff registry for preprocessing -> AR scheduler.

    The attributes (``context`` / ``prepared`` / ``inflight`` / ``aborted`` /
    ``lock``) are exposed for introspection; mutate them only through the methods
    below, each of which holds ``lock`` for the whole transition.
    """

    def __init__(self) -> None:
        self.context: CtxT | None = None
        self.prepared: dict[str, PrepT] = {}
        self.inflight: set[str] = set()
        self.aborted: set[str] = set()
        self.lock = threading.Lock()

    def set_context(self, context: CtxT) -> None:
        """Register the preprocessing context and reset the registry."""
        with self.lock:
            self.context = context
            self.prepared.clear()
            self.inflight.clear()
            self.aborted.clear()

    def clear_context(self) -> None:
        """Drop the context and reset the registry (reloads and tests)."""
        with self.lock:
            self.context = None
            self.prepared.clear()
            self.inflight.clear()
            self.aborted.clear()

    def begin(self, request_id: str) -> CtxT | None:
        """Read the context and, if present, mark ``request_id`` in flight.

        The read and the in-flight insert happen under one lock so a concurrent
        ``clear_context`` cannot strand a stale in-flight id.
        """
        with self.lock:
            context = self.context
            if context is not None:
                self.inflight.add(request_id)
            return context

    def fail_inflight(self, request_id: str) -> None:
        """Roll back an in-flight request whose preprocessing raised."""
        with self.lock:
            self.inflight.discard(request_id)
            self.aborted.discard(request_id)

    def publish(self, request_id: str, prepared: PrepT) -> bool:
        """Publish a handoff unless the request was aborted mid-flight.

        Returns ``True`` if stored, ``False`` if dropped because an abort arrived
        while preprocessing was running.
        """
        with self.lock:
            self.inflight.discard(request_id)
            aborted = request_id in self.aborted
            self.aborted.discard(request_id)
            if not aborted:
                self.prepared[request_id] = prepared
            return not aborted

    def abort(self, request_id: str) -> None:
        """Drop a published handoff, or tombstone an in-flight one.

        Only tombstone when preprocessing is actually in flight; an abort for a
        request that is neither published nor in flight leaves nothing behind.
        """
        with self.lock:
            if self.prepared.pop(request_id, None) is not None:
                return
            if request_id in self.inflight:
                self.aborted.add(request_id)

    def pop(self, request_id: str) -> PrepT | None:
        """AR side: remove and return a published handoff, or ``None`` if absent."""
        with self.lock:
            return self.prepared.pop(request_id, None)


__all__ = ["PreparedRequestQueue"]
