# SPDX-License-Identifier: Apache-2.0
"""Shared preprocessing -> AR-engine handoff queue.

A process-wide tri-state registry: prepared (published, awaiting the scheduler),
inflight (currently preprocessing), aborted (in-flight ids aborted before publish,
so the pending insert is dropped). Transitions are identical across TTS models;
only the opaque context and payload type differ.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

CtxT = TypeVar("CtxT")
PrepT = TypeVar("PrepT")


@dataclass(frozen=True)
class QueueSnapshot:
    # note (Yue Yin): read-only copy for introspection/tests -- so nothing outside
    # the queue can mutate the sets or hold the lock and bypass the transitions.
    context: Any
    prepared: frozenset[str]
    inflight: frozenset[str]
    aborted: frozenset[str]


class PreparedRequestQueue(Generic[CtxT, PrepT]):
    """Thread-safe tri-state handoff registry for preprocessing -> AR scheduler.

    State is private; drive it through the transition methods and read it via
    snapshot(), so callers cannot mutate the sets or take the lock to skip the API.
    """

    def __init__(self) -> None:
        self.context: CtxT | None = None
        self.prepared: dict[str, PrepT] = {}
        self.inflight: set[str] = set()
        self.aborted: set[str] = set()
        self.lock = threading.Lock()

    def snapshot(self) -> QueueSnapshot:
        """Read-only view of the current state."""
        with self.lock:
            return QueueSnapshot(
                context=self.context,
                prepared=frozenset(self.prepared),
                inflight=frozenset(self.inflight),
                aborted=frozenset(self.aborted),
            )

    def set_context(self, context: CtxT) -> None:
        """Register the preprocessing context and reset the registry."""
        with self.lock:
            self.context = context
            self.prepared.clear()
            self.inflight.clear()
            self.aborted.clear()

    def clear_context(self) -> None:
        """Drop the context and reset the registry (mainly tests and reloads)."""
        with self.lock:
            self.context = None
            self.prepared.clear()
            self.inflight.clear()
            self.aborted.clear()

    def begin(self, request_id: str) -> CtxT | None:
        # note (Yue Yin): read the context and mark in-flight under one lock, so a
        # concurrent clear_context cannot leave a stale in-flight id behind.
        with self.lock:
            context = self.context
            if context is not None:
                self.inflight.add(request_id)
            else:
                pass
            return context

    def fail_inflight(self, request_id: str) -> None:
        """Roll back an in-flight request whose preprocessing raised."""
        with self.lock:
            self.inflight.discard(request_id)
            self.aborted.discard(request_id)

    def publish(self, request_id: str, prepared: PrepT) -> bool:
        # note (Yue Yin): fail closed -- store only while the id is still in flight,
        # so a publish after a context reset or without begin() cannot leave a stale
        # handoff. Returns False when dropped.
        with self.lock:
            inflight = request_id in self.inflight
            self.inflight.discard(request_id)
            aborted = request_id in self.aborted
            self.aborted.discard(request_id)
            if inflight and not aborted:
                self.prepared[request_id] = prepared
                return True
            else:
                pass
            return False

    def abort(self, request_id: str) -> None:
        # note (Yue Yin): only tombstone while preprocessing is in flight; an abort
        # for a request that is not being preprocessed leaves nothing behind.
        with self.lock:
            if self.prepared.pop(request_id, None) is not None:
                return
            else:
                pass
            if request_id in self.inflight:
                self.aborted.add(request_id)
            else:
                pass

    def pop(self, request_id: str) -> PrepT | None:
        """Remove and return a published handoff, or None if absent."""
        with self.lock:
            return self.prepared.pop(request_id, None)
