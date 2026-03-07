# SPDX-License-Identifier: Apache-2.0
"""Worker router with per-request affinity."""

from __future__ import annotations

import asyncio

from sglang_omni.pipeline.stage.work import WorkDescriptor


class WorkerRouter:
    """Assign work to workers with sticky per-request affinity."""

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[WorkDescriptor | None]] = []
        self._affinity: dict[str, int] = {}
        self._rr_index = 0

    def add_worker(self) -> asyncio.Queue[WorkDescriptor | None]:
        queue: asyncio.Queue[WorkDescriptor | None] = asyncio.Queue()
        self._queues.append(queue)
        return queue

    def enqueue(self, work: WorkDescriptor) -> None:
        if not self._queues:
            raise RuntimeError("No workers available")

        idx = self._affinity.get(work.request_id)
        if idx is None:
            idx = self._rr_index % len(self._queues)
            self._rr_index += 1
            self._affinity[work.request_id] = idx

        self._queues[idx].put_nowait(work)

    def get_worker_index(self, request_id: str) -> int | None:
        """Return the worker index for a request, or None if not assigned."""
        return self._affinity.get(request_id)

    def clear_request(self, request_id: str) -> None:
        self._affinity.pop(request_id, None)

    def queue_size(self) -> int:
        return sum(queue.qsize() for queue in self._queues)

    def num_workers(self) -> int:
        return len(self._queues)
