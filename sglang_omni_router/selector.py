from __future__ import annotations

import random

from sglang_omni_router.config import Capability, RoutingPolicy
from sglang_omni_router.worker import Worker


class NoEligibleWorkerError(RuntimeError):
    pass


class WorkerSelector:
    def __init__(self, policy: RoutingPolicy, *, seed: int | None = None) -> None:
        self.policy = policy
        self._rr_index = 0
        self._random = random.Random(seed)

    def select(
        self,
        workers: list[Worker],
        *,
        capability: Capability,
    ) -> Worker:
        candidates = [
            worker
            for worker in workers
            if worker.is_healthy and worker.supports(capability)
        ]
        if not candidates:
            raise NoEligibleWorkerError("no eligible healthy workers")

        if self.policy == "round_robin":
            worker = candidates[self._rr_index % len(candidates)]
            self._rr_index = (self._rr_index + 1) % len(candidates)
            return worker

        if self.policy == "least_request":
            return min(
                candidates,
                key=lambda worker: (worker.active_requests, worker.url),
            )

        if self.policy == "random":
            return self._random.choice(candidates)

        raise ValueError(f"unsupported routing policy: {self.policy}")
