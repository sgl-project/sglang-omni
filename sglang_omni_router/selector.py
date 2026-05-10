# SPDX-License-Identifier: Apache-2.0
"""Load-balancing policy selection for healthy Omni workers."""

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
        required_capabilities: set[Capability],
        requested_model: str | None = None,
    ) -> Worker:
        candidates = [
            worker
            for worker in workers
            if worker.is_routable
            and all(worker.supports(capability) for capability in required_capabilities)
        ]
        if requested_model is not None and any(worker.model for worker in candidates):
            candidates = [
                worker for worker in candidates if worker.model == requested_model
            ]
        if not candidates:
            raise NoEligibleWorkerError("no eligible healthy workers")

        if self.policy == "round_robin":
            return self._select_round_robin(candidates)

        if self.policy == "least_request":
            min_active_requests = min(worker.active_requests for worker in candidates)
            least_loaded = [
                worker
                for worker in candidates
                if worker.active_requests == min_active_requests
            ]
            return self._select_round_robin(least_loaded)

        if self.policy == "random":
            return self._random.choice(candidates)

        raise ValueError(f"unsupported routing policy: {self.policy}")

    def _select_round_robin(self, candidates: list[Worker]) -> Worker:
        worker = candidates[self._rr_index % len(candidates)]
        self._rr_index = (self._rr_index + 1) % len(candidates)
        return worker
