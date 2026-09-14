# SPDX-License-Identifier: Apache-2.0
"""Load-balancing policy selection for healthy Omni workers."""

from __future__ import annotations

import random

from sglang_omni_router.python.config import Capability, RoutingPolicy
from sglang_omni_router.python.worker import Worker


class NoEligibleWorkerError(RuntimeError):
    pass


def eligible_workers(
    workers: list[Worker],
    *,
    required_capabilities: set[Capability],
    requested_model: str | None = None,
) -> list[Worker]:
    """Return routable workers that satisfy the request contract."""
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
    return candidates


def require_eligible_worker(
    worker: Worker | None,
    *,
    required_capabilities: set[Capability],
    requested_model: str | None = None,
) -> Worker:
    """Validate an exact worker without advancing load-balancer state."""
    candidates = eligible_workers(
        [] if worker is None else [worker],
        required_capabilities=required_capabilities,
        requested_model=requested_model,
    )
    if not candidates:
        raise NoEligibleWorkerError("no eligible healthy worker")
    return candidates[0]


class WorkerSelector:
    def __init__(
        self,
        policy: RoutingPolicy,
        *,
        seed: int | None = None,
        rr_offset: int = 0,
    ) -> None:
        self.policy = policy
        # Note (Jiaxin Deng): rr_offset staggers round-robin starts so N
        # fresh DPs do not all herd onto worker 0 at once.
        self._rr_index = rr_offset
        self._random = random.Random(seed)

    def select(
        self,
        workers: list[Worker],
        *,
        required_capabilities: set[Capability],
        requested_model: str | None = None,
    ) -> Worker:
        candidates = eligible_workers(
            workers,
            required_capabilities=required_capabilities,
            requested_model=requested_model,
        )
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
        # Note (Jiaxin Deng): the cursor stays monotonic and is reduced only when
        # indexing, so a temporarily shrunken candidate list cannot collapse every
        # data plane's cursor to 0 and erase the rr_offset stagger.
        worker = candidates[self._rr_index % len(candidates)]
        self._rr_index += 1
        return worker
