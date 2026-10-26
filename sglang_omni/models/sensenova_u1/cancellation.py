# SPDX-License-Identifier: Apache-2.0
"""Cooperative cancellation at matching SenseNova model boundaries."""

import math
import time
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Protocol

import torch
import torch.distributed as distributed

from sglang_omni.proto.request import StagePayload

SENSENOVA_DEADLINE_METADATA_KEY = "sensenova_deadline_monotonic_s"


class SenseNovaCancelledError(RuntimeError):
    """Generation stopped at a safe checkpoint and the scheduler remains reusable."""


class CancellationProbe(Protocol):
    def __call__(self, request_id: str) -> bool:
        """Return whether the request has been cancelled."""


class GenerationScope:
    def __init__(
        self,
        payloads: list[StagePayload],
        is_cancelled: CancellationProbe,
        device: torch.device,
        process_group: distributed.ProcessGroup | None,
    ) -> None:
        self.request_deadlines: dict[str, float] = {}
        for payload in payloads:
            deadline = payload.request.metadata.get(
                SENSENOVA_DEADLINE_METADATA_KEY, math.inf
            )
            if (
                isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or math.isnan(deadline)
            ):
                raise ValueError("Invalid UMM generation deadline")
            else:
                self.request_deadlines[payload.request_id] = float(deadline)
        self.is_cancelled = is_cancelled
        self.process_group = process_group
        self.cancelled_tensor: torch.Tensor | None = (
            torch.zeros((), dtype=torch.int32, device=device)
            if process_group is not None
            else None
        )
        self.token: Token[GenerationScope | None] | None = None

    def __enter__(self) -> "GenerationScope":
        self.token = CURRENT_GENERATION.set(self)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.token is not None
        CURRENT_GENERATION.reset(self.token)
        self.request_deadlines.clear()
        self.cancelled_tensor = None
        self.token = None

    def checkpoint(self) -> None:
        now = time.monotonic()
        cancelled = all(
            self.is_cancelled(request_id) or now >= deadline
            for request_id, deadline in self.request_deadlines.items()
        )
        if self.process_group is not None:
            assert self.cancelled_tensor is not None
            self.cancelled_tensor.fill_(int(cancelled))
            distributed.all_reduce(
                self.cancelled_tensor,
                op=distributed.ReduceOp.MAX,
                group=self.process_group,
            )
            cancelled = bool(self.cancelled_tensor.item())
        else:
            pass
        if cancelled:
            raise SenseNovaCancelledError("SenseNova generation cancelled or expired")
        else:
            pass


CURRENT_GENERATION: ContextVar[GenerationScope | None] = ContextVar(
    "sensenova_generation", default=None
)


def generation_checkpoint() -> None:
    generation = CURRENT_GENERATION.get()
    if generation is not None:
        generation.checkpoint()
    else:
        pass
