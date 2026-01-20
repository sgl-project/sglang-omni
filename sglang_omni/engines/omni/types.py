# SPDX-License-Identifier: Apache-2.0
"""Core types for OmniEngine - generic, model-agnostic."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SchedulerStatus(Enum):
    """Scheduler request lifecycle status."""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass
class SchedulerRequest:
    """Generic scheduler request container.

    The Scheduler only cares about:
    - request_id: identity
    - status: lifecycle state

    Everything else is stored in `data` (opaque to Scheduler).
    """

    request_id: str
    status: SchedulerStatus = SchedulerStatus.WAITING
    data: Any = None  # Model-specific, opaque to Scheduler

    # Timestamps
    arrival_time: float = 0.0
    finish_time: float | None = None


@dataclass
class SchedulerOutput:
    """Generic contract between Scheduler and ModelRunner.

    - requests: which requests to process
    - batch_data: opaque, built by BatchPlanner, consumed by InputPreparer
    """

    requests: list[SchedulerRequest]
    batch_data: Any  # Opaque - built by BatchPlanner, consumed by InputPreparer
    step_id: int = 0

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]


@dataclass
class RequestOutput:
    """Generic output for a single request.

    The `data` field contains model-specific output
    (tokens, embeddings, latents, etc.)
    """

    request_id: str
    data: Any = None  # Model-specific output
    finished: bool = False
    finish_reason: str | None = None  # "stop", "length", "abort"


@dataclass
class ModelRunnerOutput:
    """Generic output from ModelRunner."""

    # request_id -> output
    outputs: dict[str, RequestOutput]
    # Ordered request ids for this output batch.
    req_ids: list[str] = field(default_factory=list)
    # request_id -> index within req_ids
    req_id_to_index: dict[str, int] = field(default_factory=dict)
