# SPDX-License-Identifier: Apache-2.0
"""Scheduling types used by OmniScheduler and SGLang components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SchedulerStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()


@dataclass
class SchedulerRequest:
    request_id: str
    status: SchedulerStatus = SchedulerStatus.WAITING
    data: Any = None
    error: Exception | None = None
    arrival_time: float = 0.0
    finish_time: float | None = None


@dataclass
class SchedulerOutput:
    requests: list[SchedulerRequest]
    batch_data: Any
    step_id: int = 0

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]


@dataclass
class RequestOutput:
    request_id: str
    data: Any = None
    finished: bool = False
    extra: dict[str, Any] | None = None


@dataclass
class ModelRunnerOutput:
    outputs: dict[str, RequestOutput]
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
