# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RequestState(Enum):
    """State of a request in the pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RequestInfo:
    """Tracking info for a request in the coordinator."""

    request_id: str
    state: RequestState = RequestState.PENDING
    current_stage: str | None = None
    result: Any = None
    error: str | None = None
