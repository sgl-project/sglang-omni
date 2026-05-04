# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from dataclasses import dataclass, field
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


@dataclass
class OmniRequest:
    """User-facing request with inputs and parameters."""

    inputs: Any
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "_type": "OmniRequest",
            "inputs": self.inputs,
            "params": self.params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmniRequest":
        return cls(
            inputs=data.get("inputs"),
            params=data.get("params", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StagePayload:
    """Payload passed between stages with request context."""

    request_id: str
    request: OmniRequest
    data: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "_type": "StagePayload",
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StagePayload":
        request = data.get("request", {})
        if isinstance(request, dict) and request.get("_type") == "OmniRequest":
            request_obj = OmniRequest.from_dict(request)
        else:
            request_obj = OmniRequest.from_dict(request)
        return cls(
            request_id=data.get("request_id", ""),
            request=request_obj,
            data=data.get("data"),
        )
