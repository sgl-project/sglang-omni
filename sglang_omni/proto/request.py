# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sglang_omni.proto.continuation import ContinuationToken


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
    terminal_stages: set[str] | None = None
    result: Any = None
    error: str | None = None


EXPLICIT_GENERATION_PARAMS_KEY = "explicit_generation_params"


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
    continuation: ContinuationToken | None = None
    # A receiving Stage assigns this local identity before scheduler dispatch.
    # It is never serialized and prevents old cleanup from clearing re-entry.
    arrival_id: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Scheduler-local stream ingress state. These fields intentionally stay
    # out of to_dict(); they are rebuilt by the receiving scheduler and never
    # form part of the inter-stage wire contract.
    prefetched_chunks: list[Any] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    prefetched_stream_done: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "_type": "StagePayload",
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "data": self.data,
        }
        if self.continuation is not None:
            payload["continuation"] = self.continuation.to_dict()
        else:
            pass
        return payload

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
            continuation=(
                ContinuationToken.from_dict(data["continuation"])
                if data.get("continuation") is not None
                else None
            ),
        )
