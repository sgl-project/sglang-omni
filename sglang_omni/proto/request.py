# SPDX-License-Identifier: Apache-2.0
"""Request state and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang_omni.models.omni_base.types import PromptInputs


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
    """Payload passed between stages with request context and pipeline state.

    Combines request routing information with per-request pipeline state that
    flows through preprocessing, encoding, merging, thinking, and decoding stages.
    """

    # Request context
    request_id: str
    request: OmniRequest
    data: Any = None

    # Pipeline state fields (flattened for efficient access)
    raw_inputs: Any | None = None
    prompt: PromptInputs | None = None
    mm_inputs: dict[str, Any] | None = None
    encoder_inputs: dict[str, dict[str, Any]] | None = None
    encoder_outs: dict[str, Any] | None = None
    thinker_inputs: dict[str, Any] | None = None
    thinker_out: dict[str, Any] | None = None
    engine_outputs: dict[str, Any] | None = None
    stream_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for control plane)."""
        result = {
            "_type": "StagePayload",
            "request_id": self.request_id,
            "request": self.request.to_dict(),
        }

        # Include data if present
        if self.data is not None:
            result["data"] = self.data

        # Include pipeline state fields if non-None
        state_fields = [
            "raw_inputs",
            "prompt",
            "mm_inputs",
            "encoder_inputs",
            "encoder_outs",
            "thinker_inputs",
            "thinker_out",
            "engine_outputs",
            "stream_state",
        ]
        for field in state_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StagePayload":
        """Deserialize from dict."""
        request = data.get("request", {})
        request_obj = OmniRequest.from_dict(request)

        return cls(
            request_id=data.get("request_id", ""),
            request=request_obj,
            data=data.get("data"),
            raw_inputs=data.get("raw_inputs"),
            prompt=data.get("prompt"),
            mm_inputs=data.get("mm_inputs"),
            encoder_inputs=data.get("encoder_inputs"),
            encoder_outs=data.get("encoder_outs"),
            thinker_inputs=data.get("thinker_inputs"),
            thinker_out=data.get("thinker_out"),
            engine_outputs=data.get("engine_outputs"),
            stream_state=data.get("stream_state"),
        )
