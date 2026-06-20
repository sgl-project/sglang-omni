# SPDX-License-Identifier: Apache-2.0
"""Scheduling types used by OmniScheduler and SGLang components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


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
    can_run_cuda_graph: bool = False


@dataclass
class ARRequestData:
    """Backend-neutral autoregressive request state."""

    input_ids: "torch.Tensor | None" = None
    attention_mask: "torch.Tensor | None" = None
    model_inputs: dict[str, Any] = field(default_factory=dict)
    output_ids: list[int] = field(default_factory=list)
    extra_model_outputs: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    weight_version: str | None = None
    return_logprob: bool = False
    output_token_logprobs: list[Any] = field(default_factory=list)
    capture_model_output_keys: tuple[str, ...] = ()
    max_new_tokens: int | None = None
    enforce_request_limits: bool = False
    temperature: float = 0.0


def sampled_logprobs_to_list(next_token_logprobs: Any) -> list[float] | None:
    """Convert sampler-produced per-row selected-token logprobs to a list.

    The sampler owns logprob semantics such as temperature and original-logprob
    mode. Rollout code should preserve those selected-token values instead of
    recomputing logprobs from logits.
    """

    if next_token_logprobs is None:
        return None
    if hasattr(next_token_logprobs, "detach"):
        values = next_token_logprobs.detach().float().cpu().tolist()
    elif hasattr(next_token_logprobs, "tolist"):
        values = next_token_logprobs.tolist()
    else:
        values = next_token_logprobs
    if isinstance(values, (int, float)):
        return [float(values)]
    if not isinstance(values, (list, tuple)):
        return None

    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
        out.append(float(value))
    return out
