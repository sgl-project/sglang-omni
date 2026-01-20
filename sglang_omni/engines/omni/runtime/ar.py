# SPDX-License-Identifier: Apache-2.0
"""AR (Autoregressive) model support with HF KV cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..types import RequestOutput, SchedulerOutput, SchedulerRequest
from .common import SimpleResourceManager
from .interfaces import ResourceManager

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class ARRequestData:
    """AR-specific request data (stored in SchedulerRequest.data)."""

    input_ids: torch.Tensor
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    max_new_tokens: int | None = None
    temperature: float = 0.0

    # For simple HF-style KV cache
    past_key_values: tuple | None = None


@dataclass
class ARBatchData:
    """AR-specific batch data (SchedulerOutput.batch_data).

    Simple version: single request only (no batching yet).
    """

    input_ids: torch.Tensor  # [num_tokens]
    is_prefill: bool
    past_key_values: tuple | None = None


# -----------------------------------------------------------------------------
# BatchPlanner
# -----------------------------------------------------------------------------


class ARBatchPlanner:
    """Batch planner for single-request AR execution."""

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        if running:
            return [running[0]]

        if not waiting:
            return []

        request = waiting[0]
        if not resource_manager.can_allocate(request):
            return []

        resource_manager.allocate(request)
        return [request]

    def build_batch(self, requests: list[SchedulerRequest]) -> ARBatchData:
        request = requests[0]
        data: ARRequestData = request.data
        is_prefill = data.num_computed_tokens == 0

        if is_prefill:
            input_ids = data.input_ids
        else:
            last_token = data.output_ids[-1]
            input_ids = torch.tensor([last_token], dtype=torch.long)

        return ARBatchData(
            input_ids=input_ids,
            is_prefill=is_prefill,
            past_key_values=data.past_key_values,
        )


class ARResourceManager(SimpleResourceManager):
    """Resource manager that clears KV cache on free."""

    def free(self, request: SchedulerRequest) -> None:
        super().free(request)
        request.data.past_key_values = None


# -----------------------------------------------------------------------------
# InputPreparer
# -----------------------------------------------------------------------------


class ARInputPreparer:
    """AR input preparer for HF models (single request)."""

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        batch_data: ARBatchData = scheduler_output.batch_data
        input_ids = batch_data.input_ids.unsqueeze(0).to(device)  # [1, seq_len]

        result = {
            "input_ids": input_ids,
            "use_cache": True,
        }

        if batch_data.past_key_values is not None:
            result["past_key_values"] = batch_data.past_key_values

        return result


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class AROutputProcessor:
    """AR output processor with per-request sampling."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        if not hasattr(model_output, "logits"):
            raise ValueError(f"Unexpected model output type: {type(model_output)}")

        logits = model_output.logits  # [batch, seq, vocab]
        past_key_values = model_output.past_key_values

        # Sample from last position
        last_logits = logits[:, -1, :]  # [batch, vocab]

        request = scheduler_output.requests[0]
        temperature = request.data.temperature
        if temperature <= 0.0:
            next_token = last_logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=(next_token, past_key_values),
                finished=False,  # IterationController decides this
            )
        }
