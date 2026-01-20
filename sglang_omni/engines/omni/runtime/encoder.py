# SPDX-License-Identifier: Apache-2.0
"""Encoder model support - BatchPlanner, InputPreparer, OutputProcessor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..types import RequestOutput, SchedulerOutput, SchedulerRequest
from .interfaces import ResourceManager


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class EncoderRequestData:
    """Encoder-specific request data (stored in SchedulerRequest.data)."""

    input_ids: torch.Tensor
    embeddings: torch.Tensor | None = None  # Filled after execution


@dataclass
class EncoderBatchData:
    """Encoder-specific batch data (SchedulerOutput.batch_data)."""

    input_ids_list: list[torch.Tensor]
    seq_lens: list[int]


# -----------------------------------------------------------------------------
# BatchPlanner
# -----------------------------------------------------------------------------


class EncoderBatchPlanner:
    """Batch planner for encoder models."""

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        del running
        selected: list[SchedulerRequest] = []
        for request in waiting:
            if len(selected) >= self.max_batch_size:
                break
            if not resource_manager.can_allocate(request):
                break
            resource_manager.allocate(request)
            selected.append(request)
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> EncoderBatchData:
        return EncoderBatchData(
            input_ids_list=[r.data.input_ids for r in requests],
            seq_lens=[len(r.data.input_ids) for r in requests],
        )

    def update_request(
        self,
        request: SchedulerRequest,
        output: RequestOutput,
    ) -> None:
        request.data.embeddings = output.data

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return True  # Encoder always done in one pass


# -----------------------------------------------------------------------------
# InputPreparer
# -----------------------------------------------------------------------------


class EncoderInputPreparer:
    """Converts EncoderBatchData to model inputs."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        batch_data: EncoderBatchData = scheduler_output.batch_data
        max_len = max(batch_data.seq_lens)
        batch_size = len(batch_data.input_ids_list)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        for i, ids in enumerate(batch_data.input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = ids.to(device)
            attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class EncoderOutputProcessor:
    """Extracts embeddings from encoder output."""

    def __init__(self, pooling: str = "last"):
        """Initialize output processor.

        Args:
            pooling: Pooling strategy - "last", "mean", or "cls"
        """
        self.pooling = pooling

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        # Handle different output formats
        hidden_states = getattr(model_output, "last_hidden_state", None)
        if hidden_states is None:
            if not isinstance(model_output, torch.Tensor):
                raise ValueError(f"Unexpected model output type: {type(model_output)}")
            hidden_states = model_output

        batch_data: EncoderBatchData = scheduler_output.batch_data

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            seq_len = batch_data.seq_lens[i]

            if self.pooling == "last":
                emb = hidden_states[i, seq_len - 1]
            elif self.pooling == "mean":
                emb = hidden_states[i, :seq_len].mean(dim=0)
            elif self.pooling == "cls":
                emb = hidden_states[i, 0]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=emb,
                finished=True,
                finish_reason="stop",
            )

        return outputs
