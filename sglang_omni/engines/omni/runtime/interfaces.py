# SPDX-License-Identifier: Apache-2.0
"""Runtime protocols for model-type-specific components."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from ..types import Request, RequestOutput, SchedulerOutput


class BatchPlanner(Protocol):
    """Select requests and build batch data."""

    def select_requests(
        self,
        waiting: list[Request],
        running: list[Request],
        resource_manager: "ResourceManager",
    ) -> list[Request]:
        """Select which requests to include in this batch."""
        ...

    def build_batch(self, requests: list[Request]) -> Any:
        """Build model-specific batch data from requests."""
        ...


class ResourceManager(Protocol):
    """Manage model resources (memory, KV cache, etc.)."""

    def can_allocate(self, request: Request) -> bool:
        ...

    def allocate(self, request: Request) -> None:
        ...

    def free(self, request: Request) -> None:
        ...


class IterationController(Protocol):
    """Update per-request state and decide when it finishes."""

    def update_request(self, request: Request, output: RequestOutput) -> None:
        ...

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        ...


class InputPreparer(Protocol):
    """Convert SchedulerOutput to model inputs."""

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        """Convert scheduler output to model input dict.

        For Encoder: padded input_ids + attention_mask
        For AR: input_ids + positions + past_key_values + ...
        For DiT: latents + timesteps + ...
        """
        ...


class OutputProcessor(Protocol):
    """Converts model outputs to RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Convert model output to per-request outputs.

        For Encoder: extract embeddings per request
        For AR: sample tokens per request
        For DiT: extract denoised latents per request
        """
        ...
