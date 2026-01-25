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
    attention_mask: torch.Tensor | None = None
    cache_position: int = 0
    # Prefill-only model kwargs (e.g., multimodal tensors or inputs_embeds).
    # These are applied only on the first step to avoid repeated work.
    model_inputs: dict[str, Any] = field(default_factory=dict)
    # Optional model output attributes to capture during prefill.
    capture_model_output_keys: tuple[str, ...] = ()
    # Captured model outputs for stage-to-stage transitions.
    extra_model_outputs: dict[str, Any] = field(default_factory=dict)
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
    model_inputs: dict[str, Any] | None = None
    attention_mask: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None
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
            model_inputs = data.model_inputs or None
            attention_mask = data.attention_mask
            if attention_mask is None and isinstance(model_inputs, dict):
                mask_from_inputs = model_inputs.get("attention_mask")
                if isinstance(mask_from_inputs, torch.Tensor):
                    attention_mask = mask_from_inputs
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                if attention_mask.shape[0] == 1:
                    attention_mask = attention_mask[0]
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.shape[0],
                    dtype=torch.long,
                )
            data.attention_mask = attention_mask
            cache_position = torch.arange(
                0,
                input_ids.shape[0],
                dtype=torch.long,
            )
        else:
            last_token = data.output_ids[-1]
            input_ids = torch.tensor([last_token], dtype=torch.long)
            model_inputs = None
            current_len = len(data.input_ids) + len(data.output_ids)
            attention_mask = data.attention_mask
            if attention_mask is None or attention_mask.shape[0] != current_len:
                attention_mask = torch.ones(current_len, dtype=torch.long)
                data.attention_mask = attention_mask
            cache_position = torch.tensor([data.cache_position], dtype=torch.long)

        return ARBatchData(
            input_ids=input_ids,
            is_prefill=is_prefill,
            model_inputs=model_inputs,
            attention_mask=attention_mask,
            cache_position=cache_position,
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

    @staticmethod
    def _to_device(value: Any, device: torch.device) -> Any:
        """Recursively move tensors to the target device."""
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: ARInputPreparer._to_device(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [ARInputPreparer._to_device(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(ARInputPreparer._to_device(v, device) for v in value)
        return value

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

        if batch_data.attention_mask is not None:
            attention_mask = batch_data.attention_mask.unsqueeze(0).to(device)
            result["attention_mask"] = attention_mask

        if batch_data.cache_position is not None:
            result["cache_position"] = batch_data.cache_position.to(device)

        if batch_data.past_key_values is not None:
            result["past_key_values"] = batch_data.past_key_values

        if batch_data.model_inputs:
            # Multimodal kwargs should only be provided during prefill.
            result.update(self._to_device(batch_data.model_inputs, device))

        return result


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class AROutputProcessor:
    """AR output processor with per-request sampling."""

    @staticmethod
    def _maybe_detach(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach()
        return value

    def _capture_prefill_outputs(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
        request: SchedulerRequest,
    ) -> None:
        """Capture requested model outputs during prefill."""
        batch_data: ARBatchData = scheduler_output.batch_data
        if not batch_data.is_prefill:
            return

        keys = request.data.capture_model_output_keys
        if not keys:
            return

        captured: dict[str, Any] = {}
        for key in keys:
            if hasattr(model_output, key):
                captured[key] = self._maybe_detach(getattr(model_output, key))
            elif isinstance(model_output, dict) and key in model_output:
                captured[key] = self._maybe_detach(model_output[key])

        if captured:
            request.data.extra_model_outputs.update(captured)

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

        self._capture_prefill_outputs(model_output, scheduler_output, request)

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=(next_token, past_key_values),
                finished=False,  # IterationController decides this
            )
        }
