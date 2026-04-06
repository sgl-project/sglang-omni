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
    # Prefill-only model kwargs (e.g., multimodal tensors).
    # Cleared after the first step.
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

    # --- inputs_embeds prefill (for models that bypass input_ids) ---
    inputs_embeds: torch.Tensor | None = None
    prefill_seq_len: int | None = None  # auto-computed from inputs_embeds

    # --- Persistent model kwargs (passed every step, not just prefill) ---
    persistent_inputs: dict[str, Any] = field(default_factory=dict)

    # --- Sampling extensions ---
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    suppress_tokens: list[int] | None = None

    # --- Custom EOS tokens (overrides engine default when set) ---
    eos_token_ids: list[int] | None = None

    # --- Per-step hook for model-specific state propagation ---
    # Signature: (data: ARRequestData, model_output: Any, token: int) -> None
    step_hook: Any = None


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
    # inputs_embeds prefill
    inputs_embeds: torch.Tensor | None = None
    # Persistent kwargs (every step)
    persistent_inputs: dict[str, Any] | None = None


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

        inputs_embeds = None
        persistent_inputs = data.persistent_inputs if data.persistent_inputs else None

        if is_prefill:
            input_ids = data.input_ids
            model_inputs = data.model_inputs or None
            inputs_embeds = data.inputs_embeds

            attention_mask = data.attention_mask
            if attention_mask is None and isinstance(model_inputs, dict):
                mask_from_inputs = model_inputs.get("attention_mask")
                if isinstance(mask_from_inputs, torch.Tensor):
                    attention_mask = mask_from_inputs
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                if attention_mask.shape[0] == 1:
                    attention_mask = attention_mask[0]

            # Determine prefill length from inputs_embeds or input_ids
            if inputs_embeds is not None:
                if inputs_embeds.ndim == 3:
                    seq_len = inputs_embeds.shape[1]
                else:
                    seq_len = inputs_embeds.shape[0]
                if data.prefill_seq_len is None:
                    data.prefill_seq_len = seq_len
            else:
                seq_len = input_ids.shape[0]

            if attention_mask is None:
                attention_mask = torch.ones(seq_len, dtype=torch.long)
            data.attention_mask = attention_mask

            cache_position = torch.arange(0, seq_len, dtype=torch.long)
        else:
            last_token = data.output_ids[-1]
            input_ids = torch.tensor([last_token], dtype=torch.long)
            model_inputs = None

            prefill_len = (
                data.prefill_seq_len
                if data.prefill_seq_len is not None
                else len(data.input_ids)
            )
            current_len = prefill_len + len(data.output_ids)
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
            inputs_embeds=inputs_embeds,
            persistent_inputs=persistent_inputs,
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
        self, scheduler_output: SchedulerOutput, device: torch.device
    ) -> dict[str, Any]:
        batch_data: ARBatchData = scheduler_output.batch_data

        result: dict[str, Any] = {"use_cache": True}

        # --- inputs_embeds vs input_ids ---
        if batch_data.is_prefill and batch_data.inputs_embeds is not None:
            embeds = batch_data.inputs_embeds
            if embeds.ndim == 2:
                embeds = embeds.unsqueeze(0)  # -> [1, seq, hidden]
            result["inputs_embeds"] = embeds.to(device)
        else:
            input_ids = batch_data.input_ids.unsqueeze(0).to(device)  # [1, seq_len]
            result["input_ids"] = input_ids

        if batch_data.attention_mask is not None:
            attention_mask = batch_data.attention_mask
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            result["attention_mask"] = attention_mask.to(device)

        if batch_data.cache_position is not None:
            result["cache_position"] = batch_data.cache_position.to(device)

        if batch_data.past_key_values is not None:
            result["past_key_values"] = batch_data.past_key_values

        if batch_data.model_inputs:
            result.update(self._to_device(batch_data.model_inputs, device))

        if batch_data.persistent_inputs:
            result.update(self._to_device(batch_data.persistent_inputs, device))

        return result


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class AROutputProcessor:
    """AR output processor with per-request sampling.

    Supports temperature, top-k, top-p, repetition penalty, and token
    suppression.  When a request carries a ``step_hook``, it is invoked
    here inside model execution (executor thread), so heavy model-specific
    post-processing does not block the asyncio event loop.
    """

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

    # ----- Sampling helpers -----

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        output_ids: list[int],
        penalty: float,
    ) -> torch.Tensor:
        if penalty == 1.0 or not output_ids:
            return logits
        token_ids = torch.tensor(output_ids, dtype=torch.long, device=logits.device)
        scores = logits.gather(-1, token_ids.unsqueeze(0))
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        logits.scatter_(-1, token_ids.unsqueeze(0), scores)
        return logits

    @staticmethod
    def _apply_suppress_tokens(
        logits: torch.Tensor,
        suppress_tokens: list[int] | None,
    ) -> torch.Tensor:
        if not suppress_tokens:
            return logits
        logits[:, suppress_tokens] = float("-inf")
        return logits

    @staticmethod
    def _sample(
        logits: torch.Tensor, temperature: float, top_k: int, top_p: float
    ) -> int:
        if temperature <= 0.0:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        # Top-p (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            remove_mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def process(
        self, model_output: Any, scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        if not hasattr(model_output, "logits"):
            raise ValueError(f"Unexpected model output type: {type(model_output)}")

        logits = model_output.logits  # [batch, seq, vocab]
        past_key_values = model_output.past_key_values

        # Clone last position logits for manipulation
        last_logits = logits[:, -1, :].clone()  # [batch, vocab]

        request = scheduler_output.requests[0]
        data: ARRequestData = request.data

        # Apply suppress_tokens
        self._apply_suppress_tokens(last_logits, data.suppress_tokens)

        # Apply repetition_penalty
        self._apply_repetition_penalty(
            last_logits, data.output_ids, data.repetition_penalty
        )

        # Sample
        next_token = self._sample(last_logits, data.temperature, data.top_k, data.top_p)

        self._capture_prefill_outputs(model_output, scheduler_output, request)

        if data.step_hook is not None:
            data.step_hook(data, model_output, next_token)

        output_data = (next_token, past_key_values)

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=output_data,
                finished=False,  # IterationController decides this
            )
        }
