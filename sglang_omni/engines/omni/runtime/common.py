# SPDX-License-Identifier: Apache-2.0
"""Reusable runtime components."""

from __future__ import annotations

import torch

from ..types import RequestOutput, SchedulerRequest


class SimpleResourceManager:
    """Count-based resource manager."""

    def __init__(self, max_count: int = 32):
        self.max_count = max_count
        self._count = 0

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return self._count < self.max_count

    def allocate(self, request: SchedulerRequest) -> None:
        self._count += 1

    def free(self, request: SchedulerRequest) -> None:
        self._count = max(0, self._count - 1)


class SinglePassIterationController:
    """Encoder-style: always done in one pass."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        if isinstance(output.data, dict) and hasattr(request.data, "output_dict"):
            request.data.output_dict = output.data
            return
        if hasattr(request.data, "embeddings"):
            request.data.embeddings = output.data
            return
        if isinstance(request.data, dict):
            if isinstance(output.data, dict):
                request.data["output_dict"] = output.data
            else:
                request.data["embeddings"] = output.data

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return True


class EosIterationController:
    """AR-style: stop at EOS or length limit."""

    def __init__(
        self,
        eos_token_id: int | list[int],
        max_length: int = 2048,
        default_max_new_tokens: int | None = None,
    ):
        self._eos_token_ids = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )
        self._max_length = max_length
        self._default_max_new_tokens = default_max_new_tokens

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        token = output.data
        past_kv = None
        extra_outputs = None

        if isinstance(output.data, tuple):
            token, past_kv = output.data
        elif isinstance(output.data, dict):
            token = output.data.get("token")
            past_kv = output.data.get("past_key_values")
            extra_outputs = output.data.get("extra_model_outputs")

        if past_kv is not None:
            request.data.past_key_values = past_kv
        if isinstance(extra_outputs, dict):
            request.data.extra_model_outputs.update(extra_outputs)

        request.data.output_ids.append(token)
        expected_len = len(request.data.input_ids) + len(request.data.output_ids)
        attention_mask = request.data.attention_mask
        if attention_mask is None or attention_mask.shape[0] == expected_len - 1:
            if attention_mask is None:
                attention_mask = request.data.input_ids.new_ones(expected_len)
            else:
                attention_mask = attention_mask.to(dtype=torch.long)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(1)], dim=0
                )
            request.data.attention_mask = attention_mask
        elif attention_mask.shape[0] != expected_len:
            request.data.attention_mask = request.data.input_ids.new_ones(expected_len)

        if request.data.num_computed_tokens == 0:
            request.data.num_computed_tokens = len(request.data.input_ids)
            # Prepare cache position for the first decode step.
            request.data.cache_position = request.data.num_computed_tokens
            # Free large prefill-only inputs once they are no longer needed.
            request.data.model_inputs.clear()
        else:
            request.data.num_computed_tokens += 1
            request.data.cache_position += 1

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        token = output.data
        if isinstance(output.data, tuple):
            token, _ = output.data

        if token in self._eos_token_ids:
            return True

        max_new_tokens = request.data.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self._default_max_new_tokens
        if (
            max_new_tokens is not None
            and len(request.data.output_ids) >= max_new_tokens
        ):
            return True

        return request.data.num_computed_tokens >= self._max_length
