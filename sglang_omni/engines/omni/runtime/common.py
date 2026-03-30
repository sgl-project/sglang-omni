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
    """AR-style: stop at EOS or length limit.

    Supports both standard ``input_ids`` prefill and ``inputs_embeds`` prefill
    (via ``ARRequestData.prefill_seq_len``).
    """

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
            token = output.data[0]
            if len(output.data) >= 2:
                past_kv = output.data[1]
        elif isinstance(output.data, dict):
            token = output.data.get("token")
            past_kv = output.data.get("past_key_values")
            extra_outputs = output.data.get("extra_model_outputs")

        if past_kv is not None:
            request.data.past_key_values = past_kv
        if isinstance(extra_outputs, dict):
            request.data.extra_model_outputs.update(extra_outputs)

        request.data.output_ids.append(token)

        # Determine prefill length: inputs_embeds path uses prefill_seq_len.
        prefill_len = getattr(request.data, "prefill_seq_len", None)
        if prefill_len is None:
            prefill_len = len(request.data.input_ids)

        expected_len = prefill_len + len(request.data.output_ids)
        attention_mask = request.data.attention_mask
        if attention_mask is None or attention_mask.shape[0] == expected_len - 1:
            if attention_mask is None:
                attention_mask = torch.ones(expected_len, dtype=torch.long)
            else:
                attention_mask = attention_mask.to(dtype=torch.long)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(1)], dim=0
                )
            request.data.attention_mask = attention_mask
        elif attention_mask.shape[0] != expected_len:
            request.data.attention_mask = torch.ones(expected_len, dtype=torch.long)

        if request.data.num_computed_tokens == 0:
            request.data.num_computed_tokens = prefill_len
            request.data.cache_position = request.data.num_computed_tokens
            request.data.model_inputs.clear()
            if hasattr(request.data, "inputs_embeds"):
                request.data.inputs_embeds = None
        else:
            request.data.num_computed_tokens += 1
            request.data.cache_position += 1

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        token = output.data
        if isinstance(output.data, tuple):
            token = output.data[0]

        # Per-request EOS override (e.g. TTS codec_eos_token_id).
        custom_eos = getattr(request.data, "eos_token_ids", None)
        eos_ids = custom_eos if custom_eos else self._eos_token_ids

        if token in eos_ids:
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
