# SPDX-License-Identifier: Apache-2.0
"""Whisper timestamp constraints for SGLang decoding."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class WhisperTimestampLogitProcessor(CustomLogitProcessor):
    """Apply Transformers-compatible Whisper timestamp constraints."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        scores = logits.clone()
        for batch_index, params in enumerate(custom_param_list):
            if not params or not params.get("segment_timestamps"):
                continue

            timestamp_begin = int(params["timestamp_begin_id"])
            no_timestamps_token_id = int(params["no_timestamps_token_id"])
            eos_token_id = int(params["eos_token_id"])
            max_initial_timestamp_index = params.get("max_initial_timestamp_index")
            req = params["__req__"]
            sequence = list(req.output_ids)

            scores[batch_index, no_timestamps_token_id] = -float("inf")

            last_was_timestamp = bool(sequence) and sequence[-1] >= timestamp_begin
            penultimate_was_timestamp = (
                len(sequence) < 2 or sequence[-2] >= timestamp_begin
            )
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    scores[batch_index, timestamp_begin:] = -float("inf")
                else:
                    scores[batch_index, :eos_token_id] = -float("inf")

            timestamps = [
                token_id for token_id in sequence if token_id >= timestamp_begin
            ]
            if timestamps:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                scores[batch_index, timestamp_begin:timestamp_last] = -float("inf")

            if not sequence:
                scores[batch_index, :timestamp_begin] = -float("inf")
                if max_initial_timestamp_index is not None:
                    last_allowed = timestamp_begin + int(max_initial_timestamp_index)
                    scores[batch_index, last_allowed + 1 :] = -float("inf")

            logprobs = torch.nn.functional.log_softmax(
                scores[batch_index].float(), dim=-1
            )
            timestamp_logprob = logprobs[timestamp_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[:timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                scores[batch_index, :timestamp_begin] = -float("inf")

        return scores


__all__ = ["WhisperTimestampLogitProcessor"]
