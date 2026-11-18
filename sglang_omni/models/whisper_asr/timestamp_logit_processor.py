# SPDX-License-Identifier: Apache-2.0
"""Whisper timestamp constraints for SGLang decoding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
else:
    pass


class WhisperTimestampParams(TypedDict):
    segment_timestamps: bool
    timestamp_begin_id: int
    no_timestamps_token_id: int
    eos_token_id: int
    max_initial_timestamp_index: object
    __req__: Req


def apply_timestamp_rules(
    scores: torch.Tensor, sequence: Sequence[int], params: WhisperTimestampParams
) -> None:
    """Mask one row of scores in place for the token that follows sequence.

    sequence is every token the request has emitted before this row; a speculative row adds the drafts ahead of it.
    """
    timestamp_begin = int(params["timestamp_begin_id"])
    no_timestamps_token_id = int(params["no_timestamps_token_id"])
    eos_token_id = int(params["eos_token_id"])
    max_initial_timestamp_index = params.get("max_initial_timestamp_index")

    scores[no_timestamps_token_id] = -float("inf")

    last_was_timestamp = bool(sequence) and sequence[-1] >= timestamp_begin
    penultimate_was_timestamp = len(sequence) < 2 or sequence[-2] >= timestamp_begin
    if last_was_timestamp:
        if penultimate_was_timestamp:
            scores[timestamp_begin:] = -float("inf")
        else:
            scores[:eos_token_id] = -float("inf")
    else:
        pass

    timestamps = [token_id for token_id in sequence if token_id >= timestamp_begin]
    if timestamps:
        if last_was_timestamp and not penultimate_was_timestamp:
            timestamp_last = timestamps[-1]
        else:
            timestamp_last = timestamps[-1] + 1
        scores[timestamp_begin:timestamp_last] = -float("inf")
    else:
        pass

    if not sequence:
        scores[:timestamp_begin] = -float("inf")
        if max_initial_timestamp_index is not None:
            last_allowed = timestamp_begin + int(max_initial_timestamp_index)
            scores[last_allowed + 1 :] = -float("inf")
        else:
            pass
    else:
        pass

    logprobs = torch.nn.functional.log_softmax(scores.float(), dim=-1)
    timestamp_logprob = logprobs[timestamp_begin:].logsumexp(dim=-1)
    max_text_token_logprob = logprobs[:timestamp_begin].max()
    if timestamp_logprob > max_text_token_logprob:
        scores[:timestamp_begin] = -float("inf")
    else:
        pass


def apply_timestamp_rules_on_device(
    scores: torch.Tensor,
    *,
    last_token: torch.Tensor,
    penultimate_token: torch.Tensor,
    last_timestamp: torch.Tensor,
    params: WhisperTimestampParams,
) -> None:
    """apply_timestamp_rules for many rows at once, from three tokens of each row's history.

    scores is [rows, vocab] and every row's sequence is non-empty. penultimate_token holds timestamp_begin_id when the sequence has one token, which the rules treat like a timestamp, and last_timestamp holds -1 before the first timestamp. Nothing is read back to the host.
    """
    timestamp_begin = int(params["timestamp_begin_id"])
    no_timestamps_token_id = int(params["no_timestamps_token_id"])
    eos_token_id = int(params["eos_token_id"])
    vocab = torch.arange(scores.shape[-1], device=scores.device)
    is_timestamp = vocab >= timestamp_begin

    scores[:, no_timestamps_token_id] = -float("inf")
    last_was_timestamp = (last_token >= timestamp_begin)[:, None]
    penultimate_was_timestamp = (penultimate_token >= timestamp_begin)[:, None]
    masked = last_was_timestamp & penultimate_was_timestamp & is_timestamp
    masked |= last_was_timestamp & ~penultimate_was_timestamp & (vocab < eos_token_id)
    timestamp_last = torch.where(
        last_was_timestamp & ~penultimate_was_timestamp,
        last_timestamp[:, None],
        last_timestamp[:, None] + 1,
    )
    masked |= (last_timestamp >= 0)[:, None] & is_timestamp & (vocab < timestamp_last)
    scores.masked_fill_(masked, -float("inf"))

    logprobs = torch.nn.functional.log_softmax(scores.float(), dim=-1)
    timestamp_logprob = logprobs[:, timestamp_begin:].logsumexp(dim=-1)
    max_text_token_logprob = logprobs[:, :timestamp_begin].max(dim=-1).values
    scores.masked_fill_(
        (timestamp_logprob > max_text_token_logprob)[:, None] & ~is_timestamp,
        -float("inf"),
    )


class WhisperTimestampLogitProcessor(CustomLogitProcessor):
    """Apply Transformers-compatible Whisper timestamp constraints."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Sequence[WhisperTimestampParams | None] | None = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits
        else:
            pass

        scores = logits.clone()
        for batch_index, params in enumerate(custom_param_list):
            if not params or not params.get("segment_timestamps"):
                continue
            else:
                pass
            apply_timestamp_rules(
                scores[batch_index], list(params["__req__"].output_ids), params
            )

        return scores


__all__ = [
    "WhisperTimestampLogitProcessor",
    "apply_timestamp_rules",
    "apply_timestamp_rules_on_device",
]
