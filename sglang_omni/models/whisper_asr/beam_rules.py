# SPDX-License-Identifier: Apache-2.0
"""Whisper's suppress, begin-suppress and timestamp rules for one beam group."""

from __future__ import annotations

import torch

from sglang_omni.models.whisper_asr.timestamp_logit_processor import (
    WhisperTimestampParams,
    apply_timestamp_rules,
    apply_timestamp_rules_on_device,
    timestamp_state_after,
)


class WhisperBeamLogitRules:
    """Masks every beam's scores the way Hugging Face's Whisper logits processors do.

    Timestamp rules need three tokens of each beam's history; they are kept on the device and follow the survivors' parents every step.
    """

    def __init__(
        self,
        *,
        suppress_token_ids: list[int],
        begin_suppress_token_ids: list[int],
        timestamp_params: WhisperTimestampParams | None,
    ) -> None:
        self.suppress_token_ids = suppress_token_ids
        self.begin_suppress_token_ids = begin_suppress_token_ids
        self.timestamp_params = timestamp_params
        self.suppress_index: torch.Tensor | None = None
        self.last_token_ids: torch.Tensor | None = None
        self.penultimate_token_ids: torch.Tensor | None = None
        self.last_timestamp_ids: torch.Tensor | None = None

    def mask_scores(self, scores: torch.Tensor, step_index: int) -> None:
        if self.suppress_index is None:
            self.suppress_index = torch.tensor(
                self.suppress_token_ids, dtype=torch.int64, device=scores.device
            )
        else:
            pass
        scores[:, self.suppress_index] = -float("inf")
        if step_index == 0:
            scores[:, self.begin_suppress_token_ids] = -float("inf")
            if self.timestamp_params is not None:
                for row in range(scores.shape[0]):
                    apply_timestamp_rules(scores[row], [], self.timestamp_params)
            else:
                pass
        elif self.timestamp_params is not None:
            apply_timestamp_rules_on_device(
                scores,
                last_token=self.last_token_ids,
                penultimate_token=self.penultimate_token_ids,
                last_timestamp=self.last_timestamp_ids,
                params=self.timestamp_params,
            )
        else:
            pass

    def advance(
        self, parent_indices: torch.Tensor, next_token_ids: torch.Tensor
    ) -> None:
        if self.timestamp_params is None:
            return
        else:
            pass
        if self.last_token_ids is None:
            # An empty history: the rules treat the missing penultimate token as a timestamp, and no timestamp came yet.
            timestamp_begin_ids = torch.full_like(
                next_token_ids, int(self.timestamp_params["timestamp_begin_id"])
            )
            parent_state = (
                timestamp_begin_ids,
                timestamp_begin_ids,
                torch.full_like(next_token_ids, -1),
            )
        else:
            parent_state = (
                self.last_token_ids[parent_indices],
                self.penultimate_token_ids[parent_indices],
                self.last_timestamp_ids[parent_indices],
            )
        # note (zhumengzhiren): the helper keeps the token tensor it is given as the last token, and the group stashes next_token_ids after advance, so it gets a copy.
        (
            self.last_token_ids,
            self.penultimate_token_ids,
            self.last_timestamp_ids,
        ) = timestamp_state_after(
            parent_state, next_token_ids.clone(), self.timestamp_params
        )


__all__ = ["WhisperBeamLogitRules"]
