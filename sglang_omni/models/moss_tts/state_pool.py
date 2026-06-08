# SPDX-License-Identifier: Apache-2.0
"""Row-indexed MOSS-TTS decode state.

The pool stores next-step-critical decode state in stable per-request rows.
It mirrors the ownership model used by Higgs's sampler pool, while keeping
MOSS-specific state such as feedback embeddings and generated row history.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MossDecodeJournal:
    request_id: str
    row: int
    sampled_row: torch.Tensor | None
    next_token_id: int | None
    emit_output_row: bool
    finish_kind: str | None = None
    finish_value: int | None = None


class MossDecodeStatePool:
    """Per-request MOSS decode state stored as fixed row-indexed tensors."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        hidden_size: int,
        max_new_tokens: int,
        num_channels: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if int(max_batch_size) <= 0:
            raise ValueError("max_batch_size must be > 0")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be > 0")
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if int(num_channels) <= 0:
            raise ValueError("num_channels must be > 0")

        self.max_batch_size = int(max_batch_size)
        self.hidden_size = int(hidden_size)
        self.max_new_tokens = int(max_new_tokens)
        self.num_channels = int(num_channels)
        self.device = torch.device(device)
        self.dtype = dtype

        pool_size = self.max_batch_size + 1
        self.padding_row = self.max_batch_size
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(self.max_batch_size))

        self.feedback_embeds = torch.zeros(
            pool_size, self.hidden_size, dtype=self.dtype, device=self.device
        )
        self.has_feedback = torch.zeros(pool_size, dtype=torch.bool, device=self.device)
        self.delay_state = torch.zeros(pool_size, 3, dtype=torch.long, device=self.device)
        self.delay_initialized = torch.zeros(
            pool_size, dtype=torch.bool, device=self.device
        )
        self.sampling_steps = torch.zeros(
            pool_size, dtype=torch.long, device=self.device
        )
        self.generated_rows = torch.zeros(
            pool_size,
            self.max_new_tokens,
            self.num_channels,
            dtype=torch.long,
            device=self.device,
        )
        self.generated_lengths = torch.zeros(
            pool_size, dtype=torch.long, device=self.device
        )
        self.stop_pending = torch.zeros(pool_size, dtype=torch.bool, device=self.device)
        self.finish_kind = torch.zeros(pool_size, dtype=torch.long, device=self.device)
        self.finish_value = torch.zeros(pool_size, dtype=torch.long, device=self.device)

        self.reset_row(self.padding_row)

    def acquire_row(self, request_id: str) -> int:
        rid = str(request_id)
        row = self._rid_to_row.get(rid)
        if row is not None:
            return row
        if not self._free_rows:
            raise RuntimeError(
                "MOSS decode state pool exhausted "
                f"(max_batch_size={self.max_batch_size})"
            )
        row = self._free_rows.pop()
        self._rid_to_row[rid] = row
        self.reset_row(row)
        return row

    def row_for(self, request_id: str) -> int | None:
        return self._rid_to_row.get(str(request_id))

    def release_row(self, request_id: str) -> None:
        row = self._rid_to_row.pop(str(request_id), None)
        if row is None:
            return
        self.reset_row(row)
        self._free_rows.append(row)

    def reset_row(self, row: int) -> None:
        row = int(row)
        self.feedback_embeds[row].zero_()
        self.has_feedback[row] = False
        self.delay_state[row].zero_()
        self.delay_initialized[row] = False
        self.sampling_steps[row] = 0
        self.generated_rows[row].zero_()
        self.generated_lengths[row] = 0
        self.stop_pending[row] = False
        self.finish_kind[row] = 0
        self.finish_value[row] = 0

    def feedback_or_zero(self, row: int) -> torch.Tensor:
        row = int(row)
        if bool(self.has_feedback[row].item()):
            return self.feedback_embeds[row]
        return torch.zeros(self.hidden_size, dtype=self.dtype, device=self.device)

    def write_feedback(self, row: int, embed: torch.Tensor) -> None:
        row = int(row)
        self.feedback_embeds[row].copy_(
            embed.to(device=self.device, dtype=self.dtype).view(self.hidden_size)
        )
        self.has_feedback[row] = True

    def clear_feedback(self, row: int) -> None:
        row = int(row)
        self.feedback_embeds[row].zero_()
        self.has_feedback[row] = False

    def append_generated_row(self, row: int, codes: torch.Tensor) -> None:
        row = int(row)
        length = int(self.generated_lengths[row].item())
        if length >= self.max_new_tokens:
            raise RuntimeError(
                "MOSS decode generated history is full "
                f"(max_new_tokens={self.max_new_tokens})"
            )
        self.generated_rows[row, length].copy_(
            codes.to(device=self.device, dtype=torch.long).view(self.num_channels)
        )
        self.generated_lengths[row] = length + 1

    def generated_history(self, row: int) -> torch.Tensor:
        row = int(row)
        length = int(self.generated_lengths[row].item())
        return self.generated_rows[row, :length]

    def mark_stop(self, row: int, *, kind: int, value: int) -> None:
        row = int(row)
        self.stop_pending[row] = True
        self.finish_kind[row] = int(kind)
        self.finish_value[row] = int(value)
