# SPDX-License-Identifier: Apache-2.0
"""Row-indexed decode-state pool for ZONOS2 TTS."""

from __future__ import annotations

from typing import Any

import torch


class Zonos2TTSDecodeStatePool:
    """Stable per-request GPU buffers used by ZONOS2 decode."""

    def __init__(self, model: Any) -> None:
        self.model = model
        weight = model._decode_input_embedding.weight
        self.num_rows = int(weight.shape[0])
        self.hidden_size = int(weight.shape[1])
        self.device = weight.device
        self.dtype = weight.dtype
        self.n_codebooks = int(getattr(model, "n_codebooks", 9) or 9)

        self.feedback_embeds = torch.zeros(
            self.num_rows,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(self.num_rows))
        self._history: torch.Tensor | None = None
        self._history_capacity = 0
        self._history_pos_host = [0 for _ in range(self.num_rows)]
        self._history_len_host = [0 for _ in range(self.num_rows)]

    def acquire_row(self, rid: str) -> int:
        existing = self._rid_to_row.get(rid)
        if existing is not None:
            return existing
        if not self._free_rows:
            raise RuntimeError(
                "ZONOS2 decode-state pool exhausted; raise max_running_requests"
            )
        row_idx = self._free_rows.pop()
        self._rid_to_row[rid] = row_idx
        return row_idx

    def release_row(self, rid: str) -> None:
        row_idx = self._rid_to_row.pop(rid, None)
        if row_idx is None:
            return
        self.reset_row(row_idx)
        self._free_rows.append(row_idx)

    def row_for(self, rid: str) -> int | None:
        return self._rid_to_row.get(rid)

    def reset_row(self, row_idx: int) -> None:
        row_idx = int(row_idx)
        self.feedback_embeds[row_idx].zero_()
        if self._history is not None:
            self._history[row_idx].fill_(-1)
        self._history_pos_host[row_idx] = 0
        self._history_len_host[row_idx] = 0

    def prepare_active_rows(
        self,
        requests: list[Any],
    ) -> tuple[torch.Tensor, list[int]]:
        pool_rows = [self.acquire_row(sched_req.request_id) for sched_req in requests]
        return (
            torch.tensor(pool_rows, dtype=torch.long, device=self.device),
            pool_rows,
        )

    def ensure_history_capacity(self, capacity: int) -> None:
        capacity = int(capacity)
        if capacity <= self._history_capacity:
            return
        new_history = torch.full(
            (self.num_rows, capacity, self.n_codebooks),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        old_history = self._history
        old_capacity = self._history_capacity
        if old_history is not None and old_capacity > 0:
            for row_idx in range(self.num_rows):
                length = min(self._history_len_host[row_idx], old_capacity, capacity)
                if length <= 0:
                    continue
                recent = self._recent_rows(
                    row_idx,
                    length,
                    device=self.device,
                ).transpose(0, 1)
                new_history[row_idx, :length] = recent
                self._history_pos_host[row_idx] = length % capacity
                self._history_len_host[row_idx] = length
        self._history = new_history
        self._history_capacity = capacity

    def update_history(
        self,
        row_t: torch.Tensor,
        rows: torch.Tensor,
        *,
        row_indices: list[int] | None = None,
    ) -> None:
        if rows.numel() == 0:
            return
        if rows.ndim != 2:
            raise RuntimeError(
                f"ZONOS2 history rows must be [B, C], got {tuple(rows.shape)}"
            )
        if self._history is None or self._history_capacity <= 0:
            self.ensure_history_capacity(1)
        assert self._history is not None

        if row_indices is None:
            row_indices = [int(row) for row in row_t.detach().cpu().tolist()]
        if len(row_indices) != int(rows.shape[0]):
            raise RuntimeError(
                "ZONOS2 history row index mismatch: "
                f"{len(row_indices)} rows for {int(rows.shape[0])} code rows"
            )

        rows = rows[:, : self.n_codebooks].to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        for source_idx, row_idx in enumerate(row_indices):
            row_idx = int(row_idx)
            pos = self._history_pos_host[row_idx]
            self._history[row_idx, pos] = rows[source_idx]
            self._history_pos_host[row_idx] = (pos + 1) % self._history_capacity
            self._history_len_host[row_idx] = min(
                self._history_len_host[row_idx] + 1,
                self._history_capacity,
            )

    def history_length(self, row_idx: int) -> int:
        return int(self._history_len_host[int(row_idx)])

    def recent_history(
        self,
        row_idx: int,
        *,
        window: int,
        n_codebooks: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self._history is None:
            return None
        length = min(int(window), self.history_length(row_idx))
        if length <= 0:
            return None
        return self._recent_rows(int(row_idx), length, device=device)[:n_codebooks]

    def reset_for_refill(self, rid: str, output_rows: list[torch.Tensor] | Any) -> bool:
        row_idx = self.row_for(rid)
        if row_idx is None:
            return False
        self.reset_row(row_idx)
        if not output_rows:
            return True
        rows = torch.stack(list(output_rows), dim=0)
        self.ensure_history_capacity(max(int(rows.shape[0]), self._history_capacity, 1))
        row_t = torch.full(
            (int(rows.shape[0]),),
            int(row_idx),
            dtype=torch.long,
            device=self.device,
        )
        self.update_history(
            row_t,
            rows[:, : self.n_codebooks],
            row_indices=[row_idx] * int(rows.shape[0]),
        )
        return True

    def _recent_rows(
        self,
        row_idx: int,
        length: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        assert self._history is not None
        capacity = self._history_capacity
        pos = self._history_pos_host[row_idx]
        start = (pos - length) % capacity
        if start + length <= capacity:
            recent = self._history[row_idx, start : start + length]
        else:
            end_len = capacity - start
            recent = torch.cat(
                [
                    self._history[row_idx, start:],
                    self._history[row_idx, : length - end_len],
                ],
                dim=0,
            )
        return recent.to(device=device, dtype=torch.long, non_blocking=True).transpose(
            0, 1
        )
