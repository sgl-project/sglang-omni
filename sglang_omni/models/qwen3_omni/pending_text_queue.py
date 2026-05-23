# SPDX-License-Identifier: Apache-2.0
"""Device-backed FIFO for Qwen3-Omni talker future text rows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch


def _as_rows(tensor: torch.Tensor) -> torch.Tensor | None:
    try:
        tensor = tensor.detach()
    except AttributeError as exc:
        raise TypeError("pending text rows must be tensors") from exc
    if tensor.dim() == 1:
        if tensor.shape[0] == 0:
            return None
        return tensor.reshape(1, -1)
    if tensor.dim() == 2:
        if tensor.shape[0] == 0:
            return None
        if tensor.shape[1] == 0:
            raise ValueError("pending text rows must have a non-empty hidden dimension")
        return tensor
    raise ValueError("pending text rows must be a 1D row tensor or a 2D row batch")


@dataclass(slots=True)
class PendingTextTensorQueue:
    """FIFO queue backed by one tensor plus a cursor.

    The talker consumes one future text row per decode step. Keeping those rows
    as a single device tensor avoids row-wise D2H/H2D copies and Python deque
    object churn while preserving the small queue API used by the runner.
    """

    rows: torch.Tensor | None = None
    cursor: int = 0

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "PendingTextTensorQueue":
        queue = cls()
        queue.append_rows(tensor)
        return queue

    def __bool__(self) -> bool:
        return len(self) > 0

    def copy(self) -> "PendingTextTensorQueue":
        return type(self)(rows=self.rows, cursor=self.cursor)

    def __len__(self) -> int:
        if self.rows is None:
            return 0
        return max(0, int(self.rows.shape[0]) - self.cursor)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.rows is None:
            return
        for idx in range(self.cursor, int(self.rows.shape[0])):
            yield self.rows[idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if not isinstance(idx, int):
            raise TypeError("PendingTextTensorQueue indices must be integers")
        if idx < 0:
            idx += len(self)
        absolute_idx = self.cursor + idx
        if self.rows is None or idx < 0 or absolute_idx >= int(self.rows.shape[0]):
            raise IndexError(idx)
        return self.rows[absolute_idx]

    def popleft(self) -> torch.Tensor:
        row = self[0]
        self.cursor += 1
        if self.rows is not None and self.cursor >= int(self.rows.shape[0]):
            self.rows = None
            self.cursor = 0
        return row

    def append(self, row: torch.Tensor) -> None:
        self.append_rows(row)

    def append_rows(self, rows: torch.Tensor) -> None:
        rows = _as_rows(rows)
        if rows is None:
            return
        if self.rows is None or len(self) == 0:
            self.rows = rows
            self.cursor = 0
            return

        remaining = self.rows[self.cursor :]
        rows = rows.to(device=remaining.device, dtype=remaining.dtype)
        self.rows = torch.cat([remaining, rows], dim=0)
        self.cursor = 0


def coerce_pending_text_queue(value: object) -> PendingTextTensorQueue:
    if value is None:
        return PendingTextTensorQueue()
    if isinstance(value, PendingTextTensorQueue):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return PendingTextTensorQueue.from_tensor(value)
    if isinstance(value, Iterable):
        queue = PendingTextTensorQueue()
        for row in value:
            queue.append(row)
        return queue
    raise TypeError(
        "pending text queue must be None, a tensor, a PendingTextTensorQueue, "
        "or an iterable of tensors"
    )
