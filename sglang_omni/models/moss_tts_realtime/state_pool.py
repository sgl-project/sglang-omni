# SPDX-License-Identifier: Apache-2.0
"""Decode state for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.moss_tts_local.state_pool import (
    MossTTSLocalDecodeJournal,
    MossTTSLocalDecodeStatePool,
)


class MossTTSRealtimeDecodeStatePool(MossTTSLocalDecodeStatePool):
    """MOSS decode state with an exact fixed-window repetition history."""

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self.repetition_window = int(getattr(model.config, "repetition_window", 50))
        self.audio_recent_tokens = torch.full(
            (self.num_rows, self.n_vq, self.repetition_window),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.audio_recent_count = torch.zeros(
            self.num_rows,
            dtype=torch.long,
            device=self.device,
        )

    def reset_row(self, row_idx: int) -> None:
        super().reset_row(row_idx)
        if hasattr(self, "audio_recent_tokens"):
            self.audio_recent_tokens[row_idx].fill_(-1)
            self.audio_recent_count[row_idx] = 0

    def update_audio_history(self, row_t: torch.Tensor, rows: torch.Tensor) -> None:
        if row_t.numel() == 0:
            return
        if rows.ndim != 2 or int(rows.shape[1]) != self.n_vq + 1:
            raise RuntimeError(
                "MOSS-TTS-Realtime audio rows must have shape "
                f"[B, {self.n_vq + 1}], got {tuple(rows.shape)}"
            )
        row_t = row_t.to(device=self.device, dtype=torch.long)
        codes = rows[:, 1:].to(device=self.device, dtype=torch.long)
        if int(row_t.numel()) != int(codes.shape[0]):
            raise RuntimeError(
                "MOSS-TTS-Realtime audio history row index mismatch: "
                f"{int(row_t.numel())} rows for {int(codes.shape[0])} code rows"
            )
        positions = self.audio_recent_count[row_t] % self.repetition_window
        channel_t = torch.arange(self.n_vq, device=self.device).view(1, -1)
        self.audio_recent_tokens[
            row_t.view(-1, 1),
            channel_t,
            positions.view(-1, 1),
        ] = codes
        self.audio_recent_count[row_t] += 1
        self._rebuild_presence(row_t)

    def _rebuild_presence(self, row_t: torch.Tensor) -> None:
        unique_rows = torch.unique(row_t)
        self.audio_token_presence[unique_rows] = False
        history = self.audio_recent_tokens[unique_rows]
        valid = (history >= 0) & (history < self.audio_vocab_size)
        row_index = unique_rows.view(-1, 1, 1).expand_as(history)
        channels = (
            torch.arange(self.n_vq, device=self.device)
            .view(1, -1, 1)
            .expand_as(history)
        )
        self.audio_token_presence[
            row_index[valid],
            channels[valid],
            history[valid],
        ] = True

    def rebuild_audio_history(self, rid: str, output_rows: list[torch.Tensor]) -> bool:
        row_idx = self.row_for(rid)
        if row_idx is None:
            return False
        self.audio_token_presence[row_idx].zero_()
        self.audio_recent_tokens[row_idx].fill_(-1)
        self.audio_recent_count[row_idx] = 0
        row_t = torch.tensor([row_idx], dtype=torch.long, device=self.device)
        for row in output_rows[-self.repetition_window :]:
            self.update_audio_history(row_t, row.reshape(1, -1))
        return True


__all__ = [
    "MossTTSLocalDecodeJournal",
    "MossTTSRealtimeDecodeStatePool",
]
