# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Realtime scheduler model runner."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
from sglang_omni.models.moss_tts_realtime.payload_types import TEXT_PAD_TOKEN


class MossTTSRealtimeModelRunner(MossTTSLocalModelRunner):
    """Advance streamed text tokens alongside generated audio frames."""

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    def _compose_frame_rows(
        self,
        *,
        codes: torch.Tensor,
        stop_choice: torch.Tensor,
        requests: list,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        end_id = int(self.model.config.audio_end_token_id)
        next_tokens = []
        for request in requests:
            data = request.data
            text_index = int(data.prefill_text_tokens) + int(data.generation_steps)
            if text_index < len(data.text_token_ids):
                next_tokens.append(int(data.text_token_ids[text_index]))
            else:
                next_tokens.append(TEXT_PAD_TOKEN)
        next_text = torch.tensor(next_tokens, dtype=torch.long, device=device)
        next_text = torch.where(
            stop_choice.to(dtype=torch.bool),
            torch.full_like(next_text, end_id),
            next_text,
        )
        rows = torch.empty(
            (len(requests), int(self.model.config.n_vq) + 1),
            dtype=torch.long,
            device=device,
        )
        rows[:, 0] = next_text
        rows[:, 1:] = codes
        return rows, next_text, end_id


__all__ = ["MossTTSRealtimeModelRunner"]
