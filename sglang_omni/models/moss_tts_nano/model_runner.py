# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Nano model runner for OmniScheduler."""

from __future__ import annotations

import torch

from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
from sglang_omni.models.moss_tts_local.radix_hash import gpu_radix_row_hash


class MossTTSNanoModelRunner(MossTTSLocalModelRunner):
    """Reuse the local-frame scheduler with Nano-safe radix token ids."""

    def _row_radix_token_ids(
        self,
        rows: torch.Tensor,
        next_text: torch.Tensor,
        end_id: int,
    ) -> torch.Tensor:
        config = self.model.config
        hash_offset = (
            max(
                int(config.pad_token_id),
                int(config.im_start_token_id),
                int(config.im_end_token_id),
                int(config.audio_start_token_id),
                int(config.audio_end_token_id),
                int(config.audio_user_slot_token_id),
                int(config.audio_assistant_slot_token_id),
            )
            + 1
        )
        return gpu_radix_row_hash(
            rows,
            next_text,
            end_id,
            hash_space=int(config.vocab_size),
            hash_offset=hash_offset,
        )


EntryClass = MossTTSNanoModelRunner
