# SPDX-License-Identifier: Apache-2.0
"""Feature merging logic for Qwen3-ASR."""

from __future__ import annotations

import torch


def merge_encoder_outputs(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    audio_embeds: torch.Tensor | None,
    audio_token_id: int,
) -> torch.Tensor:
    """Merge encoder embeddings into the text embedding stream."""
    if audio_embeds is not None:
        # Create mask for audio tokens
        audio_mask = input_ids == audio_token_id

        # Verify counts
        audio_token_count = int(audio_mask.sum().item())
        if audio_token_count != int(audio_embeds.shape[0]):
            raise ValueError(
                f"Audio placeholder count mismatch: tokens={audio_token_count} embeds={audio_embeds.shape[0]}"
            )

        # Merge
        audio_embeds = audio_embeds.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_mask.unsqueeze(-1).expand_as(inputs_embeds), audio_embeds
        )

    return inputs_embeds
