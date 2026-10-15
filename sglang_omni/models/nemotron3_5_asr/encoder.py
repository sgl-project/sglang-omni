# SPDX-License-Identifier: Apache-2.0
"""Encode streaming windows at any request progress in one batch."""

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import (
    EncoderStateSlot,
    NemotronEncoderStatePool,
    PooledAttentionCache,
    PooledPaddingCache,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)


def encode_pooled_streaming_batch(
    model: Nemotron3_5AsrForRNNT,
    input_features: Sequence[torch.Tensor],
    prompt_ids: torch.Tensor,
    *,
    encoder_slots: Sequence[EncoderStateSlot],
    num_lookahead_tokens: int,
) -> torch.Tensor:
    """Encode mixed first/subsequent windows while updating persistent slots."""
    assert not model.training
    pool = encoder_slots[0].pool
    groups: dict[tuple[bool, int], list[int]] = defaultdict(list)
    for index, (features, slot) in enumerate(
        zip(input_features, encoder_slots, strict=True)
    ):
        groups[(slot.seen_frames == 0, features.shape[1])].append(index)
    encoded_by_row: dict[int, torch.Tensor] = {}
    for (is_first_chunk, _), indices in groups.items():
        features = torch.cat([input_features[index] for index in indices])
        slot_ids = torch.tensor(
            [encoder_slots[index].slot_id for index in indices],
            device=pool.layout.device,
        )
        encoded_frames = encode_pooled_windows(
            model,
            features,
            prompt_ids[indices],
            slot_ids,
            pool,
            is_first_chunk=is_first_chunk,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        encoded_by_row.update(zip(indices, encoded_frames.split(1), strict=True))
        for index in indices:
            encoder_slots[index].seen_frames += encoded_frames.shape[1]
    return torch.cat([encoded_by_row[index] for index in range(len(encoder_slots))])


def encode_pooled_windows(
    model: Nemotron3_5AsrForRNNT,
    input_features: torch.Tensor,
    prompt_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    pool: NemotronEncoderStatePool,
    *,
    is_first_chunk: bool,
    num_lookahead_tokens: int,
) -> torch.Tensor:
    """Encode equal-shaped windows and update pool state through tensor slot IDs."""
    padding_cache = PooledPaddingCache(
        pool=pool, slot_ids=slot_ids, is_first_chunk=is_first_chunk
    )
    hidden_states = model.encoder.subsampling(
        input_features, attention_mask=None, padding_cache=padding_cache
    )
    hidden_states *= model.encoder.input_scale
    attention_cache = PooledAttentionCache(pool=pool, slot_ids=slot_ids)
    chunk_frames = hidden_states.shape[1]
    attention_mask = attention_cache.create_mask(chunk_frames, num_lookahead_tokens)
    position_embeddings = model.encoder.encode_positions(
        hidden_states, cached_frames=attention_mask.shape[-1] - hidden_states.shape[1]
    )
    all_masked_rows = torch.all(~attention_mask, dim=-1)
    for encoder_layer in model.encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            all_masked_rows=all_masked_rows,
            position_embeddings=position_embeddings,
            past_key_values=attention_cache,
            padding_cache=padding_cache,
            use_cache=True,
        )

    one_hot = F.one_hot(
        prompt_ids.to(hidden_states.device), num_classes=model.config.num_prompts
    ).to(hidden_states.dtype)
    one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
    fused = model.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
    encoded_frames = model.encoder_projector(fused)
    pool.seen_frames.index_add_(0, slot_ids, torch.full_like(slot_ids, chunk_frames))
    return encoded_frames
