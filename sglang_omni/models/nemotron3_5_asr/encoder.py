# SPDX-License-Identifier: Apache-2.0
"""Encode streaming windows at any request progress in one batch."""

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from sglang_omni.models.nemotron3_5_asr.cache import (
    NemotronBatchAttentionCache,
    NemotronBatchPaddingCache,
)
from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    Nemotron3_5AsrForRNNT,
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
)


def encode_streaming_batch(
    model: Nemotron3_5AsrForRNNT,
    input_features: Sequence[torch.Tensor],
    prompt_ids: torch.Tensor,
    *,
    attention_caches: Sequence[DynamicCache],
    padding_caches: Sequence[NemotronAsrStreamingEncoderCausalConvPaddingCache],
    num_lookahead_tokens: int,
) -> torch.Tensor:
    assert not model.training
    encoder = model.encoder
    groups: dict[int, list[int]] = defaultdict(list)
    for index, features in enumerate(input_features):
        groups[features.shape[1]].append(index)

    # note (Li Gang): Both mel lengths produce equally long encoder windows.
    subsampled: dict[int, torch.Tensor] = {}
    for indices in groups.values():
        features = torch.cat([input_features[index] for index in indices], dim=0)
        padding_cache = NemotronBatchPaddingCache(
            [padding_caches[index] for index in indices]
        )
        hidden_states = encoder.subsampling(
            features, attention_mask=None, padding_cache=padding_cache
        )
        subsampled.update(zip(indices, hidden_states.split(1), strict=True))
    hidden_states = torch.cat(
        [subsampled[index] for index in range(len(input_features))], dim=0
    )
    hidden_states *= encoder.input_scale

    attention_cache = NemotronBatchAttentionCache(attention_caches)
    padding_cache = NemotronBatchPaddingCache(padding_caches)
    seq_length = hidden_states.shape[1]
    attention_mask = attention_cache.create_mask(
        seq_length,
        hidden_states.device,
        model.config.encoder_config.sliding_window - 1,
        num_lookahead_tokens,
    )
    position_embeddings = encoder.encode_positions(
        hidden_states, cached_frames=attention_mask.shape[-1] - seq_length
    )
    all_masked_rows = torch.all(~attention_mask, dim=-1)
    for encoder_layer in encoder.layers:
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
    return model.encoder_projector(fused)
