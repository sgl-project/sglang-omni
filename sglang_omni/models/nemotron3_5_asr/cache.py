# SPDX-License-Identifier: Apache-2.0
"""Batch encoder operations without replacing request-owned caches."""

from collections.abc import Sequence

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
)


class NemotronBatchAttentionCache:
    def __init__(self, caches: Sequence[DynamicCache]) -> None:
        self.caches = list(caches)

    def update(
        self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # note (Li Gang): Request caches must not retain another row's storage.
        updated = [
            cache.update(
                keys[index : index + 1].clone(),
                values[index : index + 1].clone(),
                layer_idx,
            )
            for index, cache in enumerate(self.caches)
        ]
        max_length = max(key.shape[-2] for key, _ in updated)
        # note (Li Gang): Left padding aligns queries and relative positions.
        return (
            torch.cat(
                [
                    F.pad(key, (0, 0, max_length - key.shape[-2], 0))
                    for key, _ in updated
                ]
            ),
            torch.cat(
                [
                    F.pad(value, (0, 0, max_length - value.shape[-2], 0))
                    for _, value in updated
                ]
            ),
        )

    def create_mask(
        self,
        query_length: int,
        device: torch.device,
        left_context: int,
        right_context: int,
    ) -> torch.Tensor:
        sizes = [cache.get_mask_sizes(query_length, 0) for cache in self.caches]
        max_length = max(length for length, _ in sizes)
        lengths = torch.tensor([length for length, _ in sizes], device=device)
        offsets = torch.tensor([offset for _, offset in sizes], device=device)
        seen = torch.tensor(
            [cache.get_seq_length() for cache in self.caches], device=device
        )
        columns = torch.arange(max_length, device=device)
        left_padding = max_length - lengths
        key_positions = columns[None, :] - left_padding[:, None] + offsets[:, None]
        query_positions = seen[:, None] + torch.arange(query_length, device=device)
        chunk_size = right_context + 1
        left_chunks = left_context // chunk_size if left_context >= 0 else 10_000
        chunk_diff = (
            torch.div(query_positions, chunk_size, rounding_mode="trunc")[:, :, None]
            - torch.div(key_positions, chunk_size, rounding_mode="trunc")[:, None, :]
        )
        visible = (columns[None, :] >= left_padding[:, None])[:, None, :]
        return (visible & (chunk_diff >= 0) & (chunk_diff <= left_chunks))[:, None]


class NemotronBatchPaddingCache:
    def __init__(
        self, caches: Sequence[NemotronAsrStreamingEncoderCausalConvPaddingCache]
    ) -> None:
        self.caches = list(caches)

    def update(
        self,
        hidden_states: torch.Tensor,
        cache_key: str,
        conv_module: torch.nn.Conv1d | torch.nn.Conv2d,
    ) -> torch.Tensor:
        return torch.cat(
            [
                cache.update(hidden_states[index : index + 1], cache_key, conv_module)
                for index, cache in enumerate(self.caches)
            ],
            dim=0,
        )
