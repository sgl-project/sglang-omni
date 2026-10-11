# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Nikita Torgashov
# Construction and graph-safe cache operations adapted from herimor/voxtream
# commit 8ec2d62159dae4716ae7058827244a962d40603c. See LICENSE-MIT.voxtream.
"""Instance-local cache operations for CUDA graph capture."""

from types import MethodType
from typing import Protocol

import torch
from torchtune.modules.kv_cache import KVCache


class CacheUpdate(Protocol):
    def __call__(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class CacheReset(Protocol):
    def __call__(self) -> None: ...


def bind_graph_safe_caches(model: torch.nn.Module) -> None:
    """Adapt only this model's KV caches; other torchtune users stay untouched."""
    for cache in model.modules():
        if not isinstance(cache, KVCache) or getattr(cache, "omni_graph_safe", False):
            continue
        else:
            pass
        original_update, original_reset = cache.update, cache.reset

        def update(
            self: KVCache,
            keys: torch.Tensor,
            values: torch.Tensor,
            original: CacheUpdate = original_update,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not self.k_cache.is_cuda or not torch.cuda.is_current_stream_capturing():
                return original(keys, values)
            else:
                pass
            batch, _, length, _ = keys.shape
            if batch > self.k_cache.shape[0]:
                raise ValueError("KV cache batch capacity exceeded")
            else:
                pass
            self.k_cache[:, :, self.cache_pos[:length]] = keys
            self.v_cache[:, :, self.cache_pos[:length]] = values
            self.cache_pos += length
            return self.k_cache, self.v_cache

        def reset(self: KVCache, original: CacheReset = original_reset) -> None:
            if not self.k_cache.is_cuda or not torch.cuda.is_current_stream_capturing():
                return original()
            else:
                pass
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.cache_pos.copy_(self.cache_pos - self.cache_pos[0])

        cache.update = MethodType(update, cache)
        cache.reset = MethodType(reset, cache)
        cache.omni_graph_safe = True
