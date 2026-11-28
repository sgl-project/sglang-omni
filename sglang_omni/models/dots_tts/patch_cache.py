# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.platforms.device_graph import ReplayableGraph

import_dots_tts()

from dots_tts.modules.backbone.dit_inference import (
    CachedDiTRunner,
    DiTInferenceContext,
    DiTKvCache,
    NextPatchStep,
)

logger = logging.getLogger(__name__)

RESIDENT_CACHE_BYTES = 2 << 30


class ResidentCacheBudget:

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = int(limit_bytes)
        self.entries: OrderedDict[
            tuple[PersistentCacheDiTRunner, tuple[int, int]], int
        ] = OrderedDict()

    @property
    def resident_bytes(self) -> int:
        return sum(self.entries.values())

    def admit(
        self, runner: PersistentCacheDiTRunner, shape_key: tuple[int, int], nbytes: int
    ) -> bool:
        if nbytes > self.limit_bytes:
            return False
        else:
            pass
        while self.entries and self.resident_bytes + nbytes > self.limit_bytes:
            owner, evicted = next(iter(self.entries))
            owner.evict(evicted)
        self.entries[(runner, shape_key)] = int(nbytes)
        return True

    def touch(
        self, runner: PersistentCacheDiTRunner, shape_key: tuple[int, int]
    ) -> None:
        self.entries.move_to_end((runner, shape_key))

    def release(
        self, runner: PersistentCacheDiTRunner, shape_key: tuple[int, int]
    ) -> None:
        self.entries.pop((runner, shape_key))


class PatchSolveGraph(NamedTuple):
    graph: ReplayableGraph
    static_inputs: dict[str, torch.Tensor]
    latent: torch.Tensor


def apply_rotary_embedding(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    half = values.shape[-1] // 2
    rotated = values * cos
    rotated[..., :half].sub_(values[..., half:] * sin[..., :half])
    rotated[..., half:].add_(values[..., :half] * sin[..., half:])
    return rotated.to(dtype=values.dtype)


class TailBufferNextPatchStep(NextPatchStep):

    def _run_cached_blocks(  # noqa: leading-underscore  # NextPatchStep hook
        self,
        x: torch.Tensor,
        *,
        all_mods: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        block_mask: None,
        sdpa_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        capacity_tokens = int(cache_k.size(-2)) - 2 * self.unit_len
        if rotary_cos.dim() == 2:
            cos, sin = rotary_cos[None, None], rotary_sin[None, None]
        elif rotary_cos.dim() == 3:
            cos, sin = rotary_cos[:, None], rotary_sin[:, None]
        else:
            cos, sin = rotary_cos, rotary_sin

        def attention(
            layer_index: int, block: nn.Module, attention_input: torch.Tensor
        ) -> torch.Tensor:
            attention_module = block.attn
            batch, tokens, _hidden = attention_input.shape
            qkv = attention_module.qkv_proj(attention_input).view(
                batch, tokens, 3, self.num_heads, self.head_dim
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q = apply_rotary_embedding(attention_module.q_norm(q), cos, sin)
            k = apply_rotary_embedding(attention_module.k_norm(k), cos, sin)
            keys, values = cache_k[layer_index], cache_v[layer_index]
            keys[:, :, capacity_tokens:].copy_(k)
            values[:, :, capacity_tokens:].copy_(v)
            output = F.scaled_dot_product_attention(
                q, keys, values, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            )
            output = output.permute(0, 2, 1, 3).reshape(batch, tokens, -1)
            return attention_module.o_dropout(attention_module.o_proj(output))

        x, final_mod = self.dit.run_modulated_blocks(
            x=x, all_mods=all_mods, attention=attention
        )
        velocity = self.dit.apply_final_layer(
            x[:, self.current_latent_slice], final_mod
        )
        prev_unit_rows = slice(capacity_tokens, capacity_tokens + self.unit_len)
        return (
            velocity,
            cache_k[:, :, :, prev_unit_rows],
            cache_v[:, :, :, prev_unit_rows],
        )


class PersistentCacheDiTRunner(CachedDiTRunner):

    def __init__(
        self,
        *,
        context: DiTInferenceContext,
        capacity_patches: int,
        capacity_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        attn_backend: Literal["sdpa", "flex"],
        compile_step: bool,
        branch_multiplier: int,
        budget: ResidentCacheBudget,
    ) -> None:
        if attn_backend != "sdpa":
            raise ValueError(
                f"dots.tts patch graphs need the sdpa KV-cache attention backend, got {attn_backend!r}"
            )
        else:
            pass
        super().__init__(
            context=context,
            capacity_patches=capacity_patches,
            capacity_tokens=capacity_tokens,
            dtype=dtype,
            device=device,
            attn_backend=attn_backend,
            compile_step=compile_step,
            branch_multiplier=branch_multiplier,
        )
        self.step = TailBufferNextPatchStep(context, attn_backend=attn_backend).eval()
        self.persistent_caches: dict[tuple[int, int], DiTKvCache] = {}
        self.patch_graphs: dict[tuple[int, int], PatchSolveGraph] = {}
        self.budget = budget

    def cache_shape(self, *, batch_size: int, nfe: int) -> tuple[int, ...]:
        return (
            int(nfe),
            self.num_layers,
            self.branch_multiplier * int(batch_size),
            self.num_heads,
            self.capacity_tokens + 2 * self.unit_len,
            self.head_dim,
        )

    def cache_bytes(self, *, batch_size: int, nfe: int) -> int:
        elements = math.prod(self.cache_shape(batch_size=batch_size, nfe=nfe))
        return 2 * elements * torch.empty((), dtype=self.dtype).element_size()

    def new_cache(self, *, batch_size: int, nfe: int) -> DiTKvCache:
        cache_k = torch.zeros(
            self.cache_shape(batch_size=batch_size, nfe=nfe),
            device=self.device,
            dtype=self.dtype,
        )
        return DiTKvCache(
            capacity_patches=self.capacity_patches,
            capacity_tokens=self.capacity_tokens,
            nfe=int(nfe),
            cache_k=cache_k,
            cache_v=torch.zeros_like(cache_k),
        )

    def allocate_cache(self, *, batch_size: int, nfe: int) -> DiTKvCache:
        shape_key = (int(batch_size), int(nfe))
        cache = self.persistent_caches.get(shape_key)
        if cache is None:
            nbytes = self.cache_bytes(batch_size=batch_size, nfe=nfe)
            if not self.budget.admit(self, shape_key, nbytes):
                return self.new_cache(batch_size=batch_size, nfe=nfe)
            else:
                pass
            try:
                cache = self.new_cache(batch_size=batch_size, nfe=nfe)
            except Exception:
                self.budget.release(self, shape_key)
                raise
            self.persistent_caches[shape_key] = cache
        else:
            self.budget.touch(self, shape_key)
            cache.cache_k.zero_()
            cache.cache_v.zero_()
        return DiTKvCache(
            capacity_patches=cache.capacity_patches,
            capacity_tokens=cache.capacity_tokens,
            nfe=cache.nfe,
            cache_k=cache.cache_k,
            cache_v=cache.cache_v,
        )

    def evict(self, shape_key: tuple[int, int]) -> None:
        self.persistent_caches.pop(shape_key)
        captured = self.patch_graphs.pop(shape_key, None)
        if captured is not None:
            captured.graph.reset()
        else:
            pass
        self.budget.release(self, shape_key)
        logger.info(
            f"dots.tts DiT patch cache evicted: capacity_patches={self.capacity_patches} "
            f"batch={shape_key[0]} ode_steps={shape_key[1]}"
        )


__all__ = [
    "RESIDENT_CACHE_BYTES",
    "PatchSolveGraph",
    "PersistentCacheDiTRunner",
    "ResidentCacheBudget",
    "TailBufferNextPatchStep",
    "apply_rotary_embedding",
]
