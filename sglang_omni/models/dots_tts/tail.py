# SPDX-License-Identifier: Apache-2.0
"""Batched MeanFlow acoustic tail for the dots.tts SGLang engine.

SGLang owns the Qwen2 AR backbone; every step of that backbone additionally
needs one MeanFlow DiT sample plus one patch-encoder step per request. Both are
small transformers with their own KV state, so running them one request at a
time would make the engine launch-bound. This module keeps that state in
slot-indexed pools sized once for ``num_slots x patch_capacity`` and runs both
networks over the whole running batch in a single call per decode step.

The flow sequence the DiT reads interleaves one projected backbone hidden row
with ``latent_patch_size`` projected latent rows per audio patch. Everything
older than the last complete unit lives in the DiT KV cache, so a slot only
keeps a ``unit_len + hidden_patch_size`` row window verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang_omni.models.dots_tts.components.backbone.encoder import (
    SemanticEncoderDecodeStep,
)
from sglang_omni.models.dots_tts.components.backbone.inference_utils import (
    fuse_qkv_projection,
    project_attention,
)

logger = logging.getLogger(__name__)

# note (chenyang): the tail attends over ~10 query rows per layer, where cuDNN's
# SDPA plan lookup costs ~700us of host time for ~10us of device work and makes
# the whole engine launch-bound. The efficient/math kernels have no such
# per-call planning cost.
_TAIL_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@dataclass(frozen=True)
class DotsTtsTailSpec:
    """Static shapes the tail pools are sized from."""

    nfe: int
    patch_capacity: int
    num_slots: int
    hidden_patch_size: int
    latent_patch_size: int
    latent_dim: int
    fm_hidden_size: int

    def __post_init__(self) -> None:
        for name in ("nfe", "patch_capacity", "num_slots"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"dots.tts tail {name} must be positive")

    @property
    def unit_len(self) -> int:
        return self.hidden_patch_size + self.latent_patch_size

    @property
    def window_len(self) -> int:
        return self.unit_len + self.hidden_patch_size

    @property
    def dit_cache_tokens(self) -> int:
        # note (chenyang): one hidden row plus latent_patch_size latent rows per
        # audio patch; the cache never has to hold the trailing window.
        return self.patch_capacity * self.unit_len


# note (chenyang): calibrated on one H100 at 32 slots (pools 24.95 GB, startup
# 38.99 GB, steady peak 63.93 GB). Everything the engine touches is either fixed
# or linear in the slot count, so the planner below inverts this to pick a slot
# count that fills a memory target instead of letting memory fall out of an
# arbitrary max_running_requests.
TAIL_FIXED_OVERHEAD_BYTES = int(6.74 * 2**30)
TAIL_ACTIVATION_BYTES_PER_SLOT = int(1.787 * 2**30) - int(0.780 * 2**30)


def tail_pool_bytes_per_slot(
    *,
    nfe: int,
    patch_capacity: int,
    unit_len: int,
    dit_layers: int,
    dit_heads: int,
    dit_head_dim: int,
    encoder_layers: int,
    encoder_heads: int,
    encoder_head_dim: int,
    encoder_block: int,
    element_size: int = 2,
) -> int:
    """Bytes one slot costs in the tail's static pools plus gather scratch."""
    cache_tokens = patch_capacity * unit_len
    query = 2 * unit_len
    encoder_tokens = patch_capacity * encoder_block
    dit_pool = nfe * dit_layers * dit_heads * cache_tokens * dit_head_dim
    dit_scratch = dit_layers * dit_heads * (cache_tokens + query) * dit_head_dim
    enc_pool = encoder_layers * encoder_heads * encoder_tokens * encoder_head_dim
    enc_scratch = (
        encoder_layers
        * encoder_heads
        * (encoder_tokens + encoder_block)
        * encoder_head_dim
    )
    return 2 * element_size * (dit_pool + dit_scratch + enc_pool + enc_scratch)


def plan_tail_slots(
    *,
    total_gpu_bytes: int,
    gpu_memory_utilization: float,
    pool_bytes_per_slot: int,
    kv_bytes_per_slot: int,
    max_slots: int | None = None,
) -> int:
    """Slot count whose static pools plus activations fill the memory target.

    Mirrors how SGLang sizes its KV pool: pick the budget first, then derive the
    batch capacity, so the engine reaches a stable footprint at startup instead
    of drifting with load.
    """
    budget = int(total_gpu_bytes * float(gpu_memory_utilization))
    per_slot = (
        int(pool_bytes_per_slot)
        + int(kv_bytes_per_slot)
        + TAIL_ACTIVATION_BYTES_PER_SLOT
    )
    slots = (budget - TAIL_FIXED_OVERHEAD_BYTES) // per_slot
    if slots < 1:
        raise ValueError(
            "dots.tts cannot fit a single tail slot in "
            f"{budget / 2**30:.1f} GiB; lower max_audio_patches or raise "
            "gpu_memory_utilization"
        )
    return int(slots if max_slots is None else min(slots, int(max_slots)))


def batched_causal_update_mask(
    *,
    capacity_tokens: int,
    valid_persistent: torch.Tensor,
    prev_len: int,
    current_len: int,
) -> torch.Tensor:
    """Attention mask for a fresh query block appended after a per-row cache.

    Returns ``[rows, 1, prev_len + current_len, capacity_tokens + q_len]``. Keys
    below a row's ``valid_persistent`` are always visible; inside the fresh block
    the first ``prev_len`` queries stay causal and the rest see the whole block.
    """
    q_len = int(prev_len) + int(current_len)
    device = valid_persistent.device
    total_kv = int(capacity_tokens) + q_len
    kv_idx = torch.arange(total_kv, device=device)
    q_idx = torch.arange(q_len, device=device).reshape(q_len, 1)
    tail_idx = kv_idx.reshape(1, total_kv) - int(capacity_tokens)
    prev_query = q_idx < int(prev_len)
    prev_causal = (tail_idx >= 0) & (tail_idx < int(prev_len)) & (tail_idx <= q_idx)
    tail = torch.where(prev_query, prev_causal, tail_idx >= 0)
    past = kv_idx.reshape(1, 1, total_kv) < valid_persistent.reshape(-1, 1, 1)
    return (past | tail.unsqueeze(0)).unsqueeze(1)


def rotary_cos_sin(
    rotary: Any, *, start_pos: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary cos/sin for ``[rows, seq_len]`` positions starting per row."""
    offsets = torch.arange(int(seq_len), device=start_pos.device, dtype=torch.float32)
    embedding = rotary(start_pos.reshape(-1, 1).to(torch.float32) + offsets)
    return embedding.cos(), embedding.sin()


class DotsTtsAcousticTail:
    """Slot-pooled MeanFlow DiT + patch encoder shared by the whole batch."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        coordinate_proj: nn.Module,
        patch_encoder: nn.Module,
        spec: DotsTtsTailSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.dit = dit
        self._coordinate_proj = coordinate_proj

        self._encoder = patch_encoder
        for layer in patch_encoder.encoder.layers:
            fuse_qkv_projection(layer.attn)
        self._encoder_step = SemanticEncoderDecodeStep(patch_encoder).eval()

        dit_attn = dit.blocks[0].attn
        self._dit_layers = len(dit.blocks)
        self._dit_rotary = dit_attn.rotary
        self._dit_heads = int(dit_attn.num_heads)
        self._dit_head_dim = int(dit_attn.head_dim)

        encoder_attn = patch_encoder.encoder.layers[0].attn
        self._encoder_layers = len(patch_encoder.encoder.layers)
        self._encoder_rotary = encoder_attn.rotary if encoder_attn.rotary_bias else None
        self._encoder_block = int(patch_encoder.out_ds_rate)
        self._encoder_heads = int(encoder_attn.num_heads)
        self._encoder_head_dim = int(encoder_attn.head_dim)

        self._allocate_pools(int(dit.fused_adaln[-1].out_features))
        self._times = torch.linspace(0.0, 1.0, spec.nfe + 1, device=device, dtype=dtype)
        self._dit_layer_index = torch.arange(self._dit_layers, device=device)
        self._encoder_layer_index = torch.arange(self._encoder_layers, device=device)

        self._fm_seq_len = [0] * spec.num_slots
        self._encoder_seq_len = [0] * spec.num_slots
        self._generators: list[torch.Generator | None] = [None] * spec.num_slots
        self._free_slots = list(reversed(range(spec.num_slots)))

    def _allocate_pools(self, mods_width: int) -> None:
        """Allocate every slot-indexed buffer the tail keeps for a whole run."""
        spec = self.spec
        zeros = partial(torch.zeros, device=self.device, dtype=self.dtype)

        # note (chenyang): the MeanFlow modulations differ per ODE step, so the
        # cached keys/values of the same flow prefix do too and the DiT pool
        # needs one cache per step.
        self._dit_k = zeros(
            spec.nfe,
            self._dit_layers,
            spec.num_slots,
            self._dit_heads,
            spec.dit_cache_tokens,
            self._dit_head_dim,
        )
        self._dit_v = torch.zeros_like(self._dit_k)
        self._encoder_k = zeros(
            self._encoder_layers,
            spec.num_slots,
            self._encoder_heads,
            spec.patch_capacity * self._encoder_block,
            self._encoder_head_dim,
        )
        self._encoder_v = torch.zeros_like(self._encoder_k)
        self._encoder_conv_tail = zeros(
            spec.num_slots,
            int(self._encoder.ds_proj.in_channels),
            int(self._encoder.ds_proj.left_padding),
        )
        self._window = zeros(spec.num_slots, spec.window_len, spec.fm_hidden_size)
        self._all_mods = zeros(spec.nfe, spec.num_slots, mods_width)

        # note (chenyang): every decode step needs the cached K/V of the active
        # rows plus room for this step's fresh K/V. Both are gathered into these
        # fixed buffers instead of being allocated per step, so the engine's
        # memory footprint is decided entirely at startup and stays flat no
        # matter how long a generation runs or how full the batch is.
        dit_query = 2 * spec.unit_len
        self._dit_scratch_k = zeros(
            self._dit_layers,
            spec.num_slots,
            self._dit_heads,
            spec.dit_cache_tokens + dit_query,
            self._dit_head_dim,
        )
        self._dit_scratch_v = torch.zeros_like(self._dit_scratch_k)
        self._dit_mask = torch.zeros(
            (spec.num_slots, 1, dit_query, spec.dit_cache_tokens + dit_query),
            device=self.device,
            dtype=torch.bool,
        )
        encoder_tokens = spec.patch_capacity * self._encoder_block
        self._encoder_scratch_k = zeros(
            self._encoder_layers,
            spec.num_slots,
            self._encoder_heads,
            encoder_tokens + self._encoder_block,
            self._encoder_head_dim,
        )
        self._encoder_scratch_v = torch.zeros_like(self._encoder_scratch_k)
        self._encoder_mask = torch.zeros(
            (
                spec.num_slots,
                1,
                self._encoder_block,
                encoder_tokens + self._encoder_block,
            ),
            device=self.device,
            dtype=torch.bool,
        )

        logger.info(
            "dots.tts acoustic tail pools: slots=%s nfe=%s patch_capacity=%s "
            "dit_cache_tokens=%s pool_bytes=%s",
            spec.num_slots,
            spec.nfe,
            spec.patch_capacity,
            spec.dit_cache_tokens,
            2
            * (
                self._dit_k.numel()
                + self._encoder_k.numel()
                + self._dit_scratch_k.numel()
                + self._encoder_scratch_k.numel()
            )
            * self._dit_k.element_size(),
        )

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def acquire_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError(
                "dots.tts acoustic tail ran out of slots; max_running_requests "
                f"must not exceed {self.spec.num_slots}"
            )
        slot = self._free_slots.pop()
        self._fm_seq_len[slot] = 0
        self._encoder_seq_len[slot] = 0
        self._generators[slot] = None
        self._encoder_conv_tail[slot].zero_()
        self._window[slot].zero_()
        return slot

    def release_slot(self, slot: int) -> None:
        slot = int(slot)
        if slot in self._free_slots:
            return
        self._fm_seq_len[slot] = 0
        self._encoder_seq_len[slot] = 0
        self._generators[slot] = None
        self._free_slots.append(slot)

    def set_slot_seed(self, slot: int, seed: int) -> None:
        """Give one slot its own noise stream so batching cannot perturb it."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        self._generators[int(slot)] = generator

    def fm_seq_len(self, slot: int) -> int:
        return self._fm_seq_len[int(slot)]

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt_patches(
        self, slot: int, prompt_latents: torch.Tensor
    ) -> torch.Tensor:
        """Backbone embeddings for one request's prompt audio spans.

        ``prompt_latents`` is ``[1, patches * latent_patch_size, latent_dim]`` in
        whatever normalization the patch encoder expects. Seeds the slot's
        patch-encoder KV cache so decode steps continue the same sequence.
        """
        encoder = self._encoder
        x = encoder.in_proj(encoder._downsample(prompt_latents))
        tokens = int(x.size(1))
        if tokens > int(self._encoder_k.size(3)):
            raise ValueError(
                "dots.tts prompt audio exceeds the patch-encoder cache: "
                f"tokens={tokens} capacity={int(self._encoder_k.size(3))}"
            )

        rotary = None
        if self._encoder_rotary is not None:
            positions = torch.arange(
                tokens, device=prompt_latents.device, dtype=torch.float32
            ).reshape(1, tokens)
            embedding = self._encoder_rotary(positions)
            rotary = (embedding.cos(), embedding.sin())

        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for layer_idx, layer in enumerate(encoder.encoder.layers):
                attn_out, key, value = project_attention(
                    layer.attn,
                    layer.attn_norm(x),
                    num_heads=self._encoder_heads,
                    head_dim=self._encoder_head_dim,
                    rotary_cos=None if rotary is None else rotary[0],
                    rotary_sin=None if rotary is None else rotary[1],
                    is_causal=True,
                    dropout_p=0.0,
                )
                self._encoder_k[layer_idx, slot, :, :tokens].copy_(key[0])
                self._encoder_v[layer_idx, slot, :, :tokens].copy_(value[0])
                x = x + attn_out
                x = x + layer.ffn(layer.ffn_norm(x))

        left_padding = int(encoder.ds_proj.left_padding)
        self._encoder_conv_tail[slot].copy_(
            prompt_latents.transpose(1, 2)[0, :, -left_padding:]
        )
        self._encoder_seq_len[slot] = tokens
        return encoder._project_embeddings(x)[0]

    @torch.no_grad()
    def seed_fm_history(
        self, slot: int, *, fm_rows: torch.Tensor, all_mods: torch.Tensor
    ) -> None:
        """Install one request's prompt flow sequence and its AdaLN modulations.

        ``fm_rows`` is ``[unit_len * patches, fm_hidden_size]``: the projected
        prompt history without the trailing hidden row, which the first tail step
        appends. Everything but the last unit is folded into the DiT KV cache.
        """
        spec = self.spec
        total = int(fm_rows.size(0))
        if total <= 0 or total % spec.unit_len != 0:
            raise ValueError(
                "dots.tts prompt flow history must be unit-aligned and non-empty: "
                f"rows={total} unit_len={spec.unit_len}"
            )
        persistent = total - spec.unit_len
        if persistent + spec.unit_len > spec.dit_cache_tokens:
            raise ValueError(
                "dots.tts prompt flow history exceeds the tail capacity: "
                f"rows={total} cache_tokens={spec.dit_cache_tokens}"
            )

        self._all_mods[:, slot].copy_(all_mods)
        self._window[slot, : spec.unit_len].copy_(fm_rows[persistent:])
        self._window[slot, spec.unit_len :].zero_()
        self._fm_seq_len[slot] = total
        if persistent == 0:
            return

        positions = torch.arange(
            persistent, device=fm_rows.device, dtype=torch.float32
        ).reshape(1, persistent)
        embedding = self._dit_rotary(positions)
        cos, sin = embedding.cos(), embedding.sin()
        prefix = fm_rows[:persistent].unsqueeze(0)
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for ode_idx in range(spec.nfe):
                keys: list[torch.Tensor] = []
                values: list[torch.Tensor] = []

                def collect(
                    _layer_idx: int, block: nn.Module, attn_in: torch.Tensor
                ) -> torch.Tensor:
                    out, key, value = project_attention(
                        block.attn,
                        attn_in,
                        num_heads=self._dit_heads,
                        head_dim=self._dit_head_dim,
                        rotary_cos=cos,
                        rotary_sin=sin,
                        is_causal=True,
                        dropout_p=0.0,
                    )
                    keys.append(key)
                    values.append(value)
                    return out

                self.dit.run_modulated_blocks(
                    x=prefix,
                    all_mods=all_mods[ode_idx : ode_idx + 1],
                    attention=collect,
                )
                self._dit_k[ode_idx, :, slot, :, :persistent].copy_(
                    torch.stack(keys)[:, 0]
                )
                self._dit_v[ode_idx, :, slot, :, :persistent].copy_(
                    torch.stack(values)[:, 0]
                )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_patches(
        self,
        slots: list[int],
        *,
        fm_hidden_rows: torch.Tensor,
        latent_proj: nn.Module,
    ) -> torch.Tensor:
        """Append one hidden row per slot, then sample one latent patch each.

        ``fm_hidden_rows`` is ``[rows, fm_hidden_size]`` already in flow space.
        Returns the normalized latent patches ``[rows, latent_patch_size,
        latent_dim]`` and advances every slot's flow history by one unit.
        """
        spec = self.spec
        slot_index = torch.tensor(slots, device=self.device, dtype=torch.long)
        for slot in slots:
            self._fm_seq_len[slot] += spec.hidden_patch_size
        self._window[slot_index, spec.unit_len :] = fm_hidden_rows.unsqueeze(1).to(
            dtype=self.dtype
        )

        persistent = [self._fm_seq_len[slot] - spec.window_len for slot in slots]
        if min(persistent) < 0:
            raise RuntimeError(
                "dots.tts tail step ran before the prompt flow history was seeded"
            )
        capacity = max(persistent)
        if capacity + spec.unit_len > spec.dit_cache_tokens:
            raise RuntimeError(
                "dots.tts flow history exceeded the DiT cache: "
                f"persistent={capacity} capacity={spec.dit_cache_tokens}"
            )

        latent = self._run_meanflow(
            slots=slots,
            slot_index=slot_index,
            persistent_index=torch.tensor(
                persistent, device=self.device, dtype=torch.long
            ),
            capacity=capacity,
        )

        carry = self._window[slot_index, spec.unit_len :]
        self._window[slot_index, : spec.hidden_patch_size] = carry
        self._window[slot_index, spec.hidden_patch_size : spec.unit_len] = latent_proj(
            latent
        ).to(dtype=self.dtype)
        for slot in slots:
            self._fm_seq_len[slot] += spec.latent_patch_size
        return latent

    def _run_meanflow(
        self,
        *,
        slots: list[int],
        slot_index: torch.Tensor,
        persistent_index: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        spec = self.spec
        rows = len(slots)
        unit = spec.unit_len
        query = 2 * unit
        prev_unit = self._window[slot_index, :unit]
        current_hidden = self._window[slot_index, unit:]
        latent_slice = slice(
            unit + spec.hidden_patch_size,
            unit + spec.hidden_patch_size + spec.latent_patch_size,
        )

        sdpa_mask = self._dit_mask[:rows, :, :, : capacity + query]
        self._fill_causal_update_mask(
            sdpa_mask,
            capacity_tokens=capacity,
            valid_persistent=persistent_index,
            prev_len=unit,
            current_len=unit,
        )
        cos, sin = rotary_cos_sin(
            self._dit_rotary, start_pos=persistent_index, seq_len=query
        )
        mods = self._all_mods.index_select(1, slot_index)
        token_index = persistent_index.reshape(1, rows, 1) + torch.arange(
            unit, device=self.device
        ).reshape(1, 1, unit)
        layer_index = self._dit_layer_index.reshape(self._dit_layers, 1, 1)
        batch_index = slot_index.reshape(1, rows, 1)
        # The fresh K/V land right after the gathered prefix, so the first
        # ``unit`` of them is exactly the block promoted into the cache below.
        promote = slice(capacity, capacity + unit)

        latent = self._sample_noise(slots)
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for ode_idx in range(spec.nfe):
                key_buffer = self._dit_scratch_k[:, :rows, :, : capacity + query, :]
                value_buffer = self._dit_scratch_v[:, :rows, :, : capacity + query, :]
                torch.index_select(
                    self._dit_k[ode_idx, :, :, :, :capacity],
                    1,
                    slot_index,
                    out=key_buffer[:, :, :, :capacity, :],
                )
                torch.index_select(
                    self._dit_v[ode_idx, :, :, :, :capacity],
                    1,
                    slot_index,
                    out=value_buffer[:, :, :, :capacity, :],
                )

                def cached_attention(
                    layer_idx: int, block: nn.Module, attn_in: torch.Tensor
                ) -> torch.Tensor:
                    out, _key, _value = project_attention(
                        block.attn,
                        attn_in,
                        num_heads=self._dit_heads,
                        head_dim=self._dit_head_dim,
                        rotary_cos=cos,
                        rotary_sin=sin,
                        kv_buffer=(key_buffer[layer_idx], value_buffer[layer_idx]),
                        kv_prefix_len=capacity,
                        attn_mask=sdpa_mask,
                        dropout_p=0.0,
                    )
                    return out

                x = torch.cat(
                    [prev_unit, current_hidden, self._coordinate_proj(latent)], dim=1
                )
                x, final_mod = self.dit.run_modulated_blocks(
                    x=x, all_mods=mods[ode_idx], attention=cached_attention
                )
                velocity = self.dit.apply_final_layer(x[:, latent_slice], final_mod)
                duration = (self._times[ode_idx + 1] - self._times[ode_idx]).expand(
                    rows
                )
                latent = (latent + duration.view(-1, 1, 1) * velocity).clone()
                self._dit_k[ode_idx][layer_index, batch_index, :, token_index] = (
                    key_buffer[:, :, :, promote, :].permute(0, 1, 3, 2, 4)
                )
                self._dit_v[ode_idx][layer_index, batch_index, :, token_index] = (
                    value_buffer[:, :, :, promote, :].permute(0, 1, 3, 2, 4)
                )
        return latent

    @staticmethod
    def _fill_causal_update_mask(
        out: torch.Tensor,
        *,
        capacity_tokens: int,
        valid_persistent: torch.Tensor,
        prev_len: int,
        current_len: int,
    ) -> None:
        """Write :func:`batched_causal_update_mask` into a preallocated view."""
        q_len = int(prev_len) + int(current_len)
        device = valid_persistent.device
        total_kv = int(capacity_tokens) + q_len
        kv_idx = torch.arange(total_kv, device=device)
        q_idx = torch.arange(q_len, device=device).reshape(q_len, 1)
        tail_idx = kv_idx.reshape(1, total_kv) - int(capacity_tokens)
        prev_causal = (tail_idx >= 0) & (tail_idx < int(prev_len)) & (tail_idx <= q_idx)
        tail = torch.where(q_idx < int(prev_len), prev_causal, tail_idx >= 0)
        past = kv_idx.reshape(1, 1, 1, total_kv) < valid_persistent.reshape(-1, 1, 1, 1)
        torch.logical_or(past, tail.reshape(1, 1, q_len, total_kv), out=out)

    def _sample_noise(self, slots: list[int]) -> torch.Tensor:
        spec = self.spec
        generators = [self._generators[slot] for slot in slots]
        shape = (spec.latent_patch_size, spec.latent_dim)
        if all(generator is None for generator in generators):
            return torch.randn(
                (len(slots), *shape), device=self.device, dtype=self.dtype
            )
        return torch.cat(
            [
                torch.randn(
                    (1, *shape),
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                for generator in generators
            ],
            dim=0,
        )

    @torch.no_grad()
    def encode_feedback(
        self, slots: list[int], latent_patches: torch.Tensor
    ) -> torch.Tensor:
        """Backbone input embedding for each freshly sampled latent patch."""
        slot_index = torch.tensor(slots, device=self.device, dtype=torch.long)
        rows = len(slots)
        block = self._encoder_block
        starts = [self._encoder_seq_len[slot] for slot in slots]
        capacity = max(starts)
        if capacity + block > int(self._encoder_k.size(3)):
            raise RuntimeError(
                "dots.tts patch-encoder cache overflow: "
                f"tokens={capacity + block} capacity={int(self._encoder_k.size(3))}"
            )
        start_index = torch.tensor(starts, device=self.device, dtype=torch.long)

        sdpa_mask = self._encoder_mask[:rows, :, :, : capacity + block]
        self._fill_causal_update_mask(
            sdpa_mask,
            capacity_tokens=capacity,
            valid_persistent=start_index,
            prev_len=block,
            current_len=0,
        )
        if self._encoder_rotary is None:
            empty = torch.empty((0,), device=self.device)
            cos, sin = empty, empty
        else:
            cos, sin = rotary_cos_sin(
                self._encoder_rotary, start_pos=start_index, seq_len=block
            )

        key_buffer = self._encoder_scratch_k[:, :rows, :, : capacity + block, :]
        value_buffer = self._encoder_scratch_v[:, :rows, :, : capacity + block, :]
        torch.index_select(
            self._encoder_k[:, :, :, :capacity],
            1,
            slot_index,
            out=key_buffer[:, :, :, :capacity, :],
        )
        torch.index_select(
            self._encoder_v[:, :, :, :capacity],
            1,
            slot_index,
            out=value_buffer[:, :, :, :capacity, :],
        )
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            embedding, conv_tail = self._encoder_step(
                latent_patches.to(dtype=self.dtype),
                self._encoder_conv_tail.index_select(0, slot_index),
                (key_buffer, value_buffer),
                capacity,
                sdpa_mask,
                cos,
                sin,
            )

        token_index = start_index.reshape(1, rows, 1) + torch.arange(
            block, device=self.device
        ).reshape(1, 1, block)
        layer_index = self._encoder_layer_index.reshape(self._encoder_layers, 1, 1)
        batch_index = slot_index.reshape(1, rows, 1)
        promote = slice(capacity, capacity + block)
        self._encoder_k[layer_index, batch_index, :, token_index] = key_buffer[
            :, :, :, promote, :
        ].permute(0, 1, 3, 2, 4)
        self._encoder_v[layer_index, batch_index, :, token_index] = value_buffer[
            :, :, :, promote, :
        ].permute(0, 1, 3, 2, 4)
        self._encoder_conv_tail[slot_index] = conv_tail
        for slot in slots:
            self._encoder_seq_len[slot] += block
        return embedding.reshape(rows, -1)


__all__ = [
    "DotsTtsAcousticTail",
    "DotsTtsTailSpec",
    "batched_causal_update_mask",
]
