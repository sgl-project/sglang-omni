# SPDX-License-Identifier: Apache-2.0
"""Batched acoustic-tail state for dots.tts."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dots_tts.modules.backbone.dit_inference import FusedAdaLNDiT
from dots_tts.modules.backbone.inference_utils import fuse_qkv_projection
from dots_tts.modules.backbone.layers import rotate_half
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger(__name__)

_TAIL_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
_GRAPH_BATCH_BUCKETS = (8, 16)
_GRAPH_CONTEXT_PATCH_BUCKETS = (16, 32, 64, 128)


def _project_attention(
    attn: nn.Module,
    value: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
    kv_prefix_len: int = 0,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``dots_tts.modules.backbone.inference_utils.project_attention`` variant
    that writes K/V into a preallocated scratch buffer and attends over the
    truncated view instead of the full-width cache."""
    batch, seq_len, _ = value.shape
    qkv = attn.qkv_proj(value).view(batch, seq_len, 3, num_heads, head_dim)
    query, key, item = qkv.permute(2, 0, 3, 1, 4)
    query = attn.q_norm(query)
    key = attn.k_norm(key)
    if rotary_cos is not None:
        cos, sin = rotary_cos, rotary_sin
        if cos.ndim == 2:
            cos, sin = cos[None, None], sin[None, None]
        elif cos.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        query = (query * cos + rotate_half(query) * sin).to(query.dtype)
        key = (key * cos + rotate_half(key) * sin).to(key.dtype)

    if kv_buffer is None:
        keys, values = key, item
    else:
        key_buffer, value_buffer = kv_buffer
        end = int(kv_prefix_len) + seq_len
        key_buffer[:, :, kv_prefix_len:end] = key
        value_buffer[:, :, kv_prefix_len:end] = item
        keys, values = key_buffer[:, :, :end], value_buffer[:, :, :end]

    output = F.scaled_dot_product_attention(
        query,
        keys,
        values,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    output = output.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
    return attn.o_dropout(attn.o_proj(output)), key, item


class _AutocastFusedDiT(FusedAdaLNDiT):
    """Upstream fused DiT plus an autocast-safe fused-modulation entry point.

    ``build_mods`` runs the fp32 timestep embedding under autocast so bf16
    weights accept it outside the solver's own autocast scope.
    """

    def build_mods(
        self,
        timesteps: torch.Tensor,
        *,
        duration: torch.Tensor | None = None,
        g_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = self.fused_adaln[-1].weight.dtype
        device_type = timesteps.device.type
        autocast = device_type == "cuda" and dtype in {
            torch.float16,
            torch.bfloat16,
        }
        autocast |= device_type == "cpu" and dtype == torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast):
            return self.fused_adaln(
                self.build_condition(timesteps, duration=duration, g_cond=g_cond)
            )


def fuse_dit_for_inference(model: nn.Module) -> _AutocastFusedDiT:
    dit = model.velocity_field_predictor
    if not isinstance(dit, _AutocastFusedDiT):
        dit = _AutocastFusedDiT(dit)
        model.velocity_field_predictor = dit
    return dit


class _SemanticEncoderDecodeStep(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        attention = encoder.encoder.layers[0].attn
        self.num_heads = int(attention.num_heads)
        self.head_dim = int(attention.head_dim)

    def forward(
        self,
        latent_patch: torch.Tensor,
        conv_tail: torch.Tensor,
        kv_buffer: tuple[torch.Tensor, torch.Tensor],
        kv_prefix_len: int,
        attn_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ):
        encoder = self.encoder
        raw = latent_patch.transpose(1, 2)
        projection = encoder.ds_proj
        value = F.conv1d(
            torch.cat([conv_tail, raw], dim=-1),
            projection.weight,
            projection.bias,
            stride=projection.stride[0],
            dilation=projection.dilation[0],
            groups=projection.groups,
        ).transpose(1, 2)
        value = encoder.in_proj(value)
        keys, values = kv_buffer
        for layer_index, layer in enumerate(encoder.encoder.layers):
            output, _, _ = _project_attention(
                layer.attn,
                layer.attn_norm(value),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                rotary_cos=rotary_cos if layer.attn.rotary_bias else None,
                rotary_sin=rotary_sin if layer.attn.rotary_bias else None,
                kv_buffer=(keys[layer_index], values[layer_index]),
                kv_prefix_len=kv_prefix_len,
                attn_mask=attn_mask,
            )
            value = value + output
            value = value + layer.ffn(layer.ffn_norm(value))
        return encoder._project_embeddings(value), raw[..., -projection.left_padding :]


@dataclass(frozen=True)
class DotsTtsTailSpec:
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
        return self.patch_capacity * self.unit_len


@dataclass
class _CapturedTailGraph:
    graph: torch.cuda.CUDAGraph
    inputs: dict[str, torch.Tensor]
    output: torch.Tensor


def batched_causal_update_mask(
    *,
    capacity_tokens: int,
    valid_persistent: torch.Tensor,
    prev_len: int,
    current_len: int,
) -> torch.Tensor:
    """Mask a fresh query block appended after variable-length cached rows.

    Per-row (``valid_persistent`` tensor) form of
    ``dots_tts.modules.backbone.inference_utils.build_causal_update_mask``.
    """
    q_len = int(prev_len) + int(current_len)
    device = valid_persistent.device
    total_kv = int(capacity_tokens) + q_len
    kv_idx = torch.arange(total_kv, device=device)
    q_idx = torch.arange(q_len, device=device).reshape(q_len, 1)
    tail_idx = kv_idx.reshape(1, total_kv) - int(capacity_tokens)
    prev_causal = (tail_idx >= 0) & (tail_idx < int(prev_len)) & (tail_idx <= q_idx)
    tail = torch.where(q_idx < int(prev_len), prev_causal, tail_idx >= 0)
    past = kv_idx.reshape(1, 1, total_kv) < valid_persistent.reshape(-1, 1, 1)
    return (past | tail.unsqueeze(0)).unsqueeze(1)


def _rotary_cos_sin(rotary: Any, starts: torch.Tensor, length: int):
    """Per-slot-start form of
    ``dots_tts.modules.backbone.inference_utils.build_rotary_cos_sin``."""
    offsets = torch.arange(length, device=starts.device, dtype=torch.float32)
    embedding = rotary(starts.reshape(-1, 1).float() + offsets)
    return embedding.cos(), embedding.sin()


class DotsTtsAcousticTail:
    """Slot-pooled MeanFlow DiT and patch encoder for one running batch."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        coordinate_proj: nn.Module,
        latent_proj: nn.Module,
        patch_encoder: nn.Module,
        spec: DotsTtsTailSpec,
        device: torch.device,
        dtype: torch.dtype,
        optimize: bool = False,
    ) -> None:
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.dit = dit
        self._coordinate_proj = coordinate_proj
        self._latent_proj = latent_proj
        self._encoder = patch_encoder
        for layer in patch_encoder.encoder.layers:
            fuse_qkv_projection(layer.attn)
        self._encoder_step = _SemanticEncoderDecodeStep(patch_encoder).eval()

        dit_attention = dit.blocks[0].attn
        self._dit_layers = len(dit.blocks)
        self._dit_rotary = dit_attention.rotary
        self._dit_heads = int(dit_attention.num_heads)
        self._dit_head_dim = int(dit_attention.head_dim)
        encoder_attention = patch_encoder.encoder.layers[0].attn
        self._encoder_layers = len(patch_encoder.encoder.layers)
        self._encoder_rotary = (
            encoder_attention.rotary if encoder_attention.rotary_bias else None
        )
        self._encoder_block = int(patch_encoder.out_ds_rate)
        self._encoder_heads = int(encoder_attention.num_heads)
        self._encoder_head_dim = int(encoder_attention.head_dim)

        self._allocate_pools(int(dit.fused_adaln[-1].out_features))
        self._times = torch.linspace(0.0, 1.0, spec.nfe + 1, device=device, dtype=dtype)
        self._dit_layer_index = torch.arange(self._dit_layers, device=device)
        self._encoder_layer_index = torch.arange(self._encoder_layers, device=device)
        self._fm_seq_len = [0] * spec.num_slots
        self._encoder_seq_len = [0] * spec.num_slots
        self._generators: list[torch.Generator | None] = [None] * spec.num_slots
        self._free_slots = list(reversed(range(spec.num_slots)))
        self._meanflow_graphs: dict[tuple[int, int], _CapturedTailGraph] = {}
        self._encoder_graphs: dict[tuple[int, int], _CapturedTailGraph] = {}
        self._graph_pool: Any | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._graph_replays: Counter[str] = Counter()
        self._graph_misses: Counter[str] = Counter()
        self._dit_contiguous_view_steps = 0
        if optimize and device.type == "cuda":
            self._capture_cuda_graphs()

    def _allocate_pools(self, mods_width: int) -> None:
        spec = self.spec
        zeros = partial(torch.zeros, device=self.device, dtype=self.dtype)
        self._dit_k = zeros(
            spec.nfe,
            self._dit_layers,
            spec.num_slots,
            self._dit_heads,
            spec.dit_cache_tokens + spec.unit_len,
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
            "dots.tts acoustic pools: slots=%s nfe=%s patch_capacity=%s",
            spec.num_slots,
            spec.nfe,
            spec.patch_capacity,
        )

    def acquire_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError(
                "dots.tts acoustic tail ran out of slots; lower concurrency"
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

    def initialize_slot_rng(self, slot: int, rng: int | torch.Tensor | None) -> None:
        if rng is None:
            return
        generator = torch.Generator(device=self.device)
        if isinstance(rng, torch.Tensor):
            generator.set_state(rng.detach().cpu())
        else:
            generator.manual_seed(int(rng))
        self._generators[int(slot)] = generator

    def slot_rng_state(self, slot: int) -> torch.Tensor | None:
        generator = self._generators[int(slot)]
        return None if generator is None else generator.get_state().clone()

    def fm_seq_len(self, slot: int) -> int:
        return self._fm_seq_len[int(slot)]

    @torch.no_grad()
    def encode_prompt_patches(
        self, slot: int, prompt_latents: torch.Tensor
    ) -> torch.Tensor:
        encoder = self._encoder
        value = encoder.in_proj(encoder._downsample(prompt_latents))
        tokens = int(value.size(1))
        if tokens > int(self._encoder_k.size(3)):
            raise ValueError(
                f"dots.tts prompt encoder needs {tokens} tokens, capacity is "
                f"{int(self._encoder_k.size(3))}"
            )
        rotary = None
        if self._encoder_rotary is not None:
            positions = torch.arange(
                tokens, device=value.device, dtype=torch.float32
            ).reshape(1, tokens)
            embedding = self._encoder_rotary(positions)
            rotary = embedding.cos(), embedding.sin()
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for layer_index, layer in enumerate(encoder.encoder.layers):
                output, key, item = _project_attention(
                    layer.attn,
                    layer.attn_norm(value),
                    num_heads=self._encoder_heads,
                    head_dim=self._encoder_head_dim,
                    rotary_cos=None if rotary is None else rotary[0],
                    rotary_sin=None if rotary is None else rotary[1],
                    is_causal=True,
                )
                self._encoder_k[layer_index, slot, :, :tokens].copy_(key[0])
                self._encoder_v[layer_index, slot, :, :tokens].copy_(item[0])
                value = value + output
                value = value + layer.ffn(layer.ffn_norm(value))
        left_padding = int(encoder.ds_proj.left_padding)
        self._encoder_conv_tail[slot].copy_(
            prompt_latents.transpose(1, 2)[0, :, -left_padding:]
        )
        self._encoder_seq_len[slot] = tokens
        return encoder._project_embeddings(value)[0]

    @torch.no_grad()
    def seed_fm_history(
        self, slot: int, *, fm_rows: torch.Tensor, all_mods: torch.Tensor
    ) -> None:
        spec = self.spec
        total = int(fm_rows.size(0))
        if total <= 0 or total % spec.unit_len:
            raise ValueError("dots.tts prompt flow history must be unit aligned")
        persistent = total - spec.unit_len
        if persistent + spec.unit_len > spec.dit_cache_tokens:
            raise ValueError("dots.tts prompt flow history exceeds tail capacity")
        self._all_mods[:, slot].copy_(all_mods)
        self._window[slot, : spec.unit_len].copy_(fm_rows[persistent:])
        self._window[slot, spec.unit_len :].zero_()
        self._fm_seq_len[slot] = total
        if persistent == 0:
            return
        positions = torch.arange(
            persistent, device=fm_rows.device, dtype=torch.float32
        ).reshape(1, persistent)
        rotary = self._dit_rotary(positions)
        prefix = fm_rows[:persistent].unsqueeze(0)
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for ode_index in range(spec.nfe):
                keys: list[torch.Tensor] = []
                values: list[torch.Tensor] = []

                def collect(
                    _index: int, block: nn.Module, value: torch.Tensor
                ) -> torch.Tensor:
                    output, key, item = _project_attention(
                        block.attn,
                        value,
                        num_heads=self._dit_heads,
                        head_dim=self._dit_head_dim,
                        rotary_cos=rotary.cos(),
                        rotary_sin=rotary.sin(),
                        is_causal=True,
                    )
                    keys.append(key)
                    values.append(item)
                    return output

                self.dit.run_modulated_blocks(
                    x=prefix,
                    all_mods=all_mods[ode_index : ode_index + 1],
                    attention=collect,
                )
                self._dit_k[ode_index, :, slot, :, :persistent].copy_(
                    torch.stack(keys)[:, 0]
                )
                self._dit_v[ode_index, :, slot, :, :persistent].copy_(
                    torch.stack(values)[:, 0]
                )

    @torch.no_grad()
    def sample_patches(
        self,
        slots: list[int],
        *,
        fm_hidden_rows: torch.Tensor,
    ) -> torch.Tensor:
        spec = self.spec
        slot_index = torch.tensor(slots, device=self.device, dtype=torch.long)
        for slot in slots:
            self._fm_seq_len[slot] += spec.hidden_patch_size
        persistent = [self._fm_seq_len[slot] - spec.window_len for slot in slots]
        if min(persistent) < 0:
            raise RuntimeError("dots.tts tail ran before prompt history was seeded")
        capacity = max(persistent)
        if capacity + spec.unit_len > spec.dit_cache_tokens:
            raise RuntimeError("dots.tts flow history exceeded the DiT cache")
        persistent_index = torch.tensor(
            persistent, device=self.device, dtype=torch.long
        )
        noise = self._sample_noise(slots)
        direct_kv = len(slots) == spec.num_slots and sorted(slots) == list(
            range(spec.num_slots)
        )
        if direct_kv:
            order = torch.argsort(slot_index)
            work_slots = torch.arange(spec.num_slots, device=self.device)
            work_persistent = persistent_index.index_select(0, order)
            work_hidden = fm_hidden_rows.index_select(0, order)
            work_noise = noise.index_select(0, order)
            self._dit_contiguous_view_steps += spec.nfe
            if self._dit_contiguous_view_steps == spec.nfe:
                logger.info("dots.tts contiguous DiT KV view is active")
        else:
            work_slots = slot_index
            work_persistent = persistent_index
            work_hidden = fm_hidden_rows
            work_noise = noise
        graph = self._select_graph(self._meanflow_graphs, len(slots), capacity)
        if graph is None:
            self._graph_misses["meanflow"] += 1
            latent = self._sample_patches_core(
                work_slots,
                work_persistent,
                capacity,
                work_hidden,
                work_noise,
                direct_kv=direct_kv,
            )
        else:
            graph.inputs["slots"].copy_(work_slots)
            graph.inputs["starts"].copy_(work_persistent)
            graph.inputs["hidden"].copy_(work_hidden)
            graph.inputs["noise"].copy_(work_noise)
            graph.graph.replay()
            latent = graph.output.clone()
            self._graph_replays["meanflow"] += 1
            if self._graph_replays["meanflow"] == 1:
                logger.info("dots.tts batched MeanFlow CUDA graph replay is active")
        if direct_kv:
            latent = latent.index_select(0, slot_index)
        for slot in slots:
            self._fm_seq_len[slot] += spec.latent_patch_size
        return latent

    def _sample_patches_core(
        self,
        slot_index: torch.Tensor,
        persistent_index: torch.Tensor,
        capacity: int,
        fm_hidden_rows: torch.Tensor,
        noise: torch.Tensor,
        *,
        direct_kv: bool = False,
    ) -> torch.Tensor:
        spec = self.spec
        window = (
            self._window[: spec.num_slots] if direct_kv else self._window[slot_index]
        )
        window[:, spec.unit_len :] = fm_hidden_rows.unsqueeze(1).to(self.dtype)
        if not direct_kv:
            self._window[slot_index] = window
        latent = self._run_meanflow(
            slot_index,
            persistent_index,
            capacity,
            noise,
            direct_kv=direct_kv,
        )
        window = (
            self._window[: spec.num_slots] if direct_kv else self._window[slot_index]
        )
        carry = window[:, spec.unit_len :]
        window[:, : spec.hidden_patch_size] = carry
        window[:, spec.hidden_patch_size : spec.unit_len] = self._latent_proj(
            latent
        ).to(self.dtype)
        if not direct_kv:
            self._window[slot_index] = window
        return latent

    def _run_meanflow(
        self,
        slot_index: torch.Tensor,
        persistent_index: torch.Tensor,
        capacity: int,
        latent: torch.Tensor,
        *,
        direct_kv: bool = False,
    ) -> torch.Tensor:
        spec = self.spec
        rows = int(slot_index.numel())
        unit = spec.unit_len
        query_len = 2 * unit
        if direct_kv:
            previous = self._window[:rows, :unit]
            hidden = self._window[:rows, unit:]
            mods = self._all_mods[:, :rows]
        else:
            previous = self._window[slot_index, :unit]
            hidden = self._window[slot_index, unit:]
            mods = self._all_mods.index_select(1, slot_index)
        latent_slice = slice(
            unit + spec.hidden_patch_size,
            unit + spec.hidden_patch_size + spec.latent_patch_size,
        )
        mask = self._dit_mask[:rows, :, :, : capacity + query_len]
        self._fill_mask(mask, capacity, persistent_index, unit, unit)
        cos, sin = _rotary_cos_sin(self._dit_rotary, persistent_index, query_len)
        token_index = persistent_index.reshape(1, rows, 1) + torch.arange(
            unit, device=self.device
        ).reshape(1, 1, unit)
        layer_index = self._dit_layer_index.reshape(self._dit_layers, 1, 1)
        batch_index = slot_index.reshape(1, rows, 1)
        promote = slice(capacity, capacity + unit)
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            for ode_index in range(spec.nfe):
                if direct_kv:
                    keys = self._dit_k[ode_index, :, :rows, :, : capacity + query_len]
                    values = self._dit_v[ode_index, :, :rows, :, : capacity + query_len]
                else:
                    keys = self._dit_scratch_k[:, :rows, :, : capacity + query_len]
                    values = self._dit_scratch_v[:, :rows, :, : capacity + query_len]
                    torch.index_select(
                        self._dit_k[ode_index, :, :, :, :capacity],
                        1,
                        slot_index,
                        out=keys[:, :, :, :capacity],
                    )
                    torch.index_select(
                        self._dit_v[ode_index, :, :, :, :capacity],
                        1,
                        slot_index,
                        out=values[:, :, :, :capacity],
                    )

                def cached_attention(
                    layer: int, block: nn.Module, value: torch.Tensor
                ) -> torch.Tensor:
                    output, _, _ = _project_attention(
                        block.attn,
                        value,
                        num_heads=self._dit_heads,
                        head_dim=self._dit_head_dim,
                        rotary_cos=cos,
                        rotary_sin=sin,
                        kv_buffer=(keys[layer], values[layer]),
                        kv_prefix_len=capacity,
                        attn_mask=mask,
                    )
                    return output

                value = torch.cat(
                    [previous, hidden, self._coordinate_proj(latent)], dim=1
                )
                value, final_mods = self.dit.run_modulated_blocks(
                    x=value,
                    all_mods=mods[ode_index],
                    attention=cached_attention,
                )
                velocity = self.dit.apply_final_layer(
                    value[:, latent_slice], final_mods
                )
                duration = self._times[ode_index + 1] - self._times[ode_index]
                latent = (latent + duration * velocity).clone()
                self._dit_k[ode_index][layer_index, batch_index, :, token_index] = keys[
                    :, :, :, promote
                ].permute(0, 1, 3, 2, 4)
                self._dit_v[ode_index][layer_index, batch_index, :, token_index] = (
                    values[:, :, :, promote].permute(0, 1, 3, 2, 4)
                )
        return latent

    @staticmethod
    def _fill_mask(
        output: torch.Tensor,
        capacity: int,
        valid: torch.Tensor,
        prev_len: int,
        current_len: int,
    ) -> None:
        output.copy_(
            batched_causal_update_mask(
                capacity_tokens=capacity,
                valid_persistent=valid,
                prev_len=prev_len,
                current_len=current_len,
            )
        )

    def _sample_noise(self, slots: list[int]) -> torch.Tensor:
        shape = (self.spec.latent_patch_size, self.spec.latent_dim)
        generators = [self._generators[slot] for slot in slots]
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
            ]
        )

    @torch.no_grad()
    def encode_feedback(
        self, slots: list[int], latent_patches: torch.Tensor
    ) -> torch.Tensor:
        slot_index = torch.tensor(slots, device=self.device, dtype=torch.long)
        rows = len(slots)
        block = self._encoder_block
        starts = [self._encoder_seq_len[slot] for slot in slots]
        capacity = max(starts)
        if capacity + block > int(self._encoder_k.size(3)):
            raise RuntimeError("dots.tts patch-encoder cache overflow")
        start_index = torch.tensor(starts, device=self.device, dtype=torch.long)
        graph = self._select_graph(self._encoder_graphs, rows, capacity)
        if graph is None:
            self._graph_misses["semantic_encoder"] += 1
            embeddings = self._encode_feedback_core(
                slot_index,
                start_index,
                capacity,
                latent_patches,
            )
        else:
            graph.inputs["slots"].copy_(slot_index)
            graph.inputs["starts"].copy_(start_index)
            graph.inputs["latent"].copy_(latent_patches)
            graph.graph.replay()
            embeddings = graph.output.clone()
            self._graph_replays["semantic_encoder"] += 1
            if self._graph_replays["semantic_encoder"] == 1:
                logger.info(
                    "dots.tts batched semantic-encoder CUDA graph replay is active"
                )
        for slot in slots:
            self._encoder_seq_len[slot] += block
        return embeddings

    def _encode_feedback_core(
        self,
        slot_index: torch.Tensor,
        start_index: torch.Tensor,
        capacity: int,
        latent_patches: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(slot_index.numel())
        block = self._encoder_block
        mask = self._encoder_mask[:rows, :, :, : capacity + block]
        self._fill_mask(mask, capacity, start_index, block, 0)
        if self._encoder_rotary is None:
            cos = sin = torch.empty(0, device=self.device)
        else:
            cos, sin = _rotary_cos_sin(self._encoder_rotary, start_index, block)
        keys = self._encoder_scratch_k[:, :rows, :, : capacity + block]
        values = self._encoder_scratch_v[:, :rows, :, : capacity + block]
        torch.index_select(
            self._encoder_k[:, :, :, :capacity],
            1,
            slot_index,
            out=keys[:, :, :, :capacity],
        )
        torch.index_select(
            self._encoder_v[:, :, :, :capacity],
            1,
            slot_index,
            out=values[:, :, :, :capacity],
        )
        with sdpa_kernel(_TAIL_SDPA_BACKENDS):
            embeddings, conv_tail = self._encoder_step(
                latent_patches.to(self.dtype),
                self._encoder_conv_tail.index_select(0, slot_index),
                (keys, values),
                capacity,
                mask,
                cos,
                sin,
            )
        token_index = start_index.reshape(1, rows, 1) + torch.arange(
            block, device=self.device
        ).reshape(1, 1, block)
        layer_index = self._encoder_layer_index.reshape(self._encoder_layers, 1, 1)
        batch_index = slot_index.reshape(1, rows, 1)
        promote = slice(capacity, capacity + block)
        self._encoder_k[layer_index, batch_index, :, token_index] = keys[
            :, :, :, promote
        ].permute(0, 1, 3, 2, 4)
        self._encoder_v[layer_index, batch_index, :, token_index] = values[
            :, :, :, promote
        ].permute(0, 1, 3, 2, 4)
        self._encoder_conv_tail[slot_index] = conv_tail
        return embeddings.reshape(rows, -1)

    @staticmethod
    def _select_graph(
        graphs: dict[tuple[int, int], _CapturedTailGraph],
        rows: int,
        capacity: int,
    ) -> _CapturedTailGraph | None:
        bucket = min(
            (
                bucket_capacity
                for batch_size, bucket_capacity in graphs
                if batch_size == rows and bucket_capacity >= capacity
            ),
            default=None,
        )
        return None if bucket is None else graphs[(rows, bucket)]

    @torch.no_grad()
    def _capture_cuda_graphs(self) -> None:
        batch_buckets = tuple(
            batch for batch in _GRAPH_BATCH_BUCKETS if batch <= self.spec.num_slots
        )
        context_buckets = tuple(
            patches
            for patches in _GRAPH_CONTEXT_PATCH_BUCKETS
            if patches < self.spec.patch_capacity
        )
        if not batch_buckets or not context_buckets:
            return

        current_stream = torch.cuda.current_stream(self.device)
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._capture_stream.wait_stream(current_stream)
        self._graph_pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(self._capture_stream):
            for batch_size in reversed(batch_buckets):
                for patches in reversed(context_buckets):
                    self._capture_graph(
                        self._meanflow_graphs,
                        batch_size,
                        patches * self.spec.unit_len,
                        kind="meanflow",
                    )
                    self._capture_graph(
                        self._encoder_graphs,
                        batch_size,
                        patches * self._encoder_block,
                        kind="semantic_encoder",
                    )
        current_stream.wait_stream(self._capture_stream)
        torch.cuda.synchronize(self.device)
        logger.info(
            "dots.tts acoustic-tail CUDA graphs: meanflow=%d semantic_encoder=%d "
            "batch_buckets=%s context_patch_buckets=%s",
            len(self._meanflow_graphs),
            len(self._encoder_graphs),
            batch_buckets,
            context_buckets,
        )

    def _capture_graph(
        self,
        destination: dict[tuple[int, int], _CapturedTailGraph],
        batch_size: int,
        capacity: int,
        *,
        kind: str,
    ) -> None:
        key = (batch_size, capacity)
        slots = torch.arange(batch_size, device=self.device, dtype=torch.long)
        starts = torch.full(
            (batch_size,), capacity, device=self.device, dtype=torch.long
        )
        if kind == "meanflow":
            inputs = {
                "slots": slots,
                "starts": starts,
                "hidden": torch.zeros(
                    batch_size,
                    self.spec.fm_hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "noise": torch.zeros(
                    batch_size,
                    self.spec.latent_patch_size,
                    self.spec.latent_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
            }
            run = lambda: self._sample_patches_core(
                inputs["slots"],
                inputs["starts"],
                capacity,
                inputs["hidden"],
                inputs["noise"],
                direct_kv=batch_size == self.spec.num_slots,
            )
        else:
            inputs = {
                "slots": slots,
                "starts": starts,
                "latent": torch.zeros(
                    batch_size,
                    self.spec.latent_patch_size,
                    self.spec.latent_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
            }
            run = lambda: self._encode_feedback_core(
                inputs["slots"],
                inputs["starts"],
                capacity,
                inputs["latent"],
            )

        graph = torch.cuda.CUDAGraph()
        try:
            for _ in range(2):
                run()
            self._capture_stream.synchronize()
            with torch.cuda.graph(
                graph,
                pool=self._graph_pool,
                stream=self._capture_stream,
                capture_error_mode="thread_local",
            ):
                output = run()
            destination[key] = _CapturedTailGraph(graph, inputs, output)
        except Exception as exc:
            graph.reset()
            logger.warning(
                "dots.tts %s CUDA graph capture failed for batch=%d capacity=%d: "
                "%s; using eager fallback",
                kind,
                batch_size,
                capacity,
                exc,
            )

    @property
    def backend(self) -> str:
        if self._meanflow_graphs and self._encoder_graphs:
            return (
                "CUDA graph batched tail with eager fallback "
                f"(meanflow={len(self._meanflow_graphs)}, "
                f"semantic_encoder={len(self._encoder_graphs)})"
            )
        return "fused eager batched tail"


__all__ = [
    "DotsTtsAcousticTail",
    "DotsTtsTailSpec",
    "batched_causal_update_mask",
    "fuse_dit_for_inference",
]
