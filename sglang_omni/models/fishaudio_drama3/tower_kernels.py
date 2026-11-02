# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from sglang_omni.models.fishaudio_drama3.sampling_kernels import untested_gpu_reason
from sglang_omni.models.fishaudio_drama3.tower_activation import (
    gate_o_proj_add,
    norm_gate_up_silu,
)
from sglang_omni.models.fishaudio_drama3.tower_attention import norm_qkv_rope_store
from sglang_omni.models.fishaudio_drama3.tower_projection import mix_eh_proj
from sglang_omni.models.fishaudio_drama3.tower_tiles import (
    FUSED_BLOCK_M,
    GATE_UP_BLOCK_N,
    GEMV_BLOCK_K,
    MAX_FUSED_ROWS,
    MIX_BLOCK_N,
    O_PROJ_BLOCK_N,
    QKV_BLOCK_K,
    NormWeights,
    SplitScratch,
    TowerLayerShape,
    TowerLayerWeights,
    split_count,
    tower_layer_shape,
)

__all__ = [
    "MAX_FUSED_ROWS",
    "FUSED_KERNELS",
    "NormWeights",
    "TowerLayerWeights",
    "TowerLayerShape",
    "SplitScratch",
    "FusedTowerLayers",
    "fused_tower_layers_unsupported_reason",
    "mix_eh_proj",
    "norm_qkv_rope_store",
    "gate_o_proj_add",
    "norm_gate_up_silu",
    "split_count",
    "tower_layer_shape",
]


try:
    import triton
    import triton.language as tl
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
    from sglang.srt.runtime_context import get_exec
    from triton.language.extra import libdevice

    from sglang_omni.platforms import current_platform

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    HybridLinearKVPool = None
    MHATokenToKVPool = None
    get_exec = None
    current_platform = None
    HAS_TRITON = False

FUSED_KERNELS = (
    "mix_eh_proj",
    "norm_qkv_rope_store",
    "gate_o_proj_add",
    "norm_gate_up_silu",
)


class QueryAttention(Protocol):
    def __call__(self, layer_index: int, query: torch.Tensor) -> torch.Tensor: ...


class FusedTowerLayers:
    def __init__(
        self,
        *,
        shape: TowerLayerShape,
        rotary_cos_sin: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.shape: TowerLayerShape = shape
        self.max_rows: int = MAX_FUSED_ROWS
        self.rotary_cos_sin: torch.Tensor = rotary_cos_sin

        def activations(width: int) -> torch.Tensor:
            return torch.zeros(
                (MAX_FUSED_ROWS, width), dtype=torch.bfloat16, device=device
            )

        self.mixed: torch.Tensor = activations(shape.dim)
        self.query: torch.Tensor = activations(shape.query_width)
        self.gate_scale: torch.Tensor = activations(shape.query_width)
        self.residual: torch.Tensor = activations(shape.dim)
        self.scaled: torch.Tensor = activations(shape.dim)
        self.activated: torch.Tensor = activations(shape.intermediate_size)
        mix_tiles = shape.dim // MIX_BLOCK_N
        o_proj_tiles = shape.dim // O_PROJ_BLOCK_N
        partial_count = FUSED_BLOCK_M * max(
            2 * shape.mix_half_split * shape.dim,
            shape.qkv_split * shape.qkv_part_count * shape.head_dim,
            shape.o_proj_split * shape.dim,
            shape.gate_up_split * 2 * shape.intermediate_size,
        )
        self.scratch: SplitScratch = SplitScratch(
            partials=torch.zeros(partial_count, dtype=torch.float32, device=device),
            sum_of_squares=torch.zeros(
                FUSED_BLOCK_M * mix_tiles * 2 * shape.mix_half_split,
                dtype=torch.float32,
                device=device,
            ),
            norm_sums=torch.zeros(
                FUSED_BLOCK_M * max(mix_tiles, o_proj_tiles),
                dtype=torch.float32,
                device=device,
            ),
            counters=torch.zeros(
                max(
                    mix_tiles,
                    o_proj_tiles,
                    shape.intermediate_size // GATE_UP_BLOCK_N,
                    shape.qkv_part_count,
                ),
                dtype=torch.int32,
                device=device,
            ),
        )

    def covers(self, row_count: int) -> bool:
        return row_count <= self.max_rows

    def forward_layer(
        self,
        *,
        weights: TowerLayerWeights,
        layer_index: int,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention: QueryAttention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_count = hidden_states.shape[0]
        shape = self.shape
        mixed = self.mixed[:row_count]
        scaled = self.scaled[:row_count]
        query = self.query[:row_count]
        gate_scale = self.gate_scale[:row_count]
        residual = self.residual[:row_count]
        activated = self.activated[:row_count]
        mix_eh_proj(
            weights=weights,
            hidden_states=hidden_states,
            code_ids=code_ids,
            label_is_semantic=label_is_semantic,
            shared_input=shared_input,
            mixed=mixed,
            scaled=scaled,
            scratch=self.scratch,
            shape=shape,
        )
        norm_qkv_rope_store(
            weights=weights,
            scaled=scaled,
            rotary_cos_sin=self.rotary_cos_sin,
            positions=positions,
            cache_slots=cache_slots,
            query=query,
            gate_scale=gate_scale,
            key_cache=key_cache,
            value_cache=value_cache,
            scratch=self.scratch,
            shape=shape,
        )
        gate_o_proj_add(
            weights=weights,
            attended=attention(layer_index, query),
            gate_scale=gate_scale,
            residual_in=mixed,
            residual_out=residual,
            scaled_out=scaled,
            scratch=self.scratch,
            shape=shape,
        )
        norm_gate_up_silu(
            weights=weights,
            scaled=scaled,
            activated=activated,
            scratch=self.scratch,
            shape=shape,
        )
        return activated, residual


def fused_tower_layers_unsupported_reason(
    *,
    layers: list[TowerLayerWeights],
    head_count: int,
    kv_head_count: int,
    rotary_dim: int,
    rotary_cos_sin: torch.Tensor,
    context_length: int,
    kv_pool: HybridLinearKVPool,
    tower_layer_ids: list[int],
    device: torch.device,
) -> str | None:
    if not HAS_TRITON:
        return "Triton is not installed"
    elif device.type != "cuda" or not current_platform.is_cuda():
        return f"device {device} is not a CUDA device of the CUDA platform"
    elif (gpu_reason := untested_gpu_reason(device)) is not None:
        return gpu_reason
    elif get_exec().deterministic.enable_deterministic_inference:
        return (
            f"deterministic inference is on, and passes above {MAX_FUSED_ROWS} rows "
            f"keep the plain path"
        )
    else:
        pass
    first_layer = layers[0]
    dim = first_layer.eh_proj.weight.shape[0]
    head_dim = first_layer.q_norm.weight.shape[0]
    intermediate_size = first_layer.gate_up_proj.weight.shape[0] // 2
    codebook_size = first_layer.token_embedding.shape[0]
    query_width = head_count * head_dim
    expected_shapes = {
        "eh_proj": (dim, 2 * dim),
        "qkv_proj": (2 * (head_count + kv_head_count) * head_dim, dim),
        "o_proj": (dim, query_width),
        "gate_up_proj": (2 * intermediate_size, dim),
    }
    if (
        dim % MIX_BLOCK_N != 0
        or dim % O_PROJ_BLOCK_N != 0
        or dim % GEMV_BLOCK_K != 0
        or dim % QKV_BLOCK_K != 0
        or intermediate_size % GATE_UP_BLOCK_N != 0
        or query_width % GEMV_BLOCK_K != 0
        or head_dim < 32
        or head_dim & (head_dim - 1) != 0
    ):
        return (
            f"dim {dim}, intermediate size {intermediate_size}, head dim {head_dim} "
            f"and {head_count} heads do not fit the kernels' blocks"
        )
    elif not (0 < rotary_dim <= head_dim and rotary_dim % 2 == 0):
        return (
            f"rotary dim {rotary_dim} is not an even width within head dim {head_dim}"
        )
    elif (
        rotary_cos_sin.dtype != torch.float32
        or not rotary_cos_sin.is_contiguous()
        or rotary_cos_sin.shape[1] != rotary_dim
        or rotary_cos_sin.shape[0] < context_length
        or rotary_cos_sin.device != device
    ):
        return (
            f"the rotary table {tuple(rotary_cos_sin.shape)} {rotary_cos_sin.dtype} is "
            f"not a contiguous float32 table of {context_length} positions and "
            f"{rotary_dim} columns on {device}"
        )
    else:
        pass
    for layer_index, layer in enumerate(layers):
        linears = {
            "eh_proj": layer.eh_proj,
            "qkv_proj": layer.qkv_proj,
            "o_proj": layer.o_proj,
            "gate_up_proj": layer.gate_up_proj,
        }
        norms = {
            "enorm": (layer.enorm, dim),
            "hnorm": (layer.hnorm, dim),
            "input_layernorm": (layer.input_norm, dim),
            "q_norm": (layer.q_norm, head_dim),
            "k_norm": (layer.k_norm, head_dim),
            "post_attention_layernorm": (layer.post_norm, dim),
        }
        token_embedding = layer.token_embedding
        for name, linear in linears.items():
            weight = linear.weight
            if type(linear) is not nn.Linear or linear.bias is not None:
                return (
                    f"tower layer {layer_index} {name} is not a bias-free torch Linear"
                )
            elif weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                return f"tower layer {layer_index} {name} weight is not contiguous bfloat16"
            elif weight.device != device:
                return (
                    f"tower layer {layer_index} {name} weight is on {weight.device}, "
                    f"not {device}"
                )
            elif tuple(weight.shape) != expected_shapes[name]:
                return (
                    f"tower layer {layer_index} {name} weight is "
                    f"{tuple(weight.shape)}, not {expected_shapes[name]}"
                )
            else:
                pass
        for name, (norm, width) in norms.items():
            if (
                tuple(norm.weight.shape) != (width,)
                or not norm.weight.is_contiguous()
                or norm.weight.device != device
            ):
                return (
                    f"tower layer {layer_index} {name} weight is not a contiguous "
                    f"vector of {width} on {device}"
                )
            else:
                pass
        if (
            token_embedding.dtype != torch.bfloat16
            or not token_embedding.is_contiguous()
            or tuple(token_embedding.shape) != (codebook_size, dim)
            or token_embedding.device != device
        ):
            return (
                f"tower layer {layer_index} token embedding is not a contiguous "
                f"bfloat16 table of {codebook_size} x {dim} on {device}"
            )
        else:
            pass
    full_pool = kv_pool.full_kv_pool
    if type(full_pool) is not MHATokenToKVPool:
        return (
            f"the full-attention KV pool is {type(full_pool).__name__}, not "
            f"MHATokenToKVPool"
        )
    elif (
        full_pool.is_quantized_kv_cache
        or full_pool.use_hnd
        or full_pool.kv_cache_layout != "nhd"
        or full_pool.dtype != torch.bfloat16
        or full_pool.store_dtype != torch.bfloat16
        or full_pool.page_size != 1
    ):
        return (
            f"the full-attention KV pool has layout {full_pool.kv_cache_layout} and "
            f"dtype {full_pool.dtype}, not an unquantized bfloat16 pool of one-slot "
            f"pages in nhd layout"
        )
    elif (full_pool.head_num, full_pool.head_dim, full_pool.v_head_dim) != (
        kv_head_count,
        head_dim,
        head_dim,
    ):
        return (
            f"the KV pool holds {full_pool.head_num} heads of {full_pool.head_dim} / "
            f"{full_pool.v_head_dim}, not {kv_head_count} of {head_dim}"
        )
    else:
        pass
    for layer_id in tower_layer_ids:
        if layer_id not in kv_pool.full_attention_layer_id_mapping:
            return f"tower layer id {layer_id} has no full-attention KV buffer"
        else:
            pass
        for buffer in (
            kv_pool.get_key_buffer(layer_id),
            kv_pool.get_value_buffer(layer_id),
        ):
            if (
                buffer.dim() != 3
                or tuple(buffer.shape[1:])
                != (
                    kv_head_count,
                    head_dim,
                )
                or not buffer.is_contiguous()
            ):
                return (
                    f"the KV buffer of tower layer id {layer_id} is "
                    f"{tuple(buffer.shape)}, not contiguous slots of "
                    f"{kv_head_count} x {head_dim}"
                )
            else:
                pass
    return None
