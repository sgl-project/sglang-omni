# SPDX-License-Identifier: Apache-2.0
"""The codec's WindowLimitedTransformer forward with its elementwise work fused into
three Triton kernels per layer, where the module launches about 480 small kernels per
window; it keeps the module's weights, intermediate dtypes and attention mask."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    WindowLimitedTransformer,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None:

    @triton.jit(
        do_not_specialize=[
            "sequence_length",
            "batch_stride",
            "sequence_stride",
            "dim_stride",
        ]
    )
    def add_scale_norm_kernel(
        x_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        gamma_ptr: torch.Tensor,
        weight_ptr: torch.Tensor,
        residual_ptr: torch.Tensor,
        output_ptr: torch.Tensor,
        sequence_length: int,
        batch_stride: int,
        sequence_stride: int,
        dim_stride: int,
        eps: float,
        DIM: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        HAS_Y: tl.constexpr,
    ) -> None:
        """residual = x + y * gamma, then output = RMSNorm(residual) * weight, per row,
        with x read by strides for the first layer's channels-first input and every
        intermediate rounded to the row's dtype where the module rounds it."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_WIDTH)
        mask = columns < DIM
        source = (
            x_ptr
            + (row // sequence_length) * batch_stride
            + (row % sequence_length) * sequence_stride
            + columns * dim_stride
        )
        x = tl.load(source, mask=mask, other=0.0)
        if HAS_Y:
            y = tl.load(y_ptr + row * DIM + columns, mask=mask, other=0.0)
            gamma = tl.load(gamma_ptr + columns, mask=mask, other=0.0)
            scaled = (y.to(tl.float32) * gamma.to(tl.float32)).to(
                y_ptr.dtype.element_ty
            )
            x = (x.to(tl.float32) + scaled.to(tl.float32)).to(x_ptr.dtype.element_ty)
        else:
            pass
        tl.store(residual_ptr + row * DIM + columns, x, mask=mask)
        x = x.to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / DIM + eps)
        normed = (x * inverse_rms).to(x_ptr.dtype.element_ty).to(tl.float32)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            output_ptr + row * DIM + columns,
            (normed * weight).to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit(do_not_specialize=["sequence_length"])
    def rotary_kernel(
        qkv_ptr: torch.Tensor,
        frequencies_ptr: torch.Tensor,
        sequence_length: int,
        PAIR_COUNT: tl.constexpr,
        PAIRS_PER_HEAD: tl.constexpr,
        ROW_WIDTH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ) -> None:
        """Rotates the query and key halves of one [q | k | v] row in place, in float32
        from the module's frequency table, as apply_rotary_emb does."""
        row = tl.program_id(0)
        pairs = tl.arange(0, BLOCK_WIDTH)
        mask = pairs < PAIR_COUNT
        frequency_ptr = (
            frequencies_ptr
            + ((row % sequence_length) * PAIRS_PER_HEAD + pairs % PAIRS_PER_HEAD) * 2
        )
        cos = tl.load(frequency_ptr, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(frequency_ptr + 1, mask=mask, other=0.0).to(tl.float32)
        for part in tl.static_range(2):
            real_ptr = qkv_ptr + row * ROW_WIDTH + part * (2 * PAIR_COUNT) + 2 * pairs
            real = tl.load(real_ptr, mask=mask, other=0.0).to(tl.float32)
            imaginary = tl.load(real_ptr + 1, mask=mask, other=0.0).to(tl.float32)
            tl.store(
                real_ptr,
                (real * cos - imaginary * sin).to(qkv_ptr.dtype.element_ty),
                mask=mask,
            )
            tl.store(
                real_ptr + 1,
                (imaginary * cos + real * sin).to(qkv_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def silu_mul_kernel(
        gate_up_ptr: torch.Tensor,
        output_ptr: torch.Tensor,
        INTERMEDIATE_SIZE: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ) -> None:
        """output = silu(gate) * up from one [gate | up] row, rounding silu to the row's
        dtype before the product as F.silu(a) * b does."""
        row = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        mask = columns < INTERMEDIATE_SIZE
        gate = tl.load(
            gate_up_ptr + row * 2 * INTERMEDIATE_SIZE + columns, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            gate_up_ptr + row * 2 * INTERMEDIATE_SIZE + INTERMEDIATE_SIZE + columns,
            mask=mask,
            other=0.0,
        )
        activated = (
            (gate * tl.sigmoid(gate)).to(output_ptr.dtype.element_ty).to(tl.float32)
        )
        product = activated * up.to(tl.float32)
        tl.store(
            output_ptr + row * INTERMEDIATE_SIZE + columns,
            product.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

else:
    pass

NORM_BLOCK_LIMIT = 8192
SILU_BLOCK_WIDTH = 1024


@dataclass(frozen=True)
class LayerWeights:
    """One TransformerBlock's weights; w13 stacks w1 over w3 for a single GEMM."""

    attention_norm: torch.Tensor
    attention_scale: torch.Tensor
    ffn_norm: torch.Tensor
    ffn_scale: torch.Tensor
    wqkv: torch.Tensor
    wo: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor


def fused_transformer_unsupported_reason(module: nn.Module) -> str | None:
    if not isinstance(module, WindowLimitedTransformer):
        return f"{type(module).__name__} is not a WindowLimitedTransformer"
    else:
        pass
    config = module.config
    device = module.norm.weight.device
    if triton is None:
        return "Triton is not installed"
    elif device.type != "cuda":
        return f"device {device} is not a CUDA device"
    elif module.training:
        return "the module is in training mode"
    elif config.pos_embed_type != "rope":
        return f"{config.pos_embed_type} position embeddings"
    elif not module.causal:
        return "non-causal attention"
    elif config.n_local_heads != config.n_head:
        return "grouped key and value heads"
    elif config.head_dim % 2 != 0:
        return f"odd head dimension {config.head_dim}"
    elif config.dim > NORM_BLOCK_LIMIT:
        return f"dimension {config.dim} above {NORM_BLOCK_LIMIT}"
    elif not all(
        isinstance(part, nn.Identity)
        for part in (module.input_proj, module.output_proj, module.look_ahead_conv)
    ):
        return "input or output projections"
    elif any(layer.attention.kv_cache is not None for layer in module.layers):
        return "a key-value cache"
    else:
        return None


class CodecTransformer:
    """WindowLimitedTransformer.forward of a codec transformer with fused kernels, for
    inputs of the module's own dtype."""

    def __init__(self, module: WindowLimitedTransformer) -> None:
        reason = fused_transformer_unsupported_reason(module)
        if reason is not None:
            raise ValueError(f"no fused codec transformer forward: {reason}")
        else:
            pass
        self.module = module
        config = module.config
        self.dim = config.dim
        self.head_count = config.n_head
        self.head_dim = config.head_dim
        self.intermediate_size = config.intermediate_size
        self.eps = config.norm_eps
        self.window_size = module.window_size
        self.dtype = module.norm.weight.dtype
        with torch.no_grad():
            self.layers = [
                LayerWeights(
                    attention_norm=layer.attention_norm.weight,
                    attention_scale=layer.attention_layer_scale.gamma,
                    ffn_norm=layer.ffn_norm.weight,
                    ffn_scale=layer.ffn_layer_scale.gamma,
                    wqkv=layer.attention.wqkv.weight,
                    wo=layer.attention.wo.weight,
                    w13=torch.cat(
                        [layer.feed_forward.w1.weight, layer.feed_forward.w3.weight]
                    ),
                    w2=layer.feed_forward.w2.weight,
                )
                for layer in module.layers
            ]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, dim, frames] in and out, as the channels-first module."""
        batch_size, dim, sequence_length = x.shape
        assert dim == self.dim and x.dtype == self.dtype
        rows = batch_size * sequence_length
        residual = torch.empty((rows, dim), dtype=self.dtype, device=x.device)
        normed = torch.empty((rows, dim), dtype=self.dtype, device=x.device)
        self.add_scale_norm(
            x,
            None,
            None,
            self.layers[0].attention_norm,
            residual,
            normed,
            sequence_length,
        )
        query_width = self.head_count * self.head_dim
        freqs_cis = self.module.freqs_cis
        if self.window_size is None or sequence_length <= self.window_size:
            attention_mask = None
        else:
            attention_mask = self.module.make_window_limited_mask(
                sequence_length, x.device
            )
        for index, layer in enumerate(self.layers):
            qkv = F.linear(normed, layer.wqkv)
            rotary_kernel[(rows,)](
                qkv,
                freqs_cis,
                sequence_length,
                PAIR_COUNT=query_width // 2,
                PAIRS_PER_HEAD=self.head_dim // 2,
                ROW_WIDTH=3 * query_width,
                BLOCK_WIDTH=triton.next_power_of_2(query_width // 2),
            )
            query, key, value = (
                part.view(
                    batch_size, sequence_length, self.head_count, self.head_dim
                ).transpose(1, 2)
                for part in qkv.split(query_width, dim=-1)
            )
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            attended = attended.transpose(1, 2).reshape(rows, query_width)
            attention_output = F.linear(attended, layer.wo)
            self.add_scale_norm(
                residual,
                attention_output,
                layer.attention_scale,
                layer.ffn_norm,
                residual,
                normed,
                sequence_length,
            )
            gate_up = F.linear(normed, layer.w13)
            activated = torch.empty(
                (rows, self.intermediate_size), dtype=self.dtype, device=x.device
            )
            silu_mul_kernel[
                (rows, triton.cdiv(self.intermediate_size, SILU_BLOCK_WIDTH))
            ](
                gate_up,
                activated,
                INTERMEDIATE_SIZE=self.intermediate_size,
                BLOCK_WIDTH=SILU_BLOCK_WIDTH,
            )
            ffn_output = F.linear(activated, layer.w2)
            if index + 1 < len(self.layers):
                next_norm_weight, output = self.layers[index + 1].attention_norm, normed
            else:
                next_norm_weight = self.module.norm.weight
                output = torch.empty((rows, dim), dtype=self.dtype, device=x.device)
            self.add_scale_norm(
                residual,
                ffn_output,
                layer.ffn_scale,
                next_norm_weight,
                residual,
                output,
                sequence_length,
            )
        return output.view(batch_size, sequence_length, dim).transpose(1, 2)

    def add_scale_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        gamma: torch.Tensor | None,
        weight: torch.Tensor,
        residual: torch.Tensor,
        output: torch.Tensor,
        sequence_length: int,
    ) -> None:
        if x.dim() == 3:
            strides = (x.stride(0), x.stride(2), x.stride(1))
        else:
            strides = (sequence_length * x.stride(0), x.stride(0), 1)
        add_scale_norm_kernel[(residual.shape[0],)](
            x,
            y if y is not None else x,
            gamma if gamma is not None else x,
            weight,
            residual,
            output,
            sequence_length,
            *strides,
            self.eps,
            DIM=self.dim,
            BLOCK_WIDTH=triton.next_power_of_2(self.dim),
            HAS_Y=y is not None,
        )


def codec_transformer(module: nn.Module) -> CodecTransformer | nn.Module:
    """The fused forward of module when its device and configuration allow one, else the
    module itself."""
    if fused_transformer_unsupported_reason(module) is None:
        return CodecTransformer(module)
    else:
        return module
