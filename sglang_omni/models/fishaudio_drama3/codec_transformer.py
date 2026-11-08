# SPDX-License-Identifier: Apache-2.0
"""The codec's WindowLimitedTransformer forward in a dozen kernels per layer, not sixty.

The module's own forward launches about 480 kernels for eight layers, so it costs a fixed
1.4 ms on an H100 however few frames it sees: broadcast rotary multiplies, RMSNorm reductions
and layer-scale and residual elementwise ops. This forward keeps the weights, the dtype of
every intermediate and the attention mask of the module, and fuses the elementwise work into
three Triton kernels per layer."""

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
        do_not_specialize=["seq_len", "stride_batch", "stride_seq", "stride_dim"]
    )
    def add_scale_norm_kernel(
        x_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        gamma_ptr: torch.Tensor,
        weight_ptr: torch.Tensor,
        residual_ptr: torch.Tensor,
        out_ptr: torch.Tensor,
        seq_len: int,
        stride_batch: int,
        stride_seq: int,
        stride_dim: int,
        eps: float,
        DIM: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_Y: tl.constexpr,
    ) -> None:
        """residual = x + y * gamma, then out = RMSNorm(residual) * weight, one row each.

        x is read with strides so the first layer reads the channels-first input; the
        contiguous residual is written back for the next layer. Each rounding matches
        the module: y * gamma, the sum, the normed row and the weighted row each round
        to the row's dtype."""
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < DIM
        source = (
            x_ptr
            + (row // seq_len) * stride_batch
            + (row % seq_len) * stride_seq
            + col * stride_dim
        )
        x = tl.load(source, mask=mask, other=0.0)
        if HAS_Y:
            y = tl.load(y_ptr + row * DIM + col, mask=mask, other=0.0)
            gamma = tl.load(gamma_ptr + col, mask=mask, other=0.0)
            scaled = (y.to(tl.float32) * gamma.to(tl.float32)).to(
                y_ptr.dtype.element_ty
            )
            x = (x.to(tl.float32) + scaled.to(tl.float32)).to(x_ptr.dtype.element_ty)
        else:
            pass
        tl.store(residual_ptr + row * DIM + col, x, mask=mask)
        x = x.to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / DIM + eps)
        normed = (x * inverse_rms).to(x_ptr.dtype.element_ty).to(tl.float32)
        weight = tl.load(weight_ptr + col, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + row * DIM + col,
            (normed * weight).to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit(do_not_specialize=["seq_len"])
    def rotary_kernel(
        qkv_ptr: torch.Tensor,
        freqs_ptr: torch.Tensor,
        seq_len: int,
        PAIRS: tl.constexpr,
        HALF_HEAD: tl.constexpr,
        ROW: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        """Rotates the query and key halves of one [q | k | v] row in place, in float32
        from the module's frequency table, as apply_rotary_emb does."""
        row = tl.program_id(0)
        pair = tl.arange(0, BLOCK)
        mask = pair < PAIRS
        frequency = freqs_ptr + ((row % seq_len) * HALF_HEAD + pair % HALF_HEAD) * 2
        cos = tl.load(frequency, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(frequency + 1, mask=mask, other=0.0).to(tl.float32)
        for part in tl.static_range(2):
            base = qkv_ptr + row * ROW + part * (2 * PAIRS) + 2 * pair
            real = tl.load(base, mask=mask, other=0.0).to(tl.float32)
            imag = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
            tl.store(
                base, (real * cos - imag * sin).to(qkv_ptr.dtype.element_ty), mask=mask
            )
            tl.store(
                base + 1,
                (imag * cos + real * sin).to(qkv_ptr.dtype.element_ty),
                mask=mask,
            )

    @triton.jit
    def silu_mul_kernel(
        gate_up_ptr: torch.Tensor,
        out_ptr: torch.Tensor,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        """out = silu(gate) * up from one [gate | up] row, rounding silu to the row's dtype
        before the product as F.silu(a) * b does."""
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < HIDDEN
        gate = tl.load(gate_up_ptr + row * 2 * HIDDEN + col, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            gate_up_ptr + row * 2 * HIDDEN + HIDDEN + col, mask=mask, other=0.0
        )
        activated = (
            (gate * tl.sigmoid(gate)).to(out_ptr.dtype.element_ty).to(tl.float32)
        )
        product = activated * up.to(tl.float32)
        tl.store(
            out_ptr + row * HIDDEN + col,
            product.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

else:
    pass

NORM_BLOCK_LIMIT = 8192
SILU_BLOCK = 1024


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
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.hidden = config.intermediate_size
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
        batch_size, dim, seq_len = x.shape
        assert dim == self.dim and x.dtype == self.dtype
        rows = batch_size * seq_len
        residual = torch.empty((rows, dim), dtype=self.dtype, device=x.device)
        normed = torch.empty((rows, dim), dtype=self.dtype, device=x.device)
        self.add_scale_norm(
            x, None, None, self.layers[0].attention_norm, residual, normed, seq_len
        )
        inner = self.n_head * self.head_dim
        freqs_cis = self.module.freqs_cis
        if self.window_size is None or seq_len <= self.window_size:
            attention_mask = None
        else:
            attention_mask = self.module.make_window_limited_mask(seq_len, x.device)
        for index, layer in enumerate(self.layers):
            qkv = F.linear(normed, layer.wqkv)
            rotary_kernel[(rows,)](
                qkv,
                freqs_cis,
                seq_len,
                PAIRS=inner // 2,
                HALF_HEAD=self.head_dim // 2,
                ROW=3 * inner,
                BLOCK=triton.next_power_of_2(inner // 2),
            )
            query, key, value = (
                part.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(
                    1, 2
                )
                for part in qkv.split(inner, dim=-1)
            )
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
            attended = attended.transpose(1, 2).reshape(rows, inner)
            attention_out = F.linear(attended, layer.wo)
            self.add_scale_norm(
                residual,
                attention_out,
                layer.attention_scale,
                layer.ffn_norm,
                residual,
                normed,
                seq_len,
            )
            gate_up = F.linear(normed, layer.w13)
            activated = torch.empty(
                (rows, self.hidden), dtype=self.dtype, device=x.device
            )
            silu_mul_kernel[(rows, triton.cdiv(self.hidden, SILU_BLOCK))](
                gate_up, activated, HIDDEN=self.hidden, BLOCK=SILU_BLOCK
            )
            ffn_out = F.linear(activated, layer.w2)
            if index + 1 < len(self.layers):
                next_norm, out = self.layers[index + 1].attention_norm, normed
            else:
                next_norm = self.module.norm.weight
                out = torch.empty((rows, dim), dtype=self.dtype, device=x.device)
            self.add_scale_norm(
                residual, ffn_out, layer.ffn_scale, next_norm, residual, out, seq_len
            )
        return out.view(batch_size, seq_len, dim).transpose(1, 2)

    def add_scale_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        gamma: torch.Tensor | None,
        weight: torch.Tensor,
        residual: torch.Tensor,
        out: torch.Tensor,
        seq_len: int,
    ) -> None:
        if x.dim() == 3:
            strides = (x.stride(0), x.stride(2), x.stride(1))
        else:
            strides = (seq_len * x.stride(0), x.stride(0), 1)
        add_scale_norm_kernel[(residual.shape[0],)](
            x,
            y if y is not None else x,
            gamma if gamma is not None else x,
            weight,
            residual,
            out,
            seq_len,
            *strides,
            self.eps,
            DIM=self.dim,
            BLOCK=triton.next_power_of_2(self.dim),
            HAS_Y=y is not None,
        )


def codec_transformer(module: nn.Module) -> CodecTransformer | nn.Module:
    """The fused forward of module when its device and configuration allow one, else the
    module itself."""
    if fused_transformer_unsupported_reason(module) is None:
        return CodecTransformer(module)
    else:
        return module
