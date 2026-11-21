# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for the Ming 3 TTS flow head's few-token transformer blocks."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    from sglang.kernels.numerics import mul_rn_f32

    HAS_TRITON = True
except ImportError as error:  # pragma: no cover - depends on runtime image
    HAS_TRITON = False
    logger.info(f"Ming 3 TTS fused kernels unavailable: {error}")


if HAS_TRITON:

    @triton.jit
    def rotary_attention_kernel(
        qkv_ptr,
        cos_sin_ptr,
        out_ptr,
        sequence_length,
        scale,
        width: tl.constexpr,
        head_dim: tl.constexpr,
        block_tokens: tl.constexpr,
    ):
        """One (batch row, head): GPT-J rotary on q and k, then softmax attention."""
        batch = tl.program_id(0)
        head = tl.program_id(1)
        half: tl.constexpr = head_dim // 2
        tokens = tl.arange(0, block_tokens)
        pairs = tl.arange(0, half)
        dims = tl.arange(0, head_dim)
        token_mask = tokens < sequence_length
        row_offsets = (batch * sequence_length + tokens[:, None]) * (3 * width)
        query = qkv_ptr + row_offsets + head * head_dim + 2 * pairs[None, :]
        key = query + width
        cos = tl.load(
            cos_sin_ptr + tokens[:, None] * head_dim + pairs[None, :],
            mask=token_mask[:, None],
            other=1.0,
        )
        sin = tl.load(
            cos_sin_ptr + tokens[:, None] * head_dim + half + pairs[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        query_even = tl.load(query, mask=token_mask[:, None], other=0.0).to(tl.float32)
        query_odd = tl.load(query + 1, mask=token_mask[:, None], other=0.0).to(
            tl.float32
        )
        key_even = tl.load(key, mask=token_mask[:, None], other=0.0).to(tl.float32)
        key_odd = tl.load(key + 1, mask=token_mask[:, None], other=0.0).to(tl.float32)
        # Rotated q and k are rounded to bfloat16, as the in-place rotary kernel stores them.
        rotated_query_even = (query_even * cos - query_odd * sin).to(tl.bfloat16)
        rotated_query_odd = (query_odd * cos + query_even * sin).to(tl.bfloat16)
        rotated_key_even = (key_even * cos - key_odd * sin).to(tl.bfloat16)
        rotated_key_odd = (key_odd * cos + key_even * sin).to(tl.bfloat16)
        scores = tl.sum(
            rotated_query_even.to(tl.float32)[:, None, :]
            * rotated_key_even.to(tl.float32)[None, :, :],
            axis=2,
        ) + tl.sum(
            rotated_query_odd.to(tl.float32)[:, None, :]
            * rotated_key_odd.to(tl.float32)[None, :, :],
            axis=2,
        )
        scores = tl.where(token_mask[None, :], scores * scale, float("-inf"))
        weights = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        weights = weights / tl.sum(weights, axis=1)[:, None]
        value = tl.load(
            qkv_ptr + row_offsets + 2 * width + head * head_dim + dims[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        attended = tl.sum(weights[:, :, None] * value[None, :, :], axis=1)
        tl.store(
            out_ptr
            + (batch * sequence_length + tokens[:, None]) * width
            + head * head_dim
            + dims[None, :],
            attended.to(tl.bfloat16),
            mask=token_mask[:, None],
        )

    @triton.jit
    def add_rmsnorm_kernel(
        states_ptr,
        update_ptr,
        weight_ptr,
        sum_ptr,
        normed_ptr,
        eps,
        width: tl.constexpr,
    ):
        """One row: the bfloat16 residual sum, then HF RMSNorm of that sum."""
        row = tl.program_id(0)
        columns = tl.arange(0, width)
        offsets = row * width + columns
        total = (
            tl.load(states_ptr + offsets).to(tl.float32)
            + tl.load(update_ptr + offsets).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(sum_ptr + offsets, total)
        total = total.to(tl.float32)
        variance = tl.sum(total * total, axis=0) / width
        normed = (total * (1.0 / tl.sqrt(variance + eps))).to(tl.bfloat16)
        weight = tl.load(weight_ptr + columns)
        tl.store(
            normed_ptr + offsets,
            (normed.to(tl.float32) * weight.to(tl.float32)).to(tl.bfloat16),
        )

    @triton.jit
    def assemble_flow_states_kernel(
        token_ptr,
        history_ptr,
        sample_ptr,
        out_ptr,
        batch_size,
        width: tl.constexpr,
        block_width: tl.constexpr,
    ):
        """Row r of [2B, 3, width]: its token row, then the shared history and sample rows."""
        row = tl.program_id(0)
        token = tl.program_id(1)
        columns = tl.program_id(2) * block_width + tl.arange(0, block_width)
        shared_row = row % batch_size
        if token == 0:
            source = token_ptr + row * width
        elif token == 1:
            source = history_ptr + shared_row * width
        else:
            source = sample_ptr + shared_row * width
        mask = columns < width
        tl.store(
            out_ptr + (row * 3 + token) * width + columns,
            tl.load(source + columns, mask=mask),
            mask=mask,
        )

    @triton.jit
    def cfg_solver_step_kernel(
        velocity_ptr,
        sample_ptr,
        guidance_ptr,
        noise_scale_ptr,
        step_size_ptr,
        noise_step_scale_ptr,
        solver_noise_ptr,
        out_ptr,
        batch_size,
        latent_dim: tl.constexpr,
        add_noise: tl.constexpr,
    ):
        """One row: guided drift from the cond/uncond velocities, then the solver update.

        Each op rounds as the eager chain does: the bf16 difference, then fp32
        products and sums in the eager order, with no fused multiply-adds.
        """
        row = tl.program_id(0)
        columns = tl.arange(0, latent_dim)
        conditional = tl.load(velocity_ptr + row * latent_dim + columns).to(tl.float32)
        unconditional = tl.load(
            velocity_ptr + (row + batch_size) * latent_dim + columns
        ).to(tl.float32)
        difference = (conditional - unconditional).to(tl.bfloat16).to(tl.float32)
        drift = conditional + mul_rn_f32(difference, tl.load(guidance_ptr + row))
        stepped = tl.load(sample_ptr + row * latent_dim + columns) + mul_rn_f32(
            tl.load(step_size_ptr), drift
        )
        if add_noise:
            scale = mul_rn_f32(
                tl.load(noise_scale_ptr + row), tl.load(noise_step_scale_ptr)
            )
            stepped = stepped + mul_rn_f32(
                scale, tl.load(solver_noise_ptr + row * latent_dim + columns)
            )
        else:
            pass
        tl.store(out_ptr + row * latent_dim + columns, stepped)

else:
    pass


def rotary_attention(
    qkv: torch.Tensor, cos_sin_cache: torch.Tensor, heads: int
) -> torch.Tensor:
    """Attention of packed [batch, tokens, 3 * width] bfloat16 q, k, v rows."""
    batch_size, sequence_length, packed_width = qkv.shape
    width = packed_width // 3
    head_dim = width // heads
    attended = torch.empty(
        (batch_size, sequence_length, width), device=qkv.device, dtype=qkv.dtype
    )
    rotary_attention_kernel[(batch_size, heads)](
        qkv,
        cos_sin_cache,
        attended,
        sequence_length,
        head_dim**-0.5,
        width=width,
        head_dim=head_dim,
        block_tokens=triton.next_power_of_2(sequence_length),
    )
    return attended


def add_rmsnorm(
    states: torch.Tensor, update: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return states + update and its RMSNorm, both bfloat16 like the eager ops."""
    total = torch.empty_like(states)
    normed = torch.empty_like(states)
    width = states.shape[-1]
    add_rmsnorm_kernel[(states.numel() // width,)](
        states, update, weight, total, normed, eps, width=width
    )
    return total, normed


def assemble_flow_states(
    token_rows: torch.Tensor,
    history_rows: torch.Tensor,
    sample_rows: torch.Tensor,
) -> torch.Tensor:
    """[2B, 3, width] block input: each row's token row, then its history and sample rows.

    The guided and unguided halves share the B history and sample rows.
    """
    rows, _tokens, width = token_rows.shape
    states = torch.empty(
        (rows, 3, width), device=token_rows.device, dtype=token_rows.dtype
    )
    block_width = 1024
    assemble_flow_states_kernel[(rows, 3, triton.cdiv(width, block_width))](
        token_rows,
        history_rows,
        sample_rows,
        states,
        history_rows.shape[0],
        width=width,
        block_width=block_width,
    )
    return states


def cfg_solver_step(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    guidance: torch.Tensor,
    noise_scale: torch.Tensor,
    step_size: torch.Tensor,
    noise_step_scale: torch.Tensor,
    solver_noise: torch.Tensor,
    add_noise: bool,
) -> torch.Tensor:
    """The next [B, 1, latent] sample from [2B, 1, latent] guided and unguided velocities."""
    batch_size, _tokens, latent_dim = sample.shape
    next_sample = torch.empty_like(sample)
    cfg_solver_step_kernel[(batch_size,)](
        velocity,
        sample,
        guidance,
        noise_scale,
        step_size,
        noise_step_scale,
        solver_noise,
        next_sample,
        batch_size,
        latent_dim=latent_dim,
        add_noise=add_noise,
    )
    return next_sample
