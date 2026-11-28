# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for the Ming 3 TTS flow head's few-token transformer blocks."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from sglang_omni.platforms import current_platform

logger = logging.getLogger(__name__)

# The custom kernels are Triton/CUDA only; other devices run the same math with
# native torch ops so the fused flow head still skips the per-step concatenations.
_TRITON_ACTIVE: bool | None = None


def use_triton_kernels() -> bool:
    """Whether the Triton implementations below are the ones to launch."""
    global _TRITON_ACTIVE
    if _TRITON_ACTIVE is None:
        _TRITON_ACTIVE = HAS_TRITON and current_platform.is_cuda()
    else:
        pass
    return _TRITON_ACTIVE


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


def apply_rotary_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Rotate the interleaved pairs of contiguous [tokens, heads, head_dim] rows."""
    kernel = current_platform.get_joint_rope_inplace_kernel()
    if kernel is not None:
        kernel(query, key, cos_sin_cache, positions, is_neox=False)
    else:
        apply_rotary_torch(query, key, cos_sin_cache, positions)


def apply_rotary_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Rotate with the cached angles when no fused rotary kernel is available."""
    half = cos_sin_cache.shape[-1] // 2
    # note (rsi): both halves see the same angles; positions index the cache rows.
    cos = cos_sin_cache[:, :half][positions].unsqueeze(1)
    sin = cos_sin_cache[:, half:][positions].unsqueeze(1)

    def rotate(states: torch.Tensor) -> torch.Tensor:
        pairs = states.view(states.shape[0], states.shape[1], half, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
        return rotated.flatten(-2).to(states.dtype)

    query.copy_(rotate(query))
    key.copy_(rotate(key))


def rotary_attention_torch(
    qkv: torch.Tensor, cos_sin_cache: torch.Tensor, heads: int
) -> torch.Tensor:
    """Rotary attention of packed q, k, v rows without the fused CUDA kernel."""
    batch_size, sequence_length, packed_width = qkv.shape
    width = packed_width // 3
    head_dim = width // heads
    query, key, value = qkv.split(width, dim=-1)
    tokens = (batch_size * sequence_length, heads, head_dim)
    query = query.reshape(tokens).contiguous()
    key = key.reshape(tokens).contiguous()
    value = value.reshape(tokens).contiguous()
    positions = torch.arange(sequence_length, device=qkv.device).repeat(batch_size)
    apply_rotary_inplace(query, key, cos_sin_cache, positions)
    query = query.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
    attended = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    return (
        attended.transpose(1, 2)
        .reshape(batch_size, sequence_length, width)
        .to(qkv.dtype)
    )


def rotary_attention_bsnd(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int
) -> torch.Tensor:
    """Ascend rotary attention with once-expanded [1, tokens, 1, head_dim] angles.

    The gathered and interleaved coefficients are constant for a whole run, so
    the per-step index and repeat work moves to cache construction.
    """
    import torch_npu

    batch_size, sequence_length, packed_width = qkv.shape
    width = packed_width // 3
    head_dim = width // heads
    query, key, value = qkv.split(width, dim=-1)
    layout = (batch_size, sequence_length, heads, head_dim)
    query = query.reshape(layout).contiguous()
    key = key.reshape(layout).contiguous()
    value = value.reshape(layout).contiguous()
    query = torch_npu.npu_rotary_mul(query, cos, sin, "interleave")
    key = torch_npu.npu_rotary_mul(key, cos, sin, "interleave")
    # note (rsi): the v3 entry is the FlashAttention SDPA dispatches to, without
    # the wrapper work around it; it stays graph-capturable and numerically equal.
    fused = getattr(torch_npu, "npu_fusion_attention_v3", None)
    if fused is not None:
        attended = fused(
            query,
            key,
            value,
            heads,
            "BSND",
            scale=head_dim**-0.5,
            keep_prob=1.0,
            sparse_mode=0,
        )[0]
    else:
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
    attended = attended.reshape(batch_size, sequence_length, width)
    return attended.to(qkv.dtype)


def rotary_attention(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads: int,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention of packed [batch, tokens, 3 * width] bfloat16 q, k, v rows."""
    if cos is not None:
        return rotary_attention_bsnd(qkv, cos, sin, heads)
    else:
        pass
    if not use_triton_kernels():
        return rotary_attention_torch(qkv, cos_sin_cache, heads)
    else:
        pass
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
    if current_platform.is_npu():
        # note (rsi): one Ascend fused kernel replaces the separate residual add
        # and the pow/mean/rsqrt chain F.rms_norm decomposes into on this backend.
        fused = current_platform.get_add_rmsnorm()
        normed, total = fused(states, update, weight, eps)
        return total, normed
    else:
        pass
    if not use_triton_kernels():
        total = states + update
        normed = F.rms_norm(
            total.reshape(-1, total.shape[-1]), (total.shape[-1],), weight, eps
        ).view(total.shape)
        return total, normed
    else:
        pass
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
    if not use_triton_kernels():
        rows, _tokens, width = token_rows.shape
        batch_size = history_rows.shape[0]
        states = torch.empty(
            (rows, 3, width), device=token_rows.device, dtype=token_rows.dtype
        )
        # note (rsi): row copies replace the per-step cat/stack; history and
        # sample rows are written to both guidance halves.
        states[:, 0].copy_(token_rows[:, 0])
        states[:batch_size, 1].copy_(history_rows[:, 0])
        states[batch_size:, 1].copy_(history_rows[:, 0])
        states[:batch_size, 2].copy_(sample_rows[:, 0])
        states[batch_size:, 2].copy_(sample_rows[:, 0])
        return states
    else:
        pass
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
    if not use_triton_kernels():
        return cfg_solver_step_torch(
            velocity,
            sample,
            guidance,
            noise_scale,
            step_size,
            noise_step_scale,
            solver_noise,
            add_noise,
        )
    else:
        pass
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


def cfg_solver_step_torch(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    guidance: torch.Tensor,
    noise_scale: torch.Tensor,
    step_size: torch.Tensor,
    noise_step_scale: torch.Tensor,
    solver_noise: torch.Tensor,
    add_noise: bool,
) -> torch.Tensor:
    """The solver step in fp32 with the same rounding order as the fused kernel."""
    batch_size = sample.shape[0]
    conditional = velocity[:batch_size].float()
    unconditional = velocity[batch_size:].float()
    difference = (conditional - unconditional).to(velocity.dtype).float()
    drift = conditional + difference * guidance.float()
    stepped = sample.float() + drift * step_size.float()
    if add_noise:
        scale = noise_scale.float() * noise_step_scale.float()
        stepped = stepped + scale * solver_noise.float()
    else:
        pass
    return stepped.to(sample.dtype)
