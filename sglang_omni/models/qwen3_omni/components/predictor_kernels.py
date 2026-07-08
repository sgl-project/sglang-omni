"""Optional Triton kernels for Qwen3-Omni talker code predictor."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only dev hosts.
    triton = None
    tl = None


def has_triton_predictor_kernels() -> bool:
    return triton is not None and tl is not None


if has_triton_predictor_kernels():

    @triton.jit
    def _stage_initial_predictor_kernel(
        layer0_code,
        talker_hidden,
        codec_weight,
        predictor_input,
        pos_codes,
        pos_summed,
        hidden_size: tl.constexpr,
        layer0_code_stride_b: tl.constexpr,
        talker_hidden_stride_b: tl.constexpr,
        talker_hidden_stride_h: tl.constexpr,
        codec_weight_stride_v: tl.constexpr,
        codec_weight_stride_h: tl.constexpr,
        predictor_input_stride_b: tl.constexpr,
        predictor_input_stride_t: tl.constexpr,
        predictor_input_stride_h: tl.constexpr,
        pos_codes_stride_b: tl.constexpr,
        pos_codes_stride_g: tl.constexpr,
        pos_summed_stride_b: tl.constexpr,
        pos_summed_stride_h: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_h)
        mask = offsets < hidden_size

        code = tl.load(layer0_code + row * layer0_code_stride_b)
        hidden = tl.load(
            talker_hidden
            + row * talker_hidden_stride_b
            + offsets * talker_hidden_stride_h,
            mask=mask,
            other=0.0,
        )
        embed = tl.load(
            codec_weight
            + code * codec_weight_stride_v
            + offsets * codec_weight_stride_h,
            mask=mask,
            other=0.0,
        )

        base = predictor_input + row * predictor_input_stride_b
        tl.store(
            base + offsets * predictor_input_stride_h,
            hidden,
            mask=mask,
        )
        tl.store(
            base + predictor_input_stride_t + offsets * predictor_input_stride_h,
            embed,
            mask=mask,
        )
        tl.store(
            pos_summed + row * pos_summed_stride_b + offsets * pos_summed_stride_h,
            embed,
            mask=mask,
        )
        tl.store(pos_codes + row * pos_codes_stride_b, code)

    @triton.jit
    def _sample_code_stage_predictor_kernel(
        logits,
        embedding_weight,
        predictor_input,
        pos_codes,
        pos_summed,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        logits_stride_b: tl.constexpr,
        logits_stride_t: tl.constexpr,
        logits_stride_v: tl.constexpr,
        embedding_weight_stride_v: tl.constexpr,
        embedding_weight_stride_h: tl.constexpr,
        predictor_input_stride_b: tl.constexpr,
        predictor_input_stride_t: tl.constexpr,
        predictor_input_stride_h: tl.constexpr,
        pos_codes_stride_b: tl.constexpr,
        pos_codes_stride_g: tl.constexpr,
        pos_summed_stride_b: tl.constexpr,
        pos_summed_stride_h: tl.constexpr,
        code_group: tl.constexpr,
        input_slot: tl.constexpr,
        block_v: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)

        vocab_offsets = tl.arange(0, block_v)
        vocab_mask = vocab_offsets < vocab_size
        row_logits = tl.load(
            logits + row * logits_stride_b + vocab_offsets * logits_stride_v,
            mask=vocab_mask,
            other=-float("inf"),
        )
        max_val = tl.max(row_logits, axis=0)
        first_max = tl.where(row_logits == max_val, vocab_offsets, block_v)
        code = tl.min(first_max, axis=0)
        tl.store(
            pos_codes + row * pos_codes_stride_b + code_group * pos_codes_stride_g,
            code,
        )

        hidden_offsets = tl.arange(0, block_h)
        hidden_mask = hidden_offsets < hidden_size
        embed = tl.load(
            embedding_weight
            + code * embedding_weight_stride_v
            + hidden_offsets * embedding_weight_stride_h,
            mask=hidden_mask,
            other=0.0,
        )
        tl.store(
            predictor_input
            + row * predictor_input_stride_b
            + input_slot * predictor_input_stride_t
            + hidden_offsets * predictor_input_stride_h,
            embed,
            mask=hidden_mask,
        )
        old_sum = tl.load(
            pos_summed
            + row * pos_summed_stride_b
            + hidden_offsets * pos_summed_stride_h,
            mask=hidden_mask,
            other=0.0,
        )
        tl.store(
            pos_summed
            + row * pos_summed_stride_b
            + hidden_offsets * pos_summed_stride_h,
            old_sum + embed,
            mask=hidden_mask,
        )


def _can_launch(*tensors: torch.Tensor) -> bool:
    return has_triton_predictor_kernels() and all(t.is_cuda for t in tensors)


def stage_initial_predictor_input_(
    *,
    layer0_code: torch.Tensor,
    talker_hidden: torch.Tensor,
    codec_weight: torch.Tensor,
    predictor_input: torch.Tensor,
    pos_codes: torch.Tensor,
    pos_summed: torch.Tensor,
) -> bool:
    if layer0_code.ndim != 1:
        raise ValueError(f"layer0_code must be 1D, got {tuple(layer0_code.shape)}")
    if talker_hidden.ndim != 2:
        raise ValueError(f"talker_hidden must be 2D, got {tuple(talker_hidden.shape)}")
    if predictor_input.ndim != 3:
        raise ValueError(
            f"predictor_input must be 3D, got {tuple(predictor_input.shape)}"
        )
    if pos_codes.ndim != 2:
        raise ValueError(f"pos_codes must be 2D, got {tuple(pos_codes.shape)}")
    if pos_summed.ndim != 2:
        raise ValueError(f"pos_summed must be 2D, got {tuple(pos_summed.shape)}")

    batch_size, hidden_size = talker_hidden.shape
    if batch_size == 0:
        return True
    if not _can_launch(
        layer0_code,
        talker_hidden,
        codec_weight,
        predictor_input,
        pos_codes,
        pos_summed,
    ):
        return False
    if codec_weight.ndim != 2 or codec_weight.shape[1] != hidden_size:
        return False
    if predictor_input.shape[0] < batch_size or predictor_input.shape[2] != hidden_size:
        return False
    if pos_codes.shape[0] < batch_size or pos_codes.shape[1] < 1:
        return False
    if pos_summed.shape[0] < batch_size or pos_summed.shape[1] != hidden_size:
        return False

    block_h = triton.next_power_of_2(hidden_size)
    _stage_initial_predictor_kernel[(batch_size,)](
        layer0_code,
        talker_hidden,
        codec_weight,
        predictor_input,
        pos_codes,
        pos_summed,
        hidden_size,
        layer0_code.stride(0),
        talker_hidden.stride(0),
        talker_hidden.stride(1),
        codec_weight.stride(0),
        codec_weight.stride(1),
        predictor_input.stride(0),
        predictor_input.stride(1),
        predictor_input.stride(2),
        pos_codes.stride(0),
        pos_codes.stride(1),
        pos_summed.stride(0),
        pos_summed.stride(1),
        block_h,
        num_warps=8,
    )
    return True


def sample_code_and_stage_(
    *,
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    predictor_input: torch.Tensor,
    pos_codes: torch.Tensor,
    pos_summed: torch.Tensor,
    layer_idx: int,
) -> bool:
    if logits.ndim != 3:
        return False
    if logits.shape[1] != 1:
        return False
    if embedding_weight.ndim != 2:
        return False
    if predictor_input.ndim != 3 or pos_codes.ndim != 2 or pos_summed.ndim != 2:
        return False

    batch_size, _, vocab_size = logits.shape
    hidden_size = embedding_weight.shape[1]
    code_group = layer_idx + 1
    input_slot = layer_idx + 2
    if batch_size == 0:
        return True
    if not _can_launch(
        logits,
        embedding_weight,
        predictor_input,
        pos_codes,
        pos_summed,
    ):
        return False
    if embedding_weight.shape[0] != vocab_size:
        return False
    if predictor_input.shape[0] < batch_size or predictor_input.shape[2] != hidden_size:
        return False
    if predictor_input.shape[1] <= input_slot:
        return False
    if pos_codes.shape[0] < batch_size or pos_codes.shape[1] <= code_group:
        return False
    if pos_summed.shape[0] < batch_size or pos_summed.shape[1] != hidden_size:
        return False

    block_v = triton.next_power_of_2(vocab_size)
    block_h = triton.next_power_of_2(hidden_size)
    _sample_code_stage_predictor_kernel[(batch_size,)](
        logits,
        embedding_weight,
        predictor_input,
        pos_codes,
        pos_summed,
        vocab_size,
        hidden_size,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        embedding_weight.stride(0),
        embedding_weight.stride(1),
        predictor_input.stride(0),
        predictor_input.stride(1),
        predictor_input.stride(2),
        pos_codes.stride(0),
        pos_codes.stride(1),
        pos_summed.stride(0),
        pos_summed.stride(1),
        code_group,
        input_slot,
        block_v,
        block_h,
        num_warps=8,
    )
    return True
