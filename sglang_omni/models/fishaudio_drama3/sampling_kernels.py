# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from sglang_omni.platforms import current_platform

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    current_platform = None
    HAS_TRITON = False

__all__ = [
    "CodebookSampler",
    "sample_codebook_head",
    "sample_semantic_candidates",
    "triton_sampling_unsupported_reason",
    "untested_gpu_reason",
    "ordered_keys",
    "key_scores",
    "key_indices",
    "exponential_noise",
    "candidate_selection",
    "selected_candidates",
    "search_threshold",
    "race_selected",
    "race_above_threshold",
    "draw_candidate",
]

SELECTED_CANDIDATES = 32
TAIL_BLOCK_WIDTH = 32
MAX_BLOCK_WIDTH = 4096
SCORE_KEY_BITS = 32
SAMPLER_WARPS = 8
PHILOX_SEED_MODULUS = 1 << 63
FIRST_DRAW_STREAM = 0
RAS_DRAW_STREAM = 1
FIRST_CODEBOOK_HEAD_STREAM = 2
VALIDATED_COMPUTE_CAPABILITY = (9, 0)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CodebookSampler:
    step: torch.Tensor
    seed: int


if HAS_TRITON:
    from sglang_omni.models.fishaudio_drama3.sampling_selection import (
        candidate_selection,
        draw_candidate,
        exponential_noise,
        key_indices,
        key_scores,
        ordered_keys,
        race_above_threshold,
        race_selected,
        search_threshold,
        selected_candidates,
    )

    @triton.jit
    def codebook_head_kernel(
        logits_pointer: tl.tensor,
        logits_row_stride: tl.tensor,
        temperature_pointer: tl.tensor,
        top_k_pointer: tl.tensor,
        top_p_pointer: tl.tensor,
        step_pointer: tl.tensor,
        sampled_codes_pointer: tl.tensor,
        seed: tl.tensor,
        stream_id: tl.tensor,
        CODEBOOK_SIZE: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        TAIL_WIDTH: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_WIDTH)
        is_column = columns < CODEBOOK_SIZE
        logits = tl.load(
            logits_pointer + row * logits_row_stride + columns,
            mask=is_column,
            other=0.0,
        ).to(tl.float32)
        temperature = tl.maximum(
            tl.load(temperature_pointer + row), MINIMUM_TEMPERATURE
        )
        (
            main_keys,
            main_weights,
            tail_keys,
            tail_weights,
            row_maximum,
            weight_sum,
            top_keys,
        ) = candidate_selection(
            logits,
            is_column,
            tl.zeros([TAIL_WIDTH], tl.float32),
            tl.arange(0, TAIL_WIDTH) < 0,
            temperature,
            INDEX_BITS,
            SELECTED,
            False,
        )
        sampled_code = draw_candidate(
            main_keys,
            main_weights,
            tail_keys,
            tail_weights,
            row_maximum,
            weight_sum,
            top_keys,
            tl.load(top_k_pointer + row),
            tl.load(top_p_pointer + row),
            seed,
            tl.load(step_pointer),
            row,
            stream_id,
            INDEX_BITS,
            KEY_BITS,
        )
        tl.store(sampled_codes_pointer + row, sampled_code.to(tl.int64))

    @triton.jit
    def semantic_kernel(
        semantic_logits_pointer: tl.tensor,
        semantic_logits_row_stride: tl.tensor,
        candidate_token_ids_pointer: tl.tensor,
        previous_main_tokens_pointer: tl.tensor,
        previous_main_tokens_row_stride: tl.tensor,
        temperature_pointer: tl.tensor,
        top_k_pointer: tl.tensor,
        top_p_pointer: tl.tensor,
        step_pointer: tl.tensor,
        main_tokens_pointer: tl.tensor,
        codebook_zero_pointer: tl.tensor,
        seed: tl.tensor,
        semantic_begin_id: tl.tensor,
        CANDIDATE_COUNT: tl.constexpr,
        MAIN_WIDTH: tl.constexpr,
        TAIL_WIDTH: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        CODEBOOK_SIZE: tl.constexpr,
        WINDOW_FRAMES: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
        RAS_TEMPERATURE: tl.constexpr,
        RAS_TOP_K: tl.constexpr,
        RAS_TOP_P: tl.constexpr,
        FIRST_STREAM: tl.constexpr,
        RAS_STREAM: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        main_candidates = tl.arange(0, MAIN_WIDTH)
        tail_candidates = MAIN_WIDTH + tl.arange(0, TAIL_WIDTH)
        main_is_candidate = main_candidates < CANDIDATE_COUNT
        tail_is_candidate = tail_candidates < CANDIDATE_COUNT
        row_logits_pointer = semantic_logits_pointer + row * semantic_logits_row_stride
        main_logits = tl.load(
            row_logits_pointer + tl.load(candidate_token_ids_pointer + main_candidates)
        ).to(tl.float32)
        tail_logits = tl.load(
            row_logits_pointer
            + tl.load(
                candidate_token_ids_pointer + tail_candidates,
                mask=tail_is_candidate,
                other=0,
            ),
            mask=tail_is_candidate,
            other=0.0,
        ).to(tl.float32)
        temperature = tl.maximum(
            tl.load(temperature_pointer + row), MINIMUM_TEMPERATURE
        )
        step = tl.load(step_pointer)
        (
            first_main_keys,
            first_main_weights,
            first_tail_keys,
            first_tail_weights,
            first_row_maximum,
            first_weight_sum,
            first_top_keys,
        ) = candidate_selection(
            main_logits,
            main_is_candidate,
            tail_logits,
            tail_is_candidate,
            temperature,
            INDEX_BITS,
            SELECTED,
            True,
        )
        if temperature == RAS_TEMPERATURE:
            ras_main_keys = first_main_keys
            ras_main_weights = first_main_weights
            ras_tail_keys = first_tail_keys
            ras_tail_weights = first_tail_weights
            ras_row_maximum = first_row_maximum
            ras_weight_sum = first_weight_sum
            ras_top_keys = first_top_keys
        else:
            (
                ras_main_keys,
                ras_main_weights,
                ras_tail_keys,
                ras_tail_weights,
                ras_row_maximum,
                ras_weight_sum,
                ras_top_keys,
            ) = candidate_selection(
                main_logits,
                main_is_candidate,
                tail_logits,
                tail_is_candidate,
                tl.full([], RAS_TEMPERATURE, tl.float32),
                INDEX_BITS,
                SELECTED,
                True,
            )
        first_candidate = draw_candidate(
            first_main_keys,
            first_main_weights,
            first_tail_keys,
            first_tail_weights,
            first_row_maximum,
            first_weight_sum,
            first_top_keys,
            tl.load(top_k_pointer + row),
            tl.load(top_p_pointer + row),
            seed,
            step,
            row,
            FIRST_STREAM,
            INDEX_BITS,
            KEY_BITS,
        )
        ras_candidate = draw_candidate(
            ras_main_keys,
            ras_main_weights,
            ras_tail_keys,
            ras_tail_weights,
            ras_row_maximum,
            ras_weight_sum,
            ras_top_keys,
            tl.full([], RAS_TOP_K, tl.int64),
            tl.full([], RAS_TOP_P, tl.float32),
            seed,
            step,
            row,
            RAS_STREAM,
            INDEX_BITS,
            KEY_BITS,
        )
        first_token = tl.load(candidate_token_ids_pointer + first_candidate)
        ras_token = tl.load(candidate_token_ids_pointer + ras_candidate)
        window = tl.load(
            previous_main_tokens_pointer
            + row * previous_main_tokens_row_stride
            + tl.arange(0, WINDOW_FRAMES)
        )
        is_repeat = tl.max((window == first_token).to(tl.int32), axis=0) > 0
        is_semantic = (first_token >= semantic_begin_id) & (
            first_token < semantic_begin_id + CODEBOOK_SIZE
        )
        main_token = tl.where(is_repeat & is_semantic, ras_token, first_token)
        tl.store(main_tokens_pointer + row, main_token)
        tl.store(
            codebook_zero_pointer + row,
            tl.minimum(
                tl.maximum(main_token - semantic_begin_id, 0), CODEBOOK_SIZE - 1
            ),
        )

else:
    pass


def main_block_width(candidate_count: int) -> int:
    return 1 << (candidate_count.bit_length() - 1)


def untested_gpu_reason(device: torch.device) -> str | None:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == VALIDATED_COMPUTE_CAPABILITY:
        return None
    else:
        return f"untested GPU sm_{major}{minor}"


def triton_sampling_unsupported_reason(
    device: torch.device, codebook_size: int, candidate_count: int
) -> str | None:
    main_width = main_block_width(candidate_count)
    if not HAS_TRITON:
        return "Triton is not installed"
    elif device.type != "cuda" or not current_platform.is_cuda():
        return f"device {device} is not a CUDA device of the CUDA platform"
    elif (gpu_reason := untested_gpu_reason(device)) is not None:
        return gpu_reason
    elif not SELECTED_CANDIDATES <= codebook_size <= MAX_BLOCK_WIDTH:
        return (
            f"codebook size {codebook_size} is outside "
            f"[{SELECTED_CANDIDATES}, {MAX_BLOCK_WIDTH}]"
        )
    elif (
        main_width < SELECTED_CANDIDATES
        or main_width > MAX_BLOCK_WIDTH
        or candidate_count - main_width > TAIL_BLOCK_WIDTH
    ):
        return (
            f"{candidate_count} main-token candidates do not fit a block of "
            f"{SELECTED_CANDIDATES} to {MAX_BLOCK_WIDTH} plus {TAIL_BLOCK_WIDTH}"
        )
    elif SELECTED_CANDIDATES in (codebook_size, candidate_count):
        return (
            f"PyTorch orders tied logits in rows of {SELECTED_CANDIDATES} columns "
            f"unlike the kernels; codebook size {codebook_size}, "
            f"{candidate_count} main-token candidates"
        )
    else:
        return None


def sample_codebook_head(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    sampler: CodebookSampler,
    stream_id: int,
    minimum_temperature: float,
) -> torch.Tensor:
    row_count, codebook_size = logits.shape
    assert logits.stride(1) == 1
    index_bits = (codebook_size - 1).bit_length()
    sampled_codes = torch.empty(row_count, dtype=torch.int64, device=logits.device)
    codebook_head_kernel[(row_count,)](
        logits,
        logits.stride(0),
        temperature.float().contiguous(),
        top_k.long().contiguous(),
        top_p.float().contiguous(),
        sampler.step,
        sampled_codes,
        sampler.seed,
        stream_id,
        CODEBOOK_SIZE=codebook_size,
        BLOCK_WIDTH=triton.next_power_of_2(codebook_size),
        INDEX_BITS=index_bits,
        KEY_BITS=SCORE_KEY_BITS + index_bits,
        SELECTED=SELECTED_CANDIDATES,
        TAIL_WIDTH=TAIL_BLOCK_WIDTH,
        MINIMUM_TEMPERATURE=minimum_temperature,
        num_warps=SAMPLER_WARPS,
    )
    return sampled_codes


def sample_semantic_candidates(
    semantic_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    semantic_begin_id: int,
    codebook_size: int,
    sampler: CodebookSampler,
    *,
    minimum_temperature: float,
    ras_temperature: float,
    ras_top_k: int,
    ras_top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = semantic_logits.shape[0]
    candidate_count = candidate_token_ids.shape[0]
    assert semantic_logits.stride(1) == 1 and previous_main_tokens.stride(1) == 1
    index_bits = (candidate_count - 1).bit_length()
    main_tokens = torch.empty(
        row_count, dtype=torch.int64, device=semantic_logits.device
    )
    codebook_zero = torch.empty_like(main_tokens)
    semantic_kernel[(row_count,)](
        semantic_logits,
        semantic_logits.stride(0),
        candidate_token_ids.long().contiguous(),
        previous_main_tokens,
        previous_main_tokens.stride(0),
        temperature.float().contiguous(),
        top_k.long().contiguous(),
        top_p.float().contiguous(),
        sampler.step,
        main_tokens,
        codebook_zero,
        sampler.seed,
        semantic_begin_id,
        CANDIDATE_COUNT=candidate_count,
        MAIN_WIDTH=main_block_width(candidate_count),
        TAIL_WIDTH=TAIL_BLOCK_WIDTH,
        INDEX_BITS=index_bits,
        KEY_BITS=SCORE_KEY_BITS + index_bits,
        SELECTED=SELECTED_CANDIDATES,
        CODEBOOK_SIZE=codebook_size,
        WINDOW_FRAMES=previous_main_tokens.shape[1],
        MINIMUM_TEMPERATURE=minimum_temperature,
        RAS_TEMPERATURE=ras_temperature,
        RAS_TOP_K=ras_top_k,
        RAS_TOP_P=ras_top_p,
        FIRST_STREAM=FIRST_DRAW_STREAM,
        RAS_STREAM=RAS_DRAW_STREAM,
        num_warps=SAMPLER_WARPS,
    )
    return main_tokens, codebook_zero
