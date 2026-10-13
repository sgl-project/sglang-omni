# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_drama3 import sampling_kernels
from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    MINIMUM_TEMPERATURE,
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    RAS_WINDOW_FRAMES,
    sample_semantic_with_ras,
    sample_top_k_top_p,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_CODEBOOK_HEAD_STREAM,
    FIRST_DRAW_STREAM,
    HAS_TRITON,
    RAS_DRAW_STREAM,
    SCORE_KEY_BITS,
    SELECTED_CANDIDATES,
    TAIL_BLOCK_WIDTH,
    VALIDATED_COMPUTE_CAPABILITY,
    CodebookSampler,
    triton_sampling_unsupported_reason,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
        candidate_selection,
        exponential_noise,
        race_above_threshold,
        race_selected,
        search_threshold,
        selected_candidates,
    )

    @triton.jit
    def noise_kernel(
        seed,
        step_pointer,
        stream_id,
        noise_pointer,
        WIDTH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_WIDTH)
        tl.store(
            noise_pointer + row * WIDTH + columns,
            exponential_noise(seed, tl.load(step_pointer), row, stream_id, columns),
            mask=columns < WIDTH,
        )

    @triton.jit
    def kept_probabilities_kernel(
        logits_pointer,
        temperature_pointer,
        top_k_pointer,
        top_p_pointer,
        step_pointer,
        seed,
        stream_id,
        force_search,
        kept_probabilities_pointer,
        sampled_codes_pointer,
        CODEBOOK_SIZE: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
        SEARCH_ROUNDS: tl.constexpr,
        SELECTED: tl.constexpr,
        TAIL_WIDTH: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_WIDTH)
        is_column = columns < CODEBOOK_SIZE
        logits = tl.load(
            logits_pointer + row * CODEBOOK_SIZE + columns, mask=is_column, other=0.0
        ).to(tl.float32)
        temperature = tl.maximum(
            tl.load(temperature_pointer + row), MINIMUM_TEMPERATURE
        )
        top_k = tl.load(top_k_pointer + row)
        top_p = tl.load(top_p_pointer + row)
        step = tl.load(step_pointer)
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
        selected_indices, selected_weights, is_kept, holds_kept_set = (
            selected_candidates(
                top_keys, row_maximum, weight_sum, top_k, top_p, INDEX_BITS
            )
        )
        uses_selected = holds_kept_set & (force_search == 0)
        if uses_selected:
            is_kept_column = (
                tl.sum(
                    (
                        (columns[:, None] == selected_indices[None, :])
                        & is_kept[None, :]
                    ).to(tl.int32),
                    axis=1,
                )
                > 0
            )
            sampled_code = race_selected(
                selected_indices, selected_weights, is_kept, seed, step, row, stream_id
            )
        else:
            threshold_key = search_threshold(
                main_keys,
                tl.math.div_rn(
                    main_weights, tl.broadcast_to(weight_sum, main_weights.shape)
                ),
                tail_keys,
                tl.math.div_rn(
                    tail_weights, tl.broadcast_to(weight_sum, tail_weights.shape)
                ),
                top_k,
                top_p,
                KEY_BITS,
                SEARCH_ROUNDS,
            )
            top_key = tl.max(top_keys, axis=0)
            is_kept_column = (main_keys >= threshold_key) | (main_keys == top_key)
            sampled_code = race_above_threshold(
                main_keys,
                main_weights,
                tail_keys,
                tail_weights,
                threshold_key,
                top_key,
                seed,
                step,
                row,
                stream_id,
                INDEX_BITS,
            )
        kept_weights = tl.where(is_kept_column, main_weights, 0.0)
        tl.store(
            kept_probabilities_pointer + row * CODEBOOK_SIZE + columns,
            kept_weights / tl.sum(kept_weights, axis=0),
            mask=is_column,
        )
        tl.store(sampled_codes_pointer + row, sampled_code.to(tl.int64))

else:
    pass

NEAR_CUT_PROBABILITY = 1e-5
NEAR_TIE_RELATIVE_RACE = 1e-6
MAXIMUM_FLAGGED_MISMATCH_FRACTION = 0.01
NEGLIGIBLE_TOTAL_VARIATION = 1e-4
SEED = 20261007
SYNTHETIC_VOCAB_SIZE = 6000
SYNTHETIC_SEMANTIC_BEGIN_ID = 1500
SYNTHETIC_EOS_TOKEN_ID = 7
UNLIMITED_TOP_K = 2**31 - 1


def require_triton_device() -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("needs a CUDA device with Triton")
    else:
        pass


def require_validated_gpu() -> None:
    require_triton_device()
    compute_capability = torch.cuda.get_device_capability()
    if compute_capability != VALIDATED_COMPUTE_CAPABILITY:
        pytest.skip(
            f"the Triton kernels are validated only on compute capability "
            f"{VALIDATED_COMPUTE_CAPABILITY}, this GPU is {compute_capability}"
        )
    else:
        pass


def sampler(step: int) -> CodebookSampler:
    return CodebookSampler(
        step=torch.tensor(step, dtype=torch.int64, device="cuda"), seed=SEED
    )


def kernel_noise(
    codebook_sampler: CodebookSampler, stream_id: int, rows: int, width: int
) -> torch.Tensor:
    noise = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    noise_kernel[(rows,)](
        codebook_sampler.seed,
        codebook_sampler.step,
        stream_id,
        noise,
        WIDTH=width,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )
    return noise


@dataclass(frozen=True, kw_only=True)
class ReferenceDraws:
    tokens: torch.Tensor
    kept_probabilities: torch.Tensor
    is_near_cut: torch.Tensor
    is_near_tie: torch.Tensor


def reference_draws(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    noise: torch.Tensor,
) -> ReferenceDraws:
    scores = logits.float() / temperature.float().clamp(
        min=MINIMUM_TEMPERATURE
    ).unsqueeze(-1)
    sorted_scores, sorted_columns = torch.sort(
        scores, dim=-1, descending=True, stable=True
    )
    cumulative_probabilities = torch.cumsum(
        torch.softmax(sorted_scores, dim=-1), dim=-1
    )
    ranks = torch.arange(scores.shape[-1], device=scores.device).unsqueeze(0)
    is_within_top_k = ranks < top_k.unsqueeze(-1)
    is_dropped = (cumulative_probabilities > top_p.float().unsqueeze(-1)) | (
        ~is_within_top_k
    )
    is_dropped[:, 0] = False
    probabilities = torch.softmax(
        sorted_scores.masked_fill(is_dropped, float("-inf")), dim=-1
    )
    race = probabilities / noise.gather(-1, sorted_columns)
    top_two_races = race.topk(2, dim=-1).values
    return ReferenceDraws(
        tokens=sorted_columns.gather(-1, race.argmax(dim=-1, keepdim=True)).squeeze(-1),
        kept_probabilities=torch.zeros_like(probabilities).scatter(
            -1, sorted_columns, probabilities
        ),
        is_near_cut=(
            (
                (cumulative_probabilities - top_p.float().unsqueeze(-1)).abs()
                < NEAR_CUT_PROBABILITY
            )
            & is_within_top_k
        ).any(dim=-1),
        is_near_tie=(top_two_races[:, 0] - top_two_races[:, 1])
        <= NEAR_TIE_RELATIVE_RACE * top_two_races[:, 0],
    )


def mixed_parameters(
    rows: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    settings = [
        (1.0, 30, 0.9),
        (0.7, 8, 1.0),
        (1.3, 32, 0.95),
        (1.0, 1, 1.0),
        (0.0, 30, 0.9),
        (1.0, 200, 0.99),
        (1.0, UNLIMITED_TOP_K, 1.0),
        (0.9, 4096, 0.97),
    ]
    order = torch.randint(0, len(settings), (rows,), generator=generator).tolist()
    return (
        torch.tensor([settings[index][0] for index in order], device="cuda"),
        torch.tensor(
            [settings[index][1] for index in order], dtype=torch.int64, device="cuda"
        ),
        torch.tensor([settings[index][2] for index in order], device="cuda"),
    )


def head_logits(rows: int, width: int, generator: torch.Generator) -> torch.Tensor:
    sharpness = torch.rand(rows, 1, generator=generator) * 4.0
    return (
        (torch.randn(rows, width, generator=generator) * sharpness)
        .to(torch.bfloat16)
        .cuda()
    )


def mismatch_problems(
    is_mismatched: torch.Tensor, is_flagged: torch.Tensor
) -> list[str]:
    unexplained_rows = torch.nonzero(is_mismatched & ~is_flagged).flatten().tolist()
    flagged_mismatch_count = int((is_mismatched & is_flagged).sum())
    allowed_count = max(1, MAXIMUM_FLAGGED_MISMATCH_FRACTION * is_mismatched.numel())
    return [
        f"row {row} differs away from the cut and ties" for row in unexplained_rows
    ] + (
        []
        if flagged_mismatch_count <= allowed_count
        else [
            f"{flagged_mismatch_count} of {int(is_flagged.sum())} rows near the cut "
            f"or a tie differ"
        ]
    )


def draw_problems(sampled: torch.Tensor, reference: ReferenceDraws) -> list[str]:
    return mismatch_problems(
        sampled != reference.tokens, reference.is_near_cut | reference.is_near_tie
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("width", [64, 1024, 4096])
@pytest.mark.parametrize("rows", [1, 37, 200])
def test_head_kernel_draws_what_the_reference_draws_with_the_same_noise(
    width: int, rows: int
) -> None:
    require_validated_gpu()
    generator = torch.Generator().manual_seed(width * 1000 + rows)
    logits = head_logits(rows, width, generator)
    temperature, top_k, top_p = mixed_parameters(rows, generator)
    stream_id = FIRST_CODEBOOK_HEAD_STREAM + 3
    codebook_sampler = sampler(11)

    sampled = sample_top_k_top_p(
        logits, temperature, top_k, top_p, codebook_sampler, stream_id
    )

    reference = reference_draws(
        logits,
        temperature,
        top_k,
        top_p,
        kernel_noise(codebook_sampler, stream_id, rows, width),
    )
    problems = draw_problems(sampled, reference)
    assert not problems, problems


def semantic_inputs(
    rows: int, codebook_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.cat(
        (
            torch.tensor([SYNTHETIC_EOS_TOKEN_ID]),
            torch.arange(
                SYNTHETIC_SEMANTIC_BEGIN_ID, SYNTHETIC_SEMANTIC_BEGIN_ID + codebook_size
            ),
        )
    ).cuda()
    sharpness = torch.rand(rows, 1, generator=generator) * 4.0
    logits = (
        torch.randn(rows, SYNTHETIC_VOCAB_SIZE, generator=generator) * sharpness
    ).to(torch.bfloat16)
    logits[:, :SYNTHETIC_EOS_TOKEN_ID] = 30.0
    return logits.float().cuda(), candidates


@pytest.mark.accelerator
@pytest.mark.parametrize("codebook_size", [64, 4096])
@pytest.mark.parametrize("rows", [1, 37, 200])
def test_semantic_kernel_draws_what_the_reference_draws_with_the_same_noise(
    codebook_size: int, rows: int
) -> None:
    require_validated_gpu()
    generator = torch.Generator().manual_seed(codebook_size * 1000 + rows)
    semantic_logits, candidates = semantic_inputs(rows, codebook_size, generator)
    temperature, top_k, top_p = mixed_parameters(rows, generator)
    candidate_logits = semantic_logits[:, candidates]
    codebook_sampler = sampler(5)
    candidate_count = candidates.shape[0]
    first_reference = reference_draws(
        candidate_logits,
        temperature,
        top_k,
        top_p,
        kernel_noise(codebook_sampler, FIRST_DRAW_STREAM, rows, candidate_count),
    )
    ras_reference = reference_draws(
        candidate_logits,
        torch.full((rows,), RAS_TEMPERATURE, device="cuda"),
        torch.full((rows,), RAS_TOP_K, dtype=torch.int64, device="cuda"),
        torch.full((rows,), RAS_TOP_P, device="cuda"),
        kernel_noise(codebook_sampler, RAS_DRAW_STREAM, rows, candidate_count),
    )
    first_tokens = candidates[first_reference.tokens]
    ras_tokens = candidates[ras_reference.tokens]
    windows = torch.randint(
        0, SYNTHETIC_VOCAB_SIZE, (rows, RAS_WINDOW_FRAMES), generator=generator
    ).cuda()
    is_repeat = torch.arange(rows, device="cuda") % 3 == 0
    windows[is_repeat, 5] = first_tokens[is_repeat]

    main_tokens, codebook_zero = sample_semantic_with_ras(
        semantic_logits,
        candidates,
        windows,
        temperature,
        top_k,
        top_p,
        SYNTHETIC_SEMANTIC_BEGIN_ID,
        codebook_size,
        codebook_sampler,
    )

    is_semantic = (first_tokens >= SYNTHETIC_SEMANTIC_BEGIN_ID) & (
        first_tokens < SYNTHETIC_SEMANTIC_BEGIN_ID + codebook_size
    )
    expected_tokens = torch.where(
        (windows == first_tokens.unsqueeze(-1)).any(dim=-1) & is_semantic,
        ras_tokens,
        first_tokens,
    )
    problems = mismatch_problems(
        main_tokens != expected_tokens,
        first_reference.is_near_cut
        | first_reference.is_near_tie
        | ras_reference.is_near_cut
        | ras_reference.is_near_tie,
    )
    assert not problems, problems
    assert torch.equal(
        codebook_zero,
        (main_tokens - SYNTHETIC_SEMANTIC_BEGIN_ID).clamp(0, codebook_size - 1),
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("band_edge", ["inside", "start", "end"])
def test_semantic_kernel_keeps_repeated_eos_and_replaces_repeated_semantic(
    band_edge: str,
) -> None:
    require_validated_gpu()
    rows, codebook_size = 64, 64
    semantic_end_id = SYNTHETIC_SEMANTIC_BEGIN_ID + codebook_size - 1
    eos_token_id, repeated_semantic_token = {
        "inside": (SYNTHETIC_EOS_TOKEN_ID, SYNTHETIC_SEMANTIC_BEGIN_ID + 5),
        "start": (SYNTHETIC_SEMANTIC_BEGIN_ID - 1, SYNTHETIC_SEMANTIC_BEGIN_ID),
        "end": (semantic_end_id + 1, semantic_end_id),
    }[band_edge]
    semantic_token_ids = torch.arange(
        SYNTHETIC_SEMANTIC_BEGIN_ID, semantic_end_id + 1, device="cuda"
    )
    candidates = torch.tensor(
        sorted([eos_token_id, *semantic_token_ids.tolist()]), device="cuda"
    )
    is_eos_row = torch.arange(rows, device="cuda") % 2 == 0
    top_token_ids = torch.where(is_eos_row, eos_token_id, repeated_semantic_token)
    semantic_logits = torch.zeros(rows, SYNTHETIC_VOCAB_SIZE, device="cuda")
    semantic_logits[:, semantic_token_ids] = torch.linspace(
        0.0, 0.3, codebook_size, device="cuda"
    )
    semantic_logits[torch.arange(rows, device="cuda"), top_token_ids] = 1.0
    windows = torch.full(
        (rows, RAS_WINDOW_FRAMES), EMPTY_RAS_SLOT, dtype=torch.int64, device="cuda"
    )
    windows[:, 3] = eos_token_id
    windows[:, 9] = repeated_semantic_token
    codebook_sampler = sampler(9)
    ras_reference = reference_draws(
        semantic_logits[:, candidates],
        torch.full((rows,), RAS_TEMPERATURE, device="cuda"),
        torch.full((rows,), RAS_TOP_K, dtype=torch.int64, device="cuda"),
        torch.full((rows,), RAS_TOP_P, device="cuda"),
        kernel_noise(codebook_sampler, RAS_DRAW_STREAM, rows, candidates.shape[0]),
    )
    ras_tokens = candidates[ras_reference.tokens]

    main_tokens, codebook_zero = sample_semantic_with_ras(
        semantic_logits,
        candidates,
        windows,
        torch.ones(rows, device="cuda"),
        torch.ones(rows, dtype=torch.int64, device="cuda"),
        torch.ones(rows, device="cuda"),
        SYNTHETIC_SEMANTIC_BEGIN_ID,
        codebook_size,
        codebook_sampler,
    )

    assert not bool((ras_reference.is_near_cut | ras_reference.is_near_tie).any())
    assert torch.equal(main_tokens, torch.where(is_eos_row, top_token_ids, ras_tokens))
    assert bool((ras_tokens[is_eos_row] != eos_token_id).any())
    assert bool((ras_tokens[~is_eos_row] != repeated_semantic_token).any())
    assert torch.equal(
        codebook_zero,
        (main_tokens - SYNTHETIC_SEMANTIC_BEGIN_ID).clamp(0, codebook_size - 1),
    )


def kept_probabilities(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    codebook_sampler: CodebookSampler,
    stream_id: int,
    force_search: bool,
    search_rounds_short_by: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = logits.shape
    index_bits = (width - 1).bit_length()
    probabilities = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    sampled_codes = torch.empty(rows, dtype=torch.int64, device="cuda")
    kept_probabilities_kernel[(rows,)](
        logits.contiguous(),
        temperature.float(),
        top_k.long(),
        top_p.float(),
        codebook_sampler.step,
        codebook_sampler.seed,
        stream_id,
        int(force_search),
        probabilities,
        sampled_codes,
        CODEBOOK_SIZE=width,
        BLOCK_WIDTH=triton.next_power_of_2(width),
        INDEX_BITS=index_bits,
        KEY_BITS=SCORE_KEY_BITS + index_bits,
        SEARCH_ROUNDS=SCORE_KEY_BITS + index_bits - search_rounds_short_by,
        SELECTED=SELECTED_CANDIDATES,
        TAIL_WIDTH=TAIL_BLOCK_WIDTH,
        MINIMUM_TEMPERATURE=MINIMUM_TEMPERATURE,
        num_warps=8,
    )
    return probabilities, sampled_codes


@pytest.mark.accelerator
@pytest.mark.parametrize("force_search", [False, True])
def test_kept_probabilities_follow_the_rule_on_both_branches(
    force_search: bool,
) -> None:
    require_validated_gpu()
    rows, width = 300, 1024
    generator = torch.Generator().manual_seed(3)
    logits = head_logits(rows, width, generator)
    temperature, top_k, top_p = mixed_parameters(rows, generator)
    codebook_sampler = sampler(2)
    stream_id = FIRST_CODEBOOK_HEAD_STREAM

    probabilities, sampled_codes = kept_probabilities(
        logits, temperature, top_k, top_p, codebook_sampler, stream_id, force_search, 0
    )

    reference = reference_draws(
        logits,
        temperature,
        top_k,
        top_p,
        kernel_noise(codebook_sampler, stream_id, rows, width),
    )
    is_away_from_cut = ~reference.is_near_cut
    has_same_kept_set = ((probabilities > 0) == (reference.kept_probabilities > 0)).all(
        dim=-1
    )
    assert bool(has_same_kept_set[is_away_from_cut].all())
    torch.testing.assert_close(
        probabilities[is_away_from_cut],
        reference.kept_probabilities[is_away_from_cut].float(),
        rtol=1e-5,
        atol=1e-7,
    )
    total_variation = 0.5 * (
        (probabilities - reference.kept_probabilities.float()).abs().sum(dim=-1)
    )
    problems = mismatch_problems(
        total_variation > NEGLIGIBLE_TOTAL_VARIATION, reference.is_near_cut
    )
    assert not problems, problems
    production_codes = sample_top_k_top_p(
        logits, temperature, top_k, top_p, codebook_sampler, stream_id
    )
    problems = mismatch_problems(
        sampled_codes != production_codes, reference.is_near_cut | reference.is_near_tie
    )
    assert not problems, problems


@pytest.mark.accelerator
def test_tied_and_signed_zero_scores_split_by_column() -> None:
    require_validated_gpu()
    logits = torch.full((2, 64), -3.0, dtype=torch.float32)
    logits[0, :40] = 0.0
    logits[0, 1::2] = -0.0
    logits[0, 40:] = -0.5
    logits[1, :40] = 2.0
    logits = logits.to(torch.bfloat16).cuda()
    parameters = (
        torch.ones(2, device="cuda"),
        torch.full((2,), 33, dtype=torch.int64, device="cuda"),
        torch.ones(2, device="cuda"),
    )
    codebook_sampler = sampler(0)

    probabilities, _ = kept_probabilities(
        logits, *parameters, codebook_sampler, FIRST_CODEBOOK_HEAD_STREAM, True, 0
    )
    reference = reference_draws(
        logits,
        *parameters,
        kernel_noise(codebook_sampler, FIRST_CODEBOOK_HEAD_STREAM, 2, 64),
    )

    for row in range(2):
        assert torch.nonzero(probabilities[row] > 0).flatten().tolist() == list(
            range(33)
        )
        assert torch.equal(
            probabilities[row] > 0, reference.kept_probabilities[row] > 0
        )


@pytest.mark.accelerator
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rows_stay_in_range_and_leave_other_rows_alone(
    bad_value: float,
) -> None:
    require_validated_gpu()
    rows, width = 16, 4096
    generator = torch.Generator().manual_seed(9)
    logits = head_logits(rows, width, generator)
    temperature, top_k, top_p = mixed_parameters(rows, generator)
    codebook_sampler = sampler(4)
    clean = sample_top_k_top_p(logits, temperature, top_k, top_p, codebook_sampler, 2)
    bad_logits = logits.clone()
    bad_logits[3, ::7] = bad_value
    bad_logits[11] = bad_value

    sampled = sample_top_k_top_p(
        bad_logits, temperature, top_k, top_p, codebook_sampler, 2
    )

    assert bool(((sampled >= 0) & (sampled < width)).all())
    other_rows = torch.ones(rows, dtype=torch.bool, device="cuda")
    other_rows[[3, 11]] = False
    assert torch.equal(sampled[other_rows], clean[other_rows])


@pytest.mark.accelerator
def test_graph_replays_read_the_step_and_parameters_at_replay() -> None:
    require_validated_gpu()
    rows, width, codebook_size = 64, 4096, 4096
    generator = torch.Generator().manual_seed(17)
    logits = head_logits(rows, width, generator)
    semantic_logits, candidates = semantic_inputs(rows, codebook_size, generator)
    windows = torch.full(
        (rows, RAS_WINDOW_FRAMES), -1, dtype=torch.int64, device="cuda"
    )
    temperature, top_k, top_p = (
        tensor.clone() for tensor in mixed_parameters(rows, generator)
    )
    codebook_sampler = sampler(0)

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        codebook_sampler.step.add_(1)
        main_tokens, _ = sample_semantic_with_ras(
            semantic_logits,
            candidates,
            windows,
            temperature,
            top_k,
            top_p,
            SYNTHETIC_SEMANTIC_BEGIN_ID,
            codebook_size,
            codebook_sampler,
        )
        return main_tokens, sample_top_k_top_p(
            logits,
            temperature,
            top_k,
            top_p,
            codebook_sampler,
            FIRST_CODEBOOK_HEAD_STREAM,
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        step()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed_main_tokens, replayed_codes = step()
    for replay_index in range(4):
        replay_step = 10 * replay_index
        codebook_sampler.step.fill_(replay_step)
        new_temperature, new_top_k, new_top_p = mixed_parameters(rows, generator)
        temperature.copy_(new_temperature)
        top_k.copy_(new_top_k)
        top_p.copy_(new_top_p)

        graph.replay()

        replay_main_tokens = replayed_main_tokens.clone()
        replay_codes = replayed_codes.clone()
        assert int(codebook_sampler.step) == replay_step + 1
        codebook_sampler.step.fill_(replay_step)
        eager_main_tokens, eager_codes = step()
        assert torch.equal(replay_main_tokens, eager_main_tokens)
        assert torch.equal(replay_codes, eager_codes)
        codebook_sampler.step.fill_(replay_step + 1)
        reference = reference_draws(
            logits,
            temperature,
            top_k,
            top_p,
            kernel_noise(codebook_sampler, FIRST_CODEBOOK_HEAD_STREAM, rows, width),
        )
        problems = draw_problems(replay_codes, reference)
        assert not problems, problems


@pytest.mark.accelerator
def test_streams_and_steps_draw_independent_noise() -> None:
    require_validated_gpu()
    rows, width = 4096, 64
    pair_count = rows // 2
    logits = torch.zeros(rows, width, dtype=torch.bfloat16, device="cuda")
    parameters = (
        torch.ones(rows, device="cuda"),
        torch.full((rows,), width, dtype=torch.int64, device="cuda"),
        torch.ones(rows, device="cuda"),
    )
    collision_probability = 1.0 / width

    def collisions(first: torch.Tensor, second: torch.Tensor) -> int:
        return int((first == second).sum())

    head_codes = [
        sample_top_k_top_p(
            logits, *parameters, sampler(step), FIRST_CODEBOOK_HEAD_STREAM + head
        )
        for step, head in ((1, 0), (1, 1), (2, 0), (3, 0))
    ]
    same_stream_and_step = sample_top_k_top_p(
        logits, *parameters, sampler(1), FIRST_CODEBOOK_HEAD_STREAM
    )

    assert torch.equal(same_stream_and_step, head_codes[0])
    expected = rows * collision_probability
    tolerance = 5 * (rows * collision_probability * (1 - collision_probability)) ** 0.5
    for first, second in ((0, 1), (0, 2), (2, 3)):
        assert abs(collisions(head_codes[first], head_codes[second]) - expected) <= (
            tolerance
        ), (first, second)
    row_pairs = collisions(head_codes[0][:pair_count], head_codes[0][pair_count:])
    assert (
        abs(row_pairs - pair_count * collision_probability)
        <= 5 * (pair_count * collision_probability * (1 - collision_probability)) ** 0.5
    )
    for codes in head_codes:
        frequencies = torch.bincount(codes, minlength=width).double() / rows
        assert (
            float((frequencies - collision_probability).abs().max())
            < 6 * (collision_probability / rows) ** 0.5
        )


def test_unsupported_devices_and_widths_keep_the_pytorch_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert triton_sampling_unsupported_reason(torch.device("cpu"), 4096, 4097)
    if HAS_TRITON and torch.cuda.is_available():
        cuda = torch.device("cuda")
        with monkeypatch.context() as validated_gpu:
            validated_gpu.setattr(
                torch.cuda,
                "get_device_capability",
                lambda device=None: VALIDATED_COMPUTE_CAPABILITY,
            )
            assert triton_sampling_unsupported_reason(cuda, 4096, 4097) is None
            assert triton_sampling_unsupported_reason(cuda, 64, 65) is None
            assert triton_sampling_unsupported_reason(cuda, 16, 17)
            assert triton_sampling_unsupported_reason(cuda, 8192, 8193)
            assert triton_sampling_unsupported_reason(cuda, 4096, 4096 + 40)
    else:
        pass


@pytest.mark.accelerator
def test_this_gpu_takes_the_triton_kernels_only_if_validated() -> None:
    require_triton_device()
    major, minor = torch.cuda.get_device_capability()

    reason = triton_sampling_unsupported_reason(torch.device("cuda"), 4096, 4097)

    if (major, minor) == VALIDATED_COMPUTE_CAPABILITY:
        assert reason is None
    else:
        assert reason == f"untested GPU sm_{major}{minor}"


def patch_cuda_platform(
    monkeypatch: pytest.MonkeyPatch, compute_capability: tuple[int, int]
) -> None:
    monkeypatch.setattr(sampling_kernels, "HAS_TRITON", True)
    monkeypatch.setattr(
        sampling_kernels, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: compute_capability
    )


@pytest.mark.parametrize(
    ("codebook_size", "candidate_count"), [(32, 4097), (4096, 32), (32, 33)]
)
def test_rows_of_32_columns_keep_the_pytorch_path(
    monkeypatch: pytest.MonkeyPatch, codebook_size: int, candidate_count: int
) -> None:
    patch_cuda_platform(monkeypatch, (9, 0))
    cuda = torch.device("cuda")

    reason = triton_sampling_unsupported_reason(cuda, codebook_size, candidate_count)

    assert reason is not None and "tied logits" in reason
    assert triton_sampling_unsupported_reason(cuda, 33, 33) is None


def test_compute_capability_9_0_takes_the_triton_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_cuda_platform(monkeypatch, (9, 0))

    assert triton_sampling_unsupported_reason(torch.device("cuda"), 4096, 4097) is None


@pytest.mark.parametrize(
    ("compute_capability", "expected_reason"),
    [
        ((8, 6), "untested GPU sm_86"),
        ((8, 9), "untested GPU sm_89"),
        ((10, 0), "untested GPU sm_100"),
        ((12, 0), "untested GPU sm_120"),
    ],
)
def test_other_compute_capabilities_keep_the_pytorch_path(
    monkeypatch: pytest.MonkeyPatch,
    compute_capability: tuple[int, int],
    expected_reason: str,
) -> None:
    patch_cuda_platform(monkeypatch, compute_capability)

    reason = triton_sampling_unsupported_reason(torch.device("cuda"), 4096, 4097)

    assert reason == expected_reason


def tied_columns(width: int) -> list[list[int]]:
    return [[0, 1], [width - 2, width - 1], [5, 17], list(range(width))]


@pytest.mark.accelerator
@pytest.mark.parametrize("width", [33, 64, 4096])
def test_head_kernel_breaks_top_k_one_ties_like_the_pytorch_sampler(
    width: int,
) -> None:
    require_validated_gpu()
    rows = tied_columns(width)
    logits = torch.zeros(len(rows), width, device="cuda")
    for row, columns in enumerate(rows):
        logits[row, columns] = 1.0
    parameters = (
        torch.ones(len(rows), device="cuda"),
        torch.ones(len(rows), dtype=torch.int64, device="cuda"),
        torch.ones(len(rows), device="cuda"),
    )

    sampled = sample_top_k_top_p(
        logits, *parameters, sampler(0), FIRST_CODEBOOK_HEAD_STREAM
    )

    assert torch.equal(
        sampled,
        sample_top_k_top_p(logits, *parameters, None, FIRST_CODEBOOK_HEAD_STREAM),
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("candidate_count", [33, 4097])
def test_semantic_kernel_breaks_top_k_one_ties_like_the_pytorch_sampler(
    candidate_count: int,
) -> None:
    require_validated_gpu()
    codebook_size = candidate_count - 1
    candidates = torch.cat(
        (
            torch.tensor([SYNTHETIC_EOS_TOKEN_ID]),
            torch.arange(
                SYNTHETIC_SEMANTIC_BEGIN_ID, SYNTHETIC_SEMANTIC_BEGIN_ID + codebook_size
            ),
        )
    ).cuda()
    rows = tied_columns(candidate_count)
    semantic_logits = torch.zeros(len(rows), SYNTHETIC_VOCAB_SIZE, device="cuda")
    for row, columns in enumerate(rows):
        semantic_logits[row, candidates[columns]] = 1.0
    windows = torch.full((len(rows), RAS_WINDOW_FRAMES), EMPTY_RAS_SLOT, device="cuda")
    parameters = (
        torch.ones(len(rows), device="cuda"),
        torch.ones(len(rows), dtype=torch.int64, device="cuda"),
        torch.ones(len(rows), device="cuda"),
    )

    sampled = sample_semantic_with_ras(
        semantic_logits,
        candidates,
        windows,
        *parameters,
        SYNTHETIC_SEMANTIC_BEGIN_ID,
        codebook_size,
        sampler(0),
    )
    expected = sample_semantic_with_ras(
        semantic_logits,
        candidates,
        windows,
        *parameters,
        SYNTHETIC_SEMANTIC_BEGIN_ID,
        codebook_size,
        None,
    )

    assert torch.equal(sampled[0], expected[0])
    assert torch.equal(sampled[1], expected[1])
