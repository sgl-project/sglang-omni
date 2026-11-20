# SPDX-License-Identifier: Apache-2.0
"""Drama 3 sampling: top-p cut, top-k, semantic bias, repetition-aware resampling."""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.sampling import (
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    sample_semantic_with_ras,
    sample_top_k_top_p,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_CODEBOOK_HEAD_STREAM,
    FIRST_DRAW_STREAM,
    RAS_DRAW_STREAM,
)

DRAW_COUNT = 2000
TEXT_VOCAB_SIZE = 48
SEMANTIC_BEGIN_ID = 8
CODEBOOK_SIZE = 32
EOS_TOKEN_ID = 2
# No token id is negative, so this window never matches.
EMPTY_WINDOW_TOKEN = -1


def repeated(values: list[float], rows: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).repeat(rows, 1)


def request_parameters(
    rows: int, temperature: float, top_k: int, top_p: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((rows,), temperature),
        torch.full((rows,), top_k, dtype=torch.int64),
        torch.full((rows,), top_p),
    )


def semantic_bias() -> torch.Tensor:
    bias = torch.full((TEXT_VOCAB_SIZE,), float("-inf"))
    bias[SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + CODEBOOK_SIZE] = 0.0
    bias[EOS_TOKEN_ID] = 0.0
    return bias


def candidate_token_ids() -> torch.Tensor:
    return torch.cat(
        (
            torch.tensor([EOS_TOKEN_ID]),
            torch.arange(SEMANTIC_BEGIN_ID, SEMANTIC_BEGIN_ID + CODEBOOK_SIZE),
        )
    )


def test_top_k_one_takes_the_argmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 32)

    sampled = sample_top_k_top_p(
        logits, *request_parameters(64, 1.0, 1, 1.0), None, FIRST_CODEBOOK_HEAD_STREAM
    )

    assert torch.equal(sampled, logits.argmax(dim=-1))


@pytest.mark.parametrize(("top_p", "kept_tokens"), [(0.8, {0, 1}), (0.79, {0})])
def test_top_p_drops_the_token_that_crosses_p(
    top_p: float, kept_tokens: set[int]
) -> None:
    """Cumulative probability counts the token itself; at 0.79 only the top stays."""
    torch.manual_seed(0)
    logits = repeated([math.log(0.5), math.log(0.3), math.log(0.2)], DRAW_COUNT)

    sampled = sample_top_k_top_p(
        logits,
        *request_parameters(DRAW_COUNT, 1.0, 3, top_p),
        None,
        FIRST_CODEBOOK_HEAD_STREAM,
    )

    assert set(sampled.tolist()) == kept_tokens


@pytest.mark.parametrize("top_p", [0.9, 0.95])
@pytest.mark.parametrize("is_repeat", [False, True])
@pytest.mark.parametrize(
    ("probabilities", "can_sample_eos"),
    [
        ([0.8917, 0.1065, 0.0018], True),
        ([0.97, 0.0282, 0.0018], False),
    ],
)
def test_semantic_nucleus_keeps_crossing_eos_without_exempting_it_from_top_p(
    top_p: float, is_repeat: bool, probabilities: list[float], can_sample_eos: bool
) -> None:
    torch.manual_seed(101)
    repeated_token_id = SEMANTIC_BEGIN_ID + 5
    logits = torch.full((DRAW_COUNT, TEXT_VOCAB_SIZE), float("-inf"))
    logits[:, repeated_token_id] = math.log(probabilities[0])
    logits[:, EOS_TOKEN_ID] = math.log(probabilities[1])
    logits[:, repeated_token_id + 1] = math.log(probabilities[2])
    window = torch.full((DRAW_COUNT, 16), EMPTY_WINDOW_TOKEN)
    if is_repeat:
        window[:, 0] = repeated_token_id
    else:
        pass

    main_tokens, codebook_zero = sample_semantic_with_ras(
        logits,
        candidate_token_ids(),
        window,
        *request_parameters(DRAW_COUNT, 1.0, 30, top_p),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    expected_tokens = (
        {repeated_token_id, EOS_TOKEN_ID} if can_sample_eos else {repeated_token_id}
    )
    assert set(main_tokens.tolist()) == expected_tokens
    assert torch.equal(
        codebook_zero, (main_tokens - SEMANTIC_BEGIN_ID).clamp(0, CODEBOOK_SIZE - 1)
    )


def test_top_k_and_temperature_limit_the_candidates() -> None:
    torch.manual_seed(0)
    logits = repeated([2.0, 1.9, 1.8, 1.7, 1.6], DRAW_COUNT)

    sampled = sample_top_k_top_p(
        logits,
        *request_parameters(DRAW_COUNT, 5.0, 2, 1.0),
        None,
        FIRST_CODEBOOK_HEAD_STREAM,
    )

    assert set(sampled.tolist()) == {0, 1}


def test_sampling_keeps_inputs_and_returns_int64_ids() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 16, dtype=torch.bfloat16)
    temperature, top_k, top_p = request_parameters(4, 0.7, 5, 0.9)
    originals = [tensor.clone() for tensor in (logits, temperature, top_k, top_p)]

    sampled = sample_top_k_top_p(
        logits, temperature, top_k, top_p, None, FIRST_CODEBOOK_HEAD_STREAM
    )

    assert sampled.dtype == torch.int64
    assert sampled.device == logits.device
    assert sampled.shape == (4,)
    for original, tensor in zip(originals, (logits, temperature, top_k, top_p)):
        assert torch.equal(original, tensor)


def ras_logits(rows: int) -> torch.Tensor:
    """Near-flat semantic logits with a clear top token, plus a huge non-band logit."""
    logits = torch.zeros(rows, TEXT_VOCAB_SIZE)
    logits[:, SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + CODEBOOK_SIZE] = torch.linspace(
        0.0, 0.3, CODEBOOK_SIZE
    )
    logits[:, SEMANTIC_BEGIN_ID + 5] = 3.0
    logits[:, 40] = 50.0
    return logits


def test_candidate_columns_match_the_same_row_taken_from_the_full_vocabulary() -> None:
    rows = 4
    logits = ras_logits(rows)
    candidates = candidate_token_ids()
    window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)
    parameters = request_parameters(rows, 1.0, 30, 0.9)
    torch.manual_seed(1)
    from_vocabulary, from_vocabulary_codes = sample_semantic_with_ras(
        logits,
        candidates,
        window,
        *parameters,
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )
    torch.manual_seed(1)
    from_columns, from_column_codes = sample_semantic_with_ras(
        logits[:, candidates],
        candidates,
        window,
        *parameters,
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    assert torch.equal(from_vocabulary, from_columns)
    assert torch.equal(from_vocabulary_codes, from_column_codes)


def test_ras_takes_the_second_draw_exactly_when_a_semantic_first_draw_repeats() -> None:
    """Both draws happen in order; the fixed-parameter one wins only on a window hit."""
    rows = 8
    logits = ras_logits(rows)
    temperature, top_k, top_p = request_parameters(rows, 1.0, 1, 1.0)
    candidates = candidate_token_ids()
    candidate_logits = logits[:, candidates]
    torch.manual_seed(3)
    first_draw = candidates[
        sample_top_k_top_p(
            candidate_logits,
            temperature,
            top_k,
            top_p,
            None,
            FIRST_DRAW_STREAM,
            keep_top_p_crossing_token=True,
        )
    ]
    second_draw = candidates[
        sample_top_k_top_p(
            candidate_logits,
            *request_parameters(rows, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P),
            None,
            RAS_DRAW_STREAM,
            keep_top_p_crossing_token=True,
        )
    ]
    window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)
    window[::2, 7] = first_draw[::2]
    is_repeat = torch.tensor([True, False] * (rows // 2))

    torch.manual_seed(3)
    main_tokens, codebook_zero = sample_semantic_with_ras(
        logits,
        candidate_token_ids(),
        window,
        temperature,
        top_k,
        top_p,
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    assert torch.equal(first_draw, torch.full((rows,), SEMANTIC_BEGIN_ID + 5))
    assert torch.equal(main_tokens, torch.where(is_repeat, second_draw, first_draw))
    assert not torch.equal(second_draw[::2], first_draw[::2])
    assert torch.equal(codebook_zero, main_tokens - SEMANTIC_BEGIN_ID)


@pytest.mark.parametrize("band_edge", ["inside", "start", "end"])
def test_ras_keeps_repeated_eos_and_replaces_repeated_semantic(band_edge: str) -> None:
    rows = 8
    semantic_end_id = SEMANTIC_BEGIN_ID + CODEBOOK_SIZE - 1
    eos_token_id, repeated_semantic_token = {
        "inside": (EOS_TOKEN_ID, SEMANTIC_BEGIN_ID + 5),
        "start": (SEMANTIC_BEGIN_ID - 1, SEMANTIC_BEGIN_ID),
        "end": (semantic_end_id + 1, semantic_end_id),
    }[band_edge]
    is_eos_row = torch.arange(rows) % 2 == 0
    top_token_ids = torch.where(is_eos_row, eos_token_id, repeated_semantic_token)
    logits = torch.zeros(rows, TEXT_VOCAB_SIZE)
    logits[:, SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + CODEBOOK_SIZE] = torch.linspace(
        0.0, 0.3, CODEBOOK_SIZE
    )
    logits[torch.arange(rows), top_token_ids] = 1.0
    temperature, top_k, top_p = request_parameters(rows, 1.0, 1, 1.0)
    candidates = torch.tensor(
        sorted([eos_token_id, *range(SEMANTIC_BEGIN_ID, semantic_end_id + 1)])
    )
    candidate_logits = logits[:, candidates]
    torch.manual_seed(3)
    first_draw = candidates[
        sample_top_k_top_p(
            candidate_logits,
            temperature,
            top_k,
            top_p,
            None,
            FIRST_DRAW_STREAM,
            keep_top_p_crossing_token=True,
        )
    ]
    second_draw = candidates[
        sample_top_k_top_p(
            candidate_logits,
            *request_parameters(rows, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P),
            None,
            RAS_DRAW_STREAM,
            keep_top_p_crossing_token=True,
        )
    ]
    window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)
    window[:, 3] = eos_token_id
    window[:, 9] = repeated_semantic_token

    torch.manual_seed(3)
    main_tokens, codebook_zero = sample_semantic_with_ras(
        logits,
        candidates,
        window,
        temperature,
        top_k,
        top_p,
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    assert torch.equal(first_draw, top_token_ids)
    assert torch.equal(main_tokens, torch.where(is_eos_row, first_draw, second_draw))
    assert bool((second_draw[is_eos_row] != eos_token_id).any())
    assert bool((second_draw[~is_eos_row] != repeated_semantic_token).any())
    assert torch.equal(
        codebook_zero, (main_tokens - SEMANTIC_BEGIN_ID).clamp(0, CODEBOOK_SIZE - 1)
    )


def test_ras_draws_the_same_random_numbers_with_or_without_a_hit() -> None:
    rows = 4
    logits = ras_logits(rows)
    parameters = request_parameters(rows, 1.0, 1, 1.0)
    hit_window = torch.full((rows, 16), SEMANTIC_BEGIN_ID + 5)
    miss_window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)
    next_numbers = []
    for window in (hit_window, miss_window):
        torch.manual_seed(5)
        sample_semantic_with_ras(
            logits,
            candidate_token_ids(),
            window,
            *parameters,
            SEMANTIC_BEGIN_ID,
            CODEBOOK_SIZE,
            None,
        )
        next_numbers.append(torch.rand(3))

    assert torch.equal(next_numbers[0], next_numbers[1])


def test_semantic_sampling_allows_only_the_band_and_eos() -> None:
    torch.manual_seed(0)
    rows = DRAW_COUNT
    logits = torch.zeros(rows, TEXT_VOCAB_SIZE)
    logits[:, 40] = 50.0
    window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)

    main_tokens, codebook_zero = sample_semantic_with_ras(
        logits,
        candidate_token_ids(),
        window,
        *request_parameters(rows, 1.0, TEXT_VOCAB_SIZE, 1.0),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )
    logits[:, EOS_TOKEN_ID] = 9.0
    eos_tokens, eos_codebook_zero = sample_semantic_with_ras(
        logits,
        candidate_token_ids(),
        window,
        *request_parameters(rows, 1.0, 1, 1.0),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    allowed = set(range(SEMANTIC_BEGIN_ID, SEMANTIC_BEGIN_ID + CODEBOOK_SIZE)) | {
        EOS_TOKEN_ID
    }
    assert set(main_tokens.tolist()) <= allowed
    assert torch.equal(codebook_zero, (main_tokens - SEMANTIC_BEGIN_ID).clamp(min=0))
    assert set(eos_tokens.tolist()) == {EOS_TOKEN_ID}
    assert set(eos_codebook_zero.tolist()) == {0}


def test_ras_window_must_hold_sixteen_frames() -> None:
    logits = torch.zeros(1, TEXT_VOCAB_SIZE)

    with pytest.raises(ValueError, match="16"):
        sample_semantic_with_ras(
            logits,
            candidate_token_ids(),
            torch.full((1, 8), EMPTY_WINDOW_TOKEN),
            *request_parameters(1, 1.0, 1, 1.0),
            SEMANTIC_BEGIN_ID,
            CODEBOOK_SIZE,
            None,
        )


def test_top_k_one_takes_the_token_the_biased_full_row_takes() -> None:
    torch.manual_seed(0)
    rows = 256
    logits = torch.randn(rows, TEXT_VOCAB_SIZE)
    parameters = request_parameters(rows, 1.0, 1, 1.0)

    main_tokens, _ = sample_semantic_with_ras(
        logits,
        candidate_token_ids(),
        torch.full((rows, 16), EMPTY_WINDOW_TOKEN),
        *parameters,
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    assert torch.equal(
        main_tokens,
        sample_top_k_top_p(
            logits + semantic_bias(), *parameters, None, FIRST_DRAW_STREAM
        ),
    )


@pytest.mark.accelerator
def test_tied_top_tokens_resolve_as_in_the_biased_full_row_on_the_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    torch.manual_seed(0)
    rows = 64
    vocab_size, semantic_begin_id, codebook_size, eos_token_id = (
        150000,
        120000,
        4096,
        100000,
    )
    candidates = torch.cat(
        (
            torch.tensor([eos_token_id]),
            torch.arange(semantic_begin_id, semantic_begin_id + codebook_size),
        )
    ).cuda()
    bias = torch.full((vocab_size,), float("-inf"), device="cuda")
    bias[candidates] = 0.0
    logits = torch.randn(rows, vocab_size, device="cuda").to(torch.bfloat16)
    top_logits = logits[:, candidates].amax(dim=-1)
    for token_id in (eos_token_id, semantic_begin_id + 5, semantic_begin_id + 900):
        logits[:, token_id] = top_logits
    parameters = tuple(
        tensor.cuda() for tensor in request_parameters(rows, 1.0, 1, 1.0)
    )

    main_tokens, _ = sample_semantic_with_ras(
        logits,
        candidates,
        torch.full((rows, 16), EMPTY_WINDOW_TOKEN, device="cuda"),
        *parameters,
        semantic_begin_id,
        codebook_size,
        None,
    )

    assert torch.equal(
        main_tokens,
        sample_top_k_top_p(logits.float() + bias, *parameters, None, FIRST_DRAW_STREAM),
    )
    assert set(main_tokens.tolist()) == {eos_token_id}


def rule_probabilities(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    probabilities = torch.softmax(logits.double() / temperature, dim=-1)
    order = torch.argsort(probabilities, descending=True)
    kept = torch.zeros_like(probabilities, dtype=torch.bool)
    cumulative_probability = 0.0
    for rank, column in enumerate(order.tolist()):
        if rank > 0 and (rank >= top_k or cumulative_probability > top_p):
            break
        else:
            kept[column] = True
            cumulative_probability += float(probabilities[column])
    kept_probabilities = probabilities * kept
    return kept_probabilities / kept_probabilities.sum()


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "is_ras_draw"),
    [
        pytest.param(1.0, 30, 0.9, False, id="request-1.0-30-0.9"),
        pytest.param(0.7, 8, 1.0, False, id="request-0.7-8-1.0"),
        pytest.param(1.0, 1, 1.0, True, id="ras-draw"),
    ],
)
def test_semantic_draws_follow_the_top_k_top_p_rule(
    temperature: float, top_k: int, top_p: float, is_ras_draw: bool
) -> None:
    torch.manual_seed(0)
    draw_count = 40000
    logits = torch.full((TEXT_VOCAB_SIZE,), 30.0)
    candidates = candidate_token_ids()
    logits[candidates] = torch.randn(candidates.numel()) * 1.5
    window = torch.full((draw_count, 16), EMPTY_WINDOW_TOKEN)
    if is_ras_draw:
        window[:, 0] = int(candidates[logits[candidates].argmax()])
        expected = rule_probabilities(
            logits[candidates], RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P
        )
    else:
        expected = rule_probabilities(logits[candidates], temperature, top_k, top_p)

    main_tokens, _ = sample_semantic_with_ras(
        logits.repeat(draw_count, 1),
        candidates,
        window,
        *request_parameters(draw_count, temperature, top_k, top_p),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        None,
    )

    columns = torch.searchsorted(candidates, main_tokens)
    assert torch.equal(candidates[columns], main_tokens)
    frequencies = torch.bincount(columns, minlength=candidates.numel()).double()
    frequencies /= draw_count
    standard_errors = (expected * (1 - expected) / draw_count).sqrt()
    assert bool(((frequencies - expected).abs() <= 5 * standard_errors + 1e-9).all())
    assert bool((frequencies[expected == 0] == 0).all())


def test_sampling_imports_and_samples_on_pytorch_without_triton() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins

                original_import = builtins.__import__

                def without_triton(name, *args, **kwargs):
                    if name == "triton" or name.startswith("triton."):
                        raise ModuleNotFoundError(name, name=name)
                    return original_import(name, *args, **kwargs)

                builtins.__import__ = without_triton

                import torch

                from sglang_omni.models.fishaudio_drama3 import sampling
                from sglang_omni.models.fishaudio_drama3 import sampling_kernels

                assert not sampling_kernels.HAS_TRITON
                assert (
                    sampling_kernels.triton_sampling_unsupported_reason(
                        torch.device("cpu"), 64, 65
                    )
                    == "Triton is not installed"
                )
                rows = 4
                parameters = (
                    torch.ones(rows),
                    torch.full((rows,), 30),
                    torch.full((rows,), 0.9),
                )
                head_codes = sampling.sample_top_k_top_p(
                    torch.randn(rows, 64),
                    *parameters,
                    None,
                    sampling_kernels.FIRST_CODEBOOK_HEAD_STREAM,
                )
                assert head_codes.dtype == torch.int64
                assert bool(((head_codes >= 0) & (head_codes < 64)).all())
                candidates = torch.cat((torch.tensor([2]), torch.arange(8, 72)))
                main_tokens, codebook_zero = sampling.sample_semantic_with_ras(
                    torch.randn(rows, 80),
                    candidates,
                    torch.full((rows, sampling.RAS_WINDOW_FRAMES), -1),
                    *parameters,
                    8,
                    64,
                    None,
                )
                assert bool(torch.isin(main_tokens, candidates).all())
                assert torch.equal(codebook_zero, (main_tokens - 8).clamp(0, 63))
                """
            ),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
