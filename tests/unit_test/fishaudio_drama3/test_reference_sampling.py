# SPDX-License-Identifier: Apache-2.0
"""Drama 3 reference sampling: the keep rule, epsilon draws, the repetition check and
residual codes inside the codec's codebooks."""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.fishaudio_drama3 import audio_decoder as audio_decoder_module
from sglang_omni.models.fishaudio_drama3.audio_decoder import (
    CODEBOOK_TOP_K,
    AudioDecoder,
)
from sglang_omni.models.fishaudio_drama3.decode_state import ras_window
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.reference_sampling import (
    RESIDUAL_CODEBOOK_SIZE,
    sample_main_tokens,
    sample_residual_codes,
    truncate_candidates,
    truncated_sample,
)
from sglang_omni.models.fishaudio_drama3.sampling import (
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
)
from tests.unit_test.fishaudio_drama3.dense_attention import DenseAttention
from tests.unit_test.fishaudio_drama3.test_audio_decoder import torch_fused_op_backend
from tests.unit_test.fishaudio_drama3.test_hf_config import fish_config_dict

EOS_TOKEN_ID = 2
SEMANTIC_BEGIN_ID = 8
CODEBOOK_SIZE = 64
# eos and the semantic tokens, ascending, as the model's candidate ids.
CANDIDATE_TOKEN_IDS = torch.tensor(
    [EOS_TOKEN_ID, *range(SEMANTIC_BEGIN_ID, SEMANTIC_BEGIN_ID + CODEBOOK_SIZE)]
)
EMPTY_WINDOW_TOKEN = -1
WINDOW_FRAMES = 16
# The candidate count of fishaudio/drama-3: eos and 4096 semantic tokens.
DRAMA3_CANDIDATE_COUNT = 4097
HEAD_WIDTH = 4096
# Probabilities of six columns, listed by column; descending they are columns
# 3, 0, 5, 1, 4, 2 with cumulative sums 0.40, 0.65, 0.80, 0.90, 0.96, 1.00.
SIX_COLUMN_PROBABILITIES = [0.25, 0.10, 0.04, 0.40, 0.06, 0.15]
DESCENDING_COLUMNS = [3, 0, 5, 1, 4, 2]


def rows_of(value: float, row_count: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((row_count,), value, dtype=dtype)


def kept_columns(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    epsilon_cutoff: float,
    legal_column_count: int | None = None,
) -> set[int]:
    """The columns of a one-row logits tensor the rules keep."""
    _, sorted_columns, is_kept = truncate_candidates(
        logits.unsqueeze(0),
        rows_of(temperature, 1, torch.float32),
        rows_of(top_k, 1, torch.int64),
        rows_of(top_p, 1, torch.float32),
        rows_of(epsilon_cutoff, 1, torch.float32),
        legal_column_count,
    )
    return set(sorted_columns[is_kept].tolist())


def rule_probabilities(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    epsilon_cutoff: float,
    legal_column_count: int | None = None,
) -> torch.Tensor:
    """Float64 kept probabilities of one row, renormalised, computed rank by rank."""
    probabilities = torch.softmax(logits.double() / max(temperature, 1e-5), dim=-1)
    order = torch.argsort(probabilities, descending=True, stable=True).tolist()
    is_kept = torch.zeros_like(probabilities, dtype=torch.bool)
    cumulative_probability = 0.0
    for rank, column in enumerate(order):
        probability = float(probabilities[column])
        cumulative_probability += probability
        is_kept[column] = (top_k < 1 or rank < top_k) and (
            rank == 0
            or (
                (top_p >= 1.0 or cumulative_probability <= top_p)
                and probability >= epsilon_cutoff
            )
        )
    if legal_column_count is not None:
        is_kept[legal_column_count:] = False
        most_probable_legal = next(
            column for column in order if column < legal_column_count
        )
        is_kept[most_probable_legal] = True
    kept_probabilities = probabilities * is_kept
    return kept_probabilities / kept_probabilities.sum()


def frequencies_of(columns: torch.Tensor, column_count: int) -> torch.Tensor:
    return torch.bincount(columns, minlength=column_count).double() / columns.numel()


def assert_follows(
    frequencies: torch.Tensor, expected: torch.Tensor, draw_count: int
) -> None:
    """Every frequency within five standard errors of its kept probability, and no
    draw outside the kept set."""
    standard_errors = (expected * (1 - expected) / draw_count).sqrt()
    assert bool(((frequencies - expected).abs() <= 5 * standard_errors + 1e-9).all())
    assert bool((frequencies[expected == 0] == 0).all())


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "epsilon_cutoff", "kept_ranks"),
    [
        pytest.param(1.0, -1, 1.0, 0.0, 6, id="every-rule-off"),
        pytest.param(1.0, 0, 1.0, 0.0, 6, id="top-k-zero-is-off"),
        pytest.param(1.0, 2, 1.0, 0.0, 2, id="top-k"),
        pytest.param(1.0, -1, 0.85, 0.0, 3, id="top-p-drops-the-crossing-rank"),
        pytest.param(1.0, -1, 0.3, 0.0, 1, id="top-p-keeps-rank-zero"),
        pytest.param(1.0, -1, 1.0, 0.12, 3, id="epsilon"),
        pytest.param(1.0, -1, 1.0, 0.05, 5, id="epsilon-lower"),
        pytest.param(1.0, -1, 1.0, 0.5, 1, id="epsilon-keeps-rank-zero"),
        pytest.param(1.0, 3, 0.95, 0.05, 3, id="joint"),
        # 0.25 would pass 0.3 only if top-k renormalised the two ranks it keeps.
        pytest.param(1.0, 2, 1.0, 0.3, 1, id="rules-read-one-distribution"),
        # At temperature 2 the five most probable columns reach 0.1.
        pytest.param(2.0, -1, 1.0, 0.1, 5, id="epsilon-after-temperature"),
    ],
)
def test_the_kept_set_follows_the_keep_rule(
    temperature: float, top_k: int, top_p: float, epsilon_cutoff: float, kept_ranks: int
) -> None:
    logits = torch.tensor([math.log(p) for p in SIX_COLUMN_PROBABILITIES])

    assert kept_columns(logits, temperature, top_k, top_p, epsilon_cutoff) == set(
        DESCENDING_COLUMNS[:kept_ranks]
    )


def epsilon_only_logits() -> torch.Tensor:
    """A Drama 3 sized row where 300 candidates hold 0.6 of the mass, each above 1e-3,
    and the other 3797 share 0.4, each below it."""
    generator = torch.Generator().manual_seed(4)
    probabilities = torch.cat(
        (
            torch.empty(300).uniform_(1.5, 2.5, generator=generator),
            torch.empty(DRAMA3_CANDIDATE_COUNT - 300).uniform_(
                0.5, 1.5, generator=generator
            ),
        )
    )
    probabilities[:300] *= 0.6 / probabilities[:300].sum()
    probabilities[300:] *= 0.4 / probabilities[300:].sum()
    permutation = torch.randperm(DRAMA3_CANDIDATE_COUNT, generator=generator)
    return probabilities[permutation].log()


def test_an_epsilon_only_row_keeps_hundreds_of_candidates_in_proportion() -> None:
    """No top-k or top-p, epsilon 1e-3: the 300 candidates above it are all drawn,
    in proportion to p, and nothing else is."""
    logits = epsilon_only_logits()
    expected = rule_probabilities(logits, 1.0, -1, 1.0, 1e-3)
    torch.manual_seed(0)
    chunk_rows, chunk_count = 5000, 6
    draws = torch.cat(
        [
            truncated_sample(
                logits.repeat(chunk_rows, 1),
                rows_of(1.0, chunk_rows, torch.float32),
                rows_of(-1, chunk_rows, torch.int64),
                rows_of(1.0, chunk_rows, torch.float32),
                rows_of(1e-3, chunk_rows, torch.float32),
            )
            for _ in range(chunk_count)
        ]
    )
    draw_count = chunk_rows * chunk_count

    assert int((expected > 0).sum()) == 300
    assert kept_columns(logits, 1.0, -1, 1.0, 1e-3) == set(
        torch.nonzero(expected).flatten().tolist()
    )
    frequencies = frequencies_of(draws, DRAMA3_CANDIDATE_COUNT)
    assert_follows(frequencies, expected, draw_count)
    is_kept = expected > 0
    chi_square = float(
        ((frequencies[is_kept] - expected[is_kept]) ** 2 / expected[is_kept]).sum()
        * draw_count
    )
    degrees_of_freedom = 299
    assert chi_square < degrees_of_freedom + 6 * math.sqrt(2 * degrees_of_freedom)


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "epsilon_cutoff"),
    [
        pytest.param(1.0, -1, 1.0, 1e-3, id="recommended"),
        pytest.param(0.9, 40, 0.9, 2e-3, id="every-rule"),
        pytest.param(1.3, -1, 0.97, 5e-3, id="top-p-and-epsilon"),
    ],
)
def test_draws_follow_the_renormalised_kept_probabilities(
    temperature: float, top_k: int, top_p: float, epsilon_cutoff: float
) -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(1024, generator=generator) * 1.5
    expected = rule_probabilities(logits, temperature, top_k, top_p, epsilon_cutoff)
    draw_count = 20000
    torch.manual_seed(1)

    draws = truncated_sample(
        logits.repeat(draw_count, 1),
        rows_of(temperature, draw_count, torch.float32),
        rows_of(top_k, draw_count, torch.int64),
        rows_of(top_p, draw_count, torch.float32),
        rows_of(epsilon_cutoff, draw_count, torch.float32),
    )

    assert kept_columns(logits, temperature, top_k, top_p, epsilon_cutoff) == set(
        torch.nonzero(expected).flatten().tolist()
    )
    assert_follows(frequencies_of(draws, 1024), expected, draw_count)


def test_every_draw_stays_in_its_rows_kept_set() -> None:
    """Rows with mixed parameters, the legal-column rule included, never draw a
    column their own rules drop."""
    generator = torch.Generator().manual_seed(5)
    row_count, column_count = 3000, 256
    logits = torch.randn(row_count, column_count, generator=generator) * (
        torch.rand(row_count, 1, generator=generator) * 4.0
    )
    temperature = torch.tensor([0.0, 0.5, 1.0, 1.7])[
        torch.randint(0, 4, (row_count,), generator=generator)
    ]
    top_k = torch.tensor([-1, 0, 1, 5, 50, 300])[
        torch.randint(0, 6, (row_count,), generator=generator)
    ]
    top_p = torch.tensor([0.3, 0.8, 0.95, 1.0])[
        torch.randint(0, 4, (row_count,), generator=generator)
    ]
    epsilon_cutoff = torch.tensor([0.0, 1e-3, 1e-2, 0.2])[
        torch.randint(0, 4, (row_count,), generator=generator)
    ]
    torch.manual_seed(2)

    for legal_column_count in (None, 100):
        draws = truncated_sample(
            logits,
            temperature,
            top_k,
            top_p,
            epsilon_cutoff,
            legal_column_count=legal_column_count,
        )
        _, sorted_columns, is_kept = truncate_candidates(
            logits, temperature, top_k, top_p, epsilon_cutoff, legal_column_count
        )
        is_kept_column = torch.zeros_like(is_kept).scatter_(-1, sorted_columns, is_kept)
        assert bool(is_kept_column[torch.arange(row_count), draws].all())


def main_logits(probabilities: dict[int, float], row_count: int) -> torch.Tensor:
    """Candidate-column logits with the given token probabilities; every other
    candidate is negligible."""
    logits = torch.full((row_count, CANDIDATE_TOKEN_IDS.numel()), -30.0)
    for token_id, probability in probabilities.items():
        column = int(torch.searchsorted(CANDIDATE_TOKEN_IDS, token_id))
        logits[:, column] = math.log(probability)
    return logits


def window_of(token_ids: list[int], row_count: int) -> torch.Tensor:
    padded = [EMPTY_WINDOW_TOKEN] * (WINDOW_FRAMES - len(token_ids)) + token_ids
    return torch.tensor(padded).repeat(row_count, 1)


def sampled_main_tokens(
    logits: torch.Tensor,
    window: torch.Tensor,
    fallback: str,
    *,
    top_k: int = -1,
    epsilon_cutoff: float = 1e-3,
) -> torch.Tensor:
    row_count = logits.shape[0]
    main_tokens, codebook_zero = sample_main_tokens(
        logits,
        CANDIDATE_TOKEN_IDS,
        window,
        rows_of(1.0, row_count, torch.float32),
        rows_of(top_k, row_count, torch.int64),
        rows_of(1.0, row_count, torch.float32),
        rows_of(epsilon_cutoff, row_count, torch.float32),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
        fallback,
    )
    assert torch.equal(
        codebook_zero, (main_tokens - SEMANTIC_BEGIN_ID).clamp(0, CODEBOOK_SIZE - 1)
    )
    return main_tokens


def test_a_repeated_first_draw_takes_the_most_probable_candidate_eos_included() -> None:
    """Drawn eos and semantic tokens found in the window both become the argmax
    token; without them in the window all three are drawn."""
    top_token, other_token = SEMANTIC_BEGIN_ID + 5, SEMANTIC_BEGIN_ID + 9
    row_count = 2000
    logits = main_logits(
        {top_token: 0.4, EOS_TOKEN_ID: 0.35, other_token: 0.25}, row_count
    )
    torch.manual_seed(3)

    repeated = sampled_main_tokens(
        logits, window_of([EOS_TOKEN_ID, other_token], row_count), "argmax"
    )
    free = sampled_main_tokens(logits, window_of([], row_count), "argmax")

    assert set(repeated.tolist()) == {top_token}
    assert set(free.tolist()) == {top_token, EOS_TOKEN_ID, other_token}


def test_the_ras_fallback_draws_at_fixed_parameters_without_the_cutoff() -> None:
    """With epsilon 0.1 the first draw is always the top token, which the window
    holds; the fallback then keeps the top 30 candidates by 1.0 / 0.95 / 30 although
    all but the top one lie below the cutoff."""
    top_token = SEMANTIC_BEGIN_ID
    row_count = 20000
    others = range(SEMANTIC_BEGIN_ID + 1, SEMANTIC_BEGIN_ID + 41)
    weights = torch.linspace(1.0, 2.0, len(others))
    probabilities = {top_token: 0.5} | {
        token_id: float(0.5 * weight / weights.sum())
        for token_id, weight in zip(others, weights, strict=True)
    }
    logits = main_logits(probabilities, row_count)
    expected = rule_probabilities(logits[0], RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, 0.0)
    torch.manual_seed(4)

    main_tokens = sampled_main_tokens(
        logits, window_of([top_token], row_count), "ras", epsilon_cutoff=0.1
    )

    columns = torch.searchsorted(CANDIDATE_TOKEN_IDS, main_tokens)
    assert int((expected > 0).sum()) == RAS_TOP_K
    assert_follows(
        frequencies_of(columns, CANDIDATE_TOKEN_IDS.numel()), expected, row_count
    )


def test_the_first_frame_window_is_the_prompt_tail() -> None:
    """A token among the last 16 prompt rows is replaced on the first frame; one only
    further back is not."""
    top_token, prompt_token = SEMANTIC_BEGIN_ID + 1, SEMANTIC_BEGIN_ID + 2
    row_count = 2000
    logits = main_logits({top_token: 0.5, prompt_token: 0.5 - 1e-3}, row_count)
    text_rows = [[20 + index, 0, 0] for index in range(20)]
    prompt_row, end_of_turn_row = [prompt_token, 2, 0], [EOS_TOKEN_ID, 0, 0]
    # The prompt token is the oldest of the last 16 rows, or one row older.
    tail_rows = text_rows + [prompt_row] + text_rows[:14] + [end_of_turn_row]
    older_rows = text_rows + [prompt_row] + text_rows[:15] + [end_of_turn_row]
    torch.manual_seed(5)

    for prompt_rows, can_repeat in ((tail_rows, False), (older_rows, True)):
        window = torch.tensor(ras_window(prompt_rows)).repeat(row_count, 1)
        main_tokens = sampled_main_tokens(logits, window, "argmax")
        assert (prompt_token in main_tokens.tolist()) == can_repeat


def head_logits(probabilities: dict[int, float], row_count: int) -> torch.Tensor:
    """Codebook-head logits with the given code probabilities; every other code is
    negligible."""
    logits = torch.full((row_count, HEAD_WIDTH), -30.0)
    for code, probability in probabilities.items():
        logits[:, code] = math.log(probability)
    return logits


def sampled_residual_codes(
    logits: torch.Tensor, *, top_k: int = -1, top_p: float = 1.0
) -> torch.Tensor:
    row_count = logits.shape[0]
    return sample_residual_codes(
        logits,
        rows_of(1.0, row_count, torch.float32),
        rows_of(top_k, row_count, torch.int64),
        rows_of(top_p, row_count, torch.float32),
    )


def test_untruncated_residual_draws_stay_inside_the_codec_codebook() -> None:
    """Codes past the codebook hold half of the mass; the draws follow the legal codes'
    probabilities renormalised."""
    row_count = 20000
    probabilities = {code: 0.04 for code in range(10)} | {
        RESIDUAL_CODEBOOK_SIZE + code: 0.06 for code in range(10)
    }
    logits = head_logits(probabilities, row_count)
    expected = rule_probabilities(
        logits[0], 1.0, -1, 1.0, 0.0, legal_column_count=RESIDUAL_CODEBOOK_SIZE
    )
    torch.manual_seed(6)

    codes = sampled_residual_codes(logits)

    assert int(codes.max()) < RESIDUAL_CODEBOOK_SIZE
    assert_follows(frequencies_of(codes, HEAD_WIDTH), expected, row_count)


@pytest.mark.parametrize(
    ("probabilities", "top_k", "top_p", "codes"),
    [
        pytest.param({3000: 0.4, 5: 0.3, 7: 0.2, 9: 0.1}, 2, 1.0, {5}, id="top-k"),
        pytest.param({3000: 0.4, 5: 0.3, 7: 0.2, 9: 0.1}, -1, 0.95, {5, 7}, id="top-p"),
        pytest.param(
            {3000: 0.6, 2500: 0.3, 5: 0.06, 7: 0.04}, 2, 1.0, {5}, id="all-illegal"
        ),
        pytest.param(
            {3000: 0.6, 2500: 0.3, 5: 0.06, 7: 0.04}, -1, 0.5, {5}, id="all-illegal-p"
        ),
    ],
)
def test_residual_rules_run_on_the_whole_row_before_illegal_codes_drop(
    probabilities: dict[int, float], top_k: int, top_p: float, codes: set[int]
) -> None:
    """Illegal codes count toward top-k and top-p; if no survivor is legal, the most
    probable legal code is kept."""
    torch.manual_seed(7)

    sampled = sampled_residual_codes(
        head_logits(probabilities, 2000), top_k=top_k, top_p=top_p
    )

    assert set(sampled.tolist()) == codes


@torch_fused_op_backend()
def test_decode_step_samples_with_the_reference_rules_once_the_engine_sets_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main token gets the batch's epsilon cutoffs and the fallback; every
    codebook head gets the request's top-k, not the cap of the other samplers."""
    torch.manual_seed(8)
    config = FishOmniConfig(**fish_config_dict())
    decoder = AudioDecoder(config).eval()
    decoder_config = config.audio_decoder_config
    main_sampler = Mock(wraps=audio_decoder_module.sample_main_tokens)
    residual_sampler = Mock(wraps=audio_decoder_module.sample_residual_codes)
    monkeypatch.setattr(audio_decoder_module, "sample_main_tokens", main_sampler)
    monkeypatch.setattr(audio_decoder_module, "sample_residual_codes", residual_sampler)
    row_count = 2
    top_k = torch.full((row_count,), CODEBOOK_TOP_K + 70)
    epsilon_cutoff = torch.full((row_count,), 0.01)

    def decode(fallback: str | None) -> torch.Tensor:
        decoder.reference_sampling_fallback = fallback
        attention = DenseAttention(
            head_count=decoder_config.n_head,
            kv_head_count=decoder_config.n_local_heads,
            head_dim=decoder_config.head_dim,
        )
        attention.bind(["a", "b"], [3, 5])
        with torch.no_grad():
            return decoder.decode_step(
                torch.randn(row_count, config.text_config.vocab_size),
                torch.randn(row_count, decoder_config.dim),
                torch.tensor([3, 5]),
                torch.zeros(
                    row_count, 1 + decoder_config.num_codebooks, dtype=torch.int64
                ),
                torch.full((row_count, WINDOW_FRAMES), EMPTY_WINDOW_TOKEN),
                torch.ones(row_count),
                top_k,
                torch.ones(row_count),
                attention,
                epsilon_cutoff,
            )

    decode(None)
    assert (main_sampler.call_count, residual_sampler.call_count) == (0, 0)

    frames = decode("ras")
    assert frames.shape == (row_count, 1 + decoder_config.num_codebooks)
    (main_call,) = main_sampler.call_args_list
    assert main_call.args[6] is epsilon_cutoff and main_call.args[-1] == "ras"
    assert residual_sampler.call_count == decoder_config.num_codebooks - 1
    assert all(call.args[2] is top_k for call in residual_sampler.call_args_list)


@pytest.mark.accelerator
def test_a_cuda_graph_replay_draws_what_the_eager_step_draws() -> None:
    """Captured inside a CUDA graph, both samplers draw from the default generator
    as they do eagerly."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    device = torch.device("cuda")
    row_count = 8
    generator = torch.Generator().manual_seed(9)
    main_logits_rows = (
        torch.randn(row_count, DRAMA3_CANDIDATE_COUNT, generator=generator) * 2.0
    ).to(device)
    head_logits_rows = (
        torch.randn(row_count, HEAD_WIDTH, generator=generator) * 2.0
    ).to(device)
    candidate_token_ids = torch.arange(100, 100 + DRAMA3_CANDIDATE_COUNT, device=device)
    window = torch.full((row_count, WINDOW_FRAMES), EMPTY_WINDOW_TOKEN, device=device)
    window[::2, 3] = candidate_token_ids[main_logits_rows[::2].argmax(dim=-1)]
    temperature = torch.ones(row_count, device=device)
    top_k = torch.full((row_count,), -1, device=device)
    top_p = torch.ones(row_count, device=device)
    epsilon_cutoff = torch.full((row_count,), 1e-3, device=device)

    def sample() -> torch.Tensor:
        main_tokens, _ = sample_main_tokens(
            main_logits_rows,
            candidate_token_ids,
            window,
            temperature,
            top_k,
            top_p,
            epsilon_cutoff,
            101,
            HEAD_WIDTH,
            "ras",
        )
        codes = sample_residual_codes(head_logits_rows, temperature, top_k, top_p)
        return torch.stack((main_tokens, codes))

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        sample()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sample()

    for seed in (10, 11):
        torch.cuda.manual_seed(seed)
        eager = sample()
        torch.cuda.manual_seed(seed)
        graph.replay()
        assert torch.equal(captured, eager)
