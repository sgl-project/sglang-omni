# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 DSpark speculative decoding: sampling exactness and draft loading."""

from __future__ import annotations

import json
import os
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dspark_components import dspark_draft as sglang_dspark_draft

from sglang_omni.models.fun_cosyvoice3 import dspark_draft, dspark_worker
from sglang_omni.models.fun_cosyvoice3.dspark_draft import (
    DRAFT_ARCHITECTURE,
    DRAFT_TOKENS_PER_STEP,
    MASK_TOKEN_ID,
    UNIFIED_MASK_TOKEN_ID,
    materialize_draft_checkpoint,
)
from sglang_omni.models.fun_cosyvoice3.dspark_sampling import (
    EMPTY_HISTORY_TOKEN,
    exact_accept,
    rejection_sample,
    sample_draft_block,
    target_probabilities,
)
from sglang_omni.models.fun_cosyvoice3.dspark_worker import (
    FunCosyVoice3DSparkWorker,
    RequestHistories,
    build_step_sampling_state,
)
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.models.fun_cosyvoice3.model_runner import (
    COSYVOICE3_RAS_WINDOW_SIZE,
    FunCosyVoice3ModelRunner,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE, VOCAB_SIZE
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def stage_distribution(
    logits: torch.Tensor,
    *,
    presence_mask: torch.Tensor,
    stop_suppressed: bool,
    repetition_penalty: float,
    temperature: float,
    top_k: int,
    top_p: float,
    recent_tokens: list[int],
) -> torch.Tensor:
    """Enumerate the stage's sample-then-redraw procedure for one row exactly."""
    logits = logits.double().clone()
    if stop_suppressed:
        logits[VOCAB_SIZE:] = float("-inf")
    else:
        pass
    logits = torch.where(
        presence_mask,
        torch.where(
            logits < 0, logits * repetition_penalty, logits / repetition_penalty
        ),
        logits,
    )
    if top_k == 1:
        return torch.nn.functional.one_hot(logits.argmax(), TOTAL_VOCAB_SIZE).double()
    else:
        pass
    tempered = torch.softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_ids = tempered.sort(descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    distribution = torch.zeros(TOTAL_VOCAB_SIZE, dtype=torch.float64)
    nucleus_mass = 0.0
    candidates: list[tuple[int, float]] = []
    for rank in range(TOTAL_VOCAB_SIZE):
        probability = float(sorted_probabilities[rank])
        if rank >= top_k or float(cumulative[rank]) - probability > top_p:
            break
        else:
            candidates.append((int(sorted_ids[rank]), probability))
            nucleus_mass += probability
    window = set(recent_tokens)
    for candidate, probability in candidates:
        weight = probability / nucleus_mass
        if candidate < VOCAB_SIZE and candidate in window:
            redraw = tempered.clone()
            redraw[candidate] = 0.0
            if float(redraw.sum()) > 0:
                distribution += weight * redraw / redraw.sum()
            else:
                distribution[candidate] += weight
        else:
            distribution[candidate] += weight
    return distribution


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_target_probabilities_match_the_stage_sampling_procedure(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    rows = 6
    logits = torch.randn(rows, TOTAL_VOCAB_SIZE, generator=generator) * 3
    # A few speech tokens dominate, as in real speech-token streams.
    hot_tokens = torch.randint(0, VOCAB_SIZE, (rows, 4), generator=generator)
    logits.scatter_(1, hot_tokens, 12.0)
    presence_mask = torch.rand(rows, TOTAL_VOCAB_SIZE, generator=generator) < 0.02
    presence_mask.scatter_(1, hot_tokens[:, :2], True)
    stop_suppressed = torch.tensor([True, False, False, True, False, False])
    repetition_penalties = torch.tensor([1.21, 1.1, 1.0, 1.3, 1.21, 1.21])
    temperatures = torch.tensor([0.7, 0.8, 1.0, 0.7, 0.7, 0.7])
    top_ks = torch.tensor([20, 15, 25, 1, 20, TOTAL_VOCAB_SIZE])
    top_ps = torch.tensor([0.8, 0.95, 0.9, 0.8, 1.0, 0.8])
    recent_tokens = torch.full((rows, COSYVOICE3_RAS_WINDOW_SIZE), EMPTY_HISTORY_TOKEN)
    recent_tokens[:, :4] = hot_tokens
    recent_tokens[:, 4] = hot_tokens[:, 0]
    recent_tokens[:, 5] = VOCAB_SIZE + 1

    probabilities = target_probabilities(
        logits,
        presence_mask=presence_mask,
        stop_suppressed=stop_suppressed,
        repetition_penalties=repetition_penalties,
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        recent_tokens=recent_tokens,
    )

    for row in range(rows):
        expected = stage_distribution(
            logits[row],
            presence_mask=presence_mask[row],
            stop_suppressed=bool(stop_suppressed[row]),
            repetition_penalty=float(repetition_penalties[row]),
            temperature=float(temperatures[row]),
            top_k=int(top_ks[row]),
            top_p=float(top_ps[row]),
            recent_tokens=[int(token) for token in recent_tokens[row] if token >= 0],
        )
        torch.testing.assert_close(
            probabilities[row].double(), expected, atol=2e-5, rtol=0
        )


@pytest.mark.parametrize("logit_gap", [28.0, 33.0])
def test_target_probabilities_redraw_a_repeated_token_that_holds_all_the_mass(
    logit_gap: float,
) -> None:
    logits = torch.zeros(1, TOTAL_VOCAB_SIZE)
    logits[0, 42] = logit_gap
    arguments = dict(
        presence_mask=torch.zeros(1, TOTAL_VOCAB_SIZE, dtype=torch.bool),
        stop_suppressed=torch.tensor([False]),
        repetition_penalties=torch.tensor([1.0]),
        temperatures=torch.tensor([0.7]),
        top_ks=torch.tensor([20]),
        top_ps=torch.tensor([0.8]),
    )
    recent_tokens = torch.full((1, COSYVOICE3_RAS_WINDOW_SIZE), EMPTY_HISTORY_TOKEN)
    recent_tokens[0, -1] = 42

    probabilities = target_probabilities(
        logits, recent_tokens=recent_tokens, **arguments
    )

    expected = stage_distribution(
        logits[0],
        presence_mask=arguments["presence_mask"][0],
        stop_suppressed=False,
        repetition_penalty=1.0,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        recent_tokens=[42],
    )
    torch.testing.assert_close(probabilities[0].double(), expected, atol=1e-6, rtol=0)
    assert float(probabilities[0, 42]) == 0.0


def test_rejection_sampling_reproduces_the_target_distribution() -> None:
    torch.manual_seed(0)
    samples, draft_count, vocab_size = 400_000, 2, 5
    target_rows = torch.tensor(
        [
            [0.1, 0.4, 0.3, 0.2, 0.0],
            [0.25, 0.25, 0.25, 0.25, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
        ]
    )
    draft_logits_rows = torch.log(
        torch.tensor([[0.5, 0.1, 0.1, 0.2, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]])
    )
    draft_probabilities = torch.softmax(draft_logits_rows, dim=-1)
    draft_tokens = torch.stack(
        [
            torch.multinomial(draft_probabilities[step], samples, replacement=True)
            for step in range(draft_count)
        ],
        dim=1,
    )
    accepted_count, next_tokens = rejection_sample(
        target=target_rows.expand(samples, -1, -1),
        draft_logits=draft_logits_rows.expand(samples, -1, -1),
        draft_tokens=draft_tokens,
        temperatures=torch.ones(samples),
        greedy=torch.zeros(samples, dtype=torch.bool),
        uniform_noise=torch.rand(samples, draft_count),
        exponential_noise=torch.empty(samples, vocab_size).exponential_(),
    )
    first_tokens = torch.where(accepted_count > 0, draft_tokens[:, 0], next_tokens)
    first_distribution = torch.bincount(first_tokens, minlength=vocab_size) / samples
    torch.testing.assert_close(first_distribution, target_rows[0], atol=4e-3, rtol=0)
    # Acceptance of a sampled proposal is the overlap of the two distributions.
    expected_acceptance = torch.minimum(target_rows[0], draft_probabilities[0]).sum()
    assert (
        abs(float((accepted_count > 0).float().mean()) - float(expected_acceptance))
        < 4e-3
    )


def test_draft_block_draws_each_step_from_the_logits_it_returns() -> None:
    torch.manual_seed(0)
    rank, steps = 4, 3
    base_logits = torch.randn(2, steps, TOTAL_VOCAB_SIZE)
    # The first stop id would win every step without the minimum-length mask.
    base_logits[:, :, VOCAB_SIZE] = 50.0
    markov_embedding = torch.randn(TOTAL_VOCAB_SIZE, rank)
    markov_projection = torch.randn(TOTAL_VOCAB_SIZE, rank)
    presence_mask = torch.zeros(2, TOTAL_VOCAB_SIZE, dtype=torch.bool)
    presence_mask[:, 7] = True
    draft_tokens, shaped_logits = sample_draft_block(
        base_logits,
        torch.tensor([7, 9]),
        torch.empty(2, steps, TOTAL_VOCAB_SIZE).exponential_(),
        markov_embedding,
        markov_projection,
        presence_mask,
        torch.tensor([1.3, 1.0]),
        torch.tensor([4, 4]),
        torch.tensor([6, 0]),
        torch.tensor([0.8, 1.0]),
        torch.tensor([False, True]),
    )

    previous = torch.tensor([7, 9])
    history = presence_mask.clone()
    for step in range(steps):
        expected = (
            base_logits[:, step] + markov_embedding[previous] @ markov_projection.T
        )
        expected[0, VOCAB_SIZE:] = (
            float("-inf") if 4 + step < 6 else expected[0, VOCAB_SIZE:]
        )
        penalized = torch.where(expected < 0, expected * 1.3, expected / 1.3)
        expected[0] = torch.where(history[0], penalized[0], expected[0])
        torch.testing.assert_close(shaped_logits[:, step], expected)
        assert int(draft_tokens[1, step]) == int(expected[1].argmax())
        history.scatter_(1, draft_tokens[:, step : step + 1], True)
        previous = draft_tokens[:, step]
    # The masked stop id never wins while the minimum length is unmet.
    assert (draft_tokens[0, :2] != VOCAB_SIZE).all()


def test_exact_accept_verifies_each_row_with_the_drafts_before_it() -> None:
    torch.manual_seed(0)
    batch, drafts = 2, 3
    rows = drafts + 1
    target_logits = torch.randn(batch * rows, TOTAL_VOCAB_SIZE)
    draft_tokens = torch.tensor([[11, 12, 13], [21, 22, 23]])
    for row in range(batch):
        for draft in range(drafts):
            target_logits[row * rows + draft, draft_tokens[row, draft]] = 20.0
    presence_mask = torch.zeros(batch, TOTAL_VOCAB_SIZE, dtype=torch.bool)
    presence_mask[:, 5] = True
    recent_tokens = torch.full((batch, COSYVOICE3_RAS_WINDOW_SIZE), EMPTY_HISTORY_TOKEN)
    recent_tokens[:, -1] = 5
    arguments = dict(
        presence_mask=presence_mask,
        repetition_penalties=torch.tensor([1.2, 1.0]),
        output_lengths=torch.tensor([3, 3]),
        min_new_tokens=torch.tensor([5, 0]),
        temperatures=torch.tensor([0.7, 0.9]),
        greedy=torch.tensor([False, False]),
        top_ks=torch.tensor([20, 15]),
        top_ps=torch.tensor([0.8, 0.95]),
        recent_tokens=recent_tokens,
    )
    # Accept every draft and draw the bonus from the last row, so that row's
    # distribution decides the next token.
    accepted_count, next_tokens = exact_accept(
        target_logits,
        torch.zeros(batch, drafts, TOTAL_VOCAB_SIZE),
        draft_tokens,
        **arguments,
        uniform_noise=torch.zeros(batch, drafts),
        exponential_noise=torch.ones(batch, TOTAL_VOCAB_SIZE),
    )

    assert accepted_count.tolist() == [drafts, drafts]
    for row in range(batch):
        history = presence_mask[row].clone()
        history[draft_tokens[row]] = True
        window = recent_tokens[row].tolist()[drafts:] + draft_tokens[row].tolist()
        expected = target_probabilities(
            target_logits[row * rows + drafts : row * rows + rows],
            presence_mask=history.unsqueeze(0),
            stop_suppressed=torch.tensor(
                [3 + drafts < int(arguments["min_new_tokens"][row])]
            ),
            repetition_penalties=arguments["repetition_penalties"][row : row + 1],
            temperatures=arguments["temperatures"][row : row + 1],
            top_ks=arguments["top_ks"][row : row + 1],
            top_ps=arguments["top_ps"][row : row + 1],
            recent_tokens=torch.tensor([window]),
        )
        assert int(next_tokens[row]) == int(expected.argmax())


def test_greedy_rows_accept_only_the_target_argmax() -> None:
    target = torch.zeros(2, 3, 4)
    target[:, :, 2] = 1.0
    target[1, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    draft_tokens = torch.tensor([[2, 2], [2, 3]])
    accepted_count, next_tokens = rejection_sample(
        target=target,
        draft_logits=torch.zeros(2, 2, 4),
        draft_tokens=draft_tokens,
        temperatures=torch.ones(2),
        greedy=torch.ones(2, dtype=torch.bool),
        uniform_noise=torch.rand(2, 2),
        exponential_noise=torch.empty(2, 4).exponential_(),
    )
    assert accepted_count.tolist() == [2, 1]
    assert next_tokens.tolist() == [2, 1]


def test_request_histories_track_each_slot_through_reuse() -> None:
    histories = RequestHistories(slot_count=4, device=torch.device("cpu"))

    def presence(*requests: SimpleNamespace) -> list[list[int]]:
        batch = SimpleNamespace(
            reqs=list(requests),
            req_pool_indices=torch.tensor(
                [request.kv.req_pool_idx for request in requests]
            ),
        )
        return [
            row.nonzero().flatten().tolist() for row in histories.presence_mask(batch)
        ]

    first = SimpleNamespace(
        rid="a", kv=SimpleNamespace(req_pool_idx=2), output_ids=array("q", [7, 9])
    )
    second = SimpleNamespace(
        rid="b", kv=SimpleNamespace(req_pool_idx=1), output_ids=array("q", [3])
    )
    assert presence(first, second) == [[7, 9], [3]]
    first.output_ids.extend([9, 11])
    assert presence(first) == [[7, 9, 11]]
    # A new request on a freed slot starts from an empty history.
    reused = SimpleNamespace(
        rid="c", kv=SimpleNamespace(req_pool_idx=2), output_ids=array("q", [5])
    )
    assert presence(reused, second) == [[5], [3]]
    # So does a new request that reuses a finished request's id and slot.
    same_id = SimpleNamespace(
        rid="b", kv=SimpleNamespace(req_pool_idx=1), output_ids=array("q", [4])
    )
    assert presence(same_id) == [[4]]


def test_target_keeps_every_token_when_no_request_filters() -> None:
    request = SimpleNamespace(
        rid="a",
        kv=SimpleNamespace(req_pool_idx=0),
        output_ids=array("q", [7]),
        sampling_params=SimpleNamespace(repetition_penalty=1.0, min_new_tokens=0),
    )
    batch = SimpleNamespace(reqs=[request], req_pool_indices=torch.tensor([0]))
    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.7]]),
        top_ks=torch.tensor([TOTAL_VOCAB_SIZE]),
        top_ps=torch.tensor([1.0]),
        need_top_p_sampling=False,
        need_top_k_sampling=False,
    )
    state = build_step_sampling_state(
        batch,
        sampling_info,
        RequestHistories(slot_count=1, device=torch.device("cpu")),
    )
    logits = torch.randn(
        1, TOTAL_VOCAB_SIZE, generator=torch.Generator().manual_seed(0)
    )

    probabilities = target_probabilities(
        logits * 3,
        presence_mask=state.draft.presence_mask,
        stop_suppressed=torch.tensor([False]),
        repetition_penalties=state.draft.repetition_penalties,
        temperatures=state.draft.temperatures,
        top_ks=state.top_ks,
        top_ps=state.top_ps,
        recent_tokens=torch.full((1, COSYVOICE3_RAS_WINDOW_SIZE), EMPTY_HISTORY_TOKEN),
    )

    # The stage samples the full softmax here, tail included.
    assert bool((probabilities > 0).all())
    torch.testing.assert_close(
        probabilities, torch.softmax(logits * 3 / 0.7, dim=-1), atol=1e-8, rtol=1e-5
    )


def test_materialized_draft_names_the_stage_vocabulary_layout(tmp_path: Path) -> None:
    source = tmp_path / "speculators"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "config.json").write_text(
        json.dumps(
            {
                "speculators_model_type": "dspark",
                "mask_token_id": UNIFIED_MASK_TOKEN_ID,
                "block_size": DRAFT_TOKENS_PER_STEP + 1,
                "aux_hidden_state_layer_ids": [2, 12, 21],
                "markov_rank": 256,
                "markov_head_type": "vanilla",
                "transformer_layer_config": {
                    "hidden_size": 896,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 14,
                    "num_key_value_heads": 2,
                    "head_dim": 64,
                    "layer_types": ["sliding_attention"] * 3,
                    "sliding_window": 2048,
                    "vocab_size": 158720,
                },
            }
        )
    )

    draft_dir = materialize_draft_checkpoint(str(source), None)

    config = json.loads((Path(draft_dir) / "config.json").read_text())
    assert config["architectures"] == [DRAFT_ARCHITECTURE]
    assert config["model_type"] == "qwen3"
    assert config["vocab_size"] == TOTAL_VOCAB_SIZE
    assert config["dflash_config"]["target_layer_ids"] == [1, 11, 20]
    assert config["mask_token_id"] == MASK_TOKEN_ID == TOTAL_VOCAB_SIZE - 1
    assert config["block_size"] == DRAFT_TOKENS_PER_STEP
    assert config["sample_from_anchor"] is False
    assert config["layer_types"] == ["sliding_attention"] * 3
    assert os.path.realpath(Path(draft_dir) / "model.safetensors") == str(
        source / "model.safetensors"
    )


@pytest.mark.parametrize(
    ("config_key", "config_value"), [("mask_token_id", 151643), ("block_size", 16)]
)
def test_materialized_draft_refuses_another_vocabulary_or_block(
    tmp_path: Path, config_key: str, config_value: int
) -> None:
    source_config = {
        "speculators_model_type": "dspark",
        "markov_head_type": "vanilla",
        "mask_token_id": UNIFIED_MASK_TOKEN_ID,
        "block_size": DRAFT_TOKENS_PER_STEP + 1,
    }
    source_config[config_key] = config_value
    (tmp_path / "config.json").write_text(json.dumps(source_config))

    with pytest.raises(ValueError, match="vocabulary"):
        materialize_draft_checkpoint(str(tmp_path), None)


def dspark_overrides(
    **server_args_overrides: object,
) -> tuple[FunCosyVoice3EngineBuilder, dict[str, object]]:
    builder = FunCosyVoice3EngineBuilder()
    builder.device = "cuda:0"
    overrides = build_generation_batch_overrides(
        server_args_overrides=server_args_overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(overrides)
    return builder, overrides


@pytest.fixture
def materialized_draft(monkeypatch: pytest.MonkeyPatch) -> str:
    draft_dir = "/tmp/materialized-draft"
    monkeypatch.setattr(
        dspark_draft,
        "materialize_draft_checkpoint",
        lambda draft_model_path, revision: draft_dir,
    )
    return draft_dir


@pytest.mark.parametrize(
    ("server_args_overrides", "error"),
    [
        ({"speculative_algorithm": "DSPARK"}, "needs speculative_draft_model_path"),
        (
            {"speculative_algorithm": "EAGLE3", "speculative_draft_model_path": "d"},
            "supports DSPARK",
        ),
        (
            {"speculative_draft_model_path": "d", "speculative_dspark_block_size": 4},
            "tokens per step",
        ),
    ],
)
def test_engine_builder_refuses_what_the_dspark_draft_cannot_serve(
    server_args_overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        dspark_overrides(**server_args_overrides)


def test_dspark_draft_reserves_its_verify_windows_in_the_kv_pool_cap(
    materialized_draft: str,
) -> None:
    builder, overrides = dspark_overrides(
        speculative_draft_model_path="yuekai/cosyvoice3_llm_dspark"
    )

    assert overrides["speculative_algorithm"] == "DSPARK"
    assert overrides["speculative_draft_model_path"] == materialized_draft
    assert overrides["speculative_dspark_block_size"] == DRAFT_TOKENS_PER_STEP
    assert overrides["max_total_tokens"] == 32 * (
        builder.context_length + 2 * (DRAFT_TOKENS_PER_STEP + 1)
    )
    assert builder.infra_kwargs() == {
        "supports_speculative_decoding": True,
        "speculative_worker_class": FunCosyVoice3DSparkWorker,
    }


def test_scheduler_admits_the_default_repetition_penalty_under_dspark() -> None:
    """Every default request sets repetition_penalty, which the DSpark verify applies."""
    scheduler = object.__new__(OmniScheduler)
    scheduler.draft_worker = SimpleNamespace()
    scheduler.model_runner = object.__new__(FunCosyVoice3ModelRunner)
    request = SimpleNamespace(
        sampling_params=SamplingParams(repetition_penalty=1.21),
        return_hidden_states=False,
    )

    assert (
        scheduler.speculative_request_error(
            request, SimpleNamespace(return_logprob=False)
        )
        is None
    )


def test_drafting_routes_only_a_mirrored_head_through_its_own_block_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGLang drafts through per-step biases, so a mirrored head draws its shaped
    block itself while every other head keeps SGLang's sampling."""
    forwarded_heads: list[object] = []
    monkeypatch.setattr(
        sglang_dspark_draft,
        "sample_draft_block",
        lambda **kwargs: forwarded_heads.append(kwargs["markov_head"])
        or "sglang block",
    )
    monkeypatch.setattr(dspark_worker, "mirrored_draft_sampling_installed", False)
    dspark_worker.install_mirrored_draft_sampling()
    mirrored_head = dspark_draft.MirroredMarkovHead(vocab_size=6, markov_rank=2)
    drafted_tokens = torch.tensor([[3, 4]])
    shaped_logits = torch.ones(1, 2, 6)
    mirrored_head.sampling_state = SimpleNamespace(
        presence_mask=None,
        repetition_penalties=None,
        output_lengths=None,
        min_new_tokens=None,
        temperatures=None,
        greedy=None,
    )
    mirrored_head.block_sampler = lambda *args: (drafted_tokens, shaped_logits)
    other_head = SimpleNamespace()
    step_inputs = dict(
        base_logits=torch.zeros(1, 2, 6),
        anchor_tokens=torch.tensor([1]),
        draft_hidden=None,
        sampling_info=SimpleNamespace(
            temperatures=torch.tensor([0.7]), top_ks=torch.tensor([20])
        ),
        noise=None,
        device=torch.device("cpu"),
        tp_sync=None,
        greedy_draft=False,
    )

    mirrored_block = sglang_dspark_draft.sample_draft_block(
        markov_head=mirrored_head, **step_inputs
    )
    other_block = sglang_dspark_draft.sample_draft_block(
        markov_head=other_head, **step_inputs
    )

    assert mirrored_block.draft_tokens is drafted_tokens
    assert mirrored_block.corrected_logits is shaped_logits
    assert mirrored_block.greedy_mask.tolist() == [False]
    assert other_block == "sglang block"
    assert forwarded_heads == [other_head]
