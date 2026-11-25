# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.beam_search.beam_group import BeamGroupState, BeamResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import GPT2Config, GPT2LMHeadModel

from sglang_omni.scheduling import beam_search
from sglang_omni.scheduling.beam_search import (
    OmniBeamCoordinator,
    OmniBeamGroup,
    append_encoder_beam_tail,
    select_beam_candidates,
    strip_encoder_beam_tail,
)
from sglang_omni.scheduling.types import BeamSearchSettings
from sglang_omni.serve.openai_errors import generation_error_status_code

EOS_TOKEN_ID = 3
PROMPT = [5, 7, 9]


def make_coordinator(
    *, spec_algorithm: SpeculativeAlgorithm = SpeculativeAlgorithm.NONE
) -> OmniBeamCoordinator:
    return OmniBeamCoordinator(
        model_config=SimpleNamespace(vocab_size=16),
        spec_algorithm=spec_algorithm,
        dllm_enabled=False,
        max_req_len=64,
        req_to_token_pool=SimpleNamespace(size=8, device=torch.device("cpu")),
        token_to_kv_pool_allocator=None,
        tree_cache=SimpleNamespace(
            supports_swa=lambda: False, supports_mamba=lambda: False
        ),
        future_map=None,
    )


def make_model(seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=16,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=EOS_TOKEN_ID,
    )
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        # Peaked, position-dependent distributions with a likely EOS exercise both early stopping and the length limit.
        model.transformer.wte.weight.normal_(0.0, 0.4)
        model.transformer.wpe.weight.normal_(0.0, 1.0)
        model.transformer.wte.weight[EOS_TOKEN_ID] *= 2.0
    return model


def run_beam_search(
    model: GPT2LMHeadModel, *, beam_width: int, max_new_tokens: int
) -> tuple[OmniBeamGroup, list[BeamResult]]:
    coordinator = make_coordinator()
    group = OmniBeamGroup(
        beam_width=beam_width,
        stop_token_ids=[EOS_TOKEN_ID],
        max_new_tokens=max_new_tokens,
        diversity_rate=0.0,
        logit_rules=None,
        device=torch.device("cpu"),
    )
    frontier = [list(PROMPT)]
    tick = 0
    while group.state == BeamGroupState.DECODING:
        with torch.no_grad():
            logits = model(torch.tensor(frontier)).logits[:, -1, :]
        next_token_ids, parent_indices = coordinator.select_group_step(
            group, logits, tick
        )
        group.commit_pending()
        if parent_indices is not None:
            frontier = [
                frontier[parent] + [token_id]
                for parent, token_id in zip(
                    parent_indices.tolist(), next_token_ids.tolist()
                )
            ]
        else:
            pass
        tick += 1
    return group, group.finalize()


def hugging_face_n_best(
    model: GPT2LMHeadModel, *, beam_width: int, max_new_tokens: int
) -> list[tuple[list[int], float]]:
    input_ids = torch.tensor([PROMPT])
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        num_beams=beam_width,
        num_return_sequences=beam_width,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        early_stopping=False,
        length_penalty=1.0,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=False,
    )
    hypotheses = []
    for sequence, score in zip(
        output.sequences.tolist(), output.sequences_scores.tolist()
    ):
        generated = sequence[len(PROMPT) :]
        if EOS_TOKEN_ID in generated:
            generated = generated[: generated.index(EOS_TOKEN_ID) + 1]
        else:
            pass
        hypotheses.append((generated, score))
    return hypotheses


def test_beam_search_matches_hugging_face_generate() -> None:
    """The N-best, its order and its length-normalized scores equal Hugging Face generate(num_beams=k, num_return_sequences=k)."""
    stopped_early = 0
    hit_length_limit = 0
    for seed, beam_width, max_new_tokens in itertools.product(range(6), (2, 4), (5, 9)):
        model = make_model(seed)
        group, results = run_beam_search(
            model, beam_width=beam_width, max_new_tokens=max_new_tokens
        )
        expected = hugging_face_n_best(
            model, beam_width=beam_width, max_new_tokens=max_new_tokens
        )
        case = f"seed={seed} beam_width={beam_width} max_new_tokens={max_new_tokens}"
        assert [result.tokens for result in results] == [
            tokens for tokens, _ in expected
        ], case
        assert [result.beam_score for result in results] == pytest.approx(
            [score for _, score in expected], abs=1e-5
        ), case
        if group.num_committed < max_new_tokens:
            stopped_early += 1
        else:
            hit_length_limit += 1
    assert stopped_early > 0 and hit_length_limit > 0


def test_diversity_rate_reorders_candidates_but_returns_raw_logprobs() -> None:
    """rate * (rank_within_parent % beam_width) only ranks candidates; the cumulative log-probs stay raw."""
    candidate_logprobs = torch.tensor(
        [[-0.10, -0.20, -3.0, -4.0], [-0.15, -0.25, -3.0, -4.0]]
    )
    candidate_token_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    stop_token_ids = torch.tensor([EOS_TOKEN_ID])
    without_bonus = select_beam_candidates(
        torch.zeros(2),
        candidate_logprobs,
        candidate_token_ids,
        stop_token_ids,
        beam_width=2,
        diversity_rate=0.0,
    )
    with_bonus = select_beam_candidates(
        torch.zeros(2),
        candidate_logprobs,
        candidate_token_ids,
        stop_token_ids,
        beam_width=2,
        diversity_rate=0.2,
    )

    assert without_bonus.next_tokens.tolist() == [10, 20]
    assert with_bonus.next_tokens.tolist() == [11, 21]
    assert with_bonus.parent_idx.tolist() == [0, 1]
    assert with_bonus.new_cum_logprobs.tolist() == pytest.approx([-0.20, -0.25])


def test_only_stops_in_the_top_beam_width_ranks_finish() -> None:
    """A stop ranked past beam_width does not finish, even with fewer than beam_width survivors ahead of it."""
    selection = select_beam_candidates(
        torch.zeros(2),
        torch.tensor([[-0.10, -0.12, -2.0, -2.1], [-0.13, -0.30, -2.2, -2.3]]),
        torch.tensor([[EOS_TOKEN_ID, 5, 8, 9], [EOS_TOKEN_ID, 6, 10, 11]]),
        torch.tensor([EOS_TOKEN_ID]),
        beam_width=2,
        diversity_rate=0.0,
    )

    assert int(selection.num_finished) == 1
    assert selection.fin_parent_idx[:1].tolist() == [0]
    assert selection.next_tokens.tolist() == [5, 6]


def test_member_rows_take_their_leaders_encoder_length() -> None:
    """Decode allocation adds encoder_lens to seq_lens row by row, so member rows need their leader's encoder length."""
    group = SimpleNamespace(
        member_rows=torch.tensor([10, 11, 12]),
        member_rows_cpu=torch.tensor([10, 11, 12]),
        retired=False,
        num_member_rows=3,
    )
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(beam_group=None), SimpleNamespace(beam_group=group)],
        beam_tail=None,
        device=torch.device("cpu"),
        model_config=SimpleNamespace(is_encoder_decoder=True),
        req_pool_indices=torch.tensor([1, 2]),
        req_pool_indices_cpu=torch.tensor([1, 2]),
        seq_lens=torch.tensor([5, 7]),
        seq_lens_cpu=torch.tensor([5, 7]),
        orig_seq_lens=torch.tensor([5, 7]),
        seq_lens_sum=12,
        input_ids=None,
        out_cache_loc=None,
        encoder_lens=torch.tensor([1500, 1499]),
        encoder_lens_cpu=[1500, 1499],
    )

    append_encoder_beam_tail(batch)

    assert batch.encoder_lens.tolist() == [1500, 1499, 1499, 1499, 1499]
    assert batch.encoder_lens_cpu == [1500, 1499, 1499, 1499, 1499]
    assert batch.encoder_lens.shape == batch.seq_lens.shape

    strip_encoder_beam_tail(batch)

    assert batch.encoder_lens.tolist() == [1500, 1499]
    assert batch.encoder_lens_cpu == [1500, 1499]
    assert batch.req_pool_indices.tolist() == [1, 2]


def make_leader() -> SimpleNamespace:
    sampling_params = SamplingParams(
        max_new_tokens=20, temperature=0.0, stop_token_ids=[EOS_TOKEN_ID]
    )
    sampling_params.normalize(tokenizer=None)
    return SimpleNamespace(
        sampling_params=sampling_params,
        origin_input_ids=list(range(10)),
        session_id=None,
        lora_id=None,
        return_logprob=False,
        return_hidden_states=False,
        eos_token_ids=None,
        tokenizer=None,
        custom_logit_processor="timestamp processor",
        skip_radix_cache_insert=False,
        beam_group=None,
    )


@pytest.fixture
def single_gpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(beam_search, "get_parallel", lambda: SimpleNamespace(pp_size=1))
    monkeypatch.setattr(
        beam_search, "get_schedule", lambda: SimpleNamespace(page_size=1)
    )
    monkeypatch.setattr(
        beam_search,
        "get_memory",
        lambda: SimpleNamespace(enable_hierarchical_cache=False, enable_hisparse=False),
    )


def test_beam_requests_are_rejected_on_a_speculative_server_as_a_caller_error(
    single_gpu_runtime: None,
) -> None:
    settings = BeamSearchSettings(beam_width=5, diversity_rate=0.0, logit_rules=None)
    speculative = make_coordinator(spec_algorithm=SpeculativeAlgorithm.STANDALONE)

    error = speculative.beam_request_error(make_leader(), settings)

    assert error == "beam_width > 1 does not support speculative decoding."
    assert generation_error_status_code(ValueError(error)) == 400
    assert make_coordinator().beam_request_error(make_leader(), settings) is None


def test_attached_leader_samples_greedily_and_leaves_finishing_to_its_group(
    single_gpu_runtime: None,
) -> None:
    """The leader's own sample is discarded; it must not run the request's logit processor or stop by itself."""
    coordinator = make_coordinator()
    leader = make_leader()

    coordinator.attach_group(
        leader, BeamSearchSettings(beam_width=5, diversity_rate=0.3, logit_rules=None)
    )

    group = leader.beam_group
    assert coordinator.has_live_groups
    assert group.leader is leader and group.prompt_len == 10
    assert group.max_new_tokens == 20 and group.diversity_rate == 0.3
    assert group.stop_token_ids.tolist() == [EOS_TOKEN_ID]
    assert leader.sampling_params.top_k == 1
    assert leader.sampling_params.ignore_eos
    assert leader.sampling_params.max_new_tokens > group.max_new_tokens
    assert leader.custom_logit_processor is None
    assert leader.skip_radix_cache_insert
