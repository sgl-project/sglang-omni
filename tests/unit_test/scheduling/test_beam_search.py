# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.beam_search.beam_group import BeamGroupState, BeamResult
from sglang.srt.managers import schedule_policy
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    WhisperTimeStampLogitsProcessor,
)

from sglang_omni.models.whisper_asr.beam_rules import WhisperBeamLogitRules
from sglang_omni.scheduling import beam_search
from sglang_omni.scheduling.beam_search import (
    BeamPrefillAdder,
    OmniBeamCoordinator,
    OmniBeamGroup,
    append_encoder_beam_tail,
    select_beam_candidates,
    strip_encoder_beam_tail,
)
from sglang_omni.scheduling.types import BeamLogitRules, BeamSearchSettings
from sglang_omni.serve.openai_errors import generation_error_status_code

EOS_TOKEN_ID = 3
PROMPT = [5, 7, 9]
NO_TIMESTAMPS_TOKEN_ID = 5
TIMESTAMP_BEGIN_ID = NO_TIMESTAMPS_TOKEN_ID + 1
# Hugging Face fills N-best slots that no hypothesis reached with entries scored near -1e9.
HUGGING_FACE_UNREACHED_SCORE = -1e8


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
    model: GPT2LMHeadModel,
    *,
    beam_width: int,
    max_new_tokens: int,
    logit_rules: BeamLogitRules | None = None,
) -> tuple[OmniBeamGroup, list[BeamResult]]:
    coordinator = make_coordinator()
    group = OmniBeamGroup(
        beam_width=beam_width,
        stop_token_ids=[EOS_TOKEN_ID],
        max_new_tokens=max_new_tokens,
        diversity_rate=0.0,
        logit_rules=logit_rules,
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
    model: GPT2LMHeadModel,
    *,
    beam_width: int,
    max_new_tokens: int,
    logits_processor: LogitsProcessorList | None = None,
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
        logits_processor=logits_processor,
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


@pytest.mark.parametrize("max_new_tokens", [1, 6])
def test_masked_candidates_never_become_hypotheses(max_new_tokens: int) -> None:
    """With max_initial_timestamp_index=0 the timestamp rules allow one first token for five beams; the N-best held masked tokens scored -inf.

    Past the masked step the N-best equals Hugging Face's, without the placeholders Hugging Face fills unreached slots with.
    """
    config = SimpleNamespace(
        eos_token_id=EOS_TOKEN_ID,
        bos_token_id=0,
        no_timestamps_token_id=NO_TIMESTAMPS_TOKEN_ID,
        max_initial_timestamp_index=0,
    )
    for seed in range(3):
        model = make_model(seed)
        rules = WhisperBeamLogitRules(
            suppress_token_ids=[],
            begin_suppress_token_ids=[EOS_TOKEN_ID],
            timestamp_params={
                "segment_timestamps": True,
                "timestamp_begin_id": TIMESTAMP_BEGIN_ID,
                "no_timestamps_token_id": NO_TIMESTAMPS_TOKEN_ID,
                "eos_token_id": EOS_TOKEN_ID,
                "max_initial_timestamp_index": 0,
                "__req__": SimpleNamespace(output_ids=[]),
            },
        )
        _, results = run_beam_search(
            model, beam_width=5, max_new_tokens=max_new_tokens, logit_rules=rules
        )
        expected = [
            (tokens, score)
            for tokens, score in hugging_face_n_best(
                model,
                beam_width=5,
                max_new_tokens=max_new_tokens,
                logits_processor=LogitsProcessorList(
                    [
                        SuppressTokensAtBeginLogitsProcessor(
                            [EOS_TOKEN_ID], begin_index=len(PROMPT)
                        ),
                        WhisperTimeStampLogitsProcessor(
                            config, begin_index=len(PROMPT)
                        ),
                    ]
                ),
            )
            if score > HUGGING_FACE_UNREACHED_SCORE
        ]

        assert [result.tokens for result in results] == [
            tokens for tokens, _ in expected
        ], seed
        assert [result.beam_score for result in results] == pytest.approx(
            [score for _, score in expected], abs=1e-5
        ), seed


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


def admission_candidate(beam_width: int | None) -> SimpleNamespace:
    group = (
        None
        if beam_width is None
        else SimpleNamespace(beam_width=beam_width, member_rows=None, retired=False)
    )
    return SimpleNamespace(
        beam_group=group,
        kv=SimpleNamespace(holds_kv=False),
        finished=lambda: False,
    )


@pytest.mark.parametrize(
    "free_rows,running_widths,candidate_widths,admitted",
    [
        # A leader and its four member rows fill five rows; the greedy request after it waits.
        (5, [], [5, None], 1),
        (6, [], [5, None], 2),
        (6, [], [5, 2], 1),
        # A running leader whose members have not spawned yet keeps its four rows.
        (5, [5], [None, None], 1),
    ],
)
def test_prefill_admission_keeps_rows_for_the_member_rows_of_admitted_leaders(
    monkeypatch: pytest.MonkeyPatch,
    free_rows: int,
    running_widths: list[int],
    candidate_widths: list[int | None],
    admitted: int,
) -> None:
    def add_one_req(adder, req, has_chunked_req, truncation_align_size):
        adder.can_run_list.append(req)
        return AddReqResult.CONTINUE

    monkeypatch.setattr(PrefillAdder, "add_one_req", add_one_req)
    adder = object.__new__(BeamPrefillAdder)
    adder.can_run_list = []
    adder.running_batch = SimpleNamespace(
        reqs=[admission_candidate(width) for width in running_widths]
    )
    adder.tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(available_size=lambda: free_rows)
    )
    candidates = [admission_candidate(width) for width in candidate_widths]

    for req in candidates:
        if adder.add_one_req(req, False, None) != AddReqResult.CONTINUE:
            break
        else:
            pass

    assert [id(req) for req in adder.can_run_list] == [
        id(req) for req in candidates[:admitted]
    ]


class RunningRequest:
    """A stand-in for a running Req, compared and hashed by identity like one."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def running_request(
    *, beam_group: SimpleNamespace | None, wait_queue_entry_time: float
) -> RunningRequest:
    return RunningRequest(
        beam_group=beam_group,
        priority=0,
        time_stats=SimpleNamespace(wait_queue_entry_time=wait_queue_entry_time),
        sampling_params=SimpleNamespace(
            ignore_eos=beam_group is not None, max_new_tokens=100
        ),
        output_ids=[],
        kv=SimpleNamespace(holds_kv=True),
        finished=lambda: False,
    )


def test_priority_preemption_skips_beam_leaders_and_admits_the_preempting_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Priority preemption released a running beam leader like any request, which left its member rows held and requeued a group that cannot be rebuilt."""

    def add_one_req(adder, req, has_chunked_req, truncation_align_size):
        adder.can_run_list.append(req)
        return AddReqResult.CONTINUE

    monkeypatch.setattr(PrefillAdder, "add_one_req", add_one_req)
    monkeypatch.setattr(
        schedule_policy,
        "get_schedule",
        lambda: SimpleNamespace(schedule_low_priority_values_first=False),
    )
    free_rows = [1]
    released_indices: list[int] = []
    # The newest request is preempted first, so the leader would be the first pick.
    leader = running_request(
        beam_group=SimpleNamespace(
            beam_width=5, member_rows=torch.tensor([4, 5, 6, 7]), retired=False
        ),
        wait_queue_entry_time=2.0,
    )
    greedy = running_request(beam_group=None, wait_queue_entry_time=1.0)

    def release_req(index: int, remaining: int) -> None:
        released_indices.append(index)
        free_rows[0] += 1

    def filter_batch(keep_indices: list[int]) -> None:
        running_batch.reqs = [running_batch.reqs[index] for index in keep_indices]

    running_batch = SimpleNamespace(
        reqs=[leader, greedy], release_req=release_req, filter_batch=filter_batch
    )
    adder = object.__new__(BeamPrefillAdder)
    adder.can_run_list = []
    adder.preempt_list = []
    adder.running_batch = running_batch
    adder.tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(available_size=lambda: free_rows[0])
    )
    adder.token_to_kv_pool_allocator = SimpleNamespace(
        request_slot_reserve=lambda **kwargs: 0
    )
    adder.memory_budget = SimpleNamespace(remaining_total=0, total_offset=0)
    adder.new_token_ratio = 1.0
    adder.priority_scheduling_preemption_threshold = 0
    queued = admission_candidate(None)
    urgent = SimpleNamespace(
        beam_group=None,
        priority=10,
        full_untruncated_fill_ids=[0] * 8,
        prefix_len=0,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        kv=SimpleNamespace(holds_kv=False, req_pool_idx=None),
    )

    assert adder.add_one_req(queued, False, None) == AddReqResult.CONTINUE
    assert adder.preempt_to_schedule(urgent)
    assert adder.add_one_req(urgent, False, None) == AddReqResult.CONTINUE

    assert released_indices == [1]
    assert running_batch.reqs == [leader]
    assert adder.preempt_list == [greedy]
    assert adder.can_run_list == [queued, urgent]


def test_aborting_a_group_frees_its_member_rows_before_the_leader_releases(
    single_gpu_runtime: None,
) -> None:
    """An abort outside the commit path frees the member rows and the decode KV they share, and rewinds the leader to its prompt."""
    freed_rows: list[int] = []
    freed_slots: list[int] = []
    coordinator = make_coordinator()
    coordinator.req_to_token_pool = SimpleNamespace(
        size=8,
        device=torch.device("cpu"),
        req_to_token=torch.arange(8 * 16).reshape(8, 16),
        free_rows=freed_rows.extend,
    )
    coordinator.token_to_kv_pool_allocator = SimpleNamespace(
        free=lambda slots: freed_slots.extend(slots.tolist())
    )
    leader = make_leader()
    coordinator.attach_group(
        leader, BeamSearchSettings(beam_width=3, diversity_rate=0.0, logit_rules=None)
    )
    group = leader.beam_group
    leader.kv = SimpleNamespace(
        req_pool_idx=0, kv_committed_len=13, kv_allocated_len=13
    )
    group.member_rows = torch.tensor([5, 6])
    group.member_rows_cpu = torch.tensor([5, 6])
    group.all_rows = torch.tensor([0, 5, 6])

    coordinator.abort_group(leader)
    coordinator.abort_group(leader)

    decode_slots = coordinator.req_to_token_pool.req_to_token[[0, 5, 6], 10:13]
    assert freed_rows == [5, 6]
    assert sorted(freed_slots) == decode_slots.flatten().tolist()
    assert leader.kv.kv_allocated_len == group.prompt_len == 10
    assert group.retired and not coordinator.has_live_groups
