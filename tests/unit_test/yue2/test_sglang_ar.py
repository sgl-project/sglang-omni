# SPDX-License-Identifier: Apache-2.0
"""YuE2 SGLang AR stage: request phases, CFG pairing and row sampling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tiktoken
import torch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

from sglang_omni.models.yue2.engine_builder import Yue2EngineBuilder
from sglang_omni.models.yue2.model_runner import (
    Yue2ModelRunner,
    rewind_sampling,
    sampling_rows,
)
from sglang_omni.models.yue2.payload_types import Yue2State
from sglang_omni.models.yue2.request import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    MUSIC_END,
    SamplingConfig,
    Yue2Tokenizer,
)
from sglang_omni.models.yue2.scheduler import (
    Yue2PrefillAdder,
    Yue2Scheduler,
    paired_queue,
)
from sglang_omni.models.yue2.sglang_request_builder import (
    Yue2ARRequestData,
    Yue2RequestBuilder,
    public_request_id,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pipeline_state import load_state
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.serve.openai_errors import generation_error_status_code


@pytest.fixture
def builder() -> Yue2RequestBuilder:
    tokenizer = object.__new__(Yue2Tokenizer)
    tokenizer.encoding = tiktoken.Encoding(
        "test-yue2",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([index]): index for index in range(256)},
        special_tokens={},
    )
    return Yue2RequestBuilder(
        tokenizer, context_tokens=24576, vocab_size=184704, max_rows=16
    )


def payload(
    request_id: str = "song", token_budget: int = 8, **options: object
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        data={},
        request=OmniRequest(
            inputs="Sing",
            metadata={"tts_params": {"instructions": "pop", "seed": 7}},
            params={
                "max_new_tokens": token_budget,
                "stage_params": {"synthesis": options},
            },
        ),
    )


def test_score_phase_runs_first_under_an_internal_id(
    builder: Yue2RequestBuilder,
) -> None:
    data = builder(payload())
    assert data.phase == "abc" and data.req.rid == "song-abc"
    assert data.prefix[-1] == ABC_START and data.req.eos_token_ids == {ABC_END}
    assert data.cfg_uncond is None
    with pytest.raises(ValueError, match="reserved"):
        builder(payload("song-cfg"))


def test_audio_phase_pairs_guided_requests_with_a_twin(
    builder: Yue2RequestBuilder,
) -> None:
    guided = builder(payload(cot="off"))
    twin = guided.cfg_uncond
    assert guided.phase == "semantic" and guided.req.rid == "song"
    assert guided.cfg_scale == pytest.approx(1.01) and guided.legacy_off
    assert twin is not None and twin.is_cfg_uncond and twin.req.rid == "song-cfg"
    assert twin.prefix == builder.tokenizer.negative_prefix(guided.request, [])
    unguided = builder(payload(abc="X:1\nK:C\nCDEF"))
    assert unguided.phase == "semantic" and unguided.cfg_uncond is None
    assert unguided.req.eos_token_ids == {MUSIC_END}


def test_guidance_needs_two_rows(builder: Yue2RequestBuilder) -> None:
    builder.max_rows = 1
    assert builder(payload(abc="X:1\nK:C\nCDEF")).cfg_uncond is None
    with pytest.raises(ValueError, match="max_concurrency") as rejected:
        builder(payload(cfg_scale=1.5))
    assert generation_error_status_code(RuntimeError(str(rejected.value))) == 400


def test_context_overflow_is_a_client_error(builder: Yue2RequestBuilder) -> None:
    with pytest.raises(ValueError, match="context") as invalid_request:
        builder(payload(token_budget=24576, cot="off"))
    assert generation_error_status_code(RuntimeError(str(invalid_request.value))) == 400


@pytest.mark.parametrize(
    "output_ids,finish_reason",
    [([CODEC_OFFSET + 1, MUSIC_END], "stop"), ([CODEC_OFFSET + 1], "length")],
)
def test_result_drops_end_token_and_reports_truncation(
    builder: Yue2RequestBuilder, output_ids: list[int], finish_reason: str
) -> None:
    data = builder(payload(cot="off"))
    data.req.output_ids.extend(output_ids)
    state = load_state(builder.result(data), Yue2State)
    assert state.semantic_tokens == [CODEC_OFFSET + 1]
    assert state.semantic_prefix == data.prefix
    assert state.finish_reason == finish_reason


def test_admission_never_splits_a_cfg_pair(builder: Yue2RequestBuilder) -> None:
    single = builder(payload("a", abc="X:1\nK:C\nCDEF"))
    guided = builder(payload("b", cot="off"))
    queue = [
        SimpleNamespace(omni_data=data, origin_input_ids=data.prefix, output_ids=[])
        for data in (single, guided, guided.cfg_uncond)
    ]
    running = SimpleNamespace(reqs=[])

    def limit(allocatable: int, budget: int) -> int:
        stub = SimpleNamespace(
            get_num_allocatable_reqs=lambda running_count: allocatable,
            max_prefill_tokens=budget,
        )
        return Yue2Scheduler.admission_limit(stub, queue, running)

    assert limit(2, 100000) == 1
    assert limit(3, 100000) == 3
    assert limit(3, len(single.prefix)) == 1


def cfg_pair(builder: Yue2RequestBuilder, request_id: str) -> list:
    guided = builder(payload(request_id, cot="off"))
    for data in (guided, guided.cfg_uncond):
        data.req.omni_data = data
    return [guided.req, guided.cfg_uncond.req]


@pytest.mark.parametrize(
    "guided_result,twin_result,admitted_rows",
    [
        (AddReqResult.CONTINUE, AddReqResult.CONTINUE, 2),
        (AddReqResult.NO_TOKEN, None, 0),
        (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN, 0),
    ],
)
def test_prefill_adder_admits_both_cfg_rows_or_neither(
    builder: Yue2RequestBuilder,
    monkeypatch: pytest.MonkeyPatch,
    guided_result: AddReqResult,
    twin_result: AddReqResult | None,
    admitted_rows: int,
) -> None:
    rows = cfg_pair(builder, "b")

    def add_one_req(adder, req, has_chunked_req, truncation_align_size):
        if req is rows[0] or twin_result is AddReqResult.CONTINUE:
            adder.can_run_list.append(req)
        else:
            pass
        return guided_result if req is rows[0] else twin_result

    monkeypatch.setattr(PrefillAdder, "add_one_req", add_one_req)
    adder = object.__new__(Yue2PrefillAdder)
    adder.can_run_list = []
    adder.tree_cache = Mock()
    for req in rows:
        if adder.add_one_req(req, False, None) != AddReqResult.CONTINUE:
            break
        else:
            pass
    assert [id(req) for req in adder.can_run_list] == [
        id(req) for req in rows[:admitted_rows]
    ]
    assert adder.tree_cache.unlock.call_count == (0 if admitted_rows else 1)


@pytest.mark.parametrize("retraction_order", [[0], [1], [0, 1], [1, 0]])
def test_retracting_one_cfg_row_requeues_the_pair(
    builder: Yue2RequestBuilder,
    monkeypatch: pytest.MonkeyPatch,
    retraction_order: list[int],
) -> None:
    unguided = builder(payload("a", abc="X:1\nK:C\nCDEF"))
    unguided.req.omni_data = unguided
    single = unguided.req
    rows = cfg_pair(builder, "b")
    retracted = [rows[index] for index in retraction_order]
    batch = SimpleNamespace(
        reqs=[single] + [row for row in rows if all(row is not r for r in retracted)],
        release_req=Mock(),
    )

    def filter_batch(keep_indices: list[int]) -> None:
        batch.reqs = [batch.reqs[index] for index in keep_indices]

    batch.filter_batch = filter_batch
    monkeypatch.setattr(
        OmniScheduler,
        "_add_request_to_queue",
        lambda scheduler, req, is_retracted=False: scheduler.waiting_queue.append(req),
    )
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.waiting_queue = []
    scheduler.decoding_batch = batch
    for req in retracted:
        scheduler._add_request_to_queue(  # noqa: leading-underscore  # upstream spelling
            req, is_retracted=True
        )
    assert [id(req) for req in scheduler.waiting_queue] == [id(req) for req in rows]
    assert [id(req) for req in batch.reqs] == [id(single)]
    assert batch.release_req.call_count == 2 - len(retracted)


@pytest.mark.parametrize("client_abort", [True, False])
def test_retracted_row_with_an_aborted_twin(
    builder: Yue2RequestBuilder, monkeypatch: pytest.MonkeyPatch, client_abort: bool
) -> None:
    guided, twin = cfg_pair(builder, "b")
    twin.to_finish = object()
    guided.to_finish = object() if client_abort else None
    batch = SimpleNamespace(reqs=[twin] if client_abort else [], release_req=Mock())

    def filter_batch(keep_indices: list[int]) -> None:
        batch.reqs = [batch.reqs[index] for index in keep_indices]

    batch.filter_batch = filter_batch
    monkeypatch.setattr(
        OmniScheduler,
        "_add_request_to_queue",
        lambda scheduler, req, is_retracted=False: scheduler.waiting_queue.append(req),
    )
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.waiting_queue = []
    scheduler.decoding_batch = batch
    scheduler.emit_request_error = Mock()
    scheduler._add_request_to_queue(  # noqa: leading-underscore  # upstream spelling
        guided, is_retracted=True
    )
    # A client abort requeues the pair whole; a twin SGLang aborted for memory fails the request.
    queued = [guided, twin] if client_abort else []
    assert [id(req) for req in scheduler.waiting_queue] == [id(req) for req in queued]
    assert batch.reqs == []
    assert scheduler.emit_request_error.call_count == (0 if client_abort else 1)


def test_cfg_pair_above_the_prefill_budget_is_admitted_whole(
    builder: Yue2RequestBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = cfg_pair(builder, "b")
    pair_tokens = sum(len(req.origin_input_ids) for req in rows)
    budgets = []
    monkeypatch.setattr(
        OmniScheduler,
        "get_new_batch_prefill",
        lambda scheduler, running_batch: budgets.append(scheduler.max_prefill_tokens),
    )
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.waiting_queue = list(rows)
    scheduler.max_prefill_tokens = pair_tokens - 1
    scheduler.get_num_allocatable_reqs = lambda running_count: 16
    scheduler.get_new_batch_prefill(SimpleNamespace(reqs=[]))
    assert budgets == [2 * pair_tokens - 1]
    assert scheduler.max_prefill_tokens == pair_tokens - 1


def capacity_scheduler(kv_tokens: int) -> Yue2Scheduler:
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(effective_max_total_num_tokens=kv_tokens)
    )
    scheduler.page_size = 1
    scheduler.emit_request_error = Mock()
    scheduler.abort = Mock()
    return scheduler


def test_cfg_pair_that_cannot_fit_the_kv_cache_is_a_client_error(
    builder: Yue2RequestBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each row fit alone, so the pair waited forever and blocked the queue."""
    enqueued = []
    monkeypatch.setattr(
        OmniScheduler,
        "enqueue_built_request",
        lambda scheduler, payload, done, data, **kwargs: enqueued.append(data),
    )
    scheduler = capacity_scheduler(4096)
    scheduler.waiting_queue = []
    guided = builder(payload("b", token_budget=2500, cot="off"))
    row_tokens = len(guided.prefix) + 2500
    assert row_tokens < 4096 < 2 * row_tokens
    pair_tokens = row_tokens + len(guided.cfg_uncond.prefix) + 2500
    # SGLang admits the pair only with a spare page per row and one free token.
    # A retracted pair replays its output while SGLang reserves the first row's budget again.
    resumed_tokens = pair_tokens + 2500 + 1
    assert capacity_scheduler(resumed_tokens + 3).kv_capacity_error(guided) is None
    assert capacity_scheduler(resumed_tokens + 2).kv_capacity_error(guided) is not None
    # SGLang rounds each prefix up to whole pages.
    paged_tokens = (
        3 * 2500
        + 16
        + sum(-(-len(row.prefix) // 16) * 16 for row in (guided, guided.cfg_uncond))
    )
    for pool, fits in ((paged_tokens + 33, True), (paged_tokens + 32, False)):
        paged = capacity_scheduler(pool)
        paged.page_size = 16
        assert (paged.kv_capacity_error(guided) is None) == fits
    scheduler.enqueue_built_request(guided.stage_payload, False, guided)
    assert enqueued == []
    request_id, error = scheduler.emit_request_error.call_args.args
    assert request_id == "b" and generation_error_status_code(error) == 400
    assert "max_new_tokens" in str(error) and "max_total_tokens" in str(error)
    scheduler.abort.assert_called_once_with("b")

    unguided = builder(payload("c", token_budget=2500, abc="X:1\nK:C\nCDEF"))
    scheduler.enqueue_built_request(unguided.stage_payload, False, unguided)
    assert enqueued == [unguided]


def test_score_budget_above_the_kv_cache_names_its_own_setting(
    builder: Yue2RequestBuilder,
) -> None:
    score = builder(payload("a"))
    error = capacity_scheduler(4096).kv_capacity_error(score)
    assert "abc_max_tokens" in error and "thinker" not in error
    assert capacity_scheduler(8192).kv_capacity_error(score) is None


def test_paused_generation_requeues_cfg_rows_next_to_their_request(
    builder: Yue2RequestBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    unguided = builder(payload("a", abc="X:1\nK:C\nCDEF"))
    unguided.req.omni_data = unguided
    single = unguided.req
    first, second = cfg_pair(builder, "b"), cfg_pair(builder, "c")
    # SGLang orders a batch by length when it retracts for memory.
    batch_order = [second[1], first[1], single, second[0], first[0]]
    monkeypatch.setattr(
        OmniScheduler,
        "retract_running_requests",
        lambda scheduler: scheduler.waiting_queue.extend(batch_order) or 5,
    )
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.waiting_queue = []
    assert scheduler.retract_running_requests() == 5
    expected = [single, *second, *first]
    assert [id(req) for req in scheduler.waiting_queue] == [id(req) for req in expected]
    lone_twin = [first[1], single]
    assert [id(req) for req in paired_queue(lone_twin)] == [
        id(req) for req in lone_twin
    ]


def test_retracted_row_resumes_sampling_from_its_committed_tokens() -> None:
    """The lookahead step SGLang drops left the position and window one token ahead."""
    data = Yue2ARRequestData(sampling=SamplingConfig(penalty_window=4))
    rewind_sampling(data, [])
    assert data.generated_tokens == 0 and data.recent_tokens is None
    data.generated_tokens = 7
    data.recent_tokens = torch.tensor([14, 15, 16, 13])
    rewind_sampling(data, [10, 11, 12, 13, 14, 15])
    assert data.generated_tokens == 6
    assert data.recent_tokens.tolist() == [14, 15, 12, 13]
    rewind_sampling(data, [10, 11])
    assert data.recent_tokens.tolist() == [10, 11, -1, -1]


def test_requeued_retracted_row_is_rewound(
    builder: Yue2RequestBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    queued = []
    monkeypatch.setattr(
        OmniScheduler,
        "_add_request_to_queue",
        lambda scheduler, req, is_retracted=False: queued.append(req),
    )
    data = builder(payload("a", abc="X:1\nK:C\nCDEF"))
    req = data.req
    req.omni_data = data
    req.output_ids = [10, 11, 12]
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.decoding_batch = None
    data.generated_tokens = 4
    scheduler._add_request_to_queue(  # noqa: leading-underscore  # upstream spelling
        req, is_retracted=False
    )
    assert data.generated_tokens == 4
    req.is_retracted = True
    scheduler._add_request_to_queue(  # noqa: leading-underscore  # upstream spelling
        req, is_retracted=True
    )
    assert data.generated_tokens == 3
    assert queued == [req, req]


def test_finished_score_starts_the_audio_request(builder: Yue2RequestBuilder) -> None:
    score = builder(payload())
    scheduler = object.__new__(Yue2Scheduler)
    scheduler.request_builder = builder
    scheduler.aborted_request_ids = set()
    scheduler.enqueue_built_request = Mock()
    scheduler.start_audio_phase(SimpleNamespace(output_ids=[11, 12, ABC_END]), score)
    message, _, audio = scheduler.enqueue_built_request.call_args.args
    assert message is score.stage_payload and audio.req.rid == "song"
    assert audio.prefix == builder.tokenizer.prefix(score.request, [11, 12])
    assert not audio.abc_truncated
    scheduler.start_audio_phase(SimpleNamespace(output_ids=[11, 12]), score)
    assert scheduler.enqueue_built_request.call_args.args[2].abc_truncated
    scheduler.enqueue_built_request.reset_mock()
    scheduler.aborted_request_ids = {"song"}
    scheduler.start_audio_phase(SimpleNamespace(output_ids=[11, ABC_END]), score)
    scheduler.enqueue_built_request.assert_not_called()


def test_rejected_request_is_logged_without_a_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler = object.__new__(Yue2Scheduler)
    for error in (ValueError("YuE2 request: style is required"), RuntimeError("bug")):
        try:
            raise error
        except Exception as raised:
            scheduler.log_request_builder_failure("song", raised)
    rejected, failed = caplog.records[-2:]
    assert rejected.exc_info is None and "style is required" in rejected.getMessage()
    assert failed.exc_info is not None


def test_internal_request_ids_map_to_the_client_request() -> None:
    assert public_request_id("song-abc") == public_request_id("song-cfg") == "song"
    assert public_request_id("song") == "song"


def runner_request(request_id: str, data: Yue2ARRequestData) -> SchedulerRequest:
    return SchedulerRequest(request_id=request_id, data=data)


def test_runner_gives_twin_its_request_token_and_gates_the_end() -> None:
    sampling = SamplingConfig(
        temperature=0.0, repetition_penalty=1.0, min_tokens=1, max_tokens=4
    )
    twin = Yue2ARRequestData(sampling=sampling, is_cfg_uncond=True)
    guided = Yue2ARRequestData(sampling=sampling, cfg_scale=2.0, cfg_uncond=twin)
    single = Yue2ARRequestData(sampling=sampling)
    requests = [
        runner_request("a", guided),
        runner_request("a-cfg", twin),
        runner_request("b", single),
    ]
    logits = torch.zeros(3, 184704)
    logits[0, CODEC_OFFSET + 1] = 2.0
    logits[0, CODEC_OFFSET + 2] = 1.5
    logits[1, CODEC_OFFSET + 2] = 3.0
    logits[2, MUSIC_END] = 9.0
    logits[2, CODEC_OFFSET + 5] = 1.0
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=logits), next_token_ids=None
    )
    runner = object.__new__(Yue2ModelRunner)
    runner.sample(result, requests)
    # CFG: 0 + 2 * (2 - 0) = 4 for token 1 beats 3 + 2 * (1.5 - 3) = 0 for token 2.
    assert result.next_token_ids.tolist() == [
        CODEC_OFFSET + 1,
        CODEC_OFFSET + 1,
        CODEC_OFFSET + 5,
    ]
    assert guided.generated_tokens == single.generated_tokens == 1
    assert twin.generated_tokens == 0
    assert guided.recent_tokens[0].item() == CODEC_OFFSET + 1


def test_runner_matches_cfg_rows_after_sglang_reorders_the_batch() -> None:
    sampling = SamplingConfig()
    twin = Yue2ARRequestData(sampling=sampling, is_cfg_uncond=True)
    guided = Yue2ARRequestData(sampling=sampling, cfg_uncond=twin)
    requests = [
        runner_request("a-cfg", twin),
        runner_request("b", Yue2ARRequestData(sampling=sampling)),
        runner_request("a", guided),
    ]
    rows = sampling_rows(requests)
    assert [(row.conditional, row.unconditional) for row in rows] == [(1, None), (2, 0)]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rows_sampled_together_match_rows_sampled_alone() -> None:
    """On CUDA, a row sampled next to other rows drew its noise for the wrong position."""
    logits = torch.randn((8, 184704), generator=torch.Generator().manual_seed(0)).cuda()

    def members() -> list[Yue2ARRequestData]:
        return [
            Yue2ARRequestData(
                sampling=SamplingConfig(),
                sampling_seed=100 + index,
                generated_tokens=300 + 7 * index,
            )
            for index in range(8)
        ]

    together = Yue2ModelRunner.sample_group(logits, members(), "semantic", False)
    alone = [
        Yue2ModelRunner.sample_group(
            logits[index : index + 1], [member], "semantic", False
        ).item()
        for index, member in enumerate(members())
    ]
    assert together.tolist() == alone


def test_runner_rejects_a_twin_without_its_request() -> None:
    sampling = SamplingConfig()
    twin = Yue2ARRequestData(sampling=sampling, is_cfg_uncond=True)
    guided = Yue2ARRequestData(sampling=sampling, cfg_uncond=twin)
    with pytest.raises(RuntimeError, match="CFG"):
        sampling_rows([runner_request("a-cfg", twin)])
    with pytest.raises(RuntimeError, match="CFG"):
        sampling_rows([runner_request("a", guided)])


def test_kv_cache_is_sized_in_tokens_unless_overridden() -> None:
    engine_builder = Yue2EngineBuilder(max_running_requests=16)
    engine_builder.context_length = 24576
    defaults: dict[str, object] = {}
    engine_builder.adjust_overrides(defaults)
    assert defaults["max_total_tokens"] == 65536
    engine_builder.context_length = 40000
    longer_context: dict[str, object] = {}
    engine_builder.adjust_overrides(longer_context)
    assert longer_context["max_total_tokens"] == 80000
    user_choice: dict[str, object] = {"max_total_tokens": 200000}
    engine_builder.adjust_overrides(user_choice)
    assert user_choice["max_total_tokens"] == 200000
