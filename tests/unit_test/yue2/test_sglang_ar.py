# SPDX-License-Identifier: Apache-2.0
"""YuE2 SGLang AR stage: request phases, CFG pairing and row sampling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tiktoken
import torch

from sglang_omni.models.yue2.model_runner import Yue2ModelRunner, sampling_rows
from sglang_omni.models.yue2.payload_types import Yue2State
from sglang_omni.models.yue2.request import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    MUSIC_END,
    SamplingConfig,
    Yue2Tokenizer,
)
from sglang_omni.models.yue2.scheduler import Yue2Scheduler
from sglang_omni.models.yue2.sglang_request_builder import (
    Yue2ARRequestData,
    Yue2RequestBuilder,
    public_request_id,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
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
    return Yue2RequestBuilder(tokenizer, context_tokens=24576, vocab_size=184704)


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
        SimpleNamespace(omni_data=data, origin_input_ids=data.prefix)
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


def test_runner_rejects_a_twin_without_its_request() -> None:
    sampling = SamplingConfig()
    twin = Yue2ARRequestData(sampling=sampling, is_cfg_uncond=True)
    guided = Yue2ARRequestData(sampling=sampling, cfg_uncond=twin)
    with pytest.raises(RuntimeError, match="CFG"):
        sampling_rows([runner_request("a-cfg", twin)])
    with pytest.raises(RuntimeError, match="CFG"):
        sampling_rows([runner_request("a", guided)])
