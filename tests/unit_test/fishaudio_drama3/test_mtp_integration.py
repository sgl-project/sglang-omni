# SPDX-License-Identifier: Apache-2.0
"""Drama 3 tower integration on CPU: requests, staging, frame labels, rejections."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import random
from array import array
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Protocol
from unittest.mock import ANY, Mock, call, patch

import pytest
import torch
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor import model_runner as sglang_model_runner
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.base import PendingStep
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.models.fishaudio_drama3 import engine_builder, sglang_model
from sglang_omni.models.fishaudio_drama3.audio_decoder import AudioDecoder
from sglang_omni.models.fishaudio_drama3.config import Drama3VocoderFactoryArgs
from sglang_omni.models.fishaudio_drama3.engine_builder import (
    DECODE_STEPS_BETWEEN_PREFILLS,
    PREFILL_CUDA_GRAPH_MAX_TOKENS,
    Drama3EngineBuilder,
)
from sglang_omni.models.fishaudio_drama3.first_frame_graphs import (
    Drama3FirstFrameGraphs,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.model_runner import (
    Drama3DecodeStatePool,
    Drama3ModelRunner,
    extend_frame_rows,
    ras_window,
    request_data_of,
)
from sglang_omni.models.fishaudio_drama3.request_builders import (
    MINIMUM_PROMPT_ROWS,
    Drama3FrameStreamer,
    Drama3GenerationInput,
    Drama3GenerationResult,
    Drama3RequestBuilder,
    Drama3RequestData,
    Drama3ResultAdapter,
)
from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import PHILOX_SEED_MODULUS
from sglang_omni.models.fishaudio_drama3.sglang_model import (
    Drama3DecodeBuffers,
    Drama3ForCausalLM,
    Drama3StepInputs,
    Drama3TowerPrefill,
    StepExchange,
    next_frame_labels,
)
from sglang_omni.models.fishaudio_drama3.tower_kernels import (
    FUSED_KERNELS,
    MAX_FUSED_ROWS,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling import sglang_backend
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.serve.openai_errors import is_bad_request_error
from sglang_omni.vendor.sglang.core import GenerationBatchResult, ScheduleBatch
from tests.unit_test.fishaudio_drama3.dense_attention import DenseAttention
from tests.unit_test.fishaudio_drama3.test_audio_decoder import (
    CODEBOOK_SIZE,
    DIM,
    EOS_TOKEN_ID,
    HEAD_COUNT,
    HEAD_DIM,
    KV_HEAD_COUNT,
    SEMANTIC_BEGIN_ID,
    TEXT_VOCAB_SIZE,
    loaded_decoder,
    random_checkpoint,
    torch_fused_op_backend,
)
from tests.unit_test.fishaudio_drama3.test_first_frame_graphs import (
    cpu_first_frame_graphs,
)
from tests.unit_test.fishaudio_drama3.test_hf_config import fish_config_dict

FRAME_WIDTH = 5
CONTEXT_LENGTH = 32
MAX_PROMPT_ROWS = 6
MAX_NEW_TOKENS_CAP = 16
# The prompt cap the serving contract asks for by default.
SERVING_PROMPT_CAP_ROWS = 8192
# The new-frame cap the serving contract asks for by default.
SERVING_MAX_NEW_TOKENS_CAP = 2048
SamplingField = Literal["max_new_tokens", "temperature", "top_k", "top_p"]


class GenerationInputChange(Protocol):
    def __call__(
        self, generation_input: Drama3GenerationInput
    ) -> Drama3GenerationInput: ...


def frame_rows_of(lengths: list[int], seed: int) -> list[list[list[int]]]:
    """Per request, alternating text rows (codes 0) and semantic rows."""
    generator = torch.Generator().manual_seed(seed)
    requests = []
    for length in lengths:
        rows = []
        for row_index in range(length):
            codes = torch.randint(0, CODEBOOK_SIZE, (4,), generator=generator).tolist()
            if row_index % 2 == 0:
                rows.append(
                    [int(torch.randint(10, 79, (1,), generator=generator))] + [0] * 4
                )
            else:
                rows.append([SEMANTIC_BEGIN_ID + codes[0]] + codes)
        requests.append(rows)
    return requests


def generation_input(prompt_rows: torch.Tensor) -> Drama3GenerationInput:
    return Drama3GenerationInput(
        prompt_rows=prompt_rows,
        max_new_tokens=4,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
    )


def built_request(
    prompt_rows: list[list[int]], request_id: str = "request-a"
) -> Drama3RequestData:
    builder = Drama3RequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=MAX_PROMPT_ROWS,
        max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
    )
    return builder(
        StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="prompt"),
            data=generation_input(torch.tensor(prompt_rows, dtype=torch.int64)),
        )
    )


def test_ras_window_is_the_last_sixteen_main_tokens_of_the_history() -> None:
    history = [[token] + [0] * 4 for token in range(10, 13)]
    assert ras_window(history) == [EMPTY_RAS_SLOT] * 13 + [10, 11, 12]

    history = [[token] + [0] * 4 for token in range(30)]
    window = ras_window(history)
    assert window == list(range(14, 30))
    history.append([81, 1, 2, 3, 4])
    assert ras_window(history) == window[1:] + [81]


def test_ras_window_of_a_prompt_keeps_its_end_of_turn_token() -> None:
    """The first frame's window is the prompt tail, eos included, never all empty."""
    prompt_tail = [21, 22, EOS_TOKEN_ID, 23, 24, 25, 26]
    window = ras_window([[token] + [0] * 4 for token in prompt_tail])

    assert EOS_TOKEN_ID in window
    assert window[-len(prompt_tail) :] == prompt_tail


def test_request_builder_keeps_the_rows_and_stops_on_eos() -> None:
    prompt_rows = frame_rows_of([5], seed=0)[0]

    data = built_request(prompt_rows)

    req = data.req
    assert req is not None
    assert req.rid == "request-a"
    assert list(req.origin_input_ids) == [row[0] for row in prompt_rows]
    assert req.eos_token_ids == {EOS_TOKEN_ID}
    assert EOS_TOKEN_ID in req.sampling_params.stop_token_ids
    assert data.frame_rows == prompt_rows
    assert data.prompt_row_count == 5
    assert data.input_ids is not None
    assert data.input_ids.tolist() == [row[0] for row in prompt_rows]
    assert (data.temperature, data.top_k, data.top_p) == (0.8, 5, 0.9)


def changed_rows(rows: torch.Tensor) -> GenerationInputChange:
    def change(generation_input: Drama3GenerationInput) -> Drama3GenerationInput:
        return replace(generation_input, prompt_rows=rows)

    return change


def changed_field(name: SamplingField, value: int | float) -> GenerationInputChange:
    def change(generation_input: Drama3GenerationInput) -> Drama3GenerationInput:
        return replace(generation_input, **{name: value})

    return change


GOOD_ROWS = torch.tensor(frame_rows_of([4], seed=1)[0], dtype=torch.int64)


def with_entry(row: int, column: int, value: int) -> torch.Tensor:
    rows = GOOD_ROWS.clone()
    rows[row, column] = value
    return rows


REJECTED_INPUTS = [
    pytest.param(changed_rows(GOOD_ROWS.float()), "int64", id="float-rows"),
    pytest.param(changed_rows(GOOD_ROWS[:, :4]), "int64", id="narrow-rows"),
    pytest.param(changed_rows(GOOD_ROWS[0, 0]), "int64", id="scalar-rows"),
    pytest.param(changed_rows(GOOD_ROWS[:1]), "at least 2 rows", id="one-row"),
    pytest.param(
        changed_rows(with_entry(0, 0, TEXT_VOCAB_SIZE)),
        "main tokens",
        id="token-past-vocab",
    ),
    pytest.param(
        changed_rows(with_entry(0, 0, -1)), "main tokens", id="negative-token"
    ),
    pytest.param(
        changed_rows(with_entry(1, 2, CODEBOOK_SIZE)), "codes", id="code-past-codebook"
    ),
    pytest.param(
        changed_field("max_new_tokens", 0), "max_new_tokens", id="no-new-tokens"
    ),
    pytest.param(
        changed_field("max_new_tokens", CONTEXT_LENGTH),
        "context length",
        id="past-context",
    ),
    pytest.param(
        changed_field("max_new_tokens", MAX_NEW_TOKENS_CAP + 1),
        f"at most {MAX_NEW_TOKENS_CAP}",
        id="past-cap",
    ),
    pytest.param(
        changed_field("temperature", -0.5), "temperature", id="negative-temperature"
    ),
    pytest.param(
        changed_field("temperature", math.nan), "temperature", id="nan-temperature"
    ),
    pytest.param(changed_field("top_k", 0), "top_k", id="zero-top-k"),
    pytest.param(changed_field("top_k", -1), "top_k", id="disabled-top-k"),
    pytest.param(changed_field("top_k", 2**31), "top_k", id="top-k-past-int32"),
    pytest.param(changed_field("top_p", 0.0), "top_p", id="zero-top-p"),
    pytest.param(changed_field("top_p", 1.5), "top_p", id="top-p-above-one"),
]


@pytest.mark.parametrize(("change", "error"), REJECTED_INPUTS)
def test_request_builder_rejects_bad_inputs(
    change: GenerationInputChange, error: str
) -> None:
    builder = Drama3RequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=MAX_PROMPT_ROWS,
        max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
    )
    payload = StagePayload(
        request_id="request-a",
        request=OmniRequest(inputs="prompt"),
        data=change(generation_input(GOOD_ROWS)),
    )

    with pytest.raises(ValueError, match=error) as rejection:
        builder(payload)

    assert is_bad_request_error(rejection.value)


def unset_max_new_tokens_payload(prompt_row_count: int) -> StagePayload:
    prompt_rows = torch.tensor(
        frame_rows_of([prompt_row_count], seed=6)[0], dtype=torch.int64
    )
    return StagePayload(
        request_id="request-a",
        request=OmniRequest(inputs="prompt"),
        data=replace(generation_input(prompt_rows), max_new_tokens=None),
    )


@pytest.mark.parametrize(
    ("context_length", "max_new_tokens_cap", "prompt_row_count", "max_new_tokens"),
    [
        pytest.param(
            CONTEXT_LENGTH, MAX_NEW_TOKENS_CAP, 4, MAX_NEW_TOKENS_CAP, id="cap"
        ),
        pytest.param(
            CONTEXT_LENGTH, CONTEXT_LENGTH, 4, CONTEXT_LENGTH - 4, id="context-left"
        ),
        pytest.param(
            MAX_PROMPT_ROWS,
            MAX_NEW_TOKENS_CAP,
            MAX_PROMPT_ROWS - 1,
            1,
            id="one-row-left",
        ),
    ],
)
def test_unset_max_new_tokens_fills_the_context_up_to_the_cap(
    context_length: int,
    max_new_tokens_cap: int,
    prompt_row_count: int,
    max_new_tokens: int,
) -> None:
    builder = Drama3RequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=context_length,
        max_prompt_rows=MAX_PROMPT_ROWS,
        max_new_tokens_cap=max_new_tokens_cap,
    )

    request_data = builder(unset_max_new_tokens_payload(prompt_row_count))

    assert request_data.req is not None
    assert request_data.max_new_tokens == max_new_tokens
    assert request_data.req.sampling_params.max_new_tokens == max_new_tokens


def test_unset_max_new_tokens_with_no_context_left_is_a_bad_request() -> None:
    builder = Drama3RequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=MAX_PROMPT_ROWS,
        max_prompt_rows=MAX_PROMPT_ROWS,
        max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
    )

    with pytest.raises(ValueError, match="leaves no room") as rejection:
        builder(unset_max_new_tokens_payload(MAX_PROMPT_ROWS))

    assert is_bad_request_error(rejection.value)


def test_request_builder_rejects_a_foreign_payload() -> None:
    builder = Drama3RequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=MAX_PROMPT_ROWS,
        max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
    )
    payload = StagePayload(
        request_id="request-a", request=OmniRequest(inputs="prompt"), data=[1, 2, 3]
    )

    with pytest.raises(TypeError, match="Drama3GenerationInput"):
        builder(payload)


def test_prompt_cap_is_inclusive_and_refused_as_a_bad_request() -> None:
    built_request(frame_rows_of([MAX_PROMPT_ROWS], seed=4)[0])

    with pytest.raises(ValueError) as rejection:
        built_request(frame_rows_of([MAX_PROMPT_ROWS + 1], seed=4)[0])

    assert is_bad_request_error(rejection.value)


class ConfigOnlyModel(Drama3ForCausalLM):
    """Drama3ForCausalLM for make_adapters, which reads only the config."""

    def __init__(self, config: FishOmniConfig) -> None:
        torch.nn.Module.__init__(self)
        self.config = config


def test_default_prompt_cap_reaches_the_request_builder() -> None:
    """8192 rows pass and 8193 are refused; the context budget is still checked."""
    request_builder, _ = Drama3EngineBuilder(
        context_length=SERVING_PROMPT_CAP_ROWS + 5, max_running_requests=1
    ).make_adapters(ConfigOnlyModel(FishOmniConfig(**fish_config_dict())))
    capped_rows, past_cap_rows = (
        torch.tensor(rows, dtype=torch.int64)
        for rows in frame_rows_of(
            [SERVING_PROMPT_CAP_ROWS, SERVING_PROMPT_CAP_ROWS + 1], seed=5
        )
    )

    def payload(prompt_rows: torch.Tensor, max_new_tokens: int) -> StagePayload:
        return StagePayload(
            request_id="request-a",
            request=OmniRequest(inputs="prompt"),
            data=replace(generation_input(prompt_rows), max_new_tokens=max_new_tokens),
        )

    request_data = request_builder(payload(capped_rows, max_new_tokens=5))
    assert request_data.prompt_row_count == SERVING_PROMPT_CAP_ROWS
    with pytest.raises(ValueError, match="exceeds the maximum allowed length"):
        request_builder(payload(past_cap_rows, max_new_tokens=4))
    with pytest.raises(ValueError, match="context length"):
        request_builder(payload(capped_rows, max_new_tokens=6))


def test_engine_builder_wires_the_default_cap_and_a_frame_streamer() -> None:
    """A request without max_new_tokens gets 2048 frames at most, and the scheduler
    streams through the frame streamer make_adapters made."""
    engine_builder = Drama3EngineBuilder(
        context_length=2 * SERVING_MAX_NEW_TOKENS_CAP, max_running_requests=1
    )
    request_builder, _ = engine_builder.make_adapters(
        ConfigOnlyModel(FishOmniConfig(**fish_config_dict()))
    )

    request_data = request_builder(unset_max_new_tokens_payload(4))
    frame_streamer = engine_builder.extra_scheduler_kwargs()["stream_output_builder"]

    window = Drama3VocoderFactoryArgs()
    assert request_data.max_new_tokens == SERVING_MAX_NEW_TOKENS_CAP
    assert isinstance(frame_streamer, Drama3FrameStreamer)
    assert frame_streamer.eos_token_id == EOS_TOKEN_ID
    assert (
        frame_streamer.stream_stride,
        frame_streamer.stream_followup_stride_ramp,
        frame_streamer.stream_followup_stride,
    ) == (
        window.stream_stride,
        window.stream_followup_stride_ramp,
        window.stream_followup_stride,
    )


@pytest.mark.parametrize(
    ("enable_async_decode", "expected_log"),
    [
        pytest.param(
            False,
            "Drama 3 async decode: off, tts_engine.factory.enable_async_decode is false",
            id="off",
        ),
        pytest.param(
            True, "Drama 3 async decode: on, lookahead from batch size 3", id="on"
        ),
    ],
)
def test_engine_builder_hands_the_async_decode_switches_to_the_scheduler(
    caplog: pytest.LogCaptureFixture, enable_async_decode: bool, expected_log: str
) -> None:
    builder = Drama3EngineBuilder(
        context_length=CONTEXT_LENGTH,
        max_running_requests=1,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=3,
    )
    builder.make_adapters(ConfigOnlyModel(FishOmniConfig(**fish_config_dict())))

    with caplog.at_level(logging.INFO, logger=engine_builder.logger.name):
        scheduler_kwargs = builder.extra_scheduler_kwargs()

    assert expected_log in caplog.text
    assert (
        scheduler_kwargs["enable_async_decode"],
        scheduler_kwargs["async_decode_min_batch_size"],
        scheduler_kwargs["async_decode_resolve_before_prefill"],
    ) == (enable_async_decode, 3, True)


@pytest.mark.parametrize(
    "max_prompt_rows",
    [
        True,
        1,
        float(SERVING_PROMPT_CAP_ROWS),
    ],
)
def test_prompt_cap_must_be_an_integer_of_at_least_two_rows(
    max_prompt_rows: int | float,
) -> None:
    with pytest.raises(ValueError, match="max_prompt_rows"):
        Drama3RequestBuilder(
            config=FishOmniConfig(**fish_config_dict()),
            context_length=CONTEXT_LENGTH,
            max_prompt_rows=max_prompt_rows,
            max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
        )


@pytest.mark.parametrize(
    ("sampled_main_tokens", "kept_count", "finish_reason"),
    [
        ([81, 84], 2, "length"),
        ([81, EOS_TOKEN_ID], 1, "stop"),
        ([EOS_TOKEN_ID], 0, "stop"),
    ],
    ids=["no-eos", "eos-last", "only-eos"],
)
def test_result_holds_the_sampled_frames_without_eos(
    sampled_main_tokens: list[int], kept_count: int, finish_reason: str
) -> None:
    data = built_request(frame_rows_of([3], seed=2)[0])
    sampled = [[token, 1, 2, 3, 0] for token in sampled_main_tokens]
    data.frame_rows.extend(sampled)
    data.finish_reason = finish_reason

    payload = Drama3ResultAdapter(eos_token_id=EOS_TOKEN_ID)(data)

    result = payload.data
    assert isinstance(result, Drama3GenerationResult)
    assert result.frames.shape == (kept_count, FRAME_WIDTH)
    assert result.frames.tolist() == sampled[:kept_count]
    assert (result.prompt_row_count, result.finish_reason) == (3, finish_reason)


def window_frame_streamer() -> Drama3FrameStreamer:
    window = Drama3VocoderFactoryArgs()
    return Drama3FrameStreamer(
        eos_token_id=EOS_TOKEN_ID,
        stream_stride=window.stream_stride,
        stream_followup_stride_ramp=window.stream_followup_stride_ramp,
        stream_followup_stride=window.stream_followup_stride,
    )


def sampled_row(frame_index: int) -> list[int]:
    return [81, frame_index, 2, 3, 0]


def streaming_request() -> Drama3RequestData:
    request_data = built_request(frame_rows_of([3], seed=6)[0])
    payload = request_data.stage_payload
    assert payload is not None
    payload.request.params["stream"] = True
    return request_data


def test_frame_streamer_sends_the_first_window_by_frame_then_whole_windows() -> None:
    request_data = built_request(frame_rows_of([3], seed=6)[0])
    payload = request_data.stage_payload
    assert payload is not None
    frame_streamer = window_frame_streamer()
    request_output = RequestOutput(request_id="request-a")
    request_data.frame_rows.append(sampled_row(0))

    assert frame_streamer("request-a", request_data, request_output) == []
    payload.request.params["stream"] = True
    sent: list[tuple[int, OutgoingMessage]] = []
    for frame_index in range(1, 130):
        request_data.frame_rows.append(sampled_row(frame_index))
        for message in frame_streamer("request-a", request_data, request_output):
            sent.append((frame_index + 1, message))

    first_window = [(frame_index, frame_index + 1) for frame_index in range(10)]
    assert request_data.prompt_row_count == 3
    assert [frame_count for frame_count, _ in sent] == [
        *range(2, 11),
        20,
        32,
        64,
        128,
    ]
    for (start, end), (_, message) in zip(
        [(0, 2), *first_window[2:], (10, 20), (20, 32), (32, 64), (64, 128)], sent
    ):
        assert (message.request_id, message.type, message.target) == (
            "request-a",
            "stream",
            "vocoder",
        )
        assert message.metadata == {"modality": "audio_codes"}
        assert isinstance(message.data, torch.Tensor)
        assert message.data.dtype == torch.int64
        assert message.data.T.tolist() == [
            sampled_row(frame_index) for frame_index in range(start, end)
        ]


def test_frame_streamer_sends_the_window_a_multi_frame_step_completes() -> None:
    request_data = streaming_request()
    frame_streamer = window_frame_streamer()
    request_output = RequestOutput(request_id="request-a")
    sent: list[list[list[int]]] = []
    for frame_indices in (range(0, 9), range(9, 11), range(11, 20), range(20, 21)):
        request_data.frame_rows.extend(map(sampled_row, frame_indices))
        for message in frame_streamer("request-a", request_data, request_output):
            assert isinstance(message.data, torch.Tensor)
            sent.append(message.data.T.tolist())

    assert sent == [
        [sampled_row(frame_index) for frame_index in range(0, 9)],
        [sampled_row(9)],
        [sampled_row(frame_index) for frame_index in range(10, 20)],
    ]


def test_frame_streamer_leaves_an_eos_on_a_window_end_to_the_result() -> None:
    request_data = streaming_request()
    frame_streamer = window_frame_streamer()
    request_output = RequestOutput(request_id="request-a")
    sent: list[OutgoingMessage] = []
    for frame_index in range(19):
        request_data.frame_rows.append(sampled_row(frame_index))
        sent += frame_streamer("request-a", request_data, request_output)
    request_data.frame_rows.append([EOS_TOKEN_ID, 0, 0, 0, 0])
    eos_step_messages = frame_streamer("request-a", request_data, request_output)

    assert len(sent) == 10
    assert eos_step_messages == []
    assert request_data.streamed_frame_count == 10


def test_streamed_and_result_frames_stay_on_cpu_under_another_default_device() -> None:
    request_data = streaming_request()
    request_data.frame_rows.extend(map(sampled_row, range(10)))
    request_data.finish_reason = "length"

    with torch.device("meta"):
        [message] = window_frame_streamer()(
            "request-a", request_data, RequestOutput(request_id="request-a")
        )
        result = Drama3ResultAdapter(eos_token_id=EOS_TOKEN_ID)(request_data).data

    assert isinstance(message.data, torch.Tensor)
    assert isinstance(result, Drama3GenerationResult)
    for frames in (message.data.T, result.frames):
        assert (frames.device.type, frames.dtype) == ("cpu", torch.int64)
        assert frames.tolist() == [
            sampled_row(frame_index) for frame_index in range(10)
        ]


def admitted_for_prefill(data: Drama3RequestData, start: int, end: int) -> None:
    """Leave the request as prefill admission does: fill ids rebuilt, extend bounds set."""
    req = data.req
    assert req is not None
    # Without a tree cache this only rebuilds fill ids; no prefix is matched.
    req.init_next_round_input()
    req.prefix_len = start
    req.extend_end = end


def test_prefill_rows_cover_the_whole_history_after_a_retract() -> None:
    prompt_rows = frame_rows_of([4], seed=3)[0]
    data = built_request(prompt_rows)
    req = data.req
    assert req is not None
    admitted_for_prefill(data, 0, 4)
    assert extend_frame_rows(data) == prompt_rows

    sampled = [[82, 2, 0, 1, 3], [85, 5, 1, 1, 1]]
    req.output_ids.extend(array("q", [row[0] for row in sampled]))
    data.frame_rows.extend(sampled)
    admitted_for_prefill(data, 0, 6)
    assert extend_frame_rows(data) == prompt_rows + sampled


@pytest.mark.parametrize(
    "extend_bounds", [(0, 3), (1, 4)], ids=["partial-chunk", "cached-prefix"]
)
def test_prefill_rows_reject_a_partial_extend(extend_bounds: tuple[int, int]) -> None:
    data = built_request(frame_rows_of([4], seed=4)[0])
    admitted_for_prefill(data, *extend_bounds)

    with pytest.raises(RuntimeError, match="whole sequence"):
        extend_frame_rows(data)


def test_prefill_rows_reject_a_history_that_is_not_the_sequence() -> None:
    data = built_request(frame_rows_of([4], seed=4)[0])
    admitted_for_prefill(data, 0, 4)
    data.frame_rows[0] = [data.frame_rows[0][0] + 1] + data.frame_rows[0][1:]

    with pytest.raises(RuntimeError, match="history"):
        extend_frame_rows(data)


def step_inputs(
    request_ids: list[str], row_count: int, marker: int
) -> Drama3StepInputs:
    request_count = len(request_ids)
    return Drama3StepInputs(
        request_ids=request_ids,
        frames=torch.full((row_count, FRAME_WIDTH), marker),
        ras_windows=torch.full((request_count, 16), EMPTY_RAS_SLOT),
        temperature=torch.ones(request_count),
        top_k=torch.ones(request_count, dtype=torch.int64),
        top_p=torch.ones(request_count),
    )


def test_step_exchange_hands_inputs_and_frames_over_once() -> None:
    exchange = StepExchange()
    with pytest.raises(RuntimeError, match="staged no inputs"):
        exchange.take_inputs(["a"], 1)

    exchange.stage(step_inputs(["a", "b"], 2, marker=1))
    assert exchange.take_inputs(["a", "b"], 2).frames[0, 0] == 1
    with pytest.raises(RuntimeError, match="staged no inputs"):
        exchange.take_inputs(["a", "b"], 2)

    exchange.stage(step_inputs(["a", "b"], 2, marker=1))
    with pytest.raises(RuntimeError, match="not the batch"):
        exchange.take_inputs(["b", "a"], 2)
    exchange.stage(step_inputs(["a"], 3, marker=1))
    with pytest.raises(RuntimeError, match="3 frames for 1 rows"):
        exchange.take_inputs(["a"], 1)

    with pytest.raises(RuntimeError, match="no forward has sampled"):
        exchange.take_frames()
    exchange.publish(torch.zeros(1, FRAME_WIDTH))
    assert exchange.take_frames().shape == (1, FRAME_WIDTH)
    with pytest.raises(RuntimeError, match="no forward has sampled"):
        exchange.take_frames()


def test_a_failed_step_never_feeds_the_next_step_of_the_same_requests() -> None:
    """Request ids repeat across steps; each step's own staging decides its input."""
    exchange = StepExchange()

    # A forward that took its inputs and then failed publishes no frames.
    exchange.stage(step_inputs(["a"], 1, marker=1))
    exchange.publish(torch.zeros(1, FRAME_WIDTH))
    exchange.take_inputs(["a"], 1)
    with pytest.raises(RuntimeError, match="no forward has sampled"):
        exchange.take_frames()
    exchange.stage(step_inputs(["a"], 1, marker=2))
    assert exchange.take_inputs(["a"], 1).frames[0, 0] == 2

    # A step that failed before its forward is replaced by the next step's staging.
    exchange.stage(step_inputs(["a"], 1, marker=3))
    exchange.stage(step_inputs(["a"], 1, marker=4))
    assert exchange.take_inputs(["a"], 1).frames[0, 0] == 4

    # A failed prefill clears what it staged.
    exchange.stage(step_inputs(["a"], 1, marker=5))
    exchange.clear()
    with pytest.raises(RuntimeError, match="staged no inputs"):
        exchange.take_inputs(["a"], 1)


def buffer_addresses(buffers: Drama3DecodeBuffers) -> list[int]:
    return [
        getattr(buffers, field.name).data_ptr() for field in dataclasses.fields(buffers)
    ]


@dataclass(frozen=True, kw_only=True)
class StagedRows:
    """The batch rows of the decode buffers, as a forward reads them."""

    frames: list[list[int]]
    ras_windows: list[list[int]]
    sampling_knobs: list[tuple[float, int, float]]


def staged_rows(buffers: Drama3DecodeBuffers, row_count: int) -> StagedRows:
    return StagedRows(
        frames=buffers.frames[:row_count].tolist(),
        ras_windows=buffers.ras_windows[:row_count].tolist(),
        sampling_knobs=list(
            zip(
                buffers.temperature[:row_count].tolist(),
                buffers.top_k[:row_count].tolist(),
                buffers.top_p[:row_count].tolist(),
                strict=True,
            )
        ),
    )


def pool_rows(
    state_pool: Drama3DecodeStatePool, request_slots: list[int]
) -> StagedRows:
    rows = torch.tensor(request_slots, dtype=torch.int64)
    return StagedRows(
        frames=state_pool.frames[rows].tolist(),
        ras_windows=state_pool.ras_windows[rows].tolist(),
        sampling_knobs=list(
            zip(
                state_pool.temperature[rows].tolist(),
                state_pool.top_k[rows].tolist(),
                state_pool.top_p[rows].tolist(),
                strict=True,
            )
        ),
    )


def history_staged_rows(batch_request_data: list[Drama3RequestData]) -> StagedRows:
    row_count = len(batch_request_data)
    buffers = Drama3DecodeBuffers.allocate(
        row_count=row_count, frame_width=FRAME_WIDTH, device=torch.device("cpu")
    )
    newest_frames = []
    for request_data in batch_request_data:
        request = request_data.req
        assert request is not None
        assert len(request_data.frame_rows) == request.seqlen
        newest_frames.append(request_data.frame_rows[-1])
    buffers.frames.copy_(torch.tensor(newest_frames, dtype=torch.int64))
    buffers.ras_windows.copy_(
        torch.tensor(
            [
                ras_window(request_data.frame_rows)
                for request_data in batch_request_data
            ],
            dtype=torch.int64,
        )
    )
    buffers.temperature.copy_(
        torch.tensor(
            [request_data.temperature for request_data in batch_request_data],
            dtype=torch.float32,
        )
    )
    buffers.top_k.copy_(
        torch.tensor(
            [request_data.top_k for request_data in batch_request_data],
            dtype=torch.int64,
        )
    )
    buffers.top_p.copy_(
        torch.tensor(
            [request_data.top_p for request_data in batch_request_data],
            dtype=torch.float32,
        )
    )
    return staged_rows(buffers, row_count)


def test_decode_buffers_start_with_valid_sampling_knobs() -> None:
    buffers = Drama3DecodeBuffers.allocate(
        row_count=3, frame_width=FRAME_WIDTH, device=torch.device("cpu")
    )

    assert buffers.frames.shape == buffers.sampled_frames.shape == (3, FRAME_WIDTH)
    assert not buffers.frames.any() and not buffers.sampled_frames.any()
    assert buffers.ras_windows.tolist() == [[EMPTY_RAS_SLOT] * RAS_WINDOW_FRAMES] * 3
    assert buffers.temperature.tolist() == [1.0] * 3
    assert buffers.top_k.tolist() == [1] * 3
    assert buffers.top_p.tolist() == [1.0] * 3


def tower_prefill_of(row_count: int) -> Drama3TowerPrefill:
    return Drama3TowerPrefill.allocate(
        audio_decoder=Mock(spec=AudioDecoder),
        tower_attention=torch.nn.ModuleList(),
        row_count=row_count,
        frame_width=FRAME_WIDTH,
        device=torch.device("cpu"),
    )


def test_tower_prefill_stages_rows_in_place_over_zeroed_buffers() -> None:
    tower_prefill = tower_prefill_of(6)
    addresses = (
        tower_prefill.input_frames.data_ptr(),
        tower_prefill.next_frames.data_ptr(),
    )
    assert tower_prefill.input_frames.shape == (6, FRAME_WIDTH)
    assert tower_prefill.next_frames.shape == (6, FRAME_WIDTH)
    assert not tower_prefill.input_frames.any()
    assert not tower_prefill.next_frames.any()
    frames = torch.tensor(frame_rows_of([4], seed=3)[0])
    next_frames = next_frame_labels(frames, torch.tensor([3]), 2)

    tower_prefill.stage(frames, next_frames)
    assert torch.equal(tower_prefill.input_frames[:4], frames)
    assert torch.equal(tower_prefill.next_frames[:4], next_frames)
    assert not tower_prefill.input_frames[4:].any()
    tower_prefill.stage(frames[:2], next_frames[:2])

    assert torch.equal(tower_prefill.input_frames[:4], frames)
    assert torch.equal(tower_prefill.next_frames[:4], next_frames)
    assert (
        tower_prefill.input_frames.data_ptr(),
        tower_prefill.next_frames.data_ptr(),
    ) == addresses


def test_tower_prefill_refuses_rows_past_its_buffers() -> None:
    tower_prefill = tower_prefill_of(3)
    frames = torch.zeros((4, FRAME_WIDTH), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="4 rows exceed the 3 tower prefill rows"):
        tower_prefill.stage(frames, frames)
    with pytest.raises(RuntimeError, match="4 rows exceed the 3 tower prefill rows"):
        tower_prefill.run(torch.zeros((4, 8)), torch.arange(4), Mock(spec=ForwardBatch))
    tower_prefill.audio_decoder.prefill.assert_not_called()


def set_up_mocked_model(
    builder: Drama3EngineBuilder,
    server_args: ServerArgs,
    checkpoint_dir: Path,
    fused_layers_reason: str | None = "the KV pool is a test double",
) -> Mock:
    model = Mock(spec=Drama3ForCausalLM)
    model.config = FishOmniConfig(**fish_config_dict())
    codebook_size = model.config.audio_decoder_config.vocab_size
    model.audio_decoder = Mock(spec=AudioDecoder)
    model.audio_decoder.codebook_size = codebook_size
    model.audio_decoder.main_token_candidate_ids = torch.arange(codebook_size + 1)
    model.audio_decoder.sampling_step = torch.zeros((), dtype=torch.int64)
    model.audio_decoder.sampler = None
    model.audio_decoder.fused_layers_unsupported_reason.return_value = (
        fused_layers_reason
    )
    model.tower_attention = torch.nn.ModuleList()
    model.model = SimpleNamespace(tower_prefill=None)
    worker = Mock(spec=ModelWorker)
    worker.model_runner = SimpleNamespace(
        model=model,
        req_to_token_pool=SimpleNamespace(size=4),
        token_to_kv_pool=Mock(spec=HybridLinearKVPool),
    )
    builder.setup_model(
        model_worker=worker,
        checkpoint_dir=str(checkpoint_dir),
        device="cpu",
        gpu_id=0,
        server_args=server_args,
    )
    return model


@pytest.mark.parametrize(
    ("overrides", "context_length", "row_count"),
    [
        (
            {"max_prefill_tokens": 16, "cuda_graph_backend_prefill": "disabled"},
            100,
            100,
        ),
        (
            {"max_prefill_tokens": 64, "cuda_graph_backend_prefill": "disabled"},
            CONTEXT_LENGTH,
            64,
        ),
        ({"max_prefill_tokens": 64, "cuda_graph_max_bs_prefill": 256}, 128, 256),
        ({"max_prefill_tokens": 512, "cuda_graph_max_bs_prefill": 256}, 128, 512),
    ],
)
def test_setup_model_gives_the_backbone_rows_for_the_largest_extend_or_bucket(
    tmp_path: Path,
    overrides: dict[str, int | str],
    context_length: int,
    row_count: int,
) -> None:
    builder, server_args = engine_server_args(tmp_path, overrides, context_length)
    builder.validate_before_infrastructure(server_args)

    model = set_up_mocked_model(builder, server_args, tmp_path)

    tower_prefill = model.model.tower_prefill
    assert isinstance(tower_prefill, Drama3TowerPrefill)
    assert tower_prefill.audio_decoder is model.audio_decoder
    assert tower_prefill.tower_attention is model.tower_attention
    assert tower_prefill.input_frames.shape == (row_count, FRAME_WIDTH)
    assert tower_prefill.next_frames.shape == (row_count, FRAME_WIDTH)
    assert not tower_prefill.input_frames.any()
    assert not tower_prefill.next_frames.any()


@pytest.mark.parametrize(
    ("triton_sampling", "unsupported_reason", "expected_log"),
    [
        (
            False,
            None,
            "Drama 3 codebook sampling: PyTorch, "
            "tts_engine.factory.triton_sampling is false",
        ),
        (
            True,
            "Triton is not installed",
            "Drama 3 codebook sampling: PyTorch, Triton is not installed",
        ),
        (
            True,
            "untested GPU sm_120",
            "Drama 3 codebook sampling: PyTorch, untested GPU sm_120",
        ),
        (True, None, "Drama 3 codebook sampling: Triton kernels"),
    ],
)
def test_setup_model_picks_the_codebook_sampler_and_logs_why(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    triton_sampling: bool,
    unsupported_reason: str | None,
    expected_log: str,
) -> None:
    builder, server_args = engine_server_args(tmp_path, {})
    builder.triton_sampling = triton_sampling
    generator_seed = 2**64 - 5
    monkeypatch.setattr(
        engine_builder,
        "triton_sampling_unsupported_reason",
        Mock(return_value=unsupported_reason),
    )
    monkeypatch.setattr(torch.cuda, "initial_seed", Mock(return_value=generator_seed))

    with caplog.at_level(logging.INFO, logger=engine_builder.logger.name):
        model = set_up_mocked_model(builder, server_args, tmp_path)

    assert expected_log in caplog.text
    audio_decoder = model.audio_decoder
    if triton_sampling and unsupported_reason is None:
        assert audio_decoder.sampler is not None
        assert audio_decoder.sampler.step is audio_decoder.sampling_step
        assert audio_decoder.sampler.seed == generator_seed % PHILOX_SEED_MODULUS
    else:
        assert audio_decoder.sampler is None


@pytest.mark.parametrize(
    ("tower_fused_layers", "unsupported_reason", "expected_log"),
    [
        (
            False,
            None,
            "Drama 3 tower fused layers: off, "
            "tts_engine.factory.tower_fused_layers is false",
        ),
        (
            True,
            "Triton is not installed",
            "Drama 3 tower fused layers: off, Triton is not installed",
        ),
        (
            True,
            None,
            f"Drama 3 tower fused layers: on for passes up to {MAX_FUSED_ROWS} rows "
            f"({', '.join(FUSED_KERNELS)})",
        ),
    ],
)
def test_setup_model_picks_the_tower_layers_and_logs_why(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    tower_fused_layers: bool,
    unsupported_reason: str | None,
    expected_log: str,
) -> None:
    builder, server_args = engine_server_args(tmp_path, {})
    builder.tower_fused_layers = tower_fused_layers

    with caplog.at_level(logging.INFO, logger=engine_builder.logger.name):
        model = set_up_mocked_model(
            builder, server_args, tmp_path, fused_layers_reason=unsupported_reason
        )

    assert expected_log in caplog.text
    audio_decoder = model.audio_decoder
    audio_decoder.fused_layers_unsupported_reason.assert_called_once_with(
        kv_pool=ANY,
        tower_layer_ids=[],
        context_length=builder.context_length,
        device=torch.device("cpu"),
    )
    if tower_fused_layers and unsupported_reason is None:
        audio_decoder.enable_fused_layers.assert_called_once_with(torch.device("cpu"))
    else:
        audio_decoder.enable_fused_layers.assert_not_called()


def first_frame_graph_setup(
    *,
    first_frame_cuda_graphs: bool,
    first_frame_cuda_graph_max_requests: int,
    decode_graph_max_batch_size: int | None,
) -> tuple[Drama3EngineBuilder, Mock]:
    builder = Drama3EngineBuilder(
        context_length=CONTEXT_LENGTH,
        max_running_requests=2,
        first_frame_cuda_graphs=first_frame_cuda_graphs,
        first_frame_cuda_graph_max_requests=first_frame_cuda_graph_max_requests,
    )
    sglang_runner = Mock(spec=SGLModelRunner)
    sglang_runner.attn_backend = Mock(spec=HybridLinearAttnBackend)
    sglang_runner.decode_cuda_graph_runner = (
        None
        if decode_graph_max_batch_size is None
        else SimpleNamespace(max_bs=decode_graph_max_batch_size)
    )
    sglang_runner.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.sglang_runner = sglang_runner
    model = Mock(spec=Drama3ForCausalLM)
    model.config = FishOmniConfig(**fish_config_dict())
    model.lm_head = SimpleNamespace(weight=torch.zeros(1))
    model.audio_decoder = SimpleNamespace(
        main_token_candidate_ids=torch.tensor(
            sorted(
                {
                    model.config.eos_token_id,
                    *range(
                        model.config.semantic_start_token_id,
                        model.config.semantic_end_token_id + 1,
                    ),
                }
            )
        )
    )
    model.first_frame_graphs = None
    return builder, model


@pytest.mark.parametrize(
    (
        "first_frame_cuda_graph_max_requests",
        "decode_graph_max_batch_size",
        "max_request_count",
    ),
    [(16, 8, 8), (3, 8, 3), (16, 16, 16)],
)
def test_first_frame_graphs_cover_requests_up_to_the_decode_graph_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    first_frame_cuda_graph_max_requests: int,
    decode_graph_max_batch_size: int,
    max_request_count: int,
) -> None:
    _, server_args = engine_server_args(tmp_path, {})
    builder, model = first_frame_graph_setup(
        first_frame_cuda_graphs=True,
        first_frame_cuda_graph_max_requests=first_frame_cuda_graph_max_requests,
        decode_graph_max_batch_size=decode_graph_max_batch_size,
    )
    graphs = cpu_first_frame_graphs([])
    capture = Mock(return_value=graphs)
    monkeypatch.setattr(Drama3FirstFrameGraphs, "capture", capture)

    with caplog.at_level(logging.INFO, logger=engine_builder.logger.name):
        builder.post_cuda_graph_setup(model, server_args)

    sglang_runner = builder.sglang_runner
    assert sglang_runner is not None
    text_config = model.config.text_config
    assert capture.call_args == call(
        sample_first_frames=model.sample_first_frames,
        attention_backend=sglang_runner.attn_backend,
        max_request_count=max_request_count,
        vocab_size=int(model.audio_decoder.main_token_candidate_ids.shape[0]),
        hidden_size=text_config.hidden_size,
        hidden_dtype=torch.bfloat16,
        frame_width=FRAME_WIDTH,
        device=torch.device("cpu"),
    )
    assert model.first_frame_graphs is graphs
    assert (
        f"Drama 3 first-frame CUDA graphs: on for 1..{max_request_count} requests "
        f"(first_frame_cuda_graph_max_requests {first_frame_cuda_graph_max_requests}"
        f", decode CUDA graph max batch {decode_graph_max_batch_size})"
    ) in caplog.text


@pytest.mark.parametrize(
    ("first_frame_cuda_graphs", "decode_graph_max_batch_size", "reason"),
    [
        (False, 8, "tts_engine.factory.first_frame_cuda_graphs is false"),
        (True, None, "no decode CUDA graph"),
    ],
)
def test_the_first_frame_step_stays_eager_when_switched_off_or_without_decode_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    first_frame_cuda_graphs: bool,
    decode_graph_max_batch_size: int | None,
    reason: str,
) -> None:
    _, server_args = engine_server_args(tmp_path, {})
    builder, model = first_frame_graph_setup(
        first_frame_cuda_graphs=first_frame_cuda_graphs,
        first_frame_cuda_graph_max_requests=16,
        decode_graph_max_batch_size=decode_graph_max_batch_size,
    )
    capture = Mock(side_effect=AssertionError("captured first-frame graphs"))
    monkeypatch.setattr(Drama3FirstFrameGraphs, "capture", capture)

    with caplog.at_level(logging.INFO, logger=engine_builder.logger.name):
        builder.post_cuda_graph_setup(model, server_args)

    capture.assert_not_called()
    assert model.first_frame_graphs is None
    assert f"Drama 3 first-frame CUDA graphs: off, {reason}" in caplog.text


def test_a_failed_first_frame_capture_stops_the_engine_without_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, server_args = engine_server_args(tmp_path, {})
    builder, model = first_frame_graph_setup(
        first_frame_cuda_graphs=True,
        first_frame_cuda_graph_max_requests=16,
        decode_graph_max_batch_size=8,
    )
    monkeypatch.setattr(
        Drama3FirstFrameGraphs,
        "capture",
        Mock(side_effect=RuntimeError("the graph for 2 requests failed")),
    )

    with pytest.raises(RuntimeError, match="the graph for 2 requests failed"):
        builder.post_cuda_graph_setup(model, server_args)

    assert model.first_frame_graphs is None


# Written by a decode forward that fails, so collecting it would show.
JUNK_TOKEN_ID = 7
WALK_SLOT_COUNT = 6
WALK_STEP_COUNT = 300
PrefillFailure = Literal["in-forward", "after-first-frames", "in-post-prefill"]


def forward_output() -> sglang_model_runner.ModelRunnerOutput:
    return sglang_model_runner.ModelRunnerOutput(
        logits_output=LogitsProcessorOutput(next_token_logits=None),
        can_run_graph=True,
    )


class FakeDecodeForward:
    """Stands in for the GPU forward behind the model worker: reads the batch rows
    of the decode buffers, then writes the next frames, or writes junk and fails."""

    def __init__(self, buffers: Drama3DecodeBuffers) -> None:
        self.buffers = buffers
        self.reads: list[StagedRows] = []
        self.next_frames: list[list[int]] = []
        self.error: RuntimeError | None = None
        self.overwrites_ras_windows = False

    def __call__(
        self, forward_batch: ForwardBatch
    ) -> sglang_model_runner.ModelRunnerOutput:
        row_count = len(forward_batch.rids)
        self.reads.append(staged_rows(self.buffers, row_count))
        if self.overwrites_ras_windows:
            self.buffers.ras_windows[:row_count].fill_(EMPTY_RAS_SLOT)
        else:
            pass
        if self.error is not None:
            self.buffers.sampled_frames[:row_count].fill_(JUNK_TOKEN_ID)
            raise self.error
        else:
            self.buffers.sampled_frames[:row_count].copy_(
                torch.tensor(self.next_frames)
            )
            return forward_output()


class FakePrefillForward:
    def __init__(self, step_exchange: StepExchange) -> None:
        self.step_exchange = step_exchange
        self.taken_inputs: list[Drama3StepInputs] = []
        self.first_frames: list[list[int]] = []
        self.error: RuntimeError | None = None

    def __call__(
        self, forward_batch: ForwardBatch
    ) -> sglang_model_runner.ModelRunnerOutput:
        self.taken_inputs.append(
            self.step_exchange.take_inputs(
                forward_batch.rids, forward_batch.extend_num_tokens
            )
        )
        if self.error is not None:
            raise self.error
        else:
            self.step_exchange.publish(
                torch.tensor(self.first_frames, dtype=torch.int64)
            )
            return forward_output()


class SuppliedBatchRunner(Drama3ModelRunner):
    """Drama3ModelRunner whose ForwardBatch comes from the test instead of SGLang."""

    def build_forward_batch(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[ForwardBatch, ScheduleBatch, bool]:
        schedule_batch = scheduler_output.batch_data
        assert schedule_batch is not None
        forward_batch = Mock(spec=ForwardBatch)
        forward_batch.rids = scheduler_output.request_ids
        forward_batch.forward_mode = schedule_batch.forward_mode
        is_prefill = schedule_batch.forward_mode.is_extend()
        if is_prefill:
            request_slots = []
            extend_row_count = 0
            for request in scheduler_output.requests:
                sglang_request = request_data_of(request).req
                assert sglang_request is not None
                assert sglang_request.extend_end is not None
                request_slots.append(sglang_request.kv.req_pool_idx)
                extend_row_count += sglang_request.extend_len
            forward_batch.req_pool_indices = torch.tensor(
                request_slots, dtype=torch.int64
            )
            forward_batch.extend_num_tokens = extend_row_count
        else:
            pass
        return forward_batch, schedule_batch, is_prefill


@dataclass(frozen=True, kw_only=True)
class DecodeRunner:
    runner: SuppliedBatchRunner
    forward: FakeDecodeForward
    prefill: FakePrefillForward
    worker: Mock
    execution_bridge: Mock


def unpinned_buffer_pair(shape: tuple[int, ...]) -> list[torch.Tensor]:
    return [torch.empty(shape, dtype=torch.int64) for _ in range(2)]


def decode_runner(row_count: int) -> DecodeRunner:
    """The real runner on stubbed SGLang infrastructure and a fake GPU forward."""
    config = FishOmniConfig(**fish_config_dict())
    text_config = config.text_config
    buffers = Drama3DecodeBuffers.allocate(
        row_count=row_count, frame_width=FRAME_WIDTH, device=torch.device("cpu")
    )
    model = Mock(spec=Drama3ForCausalLM)
    model.config = config
    model.step_exchange = StepExchange()
    model.decode_buffers = buffers
    attention_backend = Mock(spec=HybridLinearAttnBackend)
    attention_backend.full_attn_backend = Mock(spec=TritonAttnBackend)
    kv_pool = Mock(spec=HybridLinearKVPool)
    kv_pool.full_attention_layer_id_mapping = {
        layer_id: index
        for index, layer_id in enumerate(text_config.full_attention_layer_ids)
    }
    request_pool = Mock(spec=HybridReqToTokenPool)
    request_pool.mamba_map = {
        layer_id: index for index, layer_id in enumerate(text_config.linear_layer_ids)
    }
    request_pool.size = row_count
    forward = FakeDecodeForward(buffers)
    prefill = FakePrefillForward(model.step_exchange)

    def model_forward(
        forward_batch: ForwardBatch,
    ) -> sglang_model_runner.ModelRunnerOutput:
        if forward_batch.forward_mode.is_extend():
            return prefill(forward_batch)
        else:
            return forward(forward_batch)

    worker = Mock(spec=ModelWorker)
    worker.gpu_id = 0
    worker.dllm_algorithm = None
    worker.model_runner = SimpleNamespace(
        model=model,
        attn_backend=attention_backend,
        token_to_kv_pool=kv_pool,
        req_to_token_pool=request_pool,
        prefill_cuda_graph_runner=None,
        decode_cuda_graph_runner=None,
        forward=model_forward,
    )
    worker.forward_batch_generation.side_effect = partial(
        ModelWorker.forward_batch_generation, worker
    )
    # A host without an accelerator has no platform device to resolve.
    with patch.object(current_platform, "get_device", return_value=torch.device("cpu")):
        runner = SuppliedBatchRunner(worker, SGLangOutputProcessor())
    runner.row_upload_buffers = unpinned_buffer_pair((row_count,))
    runner.host_staging_buffers = unpinned_buffer_pair((row_count, FRAME_WIDTH))
    execution_bridge = Mock(spec=SGLangExecutionBridge)
    execution_bridge.forward_context.side_effect = (
        lambda schedule_batch, isolate_sampling=False: contextlib.nullcontext()
    )
    runner.bind_execution_bridge(execution_bridge)
    return DecodeRunner(
        runner=runner,
        forward=forward,
        prefill=prefill,
        worker=worker,
        execution_bridge=execution_bridge,
    )


def schedule_batch_of(forward_mode: ForwardMode) -> ScheduleBatch:
    schedule_batch = Mock(spec=ScheduleBatch)
    schedule_batch.forward_mode = forward_mode
    schedule_batch.is_prefill_only = False
    schedule_batch.sampling_info.penalizer_orchestrator = None
    schedule_batch.copy.return_value = schedule_batch
    return schedule_batch


def decode_output(
    runner: Drama3ModelRunner, requests: list[SchedulerRequest]
) -> ModelRunnerOutput:
    """One decode step through the runner's own execute order."""
    return runner.execute(
        SchedulerOutput(
            requests=requests, batch_data=schedule_batch_of(ForwardMode.DECODE)
        )
    )


def prefill_output(
    runner: Drama3ModelRunner, requests: list[SchedulerRequest]
) -> ModelRunnerOutput:
    return runner.execute(
        SchedulerOutput(
            requests=requests, batch_data=schedule_batch_of(ForwardMode.EXTEND)
        )
    )


def launched_step(
    runner: Drama3ModelRunner, requests: list[SchedulerRequest]
) -> PendingStep:
    pending_step = runner.execute_launch(
        SchedulerOutput(
            requests=requests, batch_data=schedule_batch_of(ForwardMode.DECODE)
        )
    )
    assert pending_step is not None
    return pending_step


def record_tokens(requests: list[SchedulerRequest], token_ids: list[int]) -> None:
    for request, token_id in zip(requests, token_ids, strict=True):
        sglang_request = request_data_of(request).req
        assert sglang_request is not None
        sglang_request.output_ids.append(token_id)


def decode_step(
    runner: Drama3ModelRunner,
    requests: list[SchedulerRequest],
    *,
    is_lookahead: bool = False,
) -> torch.Tensor:
    """A successful decode step, then the scheduler's bookkeeping of its tokens."""
    if is_lookahead:
        output = runner.execute_resolve(launched_step(runner, requests))
        assert output is not None
    else:
        output = decode_output(runner, requests)
    next_token_ids = output.next_token_ids
    assert next_token_ids is not None
    record_tokens(requests, next_token_ids.tolist())
    return next_token_ids


def prompted_request(
    prompt_rows: list[list[int]], request_id: str, knob_index: int
) -> SchedulerRequest:
    request_data = built_request(prompt_rows, request_id)
    request_data.temperature = 0.5 + 0.25 * (knob_index % 4)
    request_data.top_k = 1 + knob_index % 7
    request_data.top_p = 0.55 + 0.15 * (knob_index % 3)
    return SchedulerRequest(request_id=request_id, data=request_data)


def prompted_requests(lengths: list[int]) -> list[SchedulerRequest]:
    return [
        prompted_request(prompt_rows, f"request-{index}", index)
        for index, prompt_rows in enumerate(frame_rows_of(lengths, seed=11))
    ]


def assign_request_slots(
    requests: list[SchedulerRequest], request_slots: list[int]
) -> None:
    for request, request_slot in zip(requests, request_slots, strict=True):
        request_data = request_data_of(request)
        sglang_request = request_data.req
        assert sglang_request is not None
        sglang_request.kv.req_pool_idx = request_slot
        sglang_request.is_retracted = False
        admitted_for_prefill(request_data, 0, len(request_data.frame_rows))


def prefill_step(
    decode: DecodeRunner,
    requests: list[SchedulerRequest],
    request_slots: list[int],
    first_frames: list[list[int]],
) -> None:
    assign_request_slots(requests, request_slots)
    decode.prefill.first_frames = first_frames
    next_token_ids = prefill_output(decode.runner, requests).next_token_ids
    assert next_token_ids is not None
    record_tokens(requests, next_token_ids.tolist())


def decoding_scheduler_requests(
    decode: DecodeRunner, lengths: list[int]
) -> list[SchedulerRequest]:
    requests = prompted_requests(lengths)
    prefill_step(
        decode,
        requests,
        list(range(1, len(requests) + 1)),
        [[81 + index, index, 1, 2, 3] for index in range(len(requests))],
    )
    return requests


def test_decode_gathers_rows_in_place_as_the_batch_shrinks_and_grows() -> None:
    decode = decode_runner(row_count=4)
    buffers = decode.runner.decode_buffers
    addresses = buffer_addresses(buffers)
    requests = decoding_scheduler_requests(decode, [3, 4, 5, 2])
    batch_request_data = [request_data_of(request) for request in requests]

    expected = history_staged_rows(batch_request_data[:3])
    decode.forward.next_frames = [[90 + index, 4, 3, 2, 1] for index in range(3)]
    decode_step(decode.runner, requests[:3])
    assert decode.forward.reads[-1] == expected
    third_row = (buffers.frames[2].clone(), buffers.ras_windows[2].clone())

    expected = history_staged_rows(batch_request_data[:2])
    decode.forward.next_frames = [[94 + index, 1, 2, 3, 4] for index in range(2)]
    decode_step(decode.runner, requests[:2])
    assert decode.forward.reads[-1] == expected
    # Rows past a smaller batch keep earlier valid inputs for padded graph rows.
    assert torch.equal(buffers.frames[2], third_row[0])
    assert torch.equal(buffers.ras_windows[2], third_row[1])

    expected = history_staged_rows(batch_request_data)
    decode.forward.next_frames = [[84 + index, 2, 2, 2, 2] for index in range(4)]
    decode_step(decode.runner, requests)
    assert decode.forward.reads[-1] == expected
    assert buffer_addresses(buffers) == addresses


@pytest.mark.parametrize("is_lookahead", [False, True], ids=["sync", "lookahead"])
def test_a_failed_decode_forward_collects_nothing_and_the_next_step_restages(
    is_lookahead: bool,
) -> None:
    decode = decode_runner(row_count=4)
    requests = decoding_scheduler_requests(decode, [3, 4, 2])
    batch_request_data = [request_data_of(request) for request in requests]
    decode.forward.next_frames = [[90 + index, 1, 1, 1, 1] for index in range(3)]
    decode_step(decode.runner, requests, is_lookahead=is_lookahead)
    histories = [list(request_data.frame_rows) for request_data in batch_request_data]
    publish_count = decode.execution_bridge.publish_next_tokens.call_count

    expected = history_staged_rows(batch_request_data)
    decode.forward.error = RuntimeError("decode forward failed")
    with pytest.raises(RuntimeError, match="decode forward failed"):
        decode_step(decode.runner, requests, is_lookahead=is_lookahead)
    assert decode.forward.reads[-1] == expected
    assert [request_data.frame_rows for request_data in batch_request_data] == (
        histories
    )
    assert decode.execution_bridge.publish_next_tokens.call_count == publish_count

    decode.forward.error = None
    decode.forward.next_frames = [[95 + index, 2, 2, 2, 2] for index in range(3)]
    next_token_ids = decode_step(decode.runner, requests, is_lookahead=is_lookahead)
    assert decode.forward.reads[-1] == expected
    assert next_token_ids.tolist() == [95, 96, 97]
    assert [request_data.frame_rows for request_data in batch_request_data] == [
        history + [frame]
        for history, frame in zip(histories, decode.forward.next_frames, strict=True)
    ]


@pytest.mark.parametrize("is_lookahead", [False, True], ids=["sync", "lookahead"])
def test_a_forward_that_overwrites_its_ras_windows_keeps_the_history_windows(
    is_lookahead: bool,
) -> None:
    decode = decode_runner(row_count=4)
    requests = decoding_scheduler_requests(decode, [3, 4])
    batch_request_data = [request_data_of(request) for request in requests]
    decode.forward.overwrites_ras_windows = True
    for step_index in range(3):
        expected = history_staged_rows(batch_request_data)
        decode.forward.next_frames = [
            [82 + step_index, 1, 1, 1, 1],
            [85 + step_index, 2, 2, 2, 2],
        ]
        decode_step(decode.runner, requests, is_lookahead=is_lookahead)
        assert decode.forward.reads[-1] == expected


def test_next_token_ids_from_post_decode_outlive_later_steps() -> None:
    decode = decode_runner(row_count=4)
    requests = decoding_scheduler_requests(decode, [3, 4, 2])
    decode.forward.next_frames = [[90 + index, 1, 1, 1, 1] for index in range(3)]
    first_token_ids = decode_step(decode.runner, requests)

    decode.forward.next_frames = [[93 + index, 2, 2, 2, 2] for index in range(3)]
    decode_step(decode.runner, requests)
    decode.forward.error = RuntimeError("decode forward failed")
    with pytest.raises(RuntimeError, match="decode forward failed"):
        decode_output(decode.runner, requests)

    assert first_token_ids.tolist() == [90, 91, 92]


def test_a_lookahead_resolve_skips_rows_finished_or_retracted_since_launch() -> None:
    decode = decode_runner(row_count=4)
    requests = decoding_scheduler_requests(decode, [3, 4, 2, 3])
    batch_request_data = [request_data_of(request) for request in requests]
    frame_streamer = window_frame_streamer()
    for request_data in batch_request_data:
        payload = request_data.stage_payload
        assert payload is not None
        payload.request.params["stream"] = True

    def streamed_message_counts() -> list[int]:
        return [
            len(
                frame_streamer(
                    request.request_id,
                    request_data_of(request),
                    RequestOutput(request_id=request.request_id),
                )
            )
            for request in requests
        ]

    assert streamed_message_counts() == [1, 1, 1, 1]
    histories = [list(request_data.frame_rows) for request_data in batch_request_data]
    generation_steps = [
        request_data.generation_steps for request_data in batch_request_data
    ]
    eos_frame = [EOS_TOKEN_ID, 0, 0, 0, 0]
    decode.forward.next_frames = [
        [90, 1, 1, 1, 1],
        [91, 1, 1, 1, 1],
        [92, 1, 1, 1, 1],
        eos_frame,
    ]
    pending_step = launched_step(decode.runner, requests)
    finished_request = batch_request_data[1].req
    retracted_request = batch_request_data[2].req
    assert finished_request is not None and retracted_request is not None
    finished_request.finished_reason = FINISH_MATCHED_TOKEN(matched=EOS_TOKEN_ID)
    retracted_request.is_retracted = True
    output = decode.runner.execute_resolve(pending_step)

    assert output is not None and output.next_token_ids is not None
    assert output.next_token_ids.tolist() == [90, 91, 92, EOS_TOKEN_ID]
    assert [request_data.frame_rows for request_data in batch_request_data] == [
        histories[0] + [[90, 1, 1, 1, 1]],
        histories[1],
        histories[2],
        histories[3] + [eos_frame],
    ]
    assert [request_data.generation_steps for request_data in batch_request_data] == [
        generation_steps[0] + 1,
        generation_steps[1],
        generation_steps[2],
        generation_steps[3] + 1,
    ]
    assert streamed_message_counts() == [1, 0, 0, 0]


def test_row_indices_upload_only_when_the_batch_order_or_slots_change() -> None:
    decode = decode_runner(row_count=4)
    runner = decode.runner
    first, second, third = decoding_scheduler_requests(decode, [3, 4, 2])
    newcomer = prompted_request(frame_rows_of([3], seed=12)[0], "request-new", 3)
    upload_counts: list[int] = []
    with patch.object(
        runner, "pinned_pingpong", wraps=runner.pinned_pingpong
    ) as pinned_pingpong:

        def decode_and_count_uploads(batch: list[SchedulerRequest]) -> None:
            expected = history_staged_rows(
                [request_data_of(request) for request in batch]
            )
            decode.forward.next_frames = [
                [84 + index, 1, 2, 3, 4] for index in range(len(batch))
            ]
            decode_step(runner, batch)
            assert decode.forward.reads[-1] == expected
            upload_counts.append(
                sum(
                    called.args[0] == "row_upload_buffers"
                    for called in pinned_pingpong.call_args_list
                )
            )

        for batch in (
            [first, second, third],
            [first, second, third],
            [third, first, second],
            [third, first],
            [third, first],
        ):
            decode_and_count_uploads(batch)
        retracted_request = request_data_of(first).req
        assert retracted_request is not None
        retracted_request.is_retracted = True
        prefill_step(decode, [newcomer], [1], [[86, 2, 2, 2, 2]])
        prefill_step(decode, [first], [4], [[87, 3, 3, 3, 3]])
        decode_and_count_uploads([third, first])

    assert upload_counts == [1, 1, 2, 3, 3, 4]


def test_a_new_request_in_the_same_slots_keeps_the_uploaded_row_indices() -> None:
    decode = decode_runner(row_count=4)
    runner = decode.runner
    first, second = decoding_scheduler_requests(decode, [3, 4])
    newcomer = prompted_request(frame_rows_of([3], seed=12)[0], "request-new", 3)
    upload_counts: list[int] = []
    with patch.object(
        runner, "pinned_pingpong", wraps=runner.pinned_pingpong
    ) as pinned_pingpong:

        def decode_and_count_uploads(batch: list[SchedulerRequest]) -> None:
            expected = history_staged_rows(
                [request_data_of(request) for request in batch]
            )
            decode.forward.next_frames = [
                [84 + index, 1, 2, 3, 4] for index in range(len(batch))
            ]
            decode_step(runner, batch)
            assert decode.forward.reads[-1] == expected
            upload_counts.append(
                sum(
                    called.args[0] == "row_upload_buffers"
                    for called in pinned_pingpong.call_args_list
                )
            )

        decode_and_count_uploads([first, second])
        prefill_step(decode, [newcomer], [1], [[86, 2, 2, 2, 2]])
        decode_and_count_uploads([newcomer, second])

    assert upload_counts == [1, 1]


def test_the_first_decode_after_a_prefill_reads_what_the_history_gives() -> None:
    decode = decode_runner(row_count=4)
    requests = prompted_requests([3, 5])
    prefill_step(decode, requests, [3, 1], [[82, 1, 2, 3, 4], [85, 4, 3, 2, 1]])
    expected = history_staged_rows([request_data_of(request) for request in requests])

    assert pool_rows(decode.runner.state_pool, [3, 1]) == expected
    decode.forward.next_frames = [[86, 1, 1, 1, 1], [87, 2, 2, 2, 2]]
    decode_step(decode.runner, requests)
    assert decode.forward.reads[-1] == expected


@pytest.mark.parametrize("is_lookahead", [False, True], ids=["sync", "lookahead"])
def test_a_retracted_request_resumes_in_a_new_slot_from_its_whole_history(
    is_lookahead: bool,
) -> None:
    decode = decode_runner(row_count=4)
    first, second = decoding_scheduler_requests(decode, [3, 4])
    for step_index in range(3):
        decode.forward.next_frames = [
            [82 + step_index, 1, 1, 1, 1],
            [85 + step_index, 2, 2, 2, 2],
        ]
        decode_step(decode.runner, [first, second], is_lookahead=is_lookahead)
    first_request_data = request_data_of(first)
    retracted_request = first_request_data.req
    assert retracted_request is not None
    retracted_request.is_retracted = True
    newcomer = prompted_request(frame_rows_of([5], seed=13)[0], "request-new", 2)
    prefill_step(decode, [newcomer], [1], [[86, 3, 3, 3, 3]])
    history = list(first_request_data.frame_rows)
    prefill_step(decode, [first], [3], [[87, 4, 4, 4, 4]])
    assert decode.prefill.taken_inputs[-1].frames.tolist() == history

    batch = [second, newcomer, first]
    expected = history_staged_rows([request_data_of(request) for request in batch])
    decode.forward.next_frames = [[80 + index, 5, 5, 5, 5] for index in range(3)]
    decode_step(decode.runner, batch, is_lookahead=is_lookahead)
    assert decode.forward.reads[-1] == expected


@pytest.mark.parametrize(
    "request_slot", [None, 0, 5], ids=["unseated", "padding-slot", "past-the-pool"]
)
def test_a_decode_refuses_a_request_without_a_state_row(
    request_slot: int | None,
) -> None:
    decode = decode_runner(row_count=4)
    first, second = decoding_scheduler_requests(decode, [3, 4])
    sglang_request = request_data_of(second).req
    assert sglang_request is not None
    sglang_request.kv.req_pool_idx = request_slot

    with pytest.raises(RuntimeError, match="has no state row"):
        decode_output(decode.runner, [first, second])


def test_post_prefill_gets_the_staging_a_prefill_keeps_past_its_cleanup() -> None:
    decode = decode_runner(row_count=4)
    runner = decode.runner
    requests = prompted_requests([3, 4])
    stagings_seen: list[tuple[str, Drama3StepInputs | None]] = []
    cleanup_prefill = runner.cleanup_prefill
    post_prefill = runner.post_prefill

    def recorded_cleanup_prefill(
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        step_requests: list[SchedulerRequest],
    ) -> None:
        stagings_seen.append(("cleanup_prefill", runner.prefill_inputs))
        cleanup_prefill(forward_batch, schedule_batch, step_requests)

    def recorded_post_prefill(
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        step_requests: list[SchedulerRequest],
    ) -> None:
        stagings_seen.append(("post_prefill", runner.prefill_inputs))
        post_prefill(result, forward_batch, schedule_batch, step_requests)

    with (
        patch.object(runner, "cleanup_prefill", side_effect=recorded_cleanup_prefill),
        patch.object(runner, "post_prefill", side_effect=recorded_post_prefill),
    ):
        prefill_step(decode, requests, [2, 4], [[82, 1, 2, 3, 4], [85, 4, 3, 2, 1]])

    staged = decode.prefill.taken_inputs[-1]
    assert staged.request_ids == ["request-0", "request-1"]
    assert [hook for hook, _ in stagings_seen] == ["cleanup_prefill", "post_prefill"]
    assert all(staging is staged for _, staging in stagings_seen)
    assert runner.prefill_inputs is None
    assert pool_rows(runner.state_pool, [2, 4]) == history_staged_rows(
        [request_data_of(request) for request in requests]
    )


@pytest.mark.parametrize(
    ("failure", "error", "message"),
    [
        pytest.param("in-forward", RuntimeError, "prefill failed", id="in-forward"),
        pytest.param(
            "after-first-frames",
            RuntimeError,
            "prefill failed",
            id="after-first-frames",
        ),
        pytest.param(
            "in-post-prefill", IndexError, "Number of indices", id="in-post-prefill"
        ),
    ],
)
def test_a_failed_prefill_drops_its_staging_before_the_error_leaves(
    failure: PrefillFailure, error: type[Exception], message: str
) -> None:
    decode = decode_runner(row_count=4)
    runner = decode.runner
    requests = prompted_requests([3, 4])
    batch_request_data = [request_data_of(request) for request in requests]
    histories = [list(request_data.frame_rows) for request_data in batch_request_data]
    pool_before = pool_rows(runner.state_pool, [1, 2])
    assign_request_slots(requests, [1, 2])
    decode.prefill.first_frames = [[82, 1, 2, 3, 4], [85, 4, 3, 2, 1]]
    if failure == "in-forward":
        decode.prefill.error = RuntimeError("prefill failed")
    elif failure == "after-first-frames":
        decode.worker.record_prefill_cuda_graph_usage.side_effect = RuntimeError(
            "prefill failed"
        )
    else:
        decode.prefill.first_frames = decode.prefill.first_frames[:1]

    with pytest.raises(error, match=message):
        prefill_output(runner, requests)

    assert runner.prefill_inputs is None
    assert (runner.model.step_exchange.sampled_frames is not None) == (
        failure == "after-first-frames"
    )
    assert pool_rows(runner.state_pool, [1, 2]) == pool_before
    assert [request_data.frame_rows for request_data in batch_request_data] == (
        histories
    )


def test_post_prefill_refuses_a_staging_of_other_requests() -> None:
    decode = decode_runner(row_count=4)
    runner = decode.runner
    staged, current = prompted_requests([3, 4])
    prefill_step(decode, [staged], [1], [[82, 1, 1, 1, 1]])
    assign_request_slots([current], [2])
    current_history = list(request_data_of(current).frame_rows)
    pool_before = pool_rows(runner.state_pool, [1, 2])
    forward_batch = Mock(spec=ForwardBatch)
    forward_batch.req_pool_indices = torch.tensor([2])
    runner.model.step_exchange.publish(torch.tensor([[86, 2, 2, 2, 2]]))
    runner.prefill_inputs = decode.prefill.taken_inputs[-1]

    with pytest.raises(RuntimeError, match="not the batch's"):
        runner.post_prefill(
            GenerationBatchResult(),
            forward_batch,
            schedule_batch_of(ForwardMode.EXTEND),
            [current],
        )

    assert runner.prefill_inputs is None
    assert pool_rows(runner.state_pool, [1, 2]) == pool_before
    assert request_data_of(current).frame_rows == current_history


@dataclass(kw_only=True)
class StatePoolWalk:
    mismatched_steps: list[int] = dataclasses.field(default_factory=list)
    checked_rows: int = 0
    joined_requests: int = 0
    finished_requests: int = 0
    reordered_steps: int = 0
    reused_slots: int = 0
    retracted_requests: int = 0
    resumed_requests: int = 0
    resumed_full_windows: int = 0
    skipped_finished_rows: int = 0
    skipped_retracted_rows: int = 0
    reused_overrun_slots: int = 0


def random_frame(generator: random.Random) -> list[int]:
    return [SEMANTIC_BEGIN_ID + generator.randrange(CODEBOOK_SIZE)] + [
        generator.randrange(CODEBOOK_SIZE) for _ in range(FRAME_WIDTH - 1)
    ]


def state_pool_walk(*, seed: int, is_lookahead: bool) -> StatePoolWalk:
    walk = StatePoolWalk()
    generator = random.Random(seed)
    decode = decode_runner(row_count=WALK_SLOT_COUNT)
    runner = decode.runner
    free_slots = list(range(1, WALK_SLOT_COUNT + 1))
    slot_owners: dict[int, str] = {}
    overrun_slots: set[int] = set()
    running: list[SchedulerRequest] = []
    retracted: list[SchedulerRequest] = []
    in_flight: list[tuple[PendingStep, list[SchedulerRequest]]] = []

    def collect(requests: list[SchedulerRequest], token_ids: list[int]) -> None:
        for request, token_id in zip(requests, token_ids, strict=True):
            sglang_request = request_data_of(request).req
            assert sglang_request is not None
            request_slot = sglang_request.kv.req_pool_idx
            assert request_slot is not None
            if sglang_request.finished():
                walk.skipped_finished_rows += 1
                overrun_slots.add(request_slot)
            elif sglang_request.is_retracted:
                walk.skipped_retracted_rows += 1
                overrun_slots.add(request_slot)
            elif token_id == EOS_TOKEN_ID:
                sglang_request.output_ids.append(token_id)
                sglang_request.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                running.remove(request)
                free_slots.append(request_slot)
                walk.finished_requests += 1
            else:
                sglang_request.output_ids.append(token_id)

    def resolve_in_flight() -> None:
        for pending_step, requests in in_flight:
            output = runner.execute_resolve(pending_step)
            assert output is not None and output.next_token_ids is not None
            collect(requests, output.next_token_ids.tolist())
        in_flight.clear()

    for step_index in range(WALK_STEP_COUNT):
        if len(running) > 1 and generator.random() < 0.05:
            request = running.pop(generator.randrange(len(running)))
            sglang_request = request_data_of(request).req
            assert sglang_request is not None
            assert sglang_request.kv.req_pool_idx is not None
            sglang_request.is_retracted = True
            free_slots.append(sglang_request.kv.req_pool_idx)
            retracted.append(request)
            walk.retracted_requests += 1
        else:
            pass
        if free_slots and (not running or generator.random() < 0.2):
            resolve_in_flight()
            admitted: list[SchedulerRequest] = []
            request_slots: list[int] = []
            for _ in range(generator.randint(1, min(3, len(free_slots)))):
                if retracted and generator.random() < 0.6:
                    request = retracted.pop(0)
                    walk.resumed_requests += 1
                    walk.resumed_full_windows += int(
                        len(request_data_of(request).frame_rows) >= RAS_WINDOW_FRAMES
                    )
                else:
                    prompt_row_count = generator.randint(
                        MINIMUM_PROMPT_ROWS, MAX_PROMPT_ROWS
                    )
                    request = prompted_request(
                        frame_rows_of(
                            [prompt_row_count], seed=generator.randrange(1 << 30)
                        )[0],
                        f"walk-{walk.joined_requests}",
                        walk.joined_requests,
                    )
                    walk.joined_requests += 1
                request_slot = free_slots.pop(generator.randrange(len(free_slots)))
                walk.reused_slots += int(
                    slot_owners.get(request_slot, request.request_id)
                    != request.request_id
                )
                walk.reused_overrun_slots += int(request_slot in overrun_slots)
                overrun_slots.discard(request_slot)
                slot_owners[request_slot] = request.request_id
                admitted.append(request)
                request_slots.append(request_slot)
            prefill_step(
                decode,
                admitted,
                request_slots,
                [random_frame(generator) for _ in admitted],
            )
            running.extend(admitted)
        else:
            pass
        if generator.random() < 0.3:
            order = [request.request_id for request in running]
            generator.shuffle(running)
            walk.reordered_steps += int(
                [request.request_id for request in running] != order
            )
        else:
            pass
        batch = list(running)
        decode.forward.next_frames = [
            (
                [EOS_TOKEN_ID] + [0] * (FRAME_WIDTH - 1)
                if generator.random() < 0.04
                else random_frame(generator)
            )
            for _ in batch
        ]
        batch_request_data = [request_data_of(request) for request in batch]
        if is_lookahead:
            pending_step = launched_step(runner, batch)
            resolve_in_flight()
            in_flight.append((pending_step, batch))
            expected = history_staged_rows(batch_request_data)
        else:
            expected = history_staged_rows(batch_request_data)
            next_token_ids = decode_output(runner, batch).next_token_ids
            assert next_token_ids is not None
            collect(batch, next_token_ids.tolist())
        if decode.forward.reads[-1] != expected:
            walk.mismatched_steps.append(step_index)
        else:
            pass
        walk.checked_rows += len(batch)
    resolve_in_flight()
    return walk


@pytest.mark.parametrize("is_lookahead", [False, True], ids=["sync", "lookahead"])
def test_the_state_pool_feeds_each_decode_what_history_staging_did(
    is_lookahead: bool,
) -> None:
    walk = state_pool_walk(seed=23, is_lookahead=is_lookahead)

    assert walk.mismatched_steps == []
    assert walk.checked_rows > WALK_STEP_COUNT
    assert (
        min(
            walk.joined_requests,
            walk.finished_requests,
            walk.reordered_steps,
            walk.reused_slots,
            walk.retracted_requests,
            walk.resumed_requests,
            walk.resumed_full_windows,
        )
        > 0
    )
    if is_lookahead:
        assert (
            min(
                walk.skipped_finished_rows,
                walk.skipped_retracted_rows,
                walk.reused_overrun_slots,
            )
            > 0
        )
    else:
        assert walk.skipped_finished_rows == walk.skipped_retracted_rows == 0


def test_next_frame_labels_pair_rows_within_each_request() -> None:
    frames = torch.tensor(sum(frame_rows_of([3, 2], seed=6), []))
    original = frames.clone()

    labels = next_frame_labels(frames, torch.tensor([2, 4]), EOS_TOKEN_ID)

    placeholder = [EOS_TOKEN_ID, 0, 0, 0, 0]
    assert labels.tolist() == [
        original[1].tolist(),
        original[2].tolist(),
        placeholder,
        original[4].tolist(),
        placeholder,
    ]
    assert torch.equal(frames, original)


def tower_cache(attention: DenseAttention) -> dict[tuple[int, str, int], torch.Tensor]:
    return {
        (layer_index, request_id, position): torch.cat((key, value))
        for (layer_index, request_id), entries in attention.entries.items()
        for position, key, value in entries
    }


@torch_fused_op_backend()
def test_placeholder_rows_end_like_first_writes_and_a_dense_recompute() -> None:
    """The integration's labels, then the decode step's rewrite of each last slot,
    leave the cache of N-1-row prefills and of a whole dense recompute, in fp32."""
    decoder = loaded_decoder(random_checkpoint(seed=0), torch.float32, "cpu")
    lengths = [6, 2, 3]
    request_ids = ["a", "b", "c"]
    requests = frame_rows_of(lengths, seed=7)
    frames = torch.tensor(sum(requests, []))
    row_request_ids = [
        request_id
        for request_id, length in zip(request_ids, lengths, strict=True)
        for _ in range(length)
    ]
    positions = torch.tensor([row for length in lengths for row in range(length)])
    last_rows = torch.cumsum(torch.tensor(lengths), dim=0) - 1
    generator = torch.Generator().manual_seed(8)
    hidden_states = torch.randn(frames.shape[0], DIM, generator=generator)
    semantic_logits = torch.randn(len(lengths), TEXT_VOCAB_SIZE, generator=generator)
    last_positions = positions[last_rows].tolist()

    def sample(attention: DenseAttention) -> torch.Tensor:
        attention.bind(request_ids, last_positions)
        return decoder.decode_step(
            semantic_logits,
            hidden_states[last_rows],
            positions[last_rows],
            frames[last_rows],
            torch.full((len(lengths), 16), EMPTY_RAS_SLOT),
            torch.ones(len(lengths)),
            torch.ones(len(lengths), dtype=torch.int64),
            torch.ones(len(lengths)),
            attention,
        )

    def new_attention() -> DenseAttention:
        return DenseAttention(
            head_count=HEAD_COUNT, kv_head_count=KV_HEAD_COUNT, head_dim=HEAD_DIM
        )

    placeholder_attention = new_attention()
    placeholder_attention.bind(row_request_ids, positions.tolist())
    decoder.prefill(
        hidden_states,
        positions,
        frames,
        next_frame_labels(frames, last_rows, EOS_TOKEN_ID),
        placeholder_attention,
    )
    placeholder_frames = sample(placeholder_attention)

    first_write_attention = new_attention()
    kept_rows = torch.tensor(
        [row for row in range(frames.shape[0]) if row not in last_rows.tolist()]
    )
    first_write_attention.bind(
        [row_request_ids[row] for row in kept_rows.tolist()],
        positions[kept_rows].tolist(),
    )
    decoder.prefill(
        hidden_states[kept_rows],
        positions[kept_rows],
        frames[kept_rows],
        frames[kept_rows + 1],
        first_write_attention,
    )
    first_write_frames = sample(first_write_attention)

    dense_attention = new_attention()
    dense_labels = frames.roll(-1, dims=0)
    dense_labels[last_rows] = placeholder_frames
    dense_attention.bind(row_request_ids, positions.tolist())
    decoder.prefill(hidden_states, positions, frames, dense_labels, dense_attention)

    assert torch.equal(placeholder_frames, first_write_frames)
    placeholder_cache = tower_cache(placeholder_attention)
    assert placeholder_cache.keys() == tower_cache(first_write_attention).keys()
    assert placeholder_cache.keys() == tower_cache(dense_attention).keys()
    for slot, cached in placeholder_cache.items():
        torch.testing.assert_close(
            cached, tower_cache(first_write_attention)[slot], rtol=0, atol=1e-4
        )
        torch.testing.assert_close(
            cached, tower_cache(dense_attention)[slot], rtol=0, atol=1e-4
        )


def test_model_rejects_a_tower_that_cannot_share_the_backbone_pool() -> None:
    # fish_config_dict() gives the tower head_dim 12 and the backbone head_dim 8.
    with pytest.raises(ValueError, match="must equal the backbone's"):
        Drama3ForCausalLM(FishOmniConfig(**fish_config_dict()))

    config_dict = fish_config_dict()
    decoder_config = config_dict["audio_decoder_config"]
    assert isinstance(decoder_config, dict)
    decoder_config["head_dim"] = 8
    config = FishOmniConfig(**config_dict)
    config.text_config.tower_layer_count = 4
    with pytest.raises(ValueError, match="tower layers"):
        Drama3ForCausalLM(config)


@pytest.mark.parametrize(
    "pipeline_parallel_size,tensor_parallel_size", [(2, 1), (1, 2)]
)
def test_model_rejects_more_than_one_tensor_or_pipeline_rank(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> None:
    config_dict = fish_config_dict()
    decoder_config = config_dict["audio_decoder_config"]
    assert isinstance(decoder_config, dict)
    decoder_config["head_dim"] = 8
    monkeypatch.setattr(
        sglang_model,
        "get_parallel",
        lambda: SimpleNamespace(
            pp_size=pipeline_parallel_size, tp_size=tensor_parallel_size
        ),
    )
    with pytest.raises(NotImplementedError, match="one tensor and pipeline rank"):
        Drama3ForCausalLM(FishOmniConfig(**config_dict))


def engine_server_args(
    checkpoint_dir: Path,
    overrides: dict[str, object],
    context_length: int = CONTEXT_LENGTH,
) -> tuple[Drama3EngineBuilder, ServerArgs]:
    """The server args the builder resolves for a checkpoint and caller overrides."""
    (checkpoint_dir / "config.json").write_text(json.dumps(fish_config_dict()))
    builder = Drama3EngineBuilder(context_length=context_length, max_running_requests=2)
    builder.pre_infra_setup(str(checkpoint_dir))
    generation_overrides = build_generation_batch_overrides(
        server_args_overrides=overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(generation_overrides)
    return builder, sglang_backend.build_sglang_server_args(
        str(checkpoint_dir), context_length=context_length, **generation_overrides
    )


def test_engine_defaults_pass_its_own_validation(tmp_path: Path) -> None:
    builder, server_args = engine_server_args(tmp_path, {})

    builder.validate_before_infrastructure(server_args)
    graph_config = resolved_view(server_args).cuda_graph_config
    assert server_args.page_size == 1
    assert server_args.chunked_prefill_size == -1
    assert not server_args.disable_cuda_graph
    assert graph_config.decode.backend == CudaGraphBackend.FULL
    assert graph_config.prefill.backend == CudaGraphBackend.BREAKABLE
    assert max(graph_config.prefill.bs) == PREFILL_CUDA_GRAPH_MAX_TOKENS
    assert server_args.prefill_decode_interval == DECODE_STEPS_BETWEEN_PREFILLS


def test_engine_context_must_fit_the_tower_rope_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    max_position_embeddings = FishOmniConfig(
        **fish_config_dict()
    ).text_config.max_position_embeddings
    builder, server_args = engine_server_args(
        tmp_path, {}, context_length=max_position_embeddings
    )
    builder.validate_before_infrastructure(server_args)

    builder, server_args = engine_server_args(
        tmp_path, {}, context_length=max_position_embeddings + 1
    )
    with pytest.raises(ValueError, match="exceeds the model's max_position_embeddings"):
        builder.validate_before_infrastructure(server_args)


def test_engine_runs_its_prefill_eager_when_asked(tmp_path: Path) -> None:
    builder, server_args = engine_server_args(
        tmp_path, {"cuda_graph_backend_prefill": "disabled"}
    )

    builder.validate_before_infrastructure(server_args)
    graph_config = resolved_view(server_args).cuda_graph_config
    assert graph_config.prefill.backend == CudaGraphBackend.DISABLED
    assert graph_config.decode.backend == CudaGraphBackend.FULL


def test_engine_accepts_a_nested_breakable_prefill_graph(tmp_path: Path) -> None:
    builder, server_args = engine_server_args(
        tmp_path, {"cuda_graph_config": {"prefill": {"backend": "breakable"}}}
    )

    builder.validate_before_infrastructure(server_args)
    graph_config = resolved_view(server_args).cuda_graph_config
    assert graph_config.prefill.backend == CudaGraphBackend.BREAKABLE
    assert graph_config.decode.backend == CudaGraphBackend.FULL
    assert max(graph_config.prefill.bs) == PREFILL_CUDA_GRAPH_MAX_TOKENS


def test_engine_also_runs_without_cuda_graphs(tmp_path: Path) -> None:
    builder, server_args = engine_server_args(tmp_path, {"disable_cuda_graph": True})

    builder.validate_before_infrastructure(server_args)
    assert resolved_view(server_args).cuda_graph_config.decode.backend == (
        CudaGraphBackend.DISABLED
    )
    assert resolved_view(server_args).cuda_graph_config.prefill.backend == (
        CudaGraphBackend.DISABLED
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"disable_radix_cache": False}, "disable_radix_cache"),
        ({"chunked_prefill_size": 64}, "chunked_prefill_size"),
        ({"cuda_graph_backend_prefill": "full"}, "breakable CUDA graph"),
        (
            {"cuda_graph_config": {"prefill": {"backend": "full"}}},
            "breakable CUDA graph",
        ),
        ({"cuda_graph_backend_decode": "breakable"}, "full decode CUDA graphs"),
        (
            {"cuda_graph_config": {"decode": {"backend": "breakable"}}},
            "full decode CUDA graphs",
        ),
        ({"disable_overlap_schedule": False}, "overlap"),
        ({"attention_backend": "flashinfer"}, "attention backend"),
        ({"page_size": 16}, "page_size"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "model dtype"),
        ({"enable_torch_compile": True}, "torch.compile"),
    ],
    ids=[
        "radix",
        "chunked-prefill",
        "full-prefill-graph",
        "nested-prefill-graph",
        "breakable-decode-graph",
        "nested-decode-graph",
        "overlap",
        "flashinfer",
        "paged",
        "fp8-kv",
        "torch-compile",
    ],
)
def test_engine_rejects_settings_outside_the_supported_runtime(
    tmp_path: Path, overrides: dict[str, object], error: str
) -> None:
    builder, server_args = engine_server_args(tmp_path, overrides)

    with pytest.raises(ValueError, match=error):
        builder.validate_before_infrastructure(server_args)


def test_reference_sampling_replaces_both_samplers_and_defaults_the_cutoff(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    builder, server_args = engine_server_args(tmp_path, {})
    builder.reference_sampling = True
    builder.reference_sampling_fallback = "ras"

    with caplog.at_level(logging.INFO, logger=engine_builder.logger.name):
        model = set_up_mocked_model(builder, server_args, tmp_path)

    assert (
        "Drama 3 codebook sampling: reference sampling in PyTorch, ras fallback"
        in caplog.text
    )
    assert model.audio_decoder.reference_sampling_fallback == "ras"
    assert model.audio_decoder.sampler is None
    request_builder, _ = builder.make_adapters(
        ConfigOnlyModel(FishOmniConfig(**fish_config_dict()))
    )
    prompt_rows = torch.tensor(frame_rows_of([3], seed=6)[0], dtype=torch.int64)
    request_data = request_builder(
        StagePayload(
            request_id="request-a",
            request=OmniRequest(inputs="prompt"),
            data=generation_input(prompt_rows),
        )
    )
    assert request_data.epsilon_cutoff == 1e-3


def test_epsilon_cutoffs_follow_their_requests_from_prefill_to_decode() -> None:
    """A prefill stages each request's cutoff for its first frame and keeps it in the
    request's slot; decode gathers the cutoffs in batch order as the batch changes."""
    decode = decode_runner(row_count=4)
    buffers = decode.runner.decode_buffers
    requests = prompted_requests([3, 4, 2])
    cutoffs = [0.002, 0.0, 0.03]
    for request, cutoff in zip(requests, cutoffs, strict=True):
        request_data_of(request).epsilon_cutoff = cutoff
    as_float32 = torch.tensor(cutoffs, dtype=torch.float32).tolist()

    prefill_step(
        decode,
        requests,
        [3, 1, 2],
        [[81 + index, index, 1, 2, 3] for index in range(3)],
    )

    assert buffers.epsilon_cutoff[:3].tolist() == as_float32
    pool_cutoffs = decode.runner.state_pool.epsilon_cutoff
    assert pool_cutoffs[torch.tensor([3, 1, 2])].tolist() == as_float32
    decode.forward.next_frames = [[90, 1, 1, 1, 1], [91, 2, 2, 2, 2]]
    decode_step(decode.runner, [requests[2], requests[0]])
    assert buffers.epsilon_cutoff[:2].tolist() == [as_float32[2], as_float32[0]]


def test_the_first_frame_step_reads_cutoffs_at_the_decode_buffers_address() -> None:
    """The first-frame CUDA graphs replay the addresses they captured, so the cutoffs
    come from the decode buffers, not from a per-call tensor."""
    model = ConfigOnlyModel(FishOmniConfig(**fish_config_dict()))
    model.decode_buffers = Drama3DecodeBuffers.allocate(
        row_count=4, frame_width=FRAME_WIDTH, device=torch.device("cpu")
    )
    model.audio_decoder = Mock(spec=AudioDecoder)
    model.tower_attention = torch.nn.ModuleList()
    request_count = 3

    model.sample_first_frames(
        torch.zeros(request_count, 9),
        torch.zeros(request_count, DIM),
        torch.zeros(request_count, dtype=torch.int64),
        torch.zeros((request_count, FRAME_WIDTH), dtype=torch.int64),
        torch.full((request_count, RAS_WINDOW_FRAMES), EMPTY_RAS_SLOT),
        torch.ones(request_count),
        torch.ones(request_count, dtype=torch.int64),
        torch.ones(request_count),
        Mock(spec=ForwardBatch),
    )

    epsilon_cutoff = model.audio_decoder.decode_step.call_args.args[9]
    assert epsilon_cutoff.shape == (request_count,)
    assert epsilon_cutoff.data_ptr() == model.decode_buffers.epsilon_cutoff.data_ptr()
