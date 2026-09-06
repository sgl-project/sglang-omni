# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from transformers import GenerationConfig

from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.models.cosmos3.request_builders import (
    apply_text_result,
    build_cosmos3_sampling_kwargs,
    build_sglang_text_request,
    make_text_scheduler_adapters,
    make_text_stream_output_builder,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, OmniRequest, StagePayload


class _FakeTokenizer:
    eos_token_id = 151645

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [1]


def _state() -> Cosmos3PipelineState:
    return Cosmos3PipelineState(
        prompt={
            "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "prompt_text": "rendered prompt",
        }
    )


def test_sampling_defaults_follow_nano_generation_config() -> None:
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )

    kwargs = build_cosmos3_sampling_kwargs(
        {"max_new_tokens": 7, "seed": 123},
        generation_config=generation_config,
    )

    assert kwargs["max_new_tokens"] == 7
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["top_k"] == 20
    assert kwargs["sampling_seed"] == 123


def test_none_sampling_values_fall_back_to_nano_defaults() -> None:
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )

    kwargs = build_cosmos3_sampling_kwargs(
        {
            "max_new_tokens": None,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "repetition_penalty": None,
            "seed": None,
            "sampling_seed": 321,
        },
        generation_config=generation_config,
    )

    assert kwargs == {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "stop": [],
        "stop_token_ids": [],
        "sampling_seed": 321,
    }


def test_builds_plain_sglang_ar_request() -> None:
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        eos_token_id=[151645, 151643],
    )

    data = build_sglang_text_request(
        _state(),
        params={"max_new_tokens": 4, "temperature": 0.2},
        tokenizer=_FakeTokenizer(),
        vocab_size=151936,
        generation_config=generation_config,
        request_id="nano-request",
    )

    assert data.input_ids.tolist() == [11, 12, 13]
    assert data.max_new_tokens == 4
    assert data.temperature == 0.2
    assert data.req.rid == "nano-request"
    assert data.req.origin_input_ids == [11, 12, 13]
    assert data.req.eos_token_ids == {151643, 151645}
    assert data.output_ids is data.req.output_ids


def test_scheduler_adapters_store_normalized_text_result(monkeypatch) -> None:
    payload = StagePayload(
        request_id="adapter-request",
        request=OmniRequest(inputs=None, params={"max_new_tokens": 2}),
        data=_state().to_dict(),
    )
    request_builder, result_adapter = make_text_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        vocab_size=151936,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )

    data = request_builder(payload)
    data.output_ids.extend([21, 22])
    data.finish_reason = "stop"
    result = result_adapter(data)
    state = Cosmos3PipelineState.from_dict(result.data)

    assert data.stage_payload is payload
    assert state.text_out == {
        "output_ids": [21, 22],
        "is_final": True,
        "finish_reason": "stop",
    }
    assert state.engine_outputs["thinker"] == state.text_out


def test_scheduler_ignores_unset_openai_sampling_placeholders(monkeypatch) -> None:
    payload = StagePayload(
        request_id="default-sampling-request",
        request=OmniRequest(
            inputs=None,
            params={
                "max_new_tokens": 2,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        ),
        data=_state().to_dict(),
    )
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )
    request_builder, _ = make_text_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        vocab_size=151936,
        generation_config=generation_config,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )

    data = request_builder(payload)

    assert data.temperature == 0.7
    assert data.top_p == 0.8
    assert data.top_k == 20


def test_scheduler_honors_explicit_openai_sampling_values(monkeypatch) -> None:
    payload = StagePayload(
        request_id="explicit-sampling-request",
        request=OmniRequest(
            inputs=None,
            params={
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
            },
            metadata={EXPLICIT_GENERATION_PARAMS_KEY: ["temperature"]},
        ),
        data=_state().to_dict(),
    )
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    )
    request_builder, _ = make_text_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        vocab_size=151936,
        generation_config=generation_config,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )

    data = request_builder(payload)

    assert data.temperature == 1.0
    assert data.top_p == 0.8
    assert data.top_k == 20


def test_stream_builder_emits_only_for_streaming_requests() -> None:
    builder = make_text_stream_output_builder()
    payload = StagePayload(
        request_id="stream-request",
        request=OmniRequest(inputs=None, params={"stream": True}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(
            inflight_middle_chunks=0,
            sampling_params=SimpleNamespace(stop_token_ids=None, stop_strs=[]),
        ),
        stage_payload=payload,
    )

    messages = builder(
        "stream-request",
        req_data,
        SimpleNamespace(data=42),
    )

    assert len(messages) == 1
    assert messages[0].target == "decode"
    assert messages[0].data.tolist() == [42]

    payload.request.params["stream"] = False
    assert builder("stream-request", req_data, SimpleNamespace(data=43)) == []


def test_stream_builder_filters_stop_token_ids() -> None:
    builder = make_text_stream_output_builder()
    payload = StagePayload(
        request_id="stream-request",
        request=OmniRequest(inputs=None, params={"stream": True}),
        data={},
    )
    req_data = SimpleNamespace(
        req=SimpleNamespace(
            inflight_middle_chunks=0,
            sampling_params=SimpleNamespace(stop_token_ids={7}, stop_strs=[]),
        ),
        stage_payload=payload,
    )

    assert builder("stream-request", req_data, SimpleNamespace(data=7)) == []

    messages = builder("stream-request", req_data, SimpleNamespace(data=42))
    assert len(messages) == 1
    assert messages[0].data.tolist() == [42]


class _StreamTailTokenizer:
    pieces = {1: "hello", 2: " gam", 3: "ma", 4: "bit"}

    def decode(self, token_ids) -> str:
        return "".join(self.pieces.get(int(token_id), "") for token_id in token_ids)


def _stop_str_req_data(stop_strs: list[str]) -> SimpleNamespace:
    payload = StagePayload(
        request_id="stream-request",
        request=OmniRequest(inputs=None, params={"stream": True}),
        data={},
    )
    req = SimpleNamespace(
        inflight_middle_chunks=0,
        sampling_params=SimpleNamespace(
            stop_token_ids=None,
            stop_strs=stop_strs,
            stop_str_max_len=max((len(stop) for stop in stop_strs), default=0),
        ),
        output_ids=[],
        tokenizer=_StreamTailTokenizer(),
    )
    return SimpleNamespace(req=req, stage_payload=payload)


def _stream_step(builder, req_data: SimpleNamespace, token_id: int):
    # The builder runs before process_batch_result appends the token, so
    # append to output_ids only after the call, matching scheduler ordering.
    messages = builder("stream-request", req_data, SimpleNamespace(data=token_id))
    req_data.req.output_ids.append(token_id)
    return messages


def test_stream_builder_holds_possible_stop_prefix_and_releases() -> None:
    builder = make_text_stream_output_builder()
    req_data = _stop_str_req_data([" gamma"])

    messages = _stream_step(builder, req_data, 1)
    assert [message.data.tolist() for message in messages] == [[1]]

    assert _stream_step(builder, req_data, 2) == []

    messages = _stream_step(builder, req_data, 4)
    assert [message.data.tolist() for message in messages] == [[2], [4]]
    assert [message.metadata for message in messages] == [
        {"token_id": 2},
        {"token_id": 4},
    ]


def test_stream_builder_flush_ships_held_stop_tokens_with_terminal_flag() -> None:
    builder = make_text_stream_output_builder()
    req_data = _stop_str_req_data([" gamma"])

    assert [m.data.tolist() for m in _stream_step(builder, req_data, 1)] == [[1]]
    assert _stream_step(builder, req_data, 2) == []
    assert _stream_step(builder, req_data, 3) == []

    messages = builder.flush("stream-request", req_data)
    assert len(messages) == 1
    assert messages[0].data.tolist() == [2, 3]
    assert messages[0].metadata == {"terminal_flush": True}

    assert builder.flush("stream-request", req_data) == []


def test_apply_text_result_preserves_optional_metadata() -> None:
    state = Cosmos3PipelineState()
    result: Any = SimpleNamespace(
        output_ids=[4],
        finish_reason="stop",
        req=SimpleNamespace(
            finished_reason=SimpleNamespace(to_json=lambda: {"matched": "###"})
        ),
        output_token_logprobs=[-0.5],
        weight_version="v1",
    )

    output = apply_text_result(state, stage_name="thinker", result=result)

    assert output["finish_reason"] == "stop"
    assert output["matched_stop"] == "###"
    assert output["output_token_logprobs"] == [-0.5]
    assert output["weight_version"] == "v1"
