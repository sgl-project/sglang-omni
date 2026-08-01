# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.models.cosmos3.request_builders import (
    apply_text_result,
    build_cosmos3_sampling_kwargs,
    build_sglang_text_request,
    compute_cosmos3_mrope_positions,
    make_text_scheduler_adapters,
    make_text_stream_output_builder,
    project_preprocessing_to_thinker,
    project_preprocessing_to_vision_encoder,
    project_thinker_to_decode,
    resolve_preprocessing_next_stages,
    resolve_thinker_wait_sources,
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
    generation_config = SimpleNamespace(
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
    generation_config = SimpleNamespace(
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
    generation_config = SimpleNamespace(
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
    generation_config = SimpleNamespace(
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
    generation_config = SimpleNamespace(
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
        req=SimpleNamespace(inflight_middle_chunks=0),
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


def test_apply_text_result_preserves_optional_metadata() -> None:
    state = Cosmos3PipelineState()
    result: Any = SimpleNamespace(
        output_ids=[4],
        finish_reason="length",
        output_token_logprobs=[-0.5],
        weight_version="v1",
    )

    output = apply_text_result(state, stage_name="thinker", result=result)

    assert output["finish_reason"] == "length"
    assert output["output_token_logprobs"] == [-0.5]
    assert output["weight_version"] == "v1"


def test_cosmos3_image_mrope_fills_text_axes_and_uses_visual_grid() -> None:
    token_types = torch.tensor([0, 0, *([1] * 4), 0], dtype=torch.long)
    positions, delta = compute_cosmos3_mrope_positions(
        torch.arange(7),
        token_types,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        video_grid_thw=None,
        spatial_merge_size=2,
    )

    assert positions.tolist() == [
        [0, 1, 2, 2, 2, 2, 4],
        [0, 1, 2, 2, 3, 3, 4],
        [0, 1, 2, 3, 2, 3, 4],
    ]
    assert delta.tolist() == [[-2]]


def test_preprocessing_routes_vision_only_when_media_is_active() -> None:
    payload = StagePayload(
        request_id="route",
        request=OmniRequest(inputs=None),
        data=_state().to_dict(),
    )
    assert resolve_preprocessing_next_stages("route", payload) == ["thinker"]
    assert resolve_thinker_wait_sources("route", "preprocessing", payload) == [
        "preprocessing"
    ]

    state = _state()
    state.encoder_inputs = {
        "vision_encoder": {
            "pixel_values": torch.ones((4, 6)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "cache_key": "image-key",
        }
    }
    payload.data = state.to_dict()

    assert resolve_preprocessing_next_stages("route", payload) == [
        "vision_encoder",
        "thinker",
    ]
    assert resolve_thinker_wait_sources("route", "vision_encoder", payload) is None
    assert resolve_thinker_wait_sources("route", "preprocessing", payload) == [
        "preprocessing",
        "vision_encoder",
    ]
    encoder_payload = project_preprocessing_to_vision_encoder(payload)
    thinker_payload = project_preprocessing_to_thinker(payload)
    encoder_state = Cosmos3PipelineState.from_dict(encoder_payload.data)
    thinker_state = Cosmos3PipelineState.from_dict(thinker_payload.data)
    assert "pixel_values" in encoder_state.encoder_inputs["vision_encoder"]
    assert thinker_state.encoder_inputs == {
        "vision_encoder": {"cache_key": "image-key", "_active": True}
    }


def test_builds_multimodal_request_with_precomputed_vision_inputs() -> None:
    state = Cosmos3PipelineState(
        prompt={
            "input_ids": torch.tensor([10, 151655, 11]),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "prompt_text": "image prompt",
            "mm_token_type_ids": torch.tensor([0, 1, 0]),
        },
        thinker_inputs={
            "model_inputs": {
                "image_embeds": torch.ones((1, 4)),
                "deepstack_visual_embeds": [torch.ones((1, 4)) for _ in range(3)],
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            "media_cache_key": "image-key",
        },
    )
    config = SimpleNamespace(
        image_token_id=151655,
        video_token_id=151656,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )

    data = build_sglang_text_request(
        state,
        params={"max_new_tokens": 2},
        tokenizer=_FakeTokenizer(),
        vocab_size=151936,
        thinker_config=config,
    )

    assert data.req.multimodal_inputs.mrope_positions.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert data.req.omni_model_inputs["image_embeds"].shape == (1, 4)
    assert data.req.origin_input_ids[1] > 151936
    assert data.req.omni_model_inputs["pad_values"]["image"] == (
        data.req.origin_input_ids[1]
    )


def test_decode_projection_drops_prefill_only_vision_tensors() -> None:
    state = _state()
    state.mm_inputs = {"image_grid_thw": torch.tensor([[1, 2, 2]])}
    state.thinker_inputs = {"model_inputs": {"image_embeds": torch.ones((1, 4))}}
    state.text_out = {"output_ids": [42], "is_final": True}
    payload = StagePayload(
        request_id="decode-projection",
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )

    projected = Cosmos3PipelineState.from_dict(project_thinker_to_decode(payload).data)

    assert projected.thinker_inputs == {}
    assert projected.mm_inputs == {}
    assert projected.text_out == {"output_ids": [42], "is_final": True}
    assert projected.prompt is not None
