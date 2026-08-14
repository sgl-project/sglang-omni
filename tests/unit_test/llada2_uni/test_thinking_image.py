# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import copy
import re
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.llada2_uni import config, request_builders, routing, stages
from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    EOI_TOKEN,
    SOI_TOKEN,
    SYSTEM_PROMPT_T2I_THINKING,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class _Tokenizer:
    mask_token_id = 99
    eos_token_id = 2

    _special = {
        SOI_TOKEN: 10,
        EOI_TOKEN: 11,
        BOI_TOKEN: 12,
    }

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special.setdefault(token, 100 + len(self._special))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded_texts.append(text)
        parts = re.findall(r"<[^>]+>|[^<]+", text)
        return [
            (
                self.convert_tokens_to_ids(part)
                if part.startswith("<")
                else 1000 + len(part)
            )
            for part in parts
            if part
        ]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"token-{token_id}" for token_id in token_ids)


def _preprocessor() -> LLaDA2Preprocessor:
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor._max_seq_len = 8192
    preprocessor._tokenizer = _Tokenizer()
    preprocessor._image_token_offset = 321000
    preprocessor._soi_id = preprocessor._tokenizer.convert_tokens_to_ids(SOI_TOKEN)
    preprocessor._eoi_id = preprocessor._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
    preprocessor._boi_id = preprocessor._tokenizer.convert_tokens_to_ids(BOI_TOKEN)
    preprocessor._merge_size = 1
    preprocessor._factor = 16
    return preprocessor


def _thinking_payload(*, modalities: list[str] | None = None) -> StagePayload:
    return StagePayload(
        request_id="thinking-request",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "paint a red fox"}],
            metadata={
                "output_modalities": modalities or ["image"],
                "image_generation": {
                    "mode": "thinking",
                    "width": 64,
                    "height": 32,
                    "cfg_scale": 4.0,
                    "cfg_rescale": 0.7,
                },
            },
        ),
        data=None,
    )


def test_thinking_phase1_omits_image_header_and_cfg() -> None:
    preprocessor = _preprocessor()
    result = asyncio.run(preprocessor(_thinking_payload()))
    state = LLaDA2UniPipelineState.from_dict(result.data)
    prompt_ids = state.prompt["input_ids"].flatten().tolist()

    assert state.task_kind == "t2i"
    assert state.generation_state["image_grid"] == {"height": 1, "width": 2}
    assert state.generation_state["thinking"] == {"phase": 1}
    assert "cfg" not in state.generation_state
    assert _Tokenizer._special[SOI_TOKEN] not in prompt_ids
    assert _Tokenizer._special[BOI_TOKEN] not in prompt_ids
    assert SYSTEM_PROMPT_T2I_THINKING in preprocessor._tokenizer.encoded_texts[0]


def test_thinking_mode_rejects_image_editing() -> None:
    payload = StagePayload(
        request_id="thinking-edit",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "make this brighter"}],
                "images": ["unused-image"],
            },
            metadata={
                "output_modalities": ["image"],
                "image_generation": {"mode": "thinking"},
            },
        ),
        data=None,
    )

    with pytest.raises(ValueError, match="only supports text-to-image"):
        asyncio.run(_preprocessor()(payload))


def test_thinking_phase1_uses_fixed_budget() -> None:
    state = LLaDA2UniPipelineState(
        task_kind="t2i",
        generation_state={
            "image_grid": {"height": 1, "width": 2},
            "thinking": {"phase": 1},
        },
    )

    assert (
        request_builders.resolve_thinker_max_new_tokens(state, {"max_new_tokens": 17})
        == 2048
    )


def test_thinking_prevalidates_both_phase_context_budgets() -> None:
    payload = _thinking_payload()
    payload.request.metadata["image_generation"].update({"width": 4096, "height": 4096})

    unconstrained = _preprocessor()
    unconstrained._max_seq_len = None
    probe = asyncio.run(unconstrained(copy.deepcopy(payload)))
    state = LLaDA2UniPipelineState.from_dict(probe.data)
    prompt_len = int(state.prompt["input_ids"].numel())
    image_grid = state.generation_state["image_grid"]
    image_tokens = image_grid["height"] * image_grid["width"]
    required_context = prompt_len + config.DEFAULT_THINKER_MAX_NEW_TOKENS + image_tokens

    exact = _preprocessor()
    exact._max_seq_len = required_context
    asyncio.run(exact(copy.deepcopy(payload)))

    too_small = _preprocessor()
    too_small._max_seq_len = required_context - 1
    with pytest.raises(ValueError, match="context length"):
        asyncio.run(too_small(copy.deepcopy(payload)))


@dataclass
class _RequestData:
    output_ids: list[int] = field(default_factory=list)
    req: object | None = None
    stage_payload: object | None = None
    finish_reason: str | None = None


class _SamplingParams:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def normalize(self, tokenizer) -> None:
        del tokenizer

    def verify(self, vocab_size: int) -> None:
        del vocab_size


class _Req:
    def __init__(
        self,
        rid,
        origin_input_text,
        origin_input_ids,
        sampling_params,
        **kwargs,
    ) -> None:
        del origin_input_text
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.sampling_params = sampling_params
        self.output_ids = []
        self.eos_token_ids = kwargs.get("eos_token_ids")


def _install_sglang_request_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule_batch = ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.Req = _Req
    sampling_params = ModuleType("sglang.srt.sampling.sampling_params")
    sampling_params.SamplingParams = _SamplingParams
    request_data = ModuleType("sglang_omni.scheduling.sglang_backend.request_data")
    request_data.SGLangDLLMRequestData = _RequestData
    monkeypatch.setitem(sys.modules, schedule_batch.__name__, schedule_batch)
    monkeypatch.setitem(sys.modules, sampling_params.__name__, sampling_params)
    monkeypatch.setitem(sys.modules, request_data.__name__, request_data)


def test_thinking_phase1_stops_on_boi_without_cfg_or_vocab_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sglang_request_stubs(monkeypatch)
    tokenizer = _Tokenizer()
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[41, 42]])},
        task_kind="t2i",
        image_token_offset=321000,
        generation_state={
            "image_grid": {"height": 1, "width": 2},
            "thinking": {"phase": 1},
        },
    )

    data = request_builders.build_dllm_thinker_request(
        state,
        params={"max_new_tokens": 17},
        tokenizer=tokenizer,
        vocab_size=400000,
        dllm_config=object(),
        request_id="thinking-request",
    )

    assert data.req.sampling_params.max_new_tokens == 2048
    assert data.req.eos_token_ids == {tokenizer.eos_token_id, 12}
    assert not hasattr(data.req, "omni_dllm_group_spec")
    assert not hasattr(data.req, "omni_dllm_image_token_offset")


def _phase1_state(
    *, output_modalities: list[str] | None = None
) -> LLaDA2UniPipelineState:
    return LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[41, 42]])},
        task_kind="t2i",
        image_token_offset=321000,
        generation_state={
            "image_grid": {"height": 1, "width": 2},
            "output_size": {"height": 32, "width": 64},
            "resolution_multiplier": 2,
            "thinking": {"phase": 1},
        },
        request_metadata={
            "output_modalities": output_modalities or ["image"],
            "image_generation": {
                "mode": "thinking",
                "cfg_scale": 4.0,
                "cfg_rescale": 0.7,
            },
        },
    )


def test_thinking_transition_builds_phase2_cfg_and_reentry() -> None:
    tokenizer = _Tokenizer()
    state = _phase1_state()

    request_builders.transition_thinking_phase1_to_phase2(
        state,
        tokenizer=tokenizer,
        output_ids=[20, 21, 12],
    )

    assert state.prompt["input_ids"].tolist() == [[41, 42, 20, 21, 12]]
    assert state.generation_state["thinking"] == {
        "phase": 2,
        "needs_reentry": True,
        "trace": "token-20 token-21",
    }
    cfg = state.generation_state["cfg"]
    assert cfg["scale"] == 4.0
    assert cfg["rescale"] == 0.7
    assert cfg["unconditional_input_ids"][-4:] == [
        tokenizer.convert_tokens_to_ids(SOI_TOKEN),
        tokenizer.convert_tokens_to_ids("<|reserved_token_1|>"),
        tokenizer.convert_tokens_to_ids("<|reserved_token_2|>"),
        tokenizer.convert_tokens_to_ids(BOI_TOKEN),
    ]
    conditional, group = request_builders.prepare_dllm_input_group(
        state, mask_token_id=tokenizer.mask_token_id
    )
    assert group is not None
    assert len(conditional) == len(group.companions[0].input_ids)
    assert group.algorithm_args["force_image_only"] is True
    assert group.algorithm_args["image_token_offset"] == 321000
    assert state.thinker_out is None
    assert config.THINKER_STAGE not in state.engine_outputs


def test_missing_boi_transition_is_atomic() -> None:
    state = _phase1_state()
    state.engine_outputs[config.THINKER_STAGE] = {"previous": True}
    before = copy.deepcopy(state.to_dict())

    with pytest.raises(RuntimeError, match=r"did not produce <boi>.*2 token"):
        request_builders.transition_thinking_phase1_to_phase2(
            state,
            tokenizer=_Tokenizer(),
            output_ids=[20, 21],
        )

    assert torch.equal(state.prompt["input_ids"], before["prompt"]["input_ids"])
    assert state.generation_state == before["generation_state"]
    assert state.engine_outputs == before["engine_outputs"]
    assert state.thinker_out is None


def test_scheduler_adapter_transitions_phase1_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sglang_request_stubs(monkeypatch)
    tokenizer = _Tokenizer()
    state = _phase1_state()
    payload = StagePayload(
        request_id="thinking-request",
        request=OmniRequest(inputs=[], params={}),
        data=state.to_dict(),
    )
    _, result_adapter = request_builders.make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=400000,
        dllm_config=object(),
    )

    result = result_adapter(
        _RequestData(
            output_ids=[20, 21, 12],
            stage_payload=payload,
            finish_reason="stop",
        )
    )
    transitioned = LLaDA2UniPipelineState.from_dict(result.data)

    assert transitioned.generation_state["thinking"]["phase"] == 2
    assert transitioned.generation_state["thinking"]["needs_reentry"] is True
    assert transitioned.thinker_out is None


def test_scheduler_adapter_missing_boi_does_not_mutate_payload() -> None:
    state = _phase1_state()
    state.engine_outputs[config.THINKER_STAGE] = {"previous": True}
    payload = StagePayload(
        request_id="thinking-request",
        request=OmniRequest(inputs=[], params={}),
        data=state.to_dict(),
    )
    before = copy.deepcopy(payload.data)
    _, result_adapter = request_builders.make_dllm_thinker_scheduler_adapters(
        tokenizer=_Tokenizer(),
        vocab_size=400000,
        dllm_config=object(),
    )

    with pytest.raises(RuntimeError, match="did not produce <boi>"):
        result_adapter(
            _RequestData(
                output_ids=[20, 21],
                stage_payload=payload,
                finish_reason="length",
            )
        )

    assert payload.data["generation_state"] == before["generation_state"]
    assert payload.data["engine_outputs"] == before["engine_outputs"]
    assert "thinker_out" not in payload.data


def test_phase2_request_clears_reentry_and_restores_grouped_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sglang_request_stubs(monkeypatch)
    tokenizer = _Tokenizer()
    state = _phase1_state()
    state.prompt = {"input_ids": torch.tensor([[41, 42, 20, 12]])}
    state.generation_state["thinking"] = {
        "phase": 2,
        "needs_reentry": True,
        "trace": "token-20",
    }
    state.generation_state["cfg"] = {
        "unconditional_input_ids": [31, 10, 101, 102, 12],
        "scale": 4.0,
        "rescale": 0.7,
    }
    payload = StagePayload(
        request_id="thinking-request",
        request=OmniRequest(inputs=[], params={}),
        data=state.to_dict(),
    )
    request_builder, _ = request_builders.make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=400000,
        dllm_config=object(),
    )

    data = request_builder(payload)
    request_state = LLaDA2UniPipelineState.from_dict(data.stage_payload.data)

    assert request_state.generation_state["thinking"] == {
        "phase": 2,
        "trace": "token-20",
    }
    assert data.req.sampling_params.max_new_tokens == 2
    assert data.req.omni_dllm_image_token_offset == 321000
    assert data.req.omni_dllm_group_spec.algorithm_args["force_image_only"] is True


def test_thinking_router_reenters_only_between_passes() -> None:
    state = _phase1_state()
    state.generation_state["thinking"] = {
        "phase": 2,
        "needs_reentry": True,
        "trace": "reasoning",
    }
    payload = SimpleNamespace(data=state.to_dict())

    assert routing.thinker_next("thinking-request", payload) == config.THINKER_STAGE

    state.generation_state["thinking"].pop("needs_reentry")
    payload.data = state.to_dict()
    assert (
        routing.thinker_next("thinking-request", payload) == config.IMAGE_DECODE_STAGE
    )


def test_native_pipeline_declares_thinker_self_route() -> None:
    pipeline = config.Variants["omni"](model_path="checkpoint")
    thinker = next(
        stage for stage in pipeline.stages if stage.name == config.THINKER_STAGE
    )

    assert set(thinker.next) == {
        config.THINKER_STAGE,
        config.DECODE_STAGE,
        config.IMAGE_DECODE_STAGE,
    }
    assert thinker.route_fn == "sglang_omni.models.llada2_uni.routing.thinker_next"


@pytest.mark.parametrize(
    ("modalities", "expected_text"),
    [
        (["image"], None),
        (["text", "image"], "token-20 token-21"),
    ],
)
def test_thinking_trace_requires_text_and_image_modalities(
    modalities: list[str], expected_text: str | None
) -> None:
    attach_trace = getattr(stages, "attach_thinking_trace", None)
    assert attach_trace is not None
    state = _phase1_state(output_modalities=modalities)
    state.generation_state["thinking"] = {
        "phase": 2,
        "trace": "token-20 token-21",
    }
    result = {"modality": "image", "images": [{"id": "image-0"}]}

    attach_trace(result, state)

    assert result.get("text") == expected_text
