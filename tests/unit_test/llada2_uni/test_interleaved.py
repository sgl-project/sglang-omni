# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import asyncio
import copy
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from types import ModuleType, SimpleNamespace

import pytest
import torch
from fastapi import HTTPException

from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    EOI_TOKEN,
    SOI_TOKEN,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.protocol import ChatCompletionRequest


class _Tokenizer:
    mask_token_id = 99
    eos_token_id = 2
    additional_stop_token_ids = [666]

    _special = {
        SOI_TOKEN: 10,
        EOI_TOKEN: 11,
        BOI_TOKEN: 12,
        "<uncondition>": 13,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special.setdefault(token, 100 + len(self._special))

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == 20:
            return "<|reserved_token_2|>"
        if token_id == 21:
            return "<|reserved_token_3|>"
        return next(
            (token for token, value in self._special.items() if value == token_id),
            f"token-{token_id}",
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "<uncondition>":
            return [self._special["<uncondition>"]]
        if text == "<|reserved_token_2|>":
            return [20]
        if text == "<|reserved_token_3|>":
            return [21]
        parts = re.findall(r"<[^>]+>|[^<]+", text)
        return [
            self.convert_tokens_to_ids(part) if part.startswith("<") else 1000
            for part in parts
            if part
        ]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "|".join(str(token_id) for token_id in token_ids)


def _preprocessor() -> LLaDA2Preprocessor:
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor._max_seq_len = 8192
    preprocessor._tokenizer = _Tokenizer()
    preprocessor._image_token_offset = 1000
    preprocessor._soi_id = preprocessor._tokenizer.convert_tokens_to_ids(SOI_TOKEN)
    preprocessor._eoi_id = preprocessor._tokenizer.convert_tokens_to_ids(EOI_TOKEN)
    preprocessor._boi_id = preprocessor._tokenizer.convert_tokens_to_ids(BOI_TOKEN)
    preprocessor._merge_size = 1
    preprocessor._factor = 16
    return preprocessor


def _request(**overrides) -> ChatCompletionRequest:
    values = {
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "write and illustrate a story"}],
        "modalities": ["text", "image"],
        "stream": False,
        "image_generation": {"mode": "interleaved"},
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


def _load_openai_validation():
    source_path = Path(__file__).parents[3] / "sglang_omni" / "serve" / "openai_api.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    requested_names = {
        "_requested_modalities",
        "_validate_chat_image_generation_request",
    }
    requested = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in requested_names
    ]
    assert {node.name for node in requested} == requested_names
    namespace = {"HTTPException": HTTPException}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=requested, type_ignores=[])),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["_validate_chat_image_generation_request"]


def test_interleaved_request_uses_only_the_shared_image_generation_contract() -> None:
    request = _request()

    _load_openai_validation()(request)

    assert request.modalities == ["text", "image"]
    assert request.stream is False
    assert request.image_generation == {"mode": "interleaved"}
    assert "interleaved_generation" not in request.model_dump()

    source = (
        Path(__file__).parents[3] / "sglang_omni" / "serve" / "openai_api.py"
    ).read_text(encoding="utf-8")
    assert 'metadata["image_generation"] = dict(req.image_generation)' in source
    assert 'metadata["interleaved_generation"]' not in source


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"modalities": ["image", "text"]}, "exactly"),
        ({"modalities": ["text", "image", "audio"]}, "exactly"),
        ({"stream": True}, "stream"),
        ({"images": ["source.png"]}, "image input"),
        ({"image_generation": {"mode": "interleaved", "width": 512}}, "width"),
        ({"image_generation": {"mode": "interleaved", "height": 512}}, "height"),
        ({"image_generation": {"mode": "interleaved", "size": "512x512"}}, "size"),
        (
            {
                "image_generation": {
                    "mode": "interleaved",
                    "decode_mode": "decoder-turbo",
                }
            },
            "decoder-turbo",
        ),
        ({"image_generation": {"mode": "interleaved", "format": "jpeg"}}, "format"),
        (
            {"image_generation": {"mode": "interleaved", "decode_mode": "fast"}},
            "decode_mode",
        ),
        ({"image_generation": {"mode": "interleaved", "unknown": 1}}, "unsupported"),
        (
            {"image_generation": {"mode": "interleaved", "max_frames": 1.5}},
            "positive integer",
        ),
        (
            {"image_generation": {"mode": "interleaved", "max_frames": True}},
            "positive integer",
        ),
        (
            {"image_generation": {"mode": "interleaved", "max_frames": "2"}},
            "positive integer",
        ),
        (
            {"image_generation": {"mode": "interleaved", "decoder_steps": True}},
            "positive integer",
        ),
        (
            {"image_generation": {"mode": "interleaved", "seed": -1}},
            "non-negative integer",
        ),
        (
            {
                "image_generation": {
                    "mode": "interleaved",
                    "max_image_tokens": "16",
                }
            },
            "positive integer",
        ),
        (
            {"image_generation": {"mode": "interleaved", "cfg_scale": True}},
            "finite number",
        ),
        (
            {"image_generation": {"mode": "interleaved", "cfg_scale": float("nan")}},
            "finite number",
        ),
        (
            {"image_generation": {"mode": "interleaved", "cfg_scale": float("inf")}},
            "finite number",
        ),
        (
            {"image_generation": {"mode": "interleaved", "cfg_rescale": 1.1}},
            r"\[0, 1\]",
        ),
    ],
)
def test_interleaved_request_rejects_unsupported_public_combinations(
    overrides, message
) -> None:
    with pytest.raises(HTTPException, match=message):
        _load_openai_validation()(_request(**overrides))


def test_preprocessor_initializes_generation_state_for_interleaved_mode() -> None:
    payload = StagePayload(
        request_id="request-0",
        request=OmniRequest(
            inputs=[{"role": "user", "content": "write and illustrate a story"}],
            params={"stream": False},
            metadata={
                "output_modalities": ["text", "image"],
                "image_generation": {
                    "mode": "interleaved",
                    "max_frames": 2,
                    "max_image_tokens": 16,
                },
            },
        ),
        data=None,
    )

    result = asyncio.run(_preprocessor()(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)

    assert state.task_kind == "interleaved"
    assert state.image_token_offset == 1000
    assert state.generation_state["interleaved"]["phase"] == "text"
    assert state.generation_state["interleaved"]["frame_index"] == 0
    assert "stream_state" not in result.data


def test_interleaved_config_accepts_only_typed_supported_controls() -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedGenerationConfig

    config = InterleavedGenerationConfig.from_image_generation(
        {
            "mode": "interleaved",
            "max_frames": 2,
            "text_max_new_tokens": 128,
            "cfg_scale": 4,
            "cfg_text_scale": 7.5,
            "cfg_image_scale": 1.5,
            "cfg_rescale": 0.7,
            "decoder_steps": 50,
            "seed": 0,
            "max_image_tokens": 16,
            "format": "png",
            "decode_mode": "normal",
        }
    )

    assert config.max_frames == 2
    assert config.cfg_scale == 4.0
    assert config.seed == 0
    assert config.format == "png"
    assert config.decode_mode == "normal"


def _interleaved_state(*, output_ids: list[int]) -> LLaDA2UniPipelineState:
    return LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={"output_ids": output_ids, "is_final": True},
        generation_state={
            "interleaved": {
                "phase": "text",
                "frame_index": 0,
                "segment_start": 2,
                "prompt_length": 2,
                "max_seq_len": 8192,
                "max_frames": 2,
                "text_max_new_tokens": 128,
                "max_image_tokens": 16,
                "cfg_scale": 4.0,
                "cfg_text_scale": 7.5,
                "cfg_image_scale": 1.5,
                "cfg_rescale": 0.7,
                "decoder_steps": 50,
                "segments": [],
            }
        },
        request_metadata={"image_generation": {"mode": "interleaved"}},
        task_kind="interleaved",
        image_token_offset=1000,
    )


def _snapshot(state: LLaDA2UniPipelineState):
    def normalize(value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(state.to_dict())


def test_text_phase_accepts_only_a_dynamic_terminal_image_header() -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state

    tokenizer = _Tokenizer()
    state = _interleaved_state(
        output_ids=[
            77,
            tokenizer._special[SOI_TOKEN],
            20,
            21,
            tokenizer._special[BOI_TOKEN],
        ]
    )

    advance_interleaved_state(state, tokenizer, completed_phase="text")

    current = state.generation_state["interleaved"]
    assert current["phase"] == "image"
    assert current["current_frame"]["grid_h"] == 2
    assert current["current_frame"]["grid_w"] == 3
    assert current["current_frame"]["image_token_count"] == 6
    assert current["cfg_plan"]["mode"] == "simple"


def test_text_phase_context_budget_uses_longest_cfg_prefix() -> None:
    from sglang_omni.models.llada2_uni.interleaved import (
        InterleavedGenerationConfig,
        advance_interleaved_state,
        build_cfg_plan,
        parse_image_header,
    )

    tokenizer = _Tokenizer()
    output_ids = [77, 10, 20, 20, 12]
    probe = _interleaved_state(output_ids=output_ids)
    interleaved = probe.generation_state["interleaved"]
    full_ids = probe.prompt["input_ids"].flatten().tolist() + output_ids
    header = parse_image_header(output_ids, tokenizer)
    plan = build_cfg_plan(
        full_ids=full_ids,
        header=header,
        frame_index=0,
        tokenizer=tokenizer,
        config=InterleavedGenerationConfig.from_state(interleaved),
    )
    longest_prefix = max(len(prefix) for prefix in [full_ids, *plan.branches.values()])
    required_context = longest_prefix + header.image_token_count + 1
    assert longest_prefix > len(full_ids)

    exact = _interleaved_state(output_ids=output_ids)
    exact.generation_state["interleaved"]["max_seq_len"] = required_context
    advance_interleaved_state(exact, tokenizer, completed_phase="text")

    too_short = _interleaved_state(output_ids=output_ids)
    too_short.generation_state["interleaved"]["max_seq_len"] = required_context - 1
    before = copy.deepcopy(_snapshot(too_short))
    with pytest.raises(ValueError, match="context"):
        advance_interleaved_state(too_short, tokenizer, completed_phase="text")
    assert _snapshot(too_short) == before


@pytest.mark.parametrize(
    ("output_ids", "message"),
    [
        ([77, 10, 20, 12], "exactly"),
        ([1007, 10, 20, 21, 12], "text phase emitted image token"),
        ([77, 10, 20, 21, 12, 88], "end with"),
    ],
)
def test_text_phase_validation_is_request_atomic(output_ids, message) -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state

    state = _interleaved_state(output_ids=output_ids)
    before = copy.deepcopy(_snapshot(state))

    with pytest.raises(ValueError, match=message):
        advance_interleaved_state(state, _Tokenizer(), completed_phase="text")

    assert _snapshot(state) == before


def _image_phase_state() -> tuple[LLaDA2UniPipelineState, _Tokenizer]:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state

    tokenizer = _Tokenizer()
    state = _interleaved_state(
        output_ids=[
            77,
            tokenizer._special[SOI_TOKEN],
            20,
            20,
            tokenizer._special[BOI_TOKEN],
        ]
    )
    advance_interleaved_state(state, tokenizer, completed_phase="text")
    return state, tokenizer


def test_image_phase_accepts_exact_vq_tokens_followed_by_eoi() -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state

    state, tokenizer = _image_phase_state()
    state.thinker_out = {
        "output_ids": [1000, 1001, 1002, 1003, tokenizer._special[EOI_TOKEN]],
        "is_final": True,
    }

    advance_interleaved_state(state, tokenizer, completed_phase="image")

    current = state.generation_state["interleaved"]
    assert current["phase"] == "text"
    assert current["frame_index"] == 1
    assert current["emit_frame"] is True
    assert state.thinker_out["output_ids"] == [1000, 1001, 1002, 1003]


def test_image_phase_accumulates_chunked_tokens_in_prompt_cfg_and_decoder() -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens

    state, tokenizer = _image_phase_state()
    state.thinker_out = {"output_ids": [1000, 1001], "is_final": True}

    advance_interleaved_state(state, tokenizer, completed_phase="image")

    interleaved = state.generation_state["interleaved"]
    assert state.prompt["input_ids"].flatten().tolist()[-2:] == [1000, 1001]
    assert all(
        branch[-2:] == [1000, 1001]
        for branch in interleaved["cfg_plan"]["branches"].values()
    )
    assert interleaved["current_frame"]["remaining_image_tokens"] == 2

    state.thinker_out = {"output_ids": [1002, 1003, 11], "is_final": True}
    advance_interleaved_state(state, tokenizer, completed_phase="image")

    assert state.prompt["input_ids"].flatten().tolist()[-5:] == [
        1000,
        1001,
        1002,
        1003,
        11,
    ]
    assert state.thinker_out["output_ids"] == [1000, 1001, 1002, 1003]
    assert extract_image_vq_tokens(state)[:3] == ([0, 1, 2, 3], 2, 2)


@pytest.mark.parametrize(
    ("output_ids", "message"),
    [
        ([1000, 11], "remaining"),
        ([1000, 1001, 1002, 1003], "without EOI"),
        ([1000, 11, 1001], "after EOI"),
        ([1000, 11, 11], "multiple EOI"),
        ([77, 1001, 1002, 1003, 11], "non-image token"),
        ([1000, 1001, 1002, 1003, 1004, 11], "with 4 remaining"),
    ],
)
def test_image_phase_validation_is_request_atomic(output_ids, message) -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state

    state, tokenizer = _image_phase_state()
    state.thinker_out = {"output_ids": output_ids, "is_final": True}
    before = copy.deepcopy(_snapshot(state))

    with pytest.raises(ValueError, match=message):
        advance_interleaved_state(state, tokenizer, completed_phase="image")

    assert _snapshot(state) == before


def test_interleaved_image_phase_uses_shared_aligned_cfg_group() -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.request_builders import prepare_dllm_input_group

    state = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
    advance_interleaved_state(state, _Tokenizer(), completed_phase="text")

    conditional, group = prepare_dllm_input_group(state, mask_token_id=99)

    assert group is not None
    assert len(conditional) == len(group.companions[0].input_ids)
    assert tuple(companion.role for companion in group.companions) == ("unconditional",)
    assert group.algorithm_args["image_token_offset"] == 1000
    assert group.algorithm_args["allowed_stop_token_ids"] == (11,)


def test_interleaved_vocab_boundary_and_budget_apply_only_to_image_phase() -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.request_builders import (
        resolve_native_image_token_offset,
        resolve_thinker_max_new_tokens,
    )

    state = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
    assert resolve_native_image_token_offset(state, vocab_size=2000) is None
    assert resolve_thinker_max_new_tokens(state, {"max_new_tokens": 3}) == 128

    advance_interleaved_state(state, _Tokenizer(), completed_phase="text")

    assert resolve_native_image_token_offset(state, vocab_size=2000) == 1000
    assert resolve_thinker_max_new_tokens(state, {"max_new_tokens": 3}) == 5


def test_interleaved_text_budget_is_clamped_to_remaining_context() -> None:
    from sglang_omni.models.llada2_uni.request_builders import (
        resolve_thinker_max_new_tokens,
    )

    state = _interleaved_state(output_ids=[])
    state.prompt = {"input_ids": torch.zeros((1, 8190), dtype=torch.long)}

    assert resolve_thinker_max_new_tokens(state, {}) == 2


def test_interleaved_text_budget_rejects_exhausted_context() -> None:
    from sglang_omni.models.llada2_uni.request_builders import (
        resolve_thinker_max_new_tokens,
    )

    state = _interleaved_state(output_ids=[])
    state.prompt = {"input_ids": torch.zeros((1, 8192), dtype=torch.long)}

    with pytest.raises(ValueError, match="context"):
        resolve_thinker_max_new_tokens(state, {})


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


def test_interleaved_image_request_uses_eoi_as_its_only_stop_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.request_builders import (
        build_dllm_thinker_request,
    )

    _install_sglang_request_stubs(monkeypatch)
    tokenizer = _Tokenizer()
    state = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
    advance_interleaved_state(state, tokenizer, completed_phase="text")

    data = build_dllm_thinker_request(
        state,
        params={"stop": ["user-stop"], "stop_token_ids": [777]},
        tokenizer=tokenizer,
        vocab_size=2000,
        dllm_config=object(),
        request_id="request-0",
    )

    assert data.req.eos_token_ids == {tokenizer._special[EOI_TOKEN]}
    assert data.req.sampling_params.stop == []
    assert data.req.sampling_params.stop_token_ids == [tokenizer._special[EOI_TOKEN]]
    assert data.req.tokenizer is None
    assert data.req.omni_dllm_image_token_offset == 1000
    assert data.req.omni_dllm_allowed_stop_token_ids == (11,)


def test_interleaved_text_request_keeps_tokenizer_stop_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.request_builders import (
        build_dllm_thinker_request,
    )

    _install_sglang_request_stubs(monkeypatch)
    state = _interleaved_state(output_ids=[])

    data = build_dllm_thinker_request(
        state,
        params={},
        tokenizer=_Tokenizer(),
        vocab_size=2000,
        dllm_config=object(),
        request_id="request-text",
    )

    assert data.req.tokenizer is not None


def test_scheduler_forwards_independent_image_stop_token_metadata() -> None:
    source = (
        Path(__file__).parents[3] / "sglang_omni" / "scheduling" / "dllm_scheduler.py"
    ).read_text(encoding="utf-8")

    assert "forward_batch.omni_dllm_allowed_stop_token_ids" in source


def test_nonfinal_and_final_frames_have_distinct_routes() -> None:
    from sglang_omni.models.llada2_uni.config import (
        IMAGE_DECODE_STAGE,
        INTERLEAVED_COLLECT_STAGE,
        THINKER_STAGE,
    )
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.routing import thinker_next

    tokenizer = _Tokenizer()
    nonfinal = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
    advance_interleaved_state(nonfinal, tokenizer, completed_phase="text")
    nonfinal.thinker_out = {"output_ids": [1000, 1001, 1002, 1003, 11]}
    advance_interleaved_state(nonfinal, tokenizer, completed_phase="image")

    assert thinker_next("request", _payload(nonfinal)) == [
        IMAGE_DECODE_STAGE,
        THINKER_STAGE,
    ]

    final = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
    final.generation_state["interleaved"]["max_frames"] = 1
    advance_interleaved_state(final, tokenizer, completed_phase="text")
    final.thinker_out = {"output_ids": [1000, 1001, 1002, 1003, 11]}
    advance_interleaved_state(final, tokenizer, completed_phase="image")

    assert thinker_next("request", _payload(final)) == [
        IMAGE_DECODE_STAGE,
        INTERLEAVED_COLLECT_STAGE,
    ]


def test_invalid_interleaved_route_reports_the_request_id() -> None:
    from sglang_omni.models.llada2_uni.routing import thinker_next

    state = _interleaved_state(output_ids=[])

    with pytest.raises(ValueError, match="request-invalid"):
        thinker_next("request-invalid", _payload(state, request_id="request-invalid"))


def test_interleaved_frame_extracts_exact_decoder_grid_and_checkpoint_offset() -> None:
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.merge import extract_image_vq_tokens

    tokenizer = _Tokenizer()
    state = _interleaved_state(output_ids=[77, 10, 20, 21, 12])
    advance_interleaved_state(state, tokenizer, completed_phase="text")
    state.thinker_out = {"output_ids": [1000 + index for index in range(6)] + [11]}
    advance_interleaved_state(state, tokenizer, completed_phase="image")

    extracted = extract_image_vq_tokens(state)

    assert extracted == (
        [0, 1, 2, 3, 4, 5],
        2,
        3,
        {"mode": "interleaved"},
    )


def test_trailing_text_finishes_at_collector_and_accumulates_usage() -> None:
    from sglang_omni.models.llada2_uni.config import INTERLEAVED_COLLECT_STAGE
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.routing import thinker_next

    state = _interleaved_state(output_ids=[70, 71])

    advance_interleaved_state(
        state,
        _Tokenizer(),
        completed_phase="text",
        finish_reason="length",
    )

    current = state.generation_state["interleaved"]
    assert current["done"] is True
    assert current["trailing_text"] == "70|71"
    assert current["finish_reason"] == "length"
    assert current["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert thinker_next("request", _payload(state)) == INTERLEAVED_COLLECT_STAGE


def test_adapter_advances_state_and_consumes_route_flags_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.request_builders import (
        make_dllm_thinker_scheduler_adapters,
    )

    _install_sglang_request_stubs(monkeypatch)
    tokenizer = _Tokenizer()
    state = _interleaved_state(output_ids=[])
    payload = _payload(state)
    request_builder, result_adapter = make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=2000,
        dllm_config=object(),
    )
    result = result_adapter(
        _RequestData(
            output_ids=[77, 10, 20, 20, 12],
            stage_payload=payload,
            finish_reason="stop",
        )
    )
    transitioned = LLaDA2UniPipelineState.from_dict(result.data)
    assert transitioned.generation_state["interleaved"]["phase"] == "image"
    assert transitioned.generation_state["interleaved"]["needs_reentry"] is True

    built = request_builder(result)
    submitted = LLaDA2UniPipelineState.from_dict(built.stage_payload.data)
    assert "needs_reentry" not in submitted.generation_state["interleaved"]


def _payload(state, request_id="request"):
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="prompt"),
        data=state.to_dict(),
    )


def _final_state(*, frame_count: int, trailing_text: str = ""):
    return LLaDA2UniPipelineState(
        generation_state={
            "interleaved": {
                "done": True,
                "phase": "done",
                "frame_index": frame_count,
                "segments": [
                    {"frame_index": index, "text": f"text-{index}"}
                    for index in range(1, frame_count + 1)
                ],
                "trailing_text": trailing_text,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 8,
                    "total_tokens": 10,
                },
            }
        },
        task_kind="interleaved",
    )


def _frame_payload(index: int, *, request_id: str = "request") -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="prompt"),
        data={
            "kind": "interleaved_frame",
            "frame": {
                "index": index,
                "image": {
                    "id": f"image-{request_id}-{index - 1}",
                    "data": f"base64-{index}",
                    "format": "png",
                    "width": 64,
                    "height": 64,
                },
            },
        },
    )


@pytest.mark.parametrize("frame_first", [True, False])
def test_collector_orders_frames_and_uses_image_refs(frame_first: bool) -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler

    collector = InterleavedCollectorScheduler()
    final = _payload(_final_state(frame_count=1, trailing_text="after"))
    frame = _frame_payload(1)
    first, second = (frame, final) if frame_first else (final, frame)

    collector._receive("request", first)
    with pytest.raises(Empty):
        collector.outbox.get_nowait()
    collector._receive("request", second)

    result = collector.outbox.get_nowait().data.data
    assert result["images"] == [
        {
            "id": "image-request-0",
            "data": "base64-1",
            "format": "png",
            "width": 64,
            "height": 64,
        }
    ]
    assert result["content"] == [
        {"type": "text", "text": "text-1"},
        {"type": "image_ref", "image_id": "image-request-0"},
        {"type": "text", "text": "after"},
    ]
    assert "data" not in result["content"][1]


def test_collector_supports_zero_frame_text_only_completion() -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler

    collector = InterleavedCollectorScheduler()
    collector._receive(
        "request",
        _payload(_final_state(frame_count=0, trailing_text="text only")),
    )

    result = collector.outbox.get_nowait().data.data
    assert result["content"] == [{"type": "text", "text": "text only"}]
    assert result["images"] == []


def test_collector_duplicate_frame_fails_and_cleans_request_once() -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler

    collector = InterleavedCollectorScheduler()
    collector._receive("request", _frame_payload(1))

    with pytest.raises(ValueError, match="duplicate"):
        collector._receive("request", _frame_payload(1))

    assert "request" not in collector._frames
    assert "request" not in collector._final_payloads


def test_collector_abort_persistently_suppresses_late_results() -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler

    collector = InterleavedCollectorScheduler()
    collector.abort("request")
    collector._receive("request", _frame_payload(1))
    collector._receive("request", _payload(_final_state(frame_count=1)))

    with pytest.raises(Empty):
        collector.outbox.get_nowait()
    assert "request" not in collector._frames
    assert "request" not in collector._final_payloads


def test_only_interleaved_nonterminal_decoder_allows_multiple_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.components import image_decoder
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler
    from sglang_omni.models.llada2_uni.stages import create_image_decode_executor

    class _Decoder:
        def __init__(self, **kwargs):
            del kwargs

    monkeypatch.setattr(image_decoder, "LLaDA2ImageDecoder", _Decoder)
    ordinary = create_image_decode_executor("model", device="cpu")
    interleaved = create_image_decode_executor(
        "model", device="cpu", interleaved_nonterminal=True
    )
    collector = InterleavedCollectorScheduler()

    assert ordinary.allow_multiple_inflight_per_request is False
    assert interleaved.allow_multiple_inflight_per_request is True
    assert collector.allow_multiple_inflight_per_request is False


def test_real_decoder_assigns_unique_ids_to_multiple_interleaved_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.llada2_uni.components import image_decoder
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler
    from sglang_omni.models.llada2_uni.stages import create_image_decode_executor

    class _Decoder:
        def __init__(self, **kwargs):
            del kwargs

        def decode_to_bytes(self, token_ids, height, width, **kwargs):
            del token_ids, kwargs
            return b"png", width, height

    def _decoder_payload(frame_index: int) -> StagePayload:
        state = LLaDA2UniPipelineState(
            thinker_out={"output_ids": [1000, 1001, 1002, 1003]},
            generation_state={
                "interleaved": {
                    "frame_index": frame_index,
                    "segments": [
                        {
                            "frame_index": index,
                            "text": f"text-{index}",
                            "grid_h": 2,
                            "grid_w": 2,
                        }
                        for index in range(1, frame_index + 1)
                    ],
                }
            },
            request_metadata={"image_generation": {"mode": "interleaved"}},
            task_kind="interleaved",
            image_token_offset=1000,
        )
        return _payload(state, request_id="request-multi")

    monkeypatch.setattr(image_decoder, "LLaDA2ImageDecoder", _Decoder)
    scheduler = create_image_decode_executor(
        "model",
        device="cpu",
        interleaved_nonterminal=True,
    )
    collector = InterleavedCollectorScheduler()

    collector._receive("request-multi", scheduler._fn(_decoder_payload(1)))
    collector._receive("request-multi", scheduler._fn(_decoder_payload(2)))
    collector._receive(
        "request-multi",
        _payload(_final_state(frame_count=2), request_id="request-multi"),
    )

    ordinary = LLaDA2UniPipelineState(
        thinker_out={"output_ids": [1000, 1001, 1002, 1003]},
        generation_state={"image_grid": {"height": 2, "width": 2}},
        request_metadata={"image_generation": {}},
        task_kind="t2i",
        image_token_offset=1000,
    )
    ordinary_result = scheduler._fn(_payload(ordinary, request_id="request-ordinary"))

    result = collector.outbox.get_nowait().data.data
    assert [image["id"] for image in result["images"]] == [
        "image-request-multi-0",
        "image-request-multi-1",
    ]
    assert ordinary_result.data["images"][0]["id"] == "image-request-ordinary-0"


def _wire_interleaved_result() -> dict:
    return {
        "modality": "interleaved",
        "content": [
            {"type": "text", "text": "before"},
            {"type": "image_ref", "image_id": "image-request-0"},
            {"type": "text", "text": "after"},
        ],
        "images": [
            {
                "id": "image-request-0",
                "data": "base64-image",
                "format": "png",
                "width": 64,
                "height": 64,
            }
        ],
        "finish_reason": "stop",
    }


def test_client_preserves_ordered_interleaved_content_through_completion() -> None:
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest

    class _Coordinator:
        async def submit(self, request_id, request):
            del request_id, request
            return _wire_interleaved_result()

    chunk = Client._default_result_builder("request", _wire_interleaved_result())
    result = asyncio.run(
        Client(_Coordinator()).completion(
            GenerateRequest(prompt="prompt", stream=False),
            request_id="request",
        )
    )

    assert chunk.content == _wire_interleaved_result()["content"]
    assert chunk.modality == "interleaved"
    assert result.content == _wire_interleaved_result()["content"]
    assert [image.id for image in result.images] == ["image-request-0"]


def test_protocol_builds_interleaved_content_refs_with_images_as_only_data() -> None:
    from sglang_omni.client.types import CompletionImage
    from sglang_omni.serve.protocol import build_chat_completion_message

    result = SimpleNamespace(
        text="",
        content=_wire_interleaved_result()["content"],
        images=[
            CompletionImage(
                id="image-request-0",
                data="base64-image",
                width=64,
                height=64,
            )
        ],
    )

    message = build_chat_completion_message(
        result,
        ["text", "image"],
        interleaved=True,
    )

    assert message["content"] == _wire_interleaved_result()["content"]
    assert message["images"][0]["data"] == "base64-image"
    assert set(message["content"][1]) == {"type", "image_id"}


@pytest.mark.parametrize(
    ("content", "images", "message"),
    [
        (
            [
                {
                    "type": "image_ref",
                    "image_id": "image-0",
                    "data": "duplicate-base64",
                }
            ],
            [{"id": "image-0", "data": "base64-image"}],
            "fields",
        ),
        (
            [{"type": "image_ref", "image_id": "orphan"}],
            [{"id": "image-0", "data": "base64-image"}],
            "one-to-one",
        ),
        (
            [
                {"type": "image_ref", "image_id": "image-0"},
                {"type": "image_ref", "image_id": "image-0"},
            ],
            [{"id": "image-0", "data": "base64-image"}],
            "duplicate",
        ),
    ],
)
def test_protocol_rejects_invalid_interleaved_response_contract(
    content, images, message
) -> None:
    from sglang_omni.serve.protocol import normalize_interleaved_content

    with pytest.raises(ValueError, match=message):
        normalize_interleaved_content(content, images)


def test_ordinary_image_response_does_not_use_interleaved_content_array() -> None:
    from sglang_omni.client.types import CompletionImage
    from sglang_omni.serve.protocol import build_chat_completion_message

    result = SimpleNamespace(
        text="",
        content=[],
        images=[CompletionImage(id="image-0", data="base64-image")],
    )

    message = build_chat_completion_message(result, ["image"], interleaved=False)

    assert not isinstance(message.get("content"), list)
    assert message["images"][0]["id"] == "image-0"


def test_interleaved_example_selects_the_model_pipeline_config() -> None:
    import yaml

    example_path = (
        Path(__file__).parents[3]
        / "examples"
        / "configs"
        / "llada2_uni_interleaved.yaml"
    )
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))

    assert config == {
        "config_cls": "LLaDA2UniInterleavedPipelineConfig",
        "name": "llada2-uni-interleaved",
        "model_path": "inclusionAI/LLaDA2.0-Uni",
    }


def test_interleaved_pipeline_reuses_omni_gpu_memory_budgets() -> None:
    from sglang_omni.config import resolve_stage_factory_args
    from sglang_omni.models.llada2_uni import config

    omni = config.LLaDA2UniOmniPipelineConfig(model_path="checkpoint")
    interleaved = config.LLaDA2UniInterleavedPipelineConfig(model_path="checkpoint")
    omni_stages = {stage.name: stage for stage in omni.stages}
    interleaved_stages = {stage.name: stage for stage in interleaved.stages}
    colocated_names = (config.THINKER_STAGE, config.IMAGE_DECODE_STAGE)

    expected_fractions = {
        name: omni_stages[name].runtime.resources.total_gpu_memory_fraction
        for name in colocated_names
    }
    actual_fractions = {
        name: interleaved_stages[name].runtime.resources.total_gpu_memory_fraction
        for name in colocated_names
    }

    assert (
        actual_fractions
        == expected_fractions
        == {
            config.THINKER_STAGE: 0.7,
            config.IMAGE_DECODE_STAGE: 0.2,
        }
    )
    assert sum(actual_fractions.values()) <= 1.0
    assert (
        interleaved_stages[config.THINKER_STAGE].factory_args["server_args_overrides"]
        == omni_stages[config.THINKER_STAGE].factory_args["server_args_overrides"]
    )
    thinker = interleaved_stages[config.THINKER_STAGE]
    assert thinker.runtime.sglang_server_args.mem_fraction_static == 0.7
    resolved = resolve_stage_factory_args(thinker, interleaved)
    assert resolved["server_args_overrides"]["mem_fraction_static"] == 0.7


def test_collector_isolates_request_ids_under_out_of_order_completion() -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler

    collector = InterleavedCollectorScheduler()
    collector._receive("request-a", _payload(_final_state(frame_count=1), "request-a"))
    collector._receive("request-b", _payload(_final_state(frame_count=1), "request-b"))
    collector._receive("request-b", _frame_payload(1, request_id="request-b"))
    collector._receive("request-a", _frame_payload(1, request_id="request-a"))

    result_b = collector.outbox.get_nowait()
    result_a = collector.outbox.get_nowait()
    assert result_b.request_id == "request-b"
    assert result_a.request_id == "request-a"
    assert result_b.data.data["images"][0]["id"] == "image-request-b-0"
    assert result_a.data.data["images"][0]["id"] == "image-request-a-0"


def test_collector_failure_tombstone_suppresses_all_late_payloads() -> None:
    from sglang_omni.models.llada2_uni.interleaved import InterleavedCollectorScheduler

    collector = InterleavedCollectorScheduler()
    collector._receive("request", _frame_payload(1))
    with pytest.raises(ValueError, match="duplicate"):
        collector._receive("request", _frame_payload(1))

    collector._receive("request", _payload(_final_state(frame_count=1)))
    collector._receive("request", _frame_payload(1))

    with pytest.raises(Empty):
        collector.outbox.get_nowait()
    assert "request" not in collector._frames
    assert "request" not in collector._final_payloads


def test_real_stage_fans_out_isolated_frame_and_thinker_reentry() -> None:
    from sglang_omni.models.llada2_uni.config import IMAGE_DECODE_STAGE, THINKER_STAGE
    from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
    from sglang_omni.models.llada2_uni.routing import (
        project_interleaved_payload,
        thinker_next,
    )
    from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
    from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler
    from tests.unit_test.pipeline.helpers import make_stage

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        thinker_scheduler = FakeScheduler()
        decoder_scheduler = FakeScheduler()
        decoder_scheduler.allow_multiple_inflight_per_request = True
        decoder = make_stage(name=IMAGE_DECODE_STAGE, scheduler=decoder_scheduler)
        thinker = make_stage(
            name=THINKER_STAGE,
            get_next=thinker_next,
            endpoints={
                THINKER_STAGE: "inproc://thinker",
                IMAGE_DECODE_STAGE: "inproc://image-decode",
            },
            scheduler=thinker_scheduler,
            project_payload={
                THINKER_STAGE: project_interleaved_payload,
                IMAGE_DECODE_STAGE: project_interleaved_payload,
            },
            same_process_targets={THINKER_STAGE, IMAGE_DECODE_STAGE},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([thinker, decoder])
        thinker._active_requests.add("request-stage")
        state = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
        advance_interleaved_state(state, _Tokenizer(), completed_phase="text")
        state.thinker_out = {"output_ids": [1000, 1001, 1002, 1003, 11]}
        advance_interleaved_state(state, _Tokenizer(), completed_phase="image")

        await thinker._route_result(
            "request-stage",
            _payload(state, request_id="request-stage"),
        )

        frame_message = decoder_scheduler.inbox.get_nowait()
        reentry_message = thinker_scheduler.inbox.get_nowait()
        assert frame_message.request_id == reentry_message.request_id == "request-stage"
        assert frame_message.data is not reentry_message.data
        assert frame_message.data.data is not reentry_message.data.data
        assert "request-stage" in thinker._active_requests

    asyncio.run(_run())


def test_real_stage_abort_suppresses_reentry_and_collector_late_results() -> None:
    from sglang_omni.models.llada2_uni.config import IMAGE_DECODE_STAGE, THINKER_STAGE
    from sglang_omni.models.llada2_uni.interleaved import (
        InterleavedCollectorScheduler,
        advance_interleaved_state,
    )
    from sglang_omni.models.llada2_uni.routing import (
        project_interleaved_payload,
        thinker_next,
    )
    from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
    from tests.unit_test.fixtures.pipeline_fakes import FakeScheduler
    from tests.unit_test.pipeline.helpers import make_stage

    async def _run() -> None:
        thinker_scheduler = FakeScheduler()

        class _AbortBeforeReentry(LocalStageDispatcher):
            async def send_payload(self, **kwargs) -> None:
                if kwargs["to_stage"] == THINKER_STAGE:
                    thinker._on_abort(kwargs["request_id"])
                await super().send_payload(**kwargs)

        dispatcher = _AbortBeforeReentry()
        decoder_scheduler = FakeScheduler()
        decoder_scheduler.allow_multiple_inflight_per_request = True
        decoder = make_stage(name=IMAGE_DECODE_STAGE, scheduler=decoder_scheduler)
        thinker = make_stage(
            name=THINKER_STAGE,
            get_next=thinker_next,
            endpoints={
                THINKER_STAGE: "inproc://thinker",
                IMAGE_DECODE_STAGE: "inproc://image-decode",
            },
            scheduler=thinker_scheduler,
            project_payload={
                THINKER_STAGE: project_interleaved_payload,
                IMAGE_DECODE_STAGE: project_interleaved_payload,
            },
            same_process_targets={THINKER_STAGE, IMAGE_DECODE_STAGE},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([thinker, decoder])
        thinker._active_requests.add("request-abort")
        state = _interleaved_state(output_ids=[77, 10, 20, 20, 12])
        advance_interleaved_state(state, _Tokenizer(), completed_phase="text")
        state.thinker_out = {"output_ids": [1000, 1001, 1002, 1003, 11]}
        advance_interleaved_state(state, _Tokenizer(), completed_phase="image")

        await thinker._route_result(
            "request-abort",
            _payload(state, request_id="request-abort"),
        )

        assert decoder_scheduler.inbox.get_nowait().request_id == "request-abort"
        assert thinker_scheduler.inbox.empty()
        assert thinker_scheduler.aborted == ["request-abort"]

        collector = InterleavedCollectorScheduler()
        collector.abort("request-abort")
        collector._receive(
            "request-abort",
            _frame_payload(1, request_id="request-abort"),
        )
        collector._receive(
            "request-abort",
            _payload(_final_state(frame_count=1), request_id="request-abort"),
        )
        with pytest.raises(Empty):
            collector.outbox.get_nowait()

    asyncio.run(_run())
