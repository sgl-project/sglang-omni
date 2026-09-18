# SPDX-License-Identifier: Apache-2.0

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.llada2_uni.components.preprocessor import (
    DUMMY_IMAGE_TOKEN_ID,
    IMAGE_TOKEN_OFFSET,
)
from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.merge import decode_events, extract_image_vq_tokens
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    _thinking_phase1_to_phase2,
    build_dllm_thinker_request,
    make_dllm_thinker_scheduler_adapters,
    merge_image_tokens_for_thinker,
)
from sglang_omni.models.llada2_uni.routing import thinker_next
from sglang_omni.proto import OmniRequest, StagePayload


def image_state(**kwargs):
    state = LLaDA2UniPipelineState(
        task_kind="t2i",
        prompt={"input_ids": torch.tensor([[11, 12, 13]])},
        stream_state={"image_info": [{"grid_h": 2, "grid_w": 3}]},
    )
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def build_request(state, **kwargs):
    return build_dllm_thinker_request(
        state,
        params=kwargs.get("params", {}),
        tokenizer=kwargs.get("tokenizer", SimpleNamespace(eos_token_id=2)),
        vocab_size=200000,
        dllm_config=SimpleNamespace(block_size=2, mask_id=99),
    ).req


def test_generation_state_roundtrip_and_chat_wire_defaults():
    assert LLaDA2UniPipelineState().to_dict() == {}
    state = image_state(request_metadata={"image_generation": {"cfg_scale": 4}})
    restored = LLaDA2UniPipelineState.from_dict(state.to_dict())
    assert restored.task_kind == "t2i"
    assert restored.stream_state == state.stream_state
    assert restored.request_metadata == state.request_metadata
    malformed = LLaDA2UniPipelineState.from_dict(
        {"stream_state": [], "request_metadata": [], "task_kind": 5}
    )
    assert malformed.to_dict() == {}


def test_request_uses_image_grid_and_upstream_array():
    state = image_state()
    state.stream_state["dllm_steps"] = 9
    req = build_request(state, params={"max_new_tokens": 100})
    assert req.sampling_params.max_new_tokens == 6
    assert req._dllm_steps == 9
    assert req._task_kind == "t2i"
    assert isinstance(req.origin_input_ids, array)
    assert req.origin_input_ids == array("q", [11, 12, 13])


def test_request_attaches_original_three_way_cfg_contract():
    state = image_state(task_kind="edit")
    state.stream_state.update(
        uncond_input_ids=[99, 21, 22],
        uncond_left_pad_len=1,
        uncond_img_input_ids=[99, 99, 23],
        uncond_img_left_pad_len=2,
        cfg_scale=4.0,
        cfg_image_scale=1.5,
        cfg_rescale=0.6,
    )
    req = build_request(state)
    assert req._uncond_input_ids == [99, 21, 22]
    assert req._uncond_left_pad_len == 1
    assert req._uncond_img_input_ids == [99, 99, 23]
    assert req._uncond_img_left_pad_len == 2
    assert req._cfg_scale == 4
    assert req._cfg_image_scale == 1.5
    assert req._cfg_rescale == 0.6
    assert isinstance(req.origin_input_ids, array)


@pytest.mark.parametrize("branch", ["uncond", "uncond_img"])
def test_request_rejects_unaligned_cfg_branches(branch):
    state = image_state()
    state.stream_state["uncond_input_ids"] = [1, 2, 3]
    state.stream_state[f"{branch}_input_ids"] = [1]
    with pytest.raises(ValueError, match="equal physical lengths"):
        build_request(state)


def test_request_rejects_invalid_padding():
    state = image_state()
    state.stream_state.update(uncond_input_ids=[1, 2, 3], uncond_left_pad_len=-1)
    with pytest.raises(ValueError, match="left-pad length"):
        build_request(state)


def test_thinking_request_stops_at_boi_without_cfg():
    state = image_state()
    state.stream_state.update(
        thinking_mode=True, thinking_phase=1, uncond_input_ids=[1]
    )
    req = build_request(
        state,
        tokenizer=SimpleNamespace(
            eos_token_id=2, convert_tokens_to_ids=lambda token: 88
        ),
    )
    assert req._is_thinking_phase1
    assert req.sampling_params.max_new_tokens == 2048
    assert 88 in req.eos_token_ids
    assert not hasattr(req, "_uncond_input_ids")


def test_merge_source_vq_into_both_cfg_branches():
    state = image_state(task_kind="edit")
    state.prompt["input_ids"] = torch.tensor(
        [[7, DUMMY_IMAGE_TOKEN_ID, DUMMY_IMAGE_TOKEN_ID, 8]]
    )
    state.stream_state["uncond_input_ids"] = [
        99,
        DUMMY_IMAGE_TOKEN_ID,
        DUMMY_IMAGE_TOKEN_ID,
        8,
    ]
    state.stream_state["uncond_img_input_ids"] = [99, 99, 99, 8]
    state.encoder_outs[IMAGE_STAGE] = {"image_token_ids": [[1, 2]]}
    merge_image_tokens_for_thinker(state)
    assert state.prompt["input_ids"].tolist() == [
        [7, IMAGE_TOKEN_OFFSET + 1, IMAGE_TOKEN_OFFSET + 2, 8]
    ]
    assert state.stream_state["uncond_input_ids"] == [
        99,
        IMAGE_TOKEN_OFFSET + 1,
        IMAGE_TOKEN_OFFSET + 2,
        8,
    ]
    assert state.stream_state["uncond_img_input_ids"] == [99, 99, 99, 8]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_merge_rejects_unconditional_vq_mismatch_without_partial_prompt_update(count):
    state = image_state(task_kind="edit")
    ids = torch.tensor([[DUMMY_IMAGE_TOKEN_ID, DUMMY_IMAGE_TOKEN_ID]])
    state.prompt["input_ids"] = ids.clone()
    state.stream_state["uncond_input_ids"] = [DUMMY_IMAGE_TOKEN_ID] * count
    state.encoder_outs[IMAGE_STAGE] = {"image_token_ids": [[1, 2]]}
    with pytest.raises(ValueError, match="VQ token count mismatch"):
        merge_image_tokens_for_thinker(state)
    torch.testing.assert_close(state.prompt["input_ids"], ids)


@pytest.mark.parametrize("kind", ["t2i", "edit"])
def test_extract_vq_preserves_rectangular_grid_and_options(kind):
    state = image_state(
        task_kind=kind,
        thinker_out={
            "output_ids": [3] + [IMAGE_TOKEN_OFFSET + i for i in range(6)] + [2]
        },
    )
    state.request_metadata = {"image_generation": {"seed": 42}}
    assert extract_image_vq_tokens(state) == (list(range(6)), 2, 3, {"seed": 42})


def test_extract_rejects_incomplete_known_grid():
    state = image_state(thinker_out={"output_ids": [IMAGE_TOKEN_OFFSET] * 4})
    with pytest.raises(ValueError, match="does not match"):
        extract_image_vq_tokens(state)


def test_extract_skips_chat_but_does_not_truncate_image_output():
    state = image_state(
        task_kind="chat", thinker_out={"output_ids": [IMAGE_TOKEN_OFFSET] * 5}
    )
    assert extract_image_vq_tokens(state) is None
    state.task_kind = "t2i"
    state.stream_state = {}
    with pytest.raises(ValueError, match="Cannot infer an image grid"):
        extract_image_vq_tokens(state)
    state.thinker_out["output_ids"] = [IMAGE_TOKEN_OFFSET] * 4
    assert extract_image_vq_tokens(state) == ([0] * 4, 2, 2, {})


def thinking_tokenizer():
    return SimpleNamespace(
        eos_token_id=2,
        mask_token_id=99,
        convert_tokens_to_ids=lambda token: 88,
        encode=lambda text, **kwargs: [21, 22, 88],
        decode=lambda ids, **kwargs: "reasoning",
    )


def test_thinking_adapters_reenter_once_and_clear_result():
    state = image_state()
    state.stream_state.update(thinking_mode=True, thinking_phase=1, cfg_scale=4)
    payload = StagePayload("thinking", OmniRequest(inputs={}), state.to_dict())
    builder, adapter = make_dllm_thinker_scheduler_adapters(
        tokenizer=thinking_tokenizer(),
        vocab_size=200000,
        dllm_config=SimpleNamespace(block_size=2, mask_id=99),
        stage_name="custom_thinker",
    )
    data = builder(payload)
    data.output_ids = array("q", [44, 88])
    result = adapter(data)
    state2 = LLaDA2UniPipelineState.from_dict(result.data)
    assert state2.thinker_out is None
    assert "custom_thinker" not in state2.engine_outputs
    assert state2.prompt["input_ids"].tolist() == [[11, 12, 13, 44, 88]]
    assert state2.stream_state["thinking_text"] == "reasoning"
    assert thinker_next("thinking", result) == THINKER_STAGE
    data2 = builder(result)
    assert data2.req.sampling_params.max_new_tokens == 6
    assert not data2.stage_payload.data["stream_state"].get("thinking_needs_reentry")
    data2.output_ids = [IMAGE_TOKEN_OFFSET] * 6
    final = adapter(data2)
    assert thinker_next("thinking", final) == [DECODE_STAGE, IMAGE_DECODE_STAGE]
    assert final.data["thinker_out"]["output_ids"] == data2.output_ids


def test_thinking_transition_requires_boi_and_available_context():
    state = image_state(thinker_out={"output_ids": [44]})
    with pytest.raises(RuntimeError, match="did not produce"):
        _thinking_phase1_to_phase2(state, thinking_tokenizer())
    state.thinker_out = {"output_ids": [44, 88]}
    state.stream_state["max_seq_len"] = 10
    with pytest.raises(ValueError, match="maximum context length"):
        _thinking_phase1_to_phase2(state, thinking_tokenizer())


def test_text_event_interface_unchanged():
    assert (
        decode_events(thinker_out={"output_ids": []}, tokenizer=thinking_tokenizer())
        == []
    )
    events = decode_events(
        thinker_out={"output_ids": [44]}, tokenizer=thinking_tokenizer()
    )
    assert len(events) == 1
    assert events[0].payload == {"text": "reasoning"}
    assert events[0].is_final
