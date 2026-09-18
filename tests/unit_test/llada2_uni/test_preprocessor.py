# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    DUMMY_IMAGE_TOKEN_ID,
    EDIT_SYSTEM_PROMPT,
    EOI_TOKEN,
    IMAGE_TOKEN_OFFSET,
    SOI_TOKEN,
    SYSTEM_PROMPT_T2I,
    SYSTEM_PROMPT_T2I_THINKING,
    LLaDA2Preprocessor,
    _resolve_edit_cfg_scales,
    align_cfg_unconditional_input_ids,
    edit_image_pixel_values,
    preprocess_image_edit,
)
from sglang_omni.models.llada2_uni.config import IMAGE_STAGE
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    merge_image_tokens_for_thinker,
)
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    mask_token_id = 156895
    eos_token_id = 2
    tokens: ClassVar[dict[str, int]] = {
        SOI_TOKEN: 156901,
        EOI_TOKEN: 156902,
        BOI_TOKEN: 156904,
        "<uncondition>": 90,
    }

    def __len__(self):
        return 157192

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in re.split(
            r"(<\|reserved_token_\d+\|>|<\|/?image\|>|<boi>|<uncondition>)", text
        ):
            if part in self.tokens:
                ids.append(self.tokens[part])
            elif part.startswith("<|reserved_token_"):
                ids.append(10000 + int(re.search(r"\d+", part)[0]))
            else:
                ids.extend(ord(char) + 1000 for char in part)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(tid - 1000) for tid in ids if 1000 <= tid < 10000)


@pytest.fixture
def preprocessor():
    processor = LLaDA2Preprocessor.__new__(LLaDA2Preprocessor)
    processor._tokenizer = Tokenizer()
    processor._soi_id = Tokenizer.tokens[SOI_TOKEN]
    processor._boi_id = Tokenizer.tokens[BOI_TOKEN]
    processor._eoi_id = Tokenizer.tokens[EOI_TOKEN]
    processor._max_seq_len = 8192
    processor._merge_size = 1
    processor._factor = 16
    processor._image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=1,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        rescale_factor=1 / 255,
    )
    return processor


def run_preprocessor(
    processor, *, generation=None, images=None, prompt="make it green", params=None
):
    payload = StagePayload(
        request_id="test-input",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": prompt}],
                "images": images,
            },
            metadata={} if generation is None else {"image_generation": generation},
            params=params or {},
        ),
        data={},
    )
    result = asyncio.run(processor(payload))
    assert result.request is payload.request
    return LLaDA2UniPipelineState.from_dict(result.data)


def test_chat_uses_pr3_prerequisite_system_prompt(preprocessor):
    assert DEFAULT_SYSTEM_PROMPT == "You are a multimodal understanding assistant."
    state = run_preprocessor(preprocessor)
    assert state.task_kind == "chat"
    assert not state.stream_state
    assert DEFAULT_SYSTEM_PROMPT in preprocessor._tokenizer.decode(
        state.prompt["input_ids"].flatten().tolist()
    )
    assert state.encoder_inputs[IMAGE_STAGE]["_skip"]


def test_t2i_header_cfg_and_actual_context_budget(preprocessor):
    preprocessor._max_seq_len = 512
    state = run_preprocessor(
        preprocessor,
        generation={"image_h": 64, "image_w": 96, "cfg_scale": 4.0, "dllm_steps": 7},
        params={"max_new_tokens": 4096},
    )
    ids = state.prompt["input_ids"].flatten().tolist()
    ss = state.stream_state
    assert state.task_kind == "t2i"
    assert SYSTEM_PROMPT_T2I in preprocessor._tokenizer.decode(ids)
    assert ids[-4:] == [156901, 10002, 10003, 156904]
    assert ss["image_info"] == [{"grid_h": 2, "grid_w": 3}]
    assert ss["dllm_steps"] == 7
    assert len(ss["uncond_input_ids"]) == len(ids)
    assert ss["uncond_input_ids"][-4:] == ids[-4:]
    assert (
        ss["uncond_input_ids"][: ss["uncond_left_pad_len"]]
        == [156895] * ss["uncond_left_pad_len"]
    )


def test_t2i_checks_image_budget_not_requested_text_budget(preprocessor):
    preprocessor._max_seq_len = 512
    with pytest.raises(ValueError, match="maximum context length"):
        run_preprocessor(preprocessor, generation={}, params={"max_new_tokens": 1})


@pytest.mark.parametrize("dimension", [0, -32, 33])
def test_invalid_t2i_dimensions(preprocessor, dimension):
    with pytest.raises(ValueError, match="positive multiples of 32"):
        run_preprocessor(preprocessor, generation={"image_h": dimension})


def test_thinking_defers_header_and_cfg(preprocessor):
    state = run_preprocessor(
        preprocessor, generation={"mode": "thinking", "cfg_scale": 4}
    )
    ids = state.prompt["input_ids"].flatten().tolist()
    assert SYSTEM_PROMPT_T2I_THINKING in preprocessor._tokenizer.decode(ids)
    assert 156904 not in ids
    assert state.stream_state["thinking_phase"] == 1
    assert "uncond_input_ids" not in state.stream_state


def test_edit_builds_source_and_three_way_cfg(preprocessor):
    state = run_preprocessor(
        preprocessor,
        generation={"cfg_text_scale": 3, "cfg_image_scale": 2},
        images=[Image.new("RGB", (512, 512), (10, 20, 30))],
    )
    ids = state.prompt["input_ids"].flatten().tolist()
    ss = state.stream_state
    assert state.task_kind == "edit"
    assert EDIT_SYSTEM_PROMPT in preprocessor._tokenizer.decode(ids)
    assert ids.count(DUMMY_IMAGE_TOKEN_ID) == 1024
    assert ss["uncond_input_ids"].count(DUMMY_IMAGE_TOKEN_ID) == 1024
    assert DUMMY_IMAGE_TOKEN_ID not in ss["uncond_img_input_ids"]
    assert len(ids) == len(ss["uncond_input_ids"]) == len(ss["uncond_img_input_ids"])
    assert ss["cfg_scale"] == 3
    assert ss["cfg_image_scale"] == 2
    assert state.encoder_inputs[IMAGE_STAGE]["image_grid_thw"].tolist() == [[1, 32, 32]]


def test_edit_accepts_precomputed_source_tokens(preprocessor):
    source_tokens = [0, 1, 2, 3, 4, 5]
    state = run_preprocessor(
        preprocessor,
        generation={
            "source_image_tokens": {
                "token_ids": source_tokens,
                "grid_thw": [1, 2, 3],
            },
            "cfg_text_scale": 4,
        },
    )

    assert state.task_kind == "edit"
    assert state.stream_state["image_info"] == [{"grid_h": 2, "grid_w": 3}]
    assert state.encoder_inputs[IMAGE_STAGE] == {
        "_skip": True,
        "_result": {"image_token_ids": [source_tokens]},
    }
    assert state.prompt["input_ids"].flatten().tolist().count(
        DUMMY_IMAGE_TOKEN_ID
    ) == len(source_tokens)
    input_ids = state.prompt["input_ids"].flatten().tolist()
    uncond_ids = state.stream_state["uncond_input_ids"]
    input_positions = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == DUMMY_IMAGE_TOKEN_ID
    ]
    uncond_positions = [
        index
        for index, token_id in enumerate(uncond_ids)
        if token_id == DUMMY_IMAGE_TOKEN_ID
    ]
    state.encoder_outs[IMAGE_STAGE] = state.encoder_inputs[IMAGE_STAGE]["_result"]

    merge_image_tokens_for_thinker(state)

    expected_tokens = [IMAGE_TOKEN_OFFSET + token_id for token_id in source_tokens]
    input_ids = state.prompt["input_ids"].flatten().tolist()
    uncond_ids = state.stream_state["uncond_input_ids"]
    assert [input_ids[index] for index in input_positions] == expected_tokens
    assert [uncond_ids[index] for index in uncond_positions] == expected_tokens


def test_edit_rejects_raw_image_with_precomputed_tokens(preprocessor):
    with pytest.raises(ValueError, match="either images or source_image_tokens"):
        run_preprocessor(
            preprocessor,
            generation={
                "source_image_tokens": {
                    "token_ids": [0],
                    "grid_thw": [1, 1, 1],
                }
            },
            images=[Image.new("RGB", (32, 32))],
        )


def test_edit_instruction_and_single_source_required(preprocessor):
    with pytest.raises(ValueError, match="non-empty instruction"):
        run_preprocessor(
            preprocessor,
            generation={},
            images=[Image.new("RGB", (32, 32))],
            prompt="  ",
        )
    with pytest.raises(ValueError, match="exactly one"):
        run_preprocessor(
            preprocessor, generation={}, images=[Image.new("RGB", (32, 32))] * 2
        )


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, (4, 0)),
        ({"cfg_scale": 1}, (0, 0)),
        ({"cfg_scale": 3}, (3, 0)),
        ({"cfg_scale": 1, "cfg_text_scale": 2, "cfg_image_scale": 1}, (2, 1)),
    ],
)
def test_edit_cfg_legacy_scale(options, expected):
    assert _resolve_edit_cfg_scales(options) == expected


def test_cfg_alignment_contract():
    assert align_cfg_unconditional_input_ids(Tokenizer(), [1, 2, 3], [4]) == (
        [156895, 156895, 4],
        2,
    )
    with pytest.raises(ValueError, match="cannot be longer"):
        align_cfg_unconditional_input_ids(Tokenizer(), [1], [2, 3])
    with pytest.raises(ValueError, match="mask_token_id"):
        align_cfg_unconditional_input_ids(SimpleNamespace(), [1, 2], [3])


def test_edit_patch_order_and_float32_normalization():
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    output = edit_image_pixel_values(
        [Image.fromarray(pixels)],
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.1, 0.2, 0.3],
        image_std=[0.5, 0.6, 0.7],
        rescale_factor=1 / 255,
    )
    expected = []
    for block_h in range(2):
        for block_w in range(2):
            for merge_h in range(2):
                for merge_w in range(2):
                    row = (block_h * 2 + merge_h) * 2
                    col = (block_w * 2 + merge_w) * 2
                    patch = (
                        torch.tensor(pixels[row : row + 2, col : col + 2])
                        .permute(2, 0, 1)
                        .float()
                    )
                    patch = (
                        patch * (1 / 255) - torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1)
                    ) / torch.tensor([0.5, 0.6, 0.7]).view(3, 1, 1)
                    expected.append(patch[:, None].repeat(1, 2, 1, 1).flatten())
    torch.testing.assert_close(
        output["pixel_values"], torch.stack(expected), rtol=0, atol=0
    )
    assert output["image_grid_thw"].tolist() == [[1, 4, 4]]


def test_edit_crop_is_deterministic_and_within_budget():
    img = Image.new("RGB", (768, 512))
    first = preprocess_image_edit([img], 16)[0]
    second = preprocess_image_edit([img], 16)[0]
    assert first.size == second.size
    assert first.size[0] * first.size[1] <= 512**2
    assert all(size % 16 == 0 for size in first.size)


def test_mmu_placeholder_count_is_raw_patch_count(preprocessor):
    preprocessor._merge_size = 2
    preprocessor._image_processor = lambda **kwargs: {
        "pixel_values": torch.zeros(16, 8),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    state = run_preprocessor(preprocessor, images=[Image.new("RGB", (64, 64))])
    assert (
        state.prompt["input_ids"].flatten().tolist().count(DUMMY_IMAGE_TOKEN_ID) == 16
    )


def test_message_validation_precedes_image_extraction(preprocessor):
    payload = StagePayload("bad", OmniRequest(inputs=[None]), {})
    with pytest.raises(ValueError, match="Each message must be a dict"):
        asyncio.run(preprocessor(payload))
