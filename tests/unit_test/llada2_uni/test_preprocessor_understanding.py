# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from typing import ClassVar

import torch
from PIL import Image

from sglang_omni.models.llada2_uni.components.preprocessor import (
    BOI_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
    DUMMY_IMAGE_TOKEN_ID,
    EOI_TOKEN,
    SOI_TOKEN,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.config import THINKER_STAGE, LLaDA2UniPipelineConfig
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class _Tokenizer:
    tokens: ClassVar[dict[str, int]] = {
        SOI_TOKEN: 156901,
        EOI_TOKEN: 156902,
        BOI_TOKEN: 156904,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokens[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for part in re.split(r"(<\|reserved_token_\d+\|>|<\|/?image\|>|<boi>)", text):
            if part in self.tokens:
                ids.append(self.tokens[part])
            elif part.startswith("<|reserved_token_"):
                ids.append(10000 + int(re.search(r"\d+", part)[0]))
            else:
                ids.extend(ord(char) + 1000 for char in part)
        return ids

    @staticmethod
    def decode(ids: list[int]) -> str:
        return "".join(
            chr(token_id - 1000) for token_id in ids if 1000 <= token_id < 10000
        )


def _make_preprocessor() -> LLaDA2Preprocessor:
    preprocessor = LLaDA2Preprocessor.__new__(LLaDA2Preprocessor)
    preprocessor._tokenizer = _Tokenizer()
    preprocessor._boi_id = _Tokenizer.tokens[BOI_TOKEN]
    preprocessor._eoi_id = _Tokenizer.tokens[EOI_TOKEN]
    preprocessor._max_seq_len = 8192
    preprocessor._merge_size = 2
    preprocessor._factor = 32

    def process_images(**kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        return {
            "pixel_values": torch.zeros(16, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }

    preprocessor._image_processor = process_images
    return preprocessor


def test_understanding_prompt_and_sigvq_patch_count() -> None:
    preprocessor = _make_preprocessor()
    payload = StagePayload(
        request_id="mmmu",
        request=OmniRequest(
            inputs={
                "messages": [{"role": "user", "content": "What is shown?"}],
                "images": [Image.new("RGB", (64, 64))],
            }
        ),
        data={},
    )

    result = asyncio.run(preprocessor(payload))
    state = LLaDA2UniPipelineState.from_dict(result.data)
    input_ids = state.prompt["input_ids"].flatten().tolist()

    assert DEFAULT_SYSTEM_PROMPT in preprocessor._tokenizer.decode(input_ids)
    assert input_ids.count(DUMMY_IMAGE_TOKEN_ID) == 16


def test_understanding_pipeline_uses_cfg_compatible_algorithm() -> None:
    thinker = next(
        stage
        for stage in LLaDA2UniPipelineConfig(model_path="unused").stages
        if stage.name == THINKER_STAGE
    )

    assert thinker.factory.dllm_algorithm == "LowConfidenceCFG"
