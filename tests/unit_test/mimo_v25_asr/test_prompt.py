# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from sglang_omni.models.mimo_v25_asr.prompt import (
    ASR_EN_TEMPLATE,
    ASR_ZH_TEMPLATE,
    CHINESE_TAG,
    ENGLISH_TAG,
    SPEECH_EMPTY_IDS,
    build_asr_sft_prompt,
    pad_audio_codes,
    resolve_audio_tag,
    select_asr_instruction,
    strip_asr_markers,
)


class _Encoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _CharacterTokenizer:
    special: ClassVar[dict[str, int]] = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|sosp|>": 151665,
        "<|eosp|>": 151666,
        "<|empty|>": 151667,
        "<think>": 151668,
        "</think>": 151669,
    }

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.special.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        ids: list[int] = []
        cursor = 0
        tokens = sorted(self.special, key=len, reverse=True)
        while cursor < len(text):
            special = next(
                (token for token in tokens if text.startswith(token, cursor)), None
            )
            if special is not None:
                ids.append(self.special[special])
                cursor += len(special)
            else:
                ids.append(1000 + ord(text[cursor]))
                cursor += 1
        return _Encoding(ids)


def test_resolve_audio_tag_maps_openai_language() -> None:
    assert resolve_audio_tag(None) == ""
    assert resolve_audio_tag("auto") == ""
    assert resolve_audio_tag("zh") == CHINESE_TAG
    assert resolve_audio_tag("chinese") == CHINESE_TAG
    assert resolve_audio_tag("en") == ENGLISH_TAG
    assert resolve_audio_tag("english") == ENGLISH_TAG


def test_select_asr_instruction_is_deterministic() -> None:
    assert select_asr_instruction(CHINESE_TAG) == ASR_ZH_TEMPLATE
    assert select_asr_instruction(ENGLISH_TAG) == ASR_EN_TEMPLATE
    assert select_asr_instruction("") == ASR_EN_TEMPLATE


def test_asr_sft_prompt_has_official_channel_contract() -> None:
    tokenizer = _CharacterTokenizer()
    codes = torch.arange(5 * 8, dtype=torch.int64).reshape(5, 8)
    prompt = build_asr_sft_prompt(tokenizer, codes, language="zh")

    assert prompt.official_input_ids.shape[0] == 9
    assert prompt.audio_codes.shape == (8, 8)
    assert torch.equal(prompt.audio_codes[5:], prompt.audio_codes[4:5].expand(3, -1))
    assert prompt.audio_end - prompt.audio_start == 2
    assert prompt.input_ids[prompt.audio_start : prompt.audio_end] == [151667, 151667]
    assert prompt.audio_tag == CHINESE_TAG
    assert prompt.instruction == ASR_ZH_TEMPLATE
    assert torch.equal(
        prompt.official_input_ids[1:, :4],
        torch.tensor(SPEECH_EMPTY_IDS)[:, None].expand(-1, 4),
    )


def test_asr_sft_prompt_appends_english_tag() -> None:
    prompt = build_asr_sft_prompt(
        _CharacterTokenizer(),
        torch.zeros((4, 8), dtype=torch.int64),
        language="en",
    )
    assert prompt.audio_tag == ENGLISH_TAG
    assert prompt.instruction == ASR_EN_TEMPLATE


def test_strip_asr_markers_removes_language_tags() -> None:
    assert strip_asr_markers(f"{CHINESE_TAG}北京<|empty|>") == "北京"


def test_pad_audio_codes_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        pad_audio_codes(torch.empty((0, 8), dtype=torch.int64))
