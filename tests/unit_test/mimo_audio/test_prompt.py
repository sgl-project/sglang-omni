# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from sglang_omni.models.mimo_audio.prompt import (
    SPEECH_EMPTY_IDS,
    build_audio_understanding_prompt,
    pad_audio_codes,
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


def test_audio_understanding_prompt_has_official_channel_contract() -> None:
    tokenizer = _CharacterTokenizer()
    codes = torch.arange(5 * 8, dtype=torch.int64).reshape(5, 8)
    prompt = build_audio_understanding_prompt(
        tokenizer,
        codes,
        "Summarize the audio.",
    )

    assert prompt.official_input_ids.shape[0] == 9
    assert prompt.audio_codes.shape == (8, 8)
    assert torch.equal(prompt.audio_codes[5:], prompt.audio_codes[4:5].expand(3, -1))
    assert prompt.audio_end - prompt.audio_start == 2
    assert prompt.input_ids[prompt.audio_start : prompt.audio_end] == [151667, 151667]
    assert torch.equal(
        prompt.official_input_ids[1:, :4],
        torch.tensor(SPEECH_EMPTY_IDS)[:, None].expand(-1, 4),
    )


def test_frozen_official_tokenizer_prompt_parity() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer_json = (
        Path(HF_HUB_CACHE)
        / "models--XiaomiMiMo--MiMo-Audio-7B-Instruct"
        / "snapshots"
        / "c359441c22c2a1c74be5f99a91e83392680e9cc8"
        / "tokenizer.json"
    )
    try:
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_json))
    except (OSError, ValueError) as exc:
        pytest.skip(f"fixed MiMo tokenizer metadata is unavailable: {exc}")

    codes = torch.arange(8 * 8, dtype=torch.int64).reshape(8, 8)
    prompt = build_audio_understanding_prompt(
        tokenizer,
        codes,
        "Summarize the audio.",
    )
    digest = hashlib.sha256(
        prompt.official_input_ids.numpy().tobytes(order="C")
    ).hexdigest()

    # Frozen from Xiaomi's InputSegment.to_input_id contract at official
    # MiMo-Audio commit 691ce54144a6844cc641fd96046a6ba20776c8b0.
    assert digest == "6679ff48431a953d3f858bc7916135885cf62f27309af5976cc61dd503ce8a7c"


def test_prompt_rejects_missing_instruction() -> None:
    with pytest.raises(ValueError, match="instruction"):
        build_audio_understanding_prompt(
            _CharacterTokenizer(),
            torch.zeros((4, 8), dtype=torch.int64),
            "",
        )


def test_pad_audio_codes_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        pad_audio_codes(torch.empty((0, 8), dtype=torch.int64))
