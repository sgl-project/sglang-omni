# SPDX-License-Identifier: Apache-2.0
"""Layout tests for multi-turn history rendering in the Realtime prompt."""

from __future__ import annotations

import re

import numpy as np

from sglang_omni.models.moss_tts_realtime.payload_types import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_PAD_TOKEN,
    N_CODEBOOKS,
    PREFILL_TEXT_TOKENS,
)
from sglang_omni.models.moss_tts_realtime.processor import (
    MossTTSRealtimePromptProcessor,
)

_SPECIAL = re.compile(r"<\|[^|]+\|>")


class _Tokenizer:
    """Char-level stub; any ``<|...|>`` special token maps to one id."""

    audio_pad_id = 151654

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|audio_pad|>"
        return self.audio_pad_id

    def __call__(self, text: str) -> dict[str, list[int]]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            match = _SPECIAL.match(text, index)
            if match is not None:
                token = match.group(0)
                ids.append(self.audio_pad_id if token == "<|audio_pad|>" else 2000 + hash(token) % 1000)
                index = match.end()
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return {"input_ids": ids}


def _processor() -> MossTTSRealtimePromptProcessor:
    return MossTTSRealtimePromptProcessor(_Tokenizer())


def _codes(frames: int, base: int = 0) -> np.ndarray:
    return (np.arange(frames * N_CODEBOOKS).reshape(frames, N_CODEBOOKS) + base) % 1024


def test_history_turn_long_text_layout() -> None:
    processor = _processor()
    text = "x" * (PREFILL_TEXT_TOKENS + 3)
    codes = _codes(10)
    rows = processor.build_history_turn_rows(text, codes, leading_break=False)
    audio_start = PREFILL_TEXT_TOKENS
    assert rows[audio_start - 1, 1] == AUDIO_BOS_TOKEN
    np.testing.assert_array_equal(rows[audio_start : audio_start + 10, 1:], codes)
    assert rows[audio_start + 10, 1] == AUDIO_EOS_TOKEN
    # text channel carries the text ids at the head
    assert rows[0, 0] == 1000 + ord("x")


def test_history_turn_zero_frames_keeps_bos_eos_adjacent() -> None:
    processor = _processor()
    rows = processor.build_history_turn_rows("ab", np.zeros((0, N_CODEBOOKS)), leading_break=False)
    assert rows[-2, 1] == AUDIO_BOS_TOKEN
    assert rows[-1, 1] == AUDIO_EOS_TOKEN


def test_history_turn_single_frame_accepts_flat_and_transposed() -> None:
    processor = _processor()
    flat = np.arange(N_CODEBOOKS)
    column = flat.reshape(N_CODEBOOKS, 1)
    for codes in (flat, column):
        rows = processor.build_history_turn_rows("ab", codes, leading_break=True)
        eos_positions = np.flatnonzero(rows[:, 1] == AUDIO_EOS_TOKEN)
        assert eos_positions.size == 1


def test_generation_prompt_appends_history_between_ensemble_and_text() -> None:
    processor = _processor()
    history = [("hello", _codes(6)), ("world", _codes(4, base=7))]
    rows_with, text_ids, prefill = processor.build_generation_prompt(
        "next turn text", None, history=history
    )
    rows_plain, _, _ = processor.build_generation_prompt("next turn text", None)
    assert rows_with.shape[0] > rows_plain.shape[0]
    assert prefill == min(len(text_ids), PREFILL_TEXT_TOKENS)
    # current-turn prefill still ends with the audio BOS marker
    assert int(rows_with[-1, 1]) == AUDIO_BOS_TOKEN
    # both history turns' codes appear verbatim in the audio channels
    flat = rows_with[:, 1:]
    for _, codes in history:
        first = codes[0]
        matches = np.flatnonzero((flat == first).all(axis=1))
        assert matches.size >= 1


def test_generation_prompt_without_history_unchanged() -> None:
    processor = _processor()
    rows, text_ids, prefill = processor.build_generation_prompt("abc", None)
    assert prefill == 3
    assert int(rows[-1, 1]) == AUDIO_BOS_TOKEN
    assert rows.shape[1] == N_CODEBOOKS + 1
    non_pad = (np.asarray(rows[:, 1:]) != AUDIO_PAD_TOKEN).sum()
    assert int(non_pad) == 1  # only the BOS marker
