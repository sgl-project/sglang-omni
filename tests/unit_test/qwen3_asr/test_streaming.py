# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.qwen3_asr.request_builders import (
    _retained_streaming_prefix,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.models.qwen3_asr.streaming import Qwen3ASRStreamingStrategy
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(range(1, len(text) + 1))

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "abcdef"[: len(token_ids)]


@pytest.mark.parametrize(
    ("text", "rollback", "expected_ids", "expected_text"),
    [
        ("", 5, [], ""),
        ("abc", 5, [], ""),
        ("abcdef", 0, [1, 2, 3, 4, 5, 6], "abcdef"),
        ("abcdef", 2, [1, 2, 3, 4], "abcd"),
    ],
)
def test_retained_streaming_prefix_rolls_back_tokens(
    text: str,
    rollback: int,
    expected_ids: list[int],
    expected_text: str,
) -> None:
    assert _retained_streaming_prefix(Tokenizer(), text, rollback) == (
        expected_ids,
        expected_text,
    )


def test_retained_streaming_prefix_drops_incomplete_utf8_suffix() -> None:
    class Utf8Tokenizer(Tokenizer):
        def decode(self, token_ids: list[int], **_: object) -> str:
            return {3: "hello\ufffd", 2: "hello"}.get(len(token_ids), "")

    assert _retained_streaming_prefix(Utf8Tokenizer(), "four", 1) == (
        [1, 2],
        "hello",
    )


def test_request_builder_reconstructs_prefix_plus_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BuilderTokenizer(Tokenizer):
        eos_token_id = 2

        def __len__(self) -> int:
            return 256

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|audio_pad|>"
            return 42

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            if text == "<asr_text>":
                return [100, 101]
            return [30 + index for index, _char in enumerate(text)]

        def __call__(self, _text: str, *, add_special_tokens: bool = False):
            assert not add_special_tokens
            return SimpleNamespace(input_ids=[11, 42, 12, 13])

        def decode(self, token_ids: list[int], **_: object) -> str:
            pieces = {
                30: "a",
                31: "b",
                32: "c",
                33: "d",
                34: "e",
                35: "f",
            }
            return "".join(pieces.get(token_id, "") for token_id in token_ids)

    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600, dtype=np.float32),
    )
    request_builder, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=BuilderTokenizer(),
        max_new_tokens=32,
        feature_extractor=lambda *args, **kwargs: SimpleNamespace(
            input_features=torch.zeros((1, 128, 100)),
            attention_mask=torch.ones((1, 100), dtype=torch.long),
        ),
    )
    data = request_builder(
        StagePayload(
            request_id="streaming-refresh",
            request=OmniRequest(
                inputs={"audio_bytes": b"wav"},
                params={
                    "language": "English",
                    "_asr_streaming": True,
                    "_asr_streaming_prefix_text": "abcdef",
                    "_asr_streaming_rollback_tokens": 2,
                },
            ),
            data={},
        )
    )
    data.output_ids = [34, 35]
    result = result_adapter(data)

    assert data.prompt_token_ids[-4:] == [30, 31, 32, 33]
    assert result.data["text"] == "abcdef"
    assert result.data["language"] == "English"


def test_qwen_strategy_waits_for_unfixed_chunks_before_using_prefix() -> None:
    strategy = Qwen3ASRStreamingStrategy()
    state = strategy.create_state(
        model_name="qwen3-asr",
        language="English",
    )

    first = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r0"
    )
    strategy.update_hypothesis(
        generated_text="hello wor",
        language="English",
        state=state,
    )
    second = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r1"
    )
    strategy.update_hypothesis(
        generated_text="hello world",
        language="English",
        state=state,
    )
    third = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r2"
    )

    assert first.extra_params["_asr_streaming_prefix_text"] is None
    assert second.extra_params["_asr_streaming_prefix_text"] is None
    assert third.extra_params["_asr_streaming_prefix_text"] == "hello world"


def test_qwen_strategy_extracts_language_and_visible_transcript() -> None:
    strategy = Qwen3ASRStreamingStrategy()
    state = strategy.create_state(
        model_name="qwen3-asr",
        language=None,
    )

    transcript = strategy.update_hypothesis(
        generated_text="hello world",
        language="English",
        state=state,
    )

    assert transcript == "hello world"
    assert state.transcript == "hello world"
    assert state.language == "English"
