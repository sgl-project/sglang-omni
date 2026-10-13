# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sglang_omni.preprocessing.transcription as transcription
from sglang_omni.models.fun_asr.request_builders import (
    make_fun_asr_scheduler_adapters,
    retained_streaming_prefix,
)
from sglang_omni.models.fun_asr.streaming import FunASRStreamingStrategy
from sglang_omni.proto import OmniRequest, StagePayload

_AUDIO_PAD = "<|object_ref_start|>"
_AUDIO_PAD_ID = 42


class PieceTokenizer:
    """Tokenizes into the longest known piece at each position, else one char."""

    def __init__(self, pieces: tuple[str, ...] = ()) -> None:
        self.pieces = sorted(pieces, key=len, reverse=True)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert not add_special_tokens
        offset_mapping = []
        start = 0
        while start < len(text):
            piece = next(
                (piece for piece in self.pieces if text.startswith(piece, start)),
                text[start],
            )
            offset_mapping.append((start, start + len(piece)))
            start += len(piece)
        return SimpleNamespace(
            input_ids=[ord(char) for char in text], offset_mapping=offset_mapping
        )


@pytest.mark.parametrize(
    ("text", "rollback_chars", "expected_ids", "expected_text"),
    [
        ("", 8, [], ""),
        ("short", 8, [], ""),
        ("abcdef", 0, [97, 98, 99, 100, 101, 102], "abcdef"),
        # No whitespace anywhere before the cut: no safe boundary to land
        # on, so the whole run rolls back rather than keeping a fragment.
        ("abcdef", 2, [], ""),
        # Cut lands mid-word ("in this" with rollback=3 cuts inside "this"):
        # back up to the start of that word instead of keeping "in thi".
        # The boundary space itself is stripped too, so the continuation's
        # own leading space is the only separator.
        ("in this", 3, [ord(c) for c in "in"], "in"),
        # Cut already lands on a word boundary: same result.
        ("in this", 4, [ord(c) for c in "in"], "in"),
        # Mixed script: Latin still never splits a word ("wo|rld").
        ("我说hello world", 3, [ord(c) for c in "我说hello"], "我说hello"),
    ],
)
def test_retained_streaming_prefix_rolls_back_chars(
    text: str,
    rollback_chars: int,
    expected_ids: list[int],
    expected_text: str,
) -> None:
    assert retained_streaming_prefix(PieceTokenizer(), text, rollback_chars) == (
        expected_ids,
        expected_text,
    )


@pytest.mark.parametrize(
    ("text", "rollback_chars", "expected_text"),
    [
        # Pieces below are the real Fun-ASR tokenization. The cut lands inside
        # 转弯, so it backs up to the token start instead of keeping 转.
        ("前方有左急转弯，请减速慢行。", 8, "前方有左急"),
        # A cut inside a leading multi-character token rolls back everything.
        ("不断提升安全业务能力。", 8, ""),
        # A cut already on a token boundary is kept.
        ("前方有左急转弯，请减速慢行。", 9, "前方有左急"),
    ],
)
def test_retained_streaming_prefix_keeps_unspaced_tokens_whole(
    text: str,
    rollback_chars: int,
    expected_text: str,
) -> None:
    tokenizer = PieceTokenizer(
        ("前方", "转弯", "，请", "减速", "不断提升", "安全", "业务", "能力")
    )
    assert retained_streaming_prefix(tokenizer, text, rollback_chars) == (
        [ord(char) for char in expected_text],
        expected_text,
    )


class BuilderTokenizer:
    eos_token_id = 151645
    vocab_size = 151936

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert not add_special_tokens
        if _AUDIO_PAD in text:
            audio_pad_count = text.count(_AUDIO_PAD)
            input_ids = (
                [10, 11, 12, 13, 14]
                + [_AUDIO_PAD_ID] * audio_pad_count
                + [15, 16, 17, 18]
            )
            return SimpleNamespace(input_ids=input_ids)
        return PieceTokenizer()(text, return_offsets_mapping=return_offsets_mapping)

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == _AUDIO_PAD
        return _AUDIO_PAD_ID

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def make_feature_extractor(num_lfr_frames: int):
    def _call(
        audio,
        sampling_rate=None,
        return_tensors=None,
        return_attention_mask=True,
        padding="longest",
    ):
        return {
            "input_features": torch.zeros((1, 560, num_lfr_frames)),
            "attention_mask": torch.ones((1, num_lfr_frames), dtype=torch.long),
        }

    return _call


def test_request_builder_reconstructs_prefix_plus_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600 * 3, dtype=np.float32),
    )
    request_builder, result_adapter = make_fun_asr_scheduler_adapters(
        tokenizer=BuilderTokenizer(),
        max_new_tokens=32,
        feature_extractor=make_feature_extractor(17),
    )
    data = request_builder(
        StagePayload(
            request_id="fun-asr-streaming-refresh",
            request=OmniRequest(
                inputs={"audio_bytes": b"wav"},
                params={
                    "_asr_streaming": True,
                    "_asr_streaming_prefix_text": "abc def",
                    "_asr_streaming_rollback_chars": 3,
                    "repetition_penalty": 1.3,
                },
            ),
            data={},
        )
    )
    data.output_ids = [101, 102]
    result = result_adapter(data)

    assert data.prompt_token_ids[-3:] == [97, 98, 99]
    assert data.streaming_prefix_text == "abc"
    assert data.req.sampling_params.repetition_penalty == 1.3
    assert result.data["text"] == "abcef"


def test_request_builder_defaults_to_no_repetition_penalty_when_not_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transcription,
        "load_audio",
        lambda source, **kwargs: np.zeros(1600 * 3, dtype=np.float32),
    )
    request_builder, _ = make_fun_asr_scheduler_adapters(
        tokenizer=BuilderTokenizer(),
        max_new_tokens=32,
        feature_extractor=make_feature_extractor(17),
    )
    data = request_builder(
        StagePayload(
            request_id="fun-asr-offline",
            request=OmniRequest(inputs={"audio_bytes": b"wav"}, params={}),
            data={},
        )
    )

    assert data.streaming_prefix_text == ""
    assert data.req.sampling_params.repetition_penalty == 1.0


def test_fun_asr_strategy_waits_for_unfixed_chunks_before_using_prefix() -> None:
    strategy = FunASRStreamingStrategy()
    state = strategy.create_state(model_name="fun-asr-nano", language="English")

    first = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r0"
    )
    strategy.update_hypothesis(
        generated_text="hello wor", language="English", state=state
    )
    second = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r1"
    )
    strategy.update_hypothesis(
        generated_text="hello world", language="English", state=state
    )
    third = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r2"
    )

    assert first.extra_params["_asr_streaming_prefix_text"] is None
    assert first.sampling.repetition_penalty == 1.0
    assert second.extra_params["_asr_streaming_prefix_text"] is None
    assert third.extra_params["_asr_streaming_prefix_text"] == "hello world"
    assert third.extra_params["_asr_streaming_rollback_chars"] == 8
    assert third.sampling.repetition_penalty == 1.3


def test_fun_asr_strategy_final_decode_rolls_back_like_a_partial() -> None:
    # The previous hypothesis's tail was decoded from truncated audio; the
    # final decode has the full segment, so it must re-decode that tail too.
    strategy = FunASRStreamingStrategy()
    state = strategy.create_state(model_name="fun-asr-nano", language="English")
    for _ in range(2):
        strategy.build_decode_request(
            audio=b"wav", state=state, is_final=False, request_id="r"
        )
        strategy.update_hypothesis(
            generated_text="hello world", language="English", state=state
        )

    partial_request = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=False, request_id="r-partial"
    )
    final_request = strategy.build_decode_request(
        audio=b"wav", state=state, is_final=True, request_id="r-final"
    )

    assert final_request.extra_params == partial_request.extra_params
    assert final_request.extra_params["_asr_streaming_rollback_chars"] == 8


def test_fun_asr_strategy_updates_transcript_and_language() -> None:
    strategy = FunASRStreamingStrategy()
    state = strategy.create_state(model_name="fun-asr-nano", language=None)

    transcript = strategy.update_hypothesis(
        generated_text="hello world", language="English", state=state
    )

    assert transcript == "hello world"
    assert state.transcript == "hello world"
    assert state.language == "English"
    assert state.chunk_id == 1
