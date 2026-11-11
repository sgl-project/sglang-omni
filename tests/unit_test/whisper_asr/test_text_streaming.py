# SPDX-License-Identifier: Apache-2.0
"""Whisper byte-token streaming preserves native final text and request ownership."""

from array import array
from typing import Literal

import pytest
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from tokenizers.pre_tokenizers import ByteLevel
from transformers import WhisperTokenizer

from sglang_omni.models.whisper_asr.request_builders import WhisperASRRequestData
from sglang_omni.models.whisper_asr.text_streaming import (
    ExactWhisperStreamOutputBuilder,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.exact_text_streaming import ExactTextStreamState
from sglang_omni.scheduling.types import RequestOutput


@pytest.fixture
def tokenizer() -> WhisperTokenizer:
    vocabulary = {
        character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))
    }
    for token in ["<|endoftext|>", "<|startoftranscript|>", "<|notimestamps|>"]:
        vocabulary[token] = len(vocabulary)
    return WhisperTokenizer(vocab=vocabulary, merges=[])


def request_data(
    request_id: str = "request", stream: bool = True
) -> WhisperASRRequestData:
    request = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=array("i", [1, 2, 3, 4]),
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=256),
    )
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"audio_bytes": b"audio"}, params={"stream": stream}
        ),
        data={},
    )
    return WhisperASRRequestData(req=request, stage_payload=payload)


def test_incomplete_utf8_is_held_until_the_character_is_complete(
    tokenizer: WhisperTokenizer,
) -> None:
    builder = ExactWhisperStreamOutputBuilder(tokenizer, interval_seconds=0)
    state = ExactTextStreamState(decoder=builder.decode, interval_seconds=0)
    token_ids = tokenizer.encode("你好", add_special_tokens=False)
    deltas = [
        state.step(token_id, terminal=False, now_seconds=float(index))
        for index, token_id in enumerate(token_ids)
    ]
    assert deltas[:3] == ["", "", "你"]
    assert deltas[3:] == ["", "", "好"]
    assert "".join(deltas) == tokenizer.decode(token_ids, skip_special_tokens=True)


def test_interval_throttles_updates_but_terminal_flush_preserves_every_token(
    tokenizer: WhisperTokenizer,
) -> None:
    builder = ExactWhisperStreamOutputBuilder(tokenizer)
    state = ExactTextStreamState(decoder=builder.decode, interval_seconds=0.05)
    token_ids = tokenizer.encode("abc", add_special_tokens=False)
    assert state.step(token_ids[0], terminal=False, now_seconds=1) == "a"
    assert state.step(token_ids[1], terminal=False, now_seconds=1.01) == ""
    assert state.step(token_ids[2], terminal=False, now_seconds=1.02) == ""
    assert state.step(None, terminal=True, now_seconds=1.03) == "bc"
    assert state.step(None, terminal=True, now_seconds=1.04) == ""


def test_deltas_preserve_internal_spaces_and_trim_only_final_outer_whitespace(
    tokenizer: WhisperTokenizer,
) -> None:
    builder = ExactWhisperStreamOutputBuilder(tokenizer, interval_seconds=0)
    current = request_data()
    token_ids = tokenizer.encode(" hello world ", add_special_tokens=False)
    messages = []
    for token_id in token_ids:
        messages.extend(builder("request", current, RequestOutput("request", token_id)))
    messages.extend(builder.flush("request", current))
    assert messages and all(message.data["text"] for message in messages)
    assert "".join(message.data["text"] for message in messages) == "hello world"
    assert all(message.type == "stream" for message in messages)
    assert builder.flush("request", current) == []


def test_interleaved_requests_keep_independent_stream_state(
    tokenizer: WhisperTokenizer,
) -> None:
    builder = ExactWhisperStreamOutputBuilder(tokenizer, interval_seconds=0)
    first, second = request_data("first"), request_data("second")
    first_token_ids = tokenizer.encode("one", add_special_tokens=False)
    second_token_ids = tokenizer.encode("two", add_special_tokens=False)
    first_messages, second_messages = [], []
    for first_token_id, second_token_id in zip(
        first_token_ids, second_token_ids, strict=True
    ):
        first_messages.extend(
            builder("first", first, RequestOutput("first", first_token_id))
        )
        second_messages.extend(
            builder("second", second, RequestOutput("second", second_token_id))
        )
    assert "".join(message.data["text"] for message in first_messages) == "one"
    assert "".join(message.data["text"] for message in second_messages) == "two"
    assert first.exact_text_stream_state is not second.exact_text_stream_state


@pytest.mark.parametrize("mode", ["nonstream", "language", "timestamps", "prefill"])
def test_unsupported_modes_retain_the_final_only_path(
    tokenizer: WhisperTokenizer,
    mode: Literal["nonstream", "language", "timestamps", "prefill"],
) -> None:
    builder = ExactWhisperStreamOutputBuilder(tokenizer)
    current = request_data(stream=mode != "nonstream")
    current.detect_language = mode == "language"
    current.segment_timestamps = mode == "timestamps"
    if mode == "prefill":
        assert current.req is not None
        current.req.inflight_middle_chunks = 1
    else:
        pass
    token_id = tokenizer.encode("a", add_special_tokens=False)[0]
    assert builder("request", current, RequestOutput("request", token_id)) == []
    assert current.exact_text_stream_state is None


def test_terminal_flush_keeps_native_replacement_text(
    tokenizer: WhisperTokenizer,
) -> None:
    builder = ExactWhisperStreamOutputBuilder(tokenizer)
    state = ExactTextStreamState(decoder=builder.decode, interval_seconds=0.05)
    incomplete = tokenizer.encode("你", add_special_tokens=False)[:1]
    assert state.step(incomplete[0], terminal=False, now_seconds=1) == ""
    assert state.step(None, terminal=True, now_seconds=2) == tokenizer.decode(
        incomplete
    )


def test_negative_stream_interval_is_rejected(tokenizer: WhisperTokenizer) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ExactWhisperStreamOutputBuilder(tokenizer, interval_seconds=-1)
