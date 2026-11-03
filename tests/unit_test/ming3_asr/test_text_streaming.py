# SPDX-License-Identifier: Apache-2.0
"""Ming transcript deltas preserve final text across token and request boundaries."""

from array import array

import pytest
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.ming3_asr.preprocessing import extract_transcript
from sglang_omni.models.ming3_asr.text_streaming import (
    Ming3ASRRequestData,
    Ming3ASRStreamOutputBuilder,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))
    }
    vocabulary["<eos>"] = len(vocabulary)
    backend = Tokenizer(BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        clean_up_tokenization_spaces=False,
    )
    tokenizer.add_tokens(["English\thello"])
    return tokenizer


def request_data(
    *, language: str | None = None, request_id: str = "request", stream: bool = True
) -> Ming3ASRRequestData:
    request = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=array("i", [1, 2, 3]),
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=256),
    )
    return Ming3ASRRequestData(
        req=request,
        language=language,
        stage_payload=StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs={"audio_bytes": b"audio"},
                params={"stream": stream, "language": language},
            ),
            data={},
        ),
    )


def message_text(messages: list[OutgoingMessage]) -> str:
    assert all(
        message.type == "stream" and message.data["text"] for message in messages
    )
    return "".join(message.data["text"] for message in messages)


@pytest.mark.parametrize(
    ("language", "generated_text", "expected", "has_incremental_text"),
    [
        (None, "Chinese\t 你好 a\tb  🧑‍🔬 \n", "你好 a\tb  🧑‍🔬", True),
        ("en", "English is a language.", "English is a language.", True),
        ("zh", "  Chinese food a\tb  中文 \n", "Chinese food a\tb  中文", True),
        (None, "English\thello", "hello", True),
        (None, "\tEnglish is a language.", "is a language.", False),
        (None, " \tEnglish\thello \n", "hello", True),
        (None, "English hello", "hello", False),
        (None, "hello", "hello", False),
        (None, "English\t", "", False),
        (None, "", "", False),
    ],
)
def test_deltas_are_stable_prefixes_of_the_final_transcript(
    tokenizer: PreTrainedTokenizerFast,
    language: str | None,
    generated_text: str,
    expected: str,
    has_incremental_text: bool,
) -> None:
    """Leading separators used to make transcript deltas disagree with completion."""
    builder = Ming3ASRStreamOutputBuilder(tokenizer, interval_seconds=0)
    current = request_data(language=language)
    token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    emitted_text = ""
    for token_id in token_ids:
        emitted_text += message_text(
            builder("request", current, RequestOutput("request", token_id))
        )
        assert expected.startswith(emitted_text)
    assert bool(emitted_text) == has_incremental_text
    current.output_ids = token_ids + [tokenizer.eos_token_id]
    emitted_text += message_text(builder.flush("request", current))
    assert (
        emitted_text
        == expected
        == extract_transcript(generated_text, language=language)
    )
    assert builder.flush("request", current) == []
    token_id = token_ids[0] if token_ids else tokenizer.eos_token_id
    assert builder("request", current, RequestOutput("request", token_id)) == []


@pytest.mark.parametrize(
    ("final_text", "token_limit", "expected_tail"),
    [("abcd", None, "bcd"), ("ab你", 3, "b\ufffd"), ("zbcd", None, None)],
)
def test_flush_reconciles_final_tokens_without_rewriting_emitted_text(
    tokenizer: PreTrainedTokenizerFast,
    final_text: str,
    token_limit: int | None,
    expected_tail: str | None,
) -> None:
    builder = Ming3ASRStreamOutputBuilder(tokenizer, interval_seconds=0.05)
    current = request_data(language="en")
    token_ids = tokenizer.encode("ab", add_special_tokens=False)
    assert (
        message_text(
            builder("request", current, RequestOutput("request", token_ids[0]))
        )
        == "a"
    )
    state = current.text_stream_state
    assert state is not None
    assert (
        state.step(
            token_ids[1], terminal=False, now_seconds=state.last_emit_seconds + 0.01
        )
        == ""
    )
    current.output_ids = tokenizer.encode(final_text, add_special_tokens=False)[
        :token_limit
    ]
    if expected_tail is None:
        with pytest.raises(RuntimeError, match="changed an emitted prefix"):
            builder.flush("request", current)
    else:
        assert message_text(builder.flush("request", current)) == expected_tail
        assert builder.flush("request", current) == []


def test_interleaved_requests_emit_only_their_own_decode_tokens(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    builder = Ming3ASRStreamOutputBuilder(tokenizer, interval_seconds=0)
    requests = [request_data(language="en", request_id=str(i)) for i in range(2)]
    transcripts = ["one", "two"]
    token_sequences = [
        tokenizer.encode(text, add_special_tokens=False) for text in transcripts
    ]
    nonstream = request_data(language="en", stream=False)
    nonstream.output_ids = token_sequences[0]
    assert (
        builder("request", nonstream, RequestOutput("request", token_sequences[0][0]))
        == []
    )
    assert builder.flush("request", nonstream) == []
    for i, current in enumerate(requests):
        assert current.req is not None
        current.req.inflight_middle_chunks = 1
        assert (
            builder(str(i), current, RequestOutput(str(i), token_sequences[i][0])) == []
        )
        current.req.inflight_middle_chunks = 0
    emitted = ["", ""]
    for token_pair in zip(*token_sequences, strict=True):
        for i, token_id in enumerate(token_pair):
            messages = builder(str(i), requests[i], RequestOutput(str(i), token_id))
            assert all(message.request_id == str(i) for message in messages)
            emitted[i] += message_text(messages)
    assert emitted == transcripts
    for i, current in enumerate(requests):
        current.output_ids = token_sequences[i]
        assert builder.flush(str(i), current) == []


@pytest.mark.parametrize("interval_seconds", [-0.01, float("nan"), float("inf")])
def test_invalid_stream_intervals_are_rejected(
    tokenizer: PreTrainedTokenizerFast, interval_seconds: float
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        Ming3ASRStreamOutputBuilder(tokenizer, interval_seconds=interval_seconds)
