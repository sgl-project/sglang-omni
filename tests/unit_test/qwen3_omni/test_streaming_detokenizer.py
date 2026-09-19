# SPDX-License-Identifier: Apache-2.0
"""UTF-8 boundaries in Qwen3-Omni streamed text."""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    StreamingDetokenizeScheduler,
)
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.fixture
def scheduler():
    # Byte-level spellings of "�", " next", " token", and the two parts of "你".
    tokenizer = Tokenizer(
        models.BPE({"ï¿½": 0, "Ġnext": 1, "Ġtoken": 2, "ä½": 3, "ł": 4}, [])
    )
    tokenizer.decoder = decoders.ByteLevel()
    return StreamingDetokenizeScheduler(
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer),
        eos_token_id=None,
    )


def test_replacement_character_does_not_hold_following_text(scheduler):
    scheduler._on_stream_chunk("req", SimpleNamespace(data=0))
    scheduler._on_stream_chunk("req", SimpleNamespace(data=1))
    assert scheduler.outbox.get_nowait().data["text"] == "� next"

    scheduler._on_stream_chunk("req", SimpleNamespace(data=2))
    assert scheduler.outbox.get_nowait().data["text"] == " token"
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("prefix", [[], [0]])
def test_incomplete_utf8_suffix_is_held_until_complete(scheduler, prefix):
    for token in prefix + [3]:
        scheduler._on_stream_chunk("req", SimpleNamespace(data=token))
    assert scheduler.outbox.empty()

    scheduler._on_stream_chunk("req", SimpleNamespace(data=4))
    expected = "�你" if prefix else "你"
    assert scheduler.outbox.get_nowait().data["text"] == expected
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("token", [0, 3])
def test_terminal_flush_preserves_trailing_replacement_character(scheduler, token):
    scheduler._on_stream_chunk("req", SimpleNamespace(data=token))
    assert scheduler.outbox.empty()

    scheduler._on_stream_done("req")
    scheduler._on_new_request(
        "req",
        StagePayload(
            request_id="req",
            request=OmniRequest(inputs=[], params={"stream": True}),
            data={},
        ),
    )
    assert scheduler.outbox.get_nowait().data["text"] == "�"
    assert scheduler.outbox.get_nowait().type == "result"
    assert scheduler.outbox.empty()
    assert "req" not in scheduler._state
