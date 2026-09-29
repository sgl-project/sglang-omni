# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.fun_asr.request_builders import (
    make_fun_asr_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload

EOS = 999


class ByteTokenizer:
    eos_token_id = EOS

    def __init__(self, vocab: dict[int, bytes]) -> None:
        self.vocab = vocab

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return b"".join(self.vocab[tid] for tid in ids).decode(
            "utf-8", errors="replace"
        )


def make_req_data(
    *, stream: bool = True, inflight_middle_chunks: int = 0, finished: bool = False
) -> Any:
    stage_payload = StagePayload(
        request_id="r",
        request=OmniRequest(
            inputs={"audio_bytes": b""},
            params={"stream": stream},
            metadata={},
        ),
        data={},
    )
    req = SimpleNamespace(
        inflight_middle_chunks=inflight_middle_chunks, finished=lambda: finished
    )
    return SimpleNamespace(req=req, stage_payload=stage_payload)


def make_req_output(token_id: int | None) -> Any:
    return SimpleNamespace(data=token_id)


def make_builder(vocab: dict[int, bytes], *, interval_s: float = 0.0):
    return make_fun_asr_stream_output_builder(
        tokenizer=ByteTokenizer(vocab),
        min_emit_interval_s=interval_s,
    )


def test_emits_text_delta_when_streaming() -> None:
    builder = make_builder({1: b"hello"})
    rd = make_req_data(stream=True)

    msgs = builder("req-1", rd, make_req_output(1))

    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.type == "stream"
    assert msg.request_id == "req-1"
    assert msg.target is None
    assert msg.data == {"text": "hello", "modality": "text", "stage_name": "asr"}
    assert msg.metadata == {"modality": "text", "token_id": 1}


def test_silent_when_not_streaming_and_does_not_create_state() -> None:
    builder = make_builder({1: b"A"})
    rd = make_req_data(stream=False)

    assert builder("req-1", rd, make_req_output(1)) == []
    assert not hasattr(rd.req, "_fun_asr_stream_pending_ids")


def test_silent_during_chunked_prefill_then_emits_after_prefill() -> None:
    builder = make_builder({1: b"A"})
    rd = make_req_data(stream=True, inflight_middle_chunks=1)

    assert builder("req-1", rd, make_req_output(1)) == []

    rd.req.inflight_middle_chunks = 0
    msgs = builder("req-1", rd, make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["A"]


def test_incremental_token_delta_and_eos_emits_no_self_delta() -> None:
    builder = make_builder({1: b"foo", 2: b"bar", EOS: b"<eos>"})
    rd = make_req_data()

    assert [m.data["text"] for m in builder("r", rd, make_req_output(1))] == ["foo"]
    assert [m.data["text"] for m in builder("r", rd, make_req_output(2))] == ["bar"]
    assert builder("r", rd, make_req_output(EOS)) == []


def test_min_emit_interval_first_delta_immediate_then_eos_flushes() -> None:
    builder = make_builder({1: b"A", 2: b"B", EOS: b"<eos>"}, interval_s=3600.0)
    rd = make_req_data()

    assert [m.data["text"] for m in builder("r", rd, make_req_output(1))] == ["A"]
    assert builder("r", rd, make_req_output(2)) == []
    assert [m.data["text"] for m in builder("r", rd, make_req_output(EOS))] == ["B"]


def test_terminal_finish_flushes_rate_limited_pending_tokens() -> None:
    builder = make_builder({1: b"A", 2: b"B"}, interval_s=3600.0)
    rd = make_req_data()

    assert [m.data["text"] for m in builder("r", rd, make_req_output(1))] == ["A"]
    assert builder("r", rd, make_req_output(2)) == []

    rd.req.finished = lambda: True
    msgs = builder("r", rd, make_req_output(None))
    assert [m.data["text"] for m in msgs] == ["B"]


def test_missing_required_scheduler_contract_fails_visibly() -> None:
    builder = make_builder({1: b"A"})
    broken_req_data = SimpleNamespace(stage_payload=None)

    with pytest.raises(AttributeError):
        builder("r", broken_req_data, make_req_output(1))


def test_per_request_state_is_isolated() -> None:
    builder = make_builder({1: b"A", 2: b"B"})
    rd1 = make_req_data()
    rd2 = make_req_data()

    out1 = builder("r1", rd1, make_req_output(1))
    out2 = builder("r2", rd2, make_req_output(2))
    out1b = builder("r1", rd1, make_req_output(2))

    assert [m.data["text"] for m in out1] == ["A"]
    assert [m.data["text"] for m in out2] == ["B"]
    assert [m.data["text"] for m in out1b] == ["B"]
    assert (
        rd1.req._fun_asr_stream_pending_ids == []
    )  # noqa: leading-underscore  # upstream name
    assert (
        rd2.req._fun_asr_stream_pending_ids == []
    )  # noqa: leading-underscore  # upstream name


def test_utf8_partial_token_is_held_until_complete() -> None:
    builder = make_builder({1: b"\xe4", 2: b"\xbd", 3: b"\xa0"})
    rd = make_req_data()

    assert builder("r", rd, make_req_output(1)) == []
    assert builder("r", rd, make_req_output(2)) == []
    assert [m.data["text"] for m in builder("r", rd, make_req_output(3))] == ["你"]
