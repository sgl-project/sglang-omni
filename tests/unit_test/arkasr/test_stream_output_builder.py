# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.arkasr.request_builders import (
    make_arkasr_scheduler_adapters,
    make_arkasr_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload

_EOS = 999


class _ByteTokenizer:
    eos_token_id = _EOS
    all_special_ids: list[int] = []

    def __init__(self, vocab: dict[int, bytes]) -> None:
        self._vocab = vocab

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return b"".join(self._vocab[tid] for tid in ids).decode(
            "utf-8", errors="replace"
        )

    def get_added_vocab(self) -> dict[str, int]:
        return {}


def _make_req_data(
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


def _make_req_output(token_id: int | None) -> Any:
    return SimpleNamespace(data=token_id)


def _builder(vocab: dict[int, bytes], *, interval_s: float = 0.0):
    return make_arkasr_stream_output_builder(
        tokenizer=_ByteTokenizer(vocab),
        min_emit_interval_s=interval_s,
    )


def test_emits_text_delta_when_streaming() -> None:
    builder = _builder({1: b"hello"})
    rd = _make_req_data(stream=True)

    msgs = builder("req-1", rd, _make_req_output(1))

    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.type == "stream"
    assert msg.request_id == "req-1"
    assert msg.target is None
    assert msg.data == {"text": "hello", "modality": "text", "stage_name": "asr"}
    assert msg.metadata == {"modality": "text", "token_id": 1}


def test_silent_when_not_streaming_and_does_not_create_state() -> None:
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=False)

    assert builder("req-1", rd, _make_req_output(1)) == []
    assert not hasattr(rd.req, "_arkasr_stream_pending_ids")


def test_silent_during_chunked_prefill_then_emits_after_prefill() -> None:
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=True, inflight_middle_chunks=1)

    assert builder("req-1", rd, _make_req_output(1)) == []

    rd.req.inflight_middle_chunks = 0
    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["A"]


def test_incremental_token_delta_and_eos_emits_no_self_delta() -> None:
    builder = _builder({1: b"foo", 2: b"bar", _EOS: b"<eos>"})
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["foo"]
    assert [m.data["text"] for m in builder("r", rd, _make_req_output(2))] == ["bar"]
    assert builder("r", rd, _make_req_output(_EOS)) == []


def test_min_emit_interval_first_delta_immediate_then_eos_flushes() -> None:
    builder = _builder({1: b"A", 2: b"B", _EOS: b"<eos>"}, interval_s=3600.0)
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(2)) == []
    assert [m.data["text"] for m in builder("r", rd, _make_req_output(_EOS))] == ["B"]


def test_missing_required_scheduler_contract_fails_visibly() -> None:
    builder = _builder({1: b"A"})
    broken_req_data = SimpleNamespace(stage_payload=None)

    with pytest.raises(AttributeError):
        builder("r", broken_req_data, _make_req_output(1))


def test_per_request_state_is_isolated() -> None:
    builder = _builder({1: b"A", 2: b"B"})
    rd1 = _make_req_data()
    rd2 = _make_req_data()

    out1 = builder("r1", rd1, _make_req_output(1))
    out2 = builder("r2", rd2, _make_req_output(2))
    out1b = builder("r1", rd1, _make_req_output(2))

    assert [m.data["text"] for m in out1] == ["A"]
    assert [m.data["text"] for m in out2] == ["B"]
    assert [m.data["text"] for m in out1b] == ["B"]
    assert rd1.req._arkasr_stream_pending_ids == []
    assert rd2.req._arkasr_stream_pending_ids == []


def test_utf8_partial_token_is_held_until_complete() -> None:
    builder = _builder({1: b"\xe4", 2: b"\xbd", 3: b"\xa0"})
    rd = _make_req_data()

    assert builder("r", rd, _make_req_output(1)) == []
    assert builder("r", rd, _make_req_output(2)) == []
    assert [m.data["text"] for m in builder("r", rd, _make_req_output(3))] == ["你"]


def test_flush_hook_emits_non_eos_terminal_tail() -> None:
    builder = _builder({1: b"A", 2: b"B"}, interval_s=3600.0)
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(2)) == []

    rd.req.finished = lambda: True
    msgs = builder.flush("r", rd)

    assert [m.data["text"] for m in msgs] == ["B"]
    assert msgs[0].metadata == {"modality": "text", "token_id": None}


def test_suppressed_marker_tokens_do_not_appear_in_deltas() -> None:
    class _MarkerTokenizer(_ByteTokenizer):
        all_special_ids = [_EOS]
        eos_token_id = _EOS

        def get_added_vocab(self):
            return {"<tool_call>": 2, "<|audio|>": 3}

    builder = make_arkasr_stream_output_builder(
        tokenizer=_MarkerTokenizer({1: b"hello", 2: b"<tool_call>", 3: b"<|audio|>"}),
    )
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["hello"]
    assert builder("r", rd, _make_req_output(2)) == []
    assert builder("r", rd, _make_req_output(3)) == []


def test_concatenated_deltas_strip_matches_result_adapter_text() -> None:
    """done.text is authoritative; join(deltas) may differ only by strip()."""

    class _Tok(_ByteTokenizer):
        vocab_size = 1000
        all_special_ids = [_EOS]
        eos_token_id = _EOS

        def get_added_vocab(self):
            return {"<tool_call>": 3}

    tokenizer = _Tok({1: b" hello", 2: b" world ", 3: b"<tool_call>", _EOS: b"<eos>"})
    _, result_adapter = make_arkasr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=16,
        feature_extractor=object(),
    )
    builder = make_arkasr_stream_output_builder(tokenizer)
    rd = _make_req_data()

    deltas: list[str] = []
    for token_id in (1, 3, 2, _EOS):
        deltas.extend(
            m.data["text"] for m in builder("r", rd, _make_req_output(token_id))
        )

    done = result_adapter(
        SimpleNamespace(
            stage_payload=rd.stage_payload,
            output_ids=[1, 3, 2],
            engine_start_s=None,
            language="en",
            audio_duration_s=0.0,
        )
    )
    joined = "".join(deltas)
    assert joined == " hello world "
    assert joined != done.data["text"]
    assert joined.strip() == done.data["text"] == "hello world"
