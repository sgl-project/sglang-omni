# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    MOSS_TD_MARKER_LOOP_REASON,
    make_moss_transcribe_diarize_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload

_EOS = 999


class _ByteTokenizer:
    """Token id → fixed bytes; UTF-8 decode with errors='replace'."""

    eos_token_id = _EOS

    def __init__(
        self,
        vocab: dict[int, bytes],
        special_token_ids: set[int] | None = None,
    ):
        self._vocab = vocab
        self._special = special_token_ids or set()

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        chunks = [
            self._vocab[tid]
            for tid in ids
            if not (skip_special_tokens and tid in self._special)
        ]
        return b"".join(chunks).decode("utf-8", errors="replace")


def _make_req_data(*, stream: bool = True, inflight_middle_chunks: int = 0) -> Any:
    """Minimal req_data as OmniScheduler passes to stream_output_builder."""
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
        inflight_middle_chunks=inflight_middle_chunks,
        finished_reason=None,
        output_ids=[],
        to_finish=None,
    )
    return SimpleNamespace(
        req=req,
        stage_payload=stage_payload,
        no_progress_termination_reason=None,
        no_progress_completed_segments=0,
        no_progress_marker_only_segments=0,
        no_progress_repeated_segments=0,
        no_progress_detected_completion_tokens=0,
    )


def _make_req_output(token_id: int | None) -> Any:
    return SimpleNamespace(data=token_id)


def _builder(vocab: dict[int, bytes], special: set[int] | None = None):
    return make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer(vocab, special_token_ids=special),
    )


def test_emits_text_delta_when_streaming():
    builder = _builder({1: b"[0.00]"})
    rd = _make_req_data(stream=True)

    msgs = builder("req-1", rd, _make_req_output(1))

    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.type == "stream"
    assert msg.request_id == "req-1"
    assert msg.target is None
    assert msg.data == {"text": "[0.00]", "modality": "text", "stage_name": "asr"}
    assert msg.metadata == {"modality": "text", "token_id": 1}


def test_silent_when_not_streaming():
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=False)

    assert builder("req-1", rd, _make_req_output(1)) == []
    assert not hasattr(rd.req, "_moss_stream_pending_ids")


def test_silent_during_chunked_prefill():
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=True, inflight_middle_chunks=1)

    assert builder("req-1", rd, _make_req_output(1)) == []

    rd.req.inflight_middle_chunks = 0
    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["A"]


def test_silent_when_no_token_this_step():
    builder = _builder({1: b"A"})
    rd = _make_req_data(stream=True)

    assert builder("req-1", rd, _make_req_output(None)) == []


def test_silent_when_req_or_payload_missing():
    builder = _builder({1: b"A"})

    no_req = SimpleNamespace(req=None, stage_payload=None)
    assert builder("req-1", no_req, _make_req_output(1)) == []

    no_payload = SimpleNamespace(
        req=SimpleNamespace(inflight_middle_chunks=0), stage_payload=None
    )
    assert builder("req-1", no_payload, _make_req_output(1)) == []


def test_incremental_deltas_across_tokens():
    builder = _builder({1: b"[S01]", 2: b" hello", 3: b" world"})
    rd = _make_req_data()

    deltas = []
    for tid in (1, 2, 3):
        for msg in builder("req-1", rd, _make_req_output(tid)):
            deltas.append(msg.data["text"])

    assert deltas == ["[S01]", " hello", " world"]


def test_utf8_multibyte_hold_then_emit():
    """A 3-byte CJK char split across 3 tokens must hold until complete."""
    builder = _builder({1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 4: b"ok"})
    rd = _make_req_data()

    assert builder("req-1", rd, _make_req_output(1)) == []
    assert builder("req-1", rd, _make_req_output(2)) == []
    msgs = builder("req-1", rd, _make_req_output(3))
    assert [m.data["text"] for m in msgs] == ["你"]

    msgs = builder("req-1", rd, _make_req_output(4))
    assert [m.data["text"] for m in msgs] == ["ok"]


def test_interior_replacement_char_does_not_stall_stream():
    """Only a TRAILING U+FFFD is held; an interior one must flush normally."""
    builder = _builder({1: b"\x80", 2: b"ok"})
    rd = _make_req_data()

    assert builder("req-1", rd, _make_req_output(1)) == []
    msgs = builder("req-1", rd, _make_req_output(2))
    assert [m.data["text"] for m in msgs] == ["\ufffdok"]


def test_eos_token_emits_no_delta():
    builder = _builder({1: b"hi", _EOS: b"<eos>"})
    rd = _make_req_data()

    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["hi"]
    assert builder("req-1", rd, _make_req_output(_EOS)) == []


def test_special_token_emits_no_delta():
    """Tokens dropped by skip_special_tokens must not produce a chunk."""
    builder = _builder({1: b"hi", 2: b"<|im_end|>"}, special={2})
    rd = _make_req_data()

    msgs = builder("req-1", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["hi"]
    assert builder("req-1", rd, _make_req_output(2)) == []


def test_per_request_state_is_isolated():
    """Concurrent requests keep independent token/text state on their req."""
    builder = _builder({1: b"A", 2: b"B"})
    rd1 = _make_req_data()
    rd2 = _make_req_data()

    out1 = builder("r1", rd1, _make_req_output(1))
    out2 = builder("r2", rd2, _make_req_output(2))
    out1b = builder("r1", rd1, _make_req_output(2))

    assert [m.data["text"] for m in out1] == ["A"]
    assert [m.data["text"] for m in out2] == ["B"]
    assert [m.data["text"] for m in out1b] == ["B"]
    assert rd1.req._moss_stream_pending_ids == []
    assert rd2.req._moss_stream_pending_ids == []


def _interval_builder(vocab: dict[int, bytes], interval_s: float):
    return make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer(vocab),
        min_emit_interval_s=interval_s,
    )


def test_min_emit_interval_first_delta_is_immediate():
    builder = _interval_builder({1: b"A"}, interval_s=3600.0)
    rd = _make_req_data()

    msgs = builder("r", rd, _make_req_output(1))
    assert [m.data["text"] for m in msgs] == ["A"]


def test_min_emit_interval_holds_then_eos_flushes():
    """Tokens within the interval are held and flushed as one delta on EOS."""
    builder = _interval_builder({1: b"A", 2: b"B", 3: b"C"}, interval_s=3600.0)
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(2)) == []
    assert builder("r", rd, _make_req_output(3)) == []

    msgs = builder("r", rd, _make_req_output(_EOS))
    assert [m.data["text"] for m in msgs] == ["BC"]


def test_min_emit_interval_elapsed_flushes_batch():
    builder = _interval_builder({1: b"A", 2: b"B", 3: b"C"}, interval_s=0.01)
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(2)) == []

    import time as _time

    _time.sleep(0.02)
    msgs = builder("r", rd, _make_req_output(3))
    assert [m.data["text"] for m in msgs] == ["BC"]


def test_eos_with_empty_pending_emits_nothing():
    builder = _interval_builder({1: b"A"}, interval_s=3600.0)
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(_EOS)) == []


def test_explicit_eos_token_id_overrides_tokenizer():
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer({1: b"A", 7: b"<stop>"}),
        eos_token_id=7,
    )
    rd = _make_req_data()

    assert [m.data["text"] for m in builder("r", rd, _make_req_output(1))] == ["A"]
    assert builder("r", rd, _make_req_output(7)) == []


def _guarded_builder(
    vocab: dict[int, bytes],
    *,
    marker_segments: int = 0,
    repeat_segments: int = 0,
):
    return make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer(vocab),
        buffered_no_progress_marker_segments=marker_segments,
        buffered_no_progress_repeat_segments=repeat_segments,
    )


def _observe(builder, rd, *token_ids: int) -> list[str]:
    deltas: list[str] = []
    for token_id in token_ids:
        messages = builder("r", rd, _make_req_output(token_id))
        deltas.extend(message.data["text"] for message in messages)
        rd.req.output_ids.append(token_id)
    return deltas


def _matched_finish_reason(rd) -> str | None:
    if rd.req.to_finish is None:
        return None
    return rd.req.to_finish.to_json().get("matched")


def test_no_progress_guard_is_zero_diff_when_disabled() -> None:
    builder = _guarded_builder({1: b"[0.00][S01][0.10]"})
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 1, 1, 1)

    assert rd.req.to_finish is None
    assert not hasattr(rd.req, "_moss_no_progress_state")
    assert rd.no_progress_termination_reason is None


def test_marker_only_loop_stops_at_complete_segment_boundary() -> None:
    vocab = {
        1: b"[0.00][S01][0.10]",
        2: b"[0.11][S02][0.20]",
        3: b"[0.21][S01]",
        4: b"[0.30]",
    }
    builder = _guarded_builder(vocab, marker_segments=3)
    rd = _make_req_data(stream=False)

    deltas = _observe(builder, rd, 1, 2, 3)
    assert rd.req.to_finish is None

    deltas.extend(_observe(builder, rd, 4))

    assert _matched_finish_reason(rd) == "moss_td_no_progress_marker_loop"
    assert rd.no_progress_termination_reason == "moss_td_no_progress_marker_loop"
    assert rd.no_progress_completed_segments == 3
    assert rd.no_progress_marker_only_segments == 3
    assert rd.no_progress_repeated_segments == 0
    assert rd.no_progress_detected_completion_tokens == 4
    assert deltas == []


def test_first_prefill_marker_decision_waits_for_a_decode_boundary() -> None:
    builder = _guarded_builder(
        {1: b"[0.00][S01][0.10]"},
        marker_segments=1,
    )
    rd = _make_req_data(stream=False)

    # OmniScheduler invokes the callback before SGLang commits the first
    # prefill token to output_ids. Finishing here would make SGLang drop it.
    assert rd.req.output_ids == []
    assert builder("r", rd, _make_req_output(1)) == []

    assert rd.req.to_finish is None
    assert rd.no_progress_termination_reason is None
    assert rd.req._moss_no_progress_state.disabled is False

    rd.req.output_ids.append(1)
    assert builder("r", rd, _make_req_output(1)) == []
    assert rd.req.to_finish is not None
    assert rd.no_progress_termination_reason == MOSS_TD_MARKER_LOOP_REASON


def test_content_segment_resets_marker_only_progress_counter() -> None:
    vocab = {
        1: b"[0.00][S01][0.10]",
        2: b"[0.11][S01][0.20]",
        3: b"[0.21][S01]spoken words[0.30]",
        4: b"[0.31][S01][0.40]",
        5: b"[0.41][S01][0.50]",
        6: b"[0.51][S01][0.60]",
    }
    builder = _guarded_builder(vocab, marker_segments=3)
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2, 3, 4, 5)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.marker_only_segments == 2

    _observe(builder, rd, 6)
    assert _matched_finish_reason(rd) == "moss_td_no_progress_marker_loop"


def test_progress_later_in_same_token_cancels_pending_marker_stop() -> None:
    builder = _guarded_builder(
        {
            1: (
                b"[0.00][S01][0.10]"
                b"[0.11][S01][0.20]"
                b"[0.21][S01]actual speech[0.30]"
            )
        },
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.completed_segments == 3
    assert rd.req._moss_no_progress_state.marker_only_segments == 0


def test_advancing_timestamps_preserve_legitimately_repeated_content() -> None:
    vocab = {
        1: b"[0.00][S01]yes[0.10]",
        2: b"[0.11][S01]yes[0.20]",
        3: b"[0.21][S01]yes[0.30]",
        4: b"[0.31][S01]yes[0.40]",
    }
    builder = _guarded_builder(vocab, repeat_segments=3)
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2, 3, 4)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.repeated_segments == 0


def test_exact_content_segment_loop_stops_only_after_configured_repeat() -> None:
    builder = _guarded_builder(
        {1: b"[0.00][S01]same words[0.10]"},
        repeat_segments=3,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 1, 1)
    assert rd.req.to_finish is None

    _observe(builder, rd, 1)

    assert _matched_finish_reason(rd) == "moss_td_no_progress_repeated_segment"
    assert rd.no_progress_termination_reason == ("moss_td_no_progress_repeated_segment")
    assert rd.no_progress_marker_only_segments == 0
    assert rd.no_progress_repeated_segments == 3
    assert rd.no_progress_detected_completion_tokens == 4


def test_full_token_decode_keeps_distinct_unicode_bodies_non_additive() -> None:
    vocab = {
        1: b"[0][S01]",
        2: b"\xe4",
        3: b"\xbd\xa0[1]",
        4: b"\xe5",
        5: b"\xa5\xbd[1]",
    }
    tokenizer = _ByteTokenizer(vocab)
    assert (
        "".join(tokenizer.decode([token_id]) for token_id in (2, 3))
        == "\ufffd\ufffd\ufffd[1]"
    )
    assert (
        "".join(tokenizer.decode([token_id]) for token_id in (4, 5))
        == "\ufffd\ufffd\ufffd[1]"
    )
    assert tokenizer.decode([2, 3]) == "\u4f60[1]"
    assert tokenizer.decode([4, 5]) == "\u597d[1]"
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=tokenizer,
        buffered_no_progress_repeat_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2, 3, 1, 4, 5)

    assert rd.req.to_finish is None
    assert rd.no_progress_termination_reason is None


def test_distinct_malformed_body_bytes_never_form_an_exact_repeat() -> None:
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer(
            {
                1: b"[0][S01]",
                2: b"\x80[1]",
                3: b"\x81[1]",
            }
        ),
        buffered_no_progress_repeat_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2, 1, 3)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.disabled is True


def test_interior_replacement_disables_repeat_detection() -> None:
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer({1: b"[0][S01]ok\x80[1]"}),
        buffered_no_progress_repeat_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 1)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.disabled is True


@pytest.mark.parametrize("error_type", [ValueError, AttributeError, OSError])
def test_decode_exception_disables_guard_fail_open_without_escaping(
    error_type: type[Exception],
) -> None:
    class _RaisingTokenizer:
        eos_token_id = _EOS

        def __init__(self) -> None:
            self.decode_calls = 0

        def decode(self, ids, **kwargs) -> str:
            self.decode_calls += 1
            raise error_type("malformed token sequence")

    tokenizer = _RaisingTokenizer()
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=tokenizer,
        buffered_no_progress_marker_segments=2,
    )
    rd = _make_req_data(stream=False)
    escaped: Exception | None = None

    try:
        builder("r", rd, _make_req_output(1))
    except Exception as error:
        escaped = error

    assert escaped is None
    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.disabled is True
    assert tokenizer.decode_calls == 1

    assert builder("r", rd, _make_req_output(1)) == []
    assert tokenizer.decode_calls == 1


def test_incomplete_utf8_waits_for_byte_faithful_valid_decode() -> None:
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer(
            {
                1: b"[0][S01]",
                2: b"\xe4",
                3: b"\xbd\xa0[1]",
            }
        ),
        buffered_no_progress_repeat_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2)
    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.completed_segments == 0

    _observe(builder, rd, 3, 1, 2, 3, 1, 2, 3)

    assert _matched_finish_reason(rd) == "moss_td_no_progress_repeated_segment"
    assert rd.req._moss_no_progress_state.disabled is False


def test_cached_qwen2_trailing_replacement_disables_bounded_guard_fail_open() -> None:
    snapshot_value = os.environ.get("MOSS_TD_TEST_TOKENIZER_SNAPSHOT")
    if not snapshot_value:
        pytest.skip("set MOSS_TD_TEST_TOKENIZER_SNAPSHOT to the cached MOSS snapshot")

    from transformers import Qwen2Tokenizer

    snapshot = Path(snapshot_value)
    tokenizer = Qwen2Tokenizer.from_pretrained(snapshot, local_files_only=True)
    assert tokenizer.decode(
        [94], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).endswith("\ufffd")

    decode_lengths: list[int] = []
    original_decode = tokenizer.decode

    def recording_decode(token_ids, *args, **kwargs):
        decode_lengths.append(len(token_ids))
        return original_decode(token_ids, *args, **kwargs)

    tokenizer.decode = recording_decode
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=tokenizer,
        buffered_no_progress_marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    for _ in range(1025):
        assert builder("req-1", rd, _make_req_output(94)) == []

    state = rd.req._moss_no_progress_state
    assert state.disabled is True
    assert state.pending_token_ids == []
    assert state.buffer == ""
    assert max(decode_lengths) <= 256
    assert rd.req.to_finish is None
    assert rd.no_progress_termination_reason is None

    observed_tokens = state.observed_tokens
    decode_calls = len(decode_lengths)
    assert builder("req-1", rd, _make_req_output(94)) == []
    assert state.observed_tokens == observed_tokens
    assert len(decode_lengths) == decode_calls


def test_buffered_thresholds_do_not_enable_streaming_termination() -> None:
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=_ByteTokenizer({1: b"[0][S01][1]"}),
        buffered_no_progress_marker_segments=2,
    )
    buffered = _make_req_data(stream=False)
    streaming = _make_req_data(stream=True)

    _observe(builder, buffered, 1, 1)
    _observe(builder, streaming, 1, 1)

    assert _matched_finish_reason(buffered) == "moss_td_no_progress_marker_loop"
    assert streaming.req.to_finish is None
    assert not hasattr(streaming.req, "_moss_no_progress_state")


@pytest.mark.parametrize(
    "second_segment",
    [b"[0][S01] body[1]", b"[0][s01]body[1]"],
)
def test_exact_repeat_identity_preserves_body_and_speaker_bytes(
    second_segment: bytes,
) -> None:
    builder = _guarded_builder(
        {1: b"[0][S01]body[1]", 2: second_segment},
        repeat_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.repeated_segments == 0


def test_marker_only_silence_stops_without_inventing_content() -> None:
    builder = _guarded_builder(
        {
            1: b"[0.00][S01][0.10]",
            2: b"[0.11][S01][0.20]",
        },
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2)

    assert _matched_finish_reason(rd) == "moss_td_no_progress_marker_loop"
    assert rd.no_progress_completed_segments == 2


def test_incomplete_or_malformed_segment_never_triggers() -> None:
    builder = _guarded_builder(
        {
            1: b"[0.00][S01][0.10]",
            2: b"[0.11][S01]",
            3: b"[bad",
            4: b"]not-a-segment",
        },
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2, 3, 4)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.completed_segments == 1


def test_chunked_prefill_tokens_do_not_advance_no_progress_state() -> None:
    builder = _guarded_builder(
        {1: b"[0.00][S01][0.10]"},
        marker_segments=2,
    )
    rd = _make_req_data(stream=False, inflight_middle_chunks=1)

    _observe(builder, rd, 1, 1)
    assert not hasattr(rd.req, "_moss_no_progress_state")

    rd.req.inflight_middle_chunks = 0
    _observe(builder, rd, 1)
    assert rd.req.to_finish is None


@pytest.mark.parametrize("terminal_field", ["finished_reason", "to_finish"])
def test_terminal_or_cancelled_request_does_not_allocate_guard_state(
    terminal_field: str,
) -> None:
    builder = _guarded_builder(
        {1: b"[0.00][S01][0.10]"},
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)
    setattr(rd.req, terminal_field, object())

    _observe(builder, rd, 1, 1)

    assert not hasattr(rd.req, "_moss_no_progress_state")
    assert rd.no_progress_termination_reason is None


def test_guard_stops_observing_after_terminal_decision() -> None:
    builder = _guarded_builder(
        {1: b"[0.00][S01][0.10]"},
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 1)
    completed_segments = rd.req._moss_no_progress_state.completed_segments
    _observe(builder, rd, 1)

    assert completed_segments == 2
    assert rd.req._moss_no_progress_state.completed_segments == completed_segments


def test_oversized_unrecognized_suffix_disables_guard_fail_open() -> None:
    builder = _guarded_builder(
        {
            1: b"x" * 65537,
            2: b"[0.00][S01][0.10]",
        },
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1, 2, 2)

    assert rd.req.to_finish is None
    assert rd.req._moss_no_progress_state.disabled is True


def test_many_small_incomplete_tokens_exhaust_decode_work_budget() -> None:
    class _CountingTokenizer(_ByteTokenizer):
        def __init__(self) -> None:
            super().__init__({1: b"x"})
            self.decode_calls = 0

        def decode(self, ids, **kwargs) -> str:
            self.decode_calls += 1
            return super().decode(ids, **kwargs)

    tokenizer = _CountingTokenizer()
    builder = make_moss_transcribe_diarize_stream_output_builder(
        tokenizer=tokenizer,
        buffered_no_progress_marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, *([1] * 300))

    state = rd.req._moss_no_progress_state
    assert state.disabled is True
    assert tokenizer.decode_calls <= 256
    decode_calls = tokenizer.decode_calls

    _observe(builder, rd, 1)
    assert tokenizer.decode_calls == decode_calls


def test_guard_warning_hashes_unbounded_request_id(caplog) -> None:
    raw_request_id = "request-id-with-newline\nsecret-" * 300
    request_id_sha256 = hashlib.sha256(raw_request_id.encode("utf-8")).hexdigest()
    builder = _guarded_builder(
        {1: b"[0.00][S01][0.10]"},
        marker_segments=2,
    )
    rd = _make_req_data(stream=False)

    _observe(builder, rd, 1)
    builder(raw_request_id, rd, _make_req_output(1))

    messages = [record.getMessage() for record in caplog.records]
    assert raw_request_id not in "\n".join(messages)
    assert any(
        f"request_id_sha256={request_id_sha256}" in message for message in messages
    )
