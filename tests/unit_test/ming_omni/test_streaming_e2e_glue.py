# SPDX-License-Identifier: Apache-2.0
"""Glue tests for streaming TTS: thinker emits + client merges."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.ming_omni.bootstrap import make_thinker_stream_output_builder


class _FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            5: "Hello",
            6: " world",
            7: ".",
            8: " Tail",
        }

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab.get(int(i), "") for i in ids)


def _make_req():
    return SimpleNamespace(
        is_chunked=0,
        _ming_stream_token_ids=None,
        _ming_stream_emitted_text="",
    )


def _make_req_data(req):
    return SimpleNamespace(req=req)


def _make_req_output(token_id):
    return SimpleNamespace(data=token_id)


def test_thinker_stream_builder_emits_to_segmenter():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
    )
    req = _make_req()
    req_data = _make_req_data(req)

    msgs = builder("req-1", req_data, _make_req_output(5))
    # Thinker is not terminal, so only inter-stage stream to segmenter.
    assert len(msgs) == 1
    assert msgs[0].target == "segmenter"
    assert msgs[0].data.dtype.is_floating_point is False  # uint8 tensor
    assert bytes(msgs[0].data.tolist()).decode("utf-8") == "Hello"


def test_thinker_stream_builder_suppresses_during_chunked_prefill():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
    )
    req = _make_req()
    req.is_chunked = 1  # still consuming prompt chunks
    req_data = _make_req_data(req)

    msgs = builder("req-2", req_data, _make_req_output(5))
    assert msgs == []


def test_thinker_stream_builder_buffers_incomplete_utf8():
    # Tokenizer that produces an incomplete UTF-8 sequence on first call.
    class _IncompleteThenComplete:
        calls = 0

        def decode(self, ids, skip_special_tokens=True):
            type(self).calls += 1
            return "Hello\ufffd" if type(self).calls == 1 else "Hello\u4e16"

    builder = make_thinker_stream_output_builder(
        tokenizer=_IncompleteThenComplete(),
        eos_token_id=None,
    )
    req = _make_req()
    req_data = _make_req_data(req)
    # First token: incomplete -> no emit.
    msgs1 = builder("req-3", req_data, _make_req_output(5))
    assert msgs1 == []
    # Second token: completes UTF-8 -> emit one segmenter message with full delta.
    msgs2 = builder("req-3", req_data, _make_req_output(6))
    assert len(msgs2) == 1
    assert msgs2[0].target == "segmenter"


def test_client_result_builder_merges_decode_with_talker_stream():
    audio_bytes = (0).to_bytes(4, "little") * 8  # 8 float32 zero samples
    merged = {
        "decode": {"text": "Hello world.", "modality": "text"},
        "talker_stream": {
            "modality": "audio",
            "audio_waveform": audio_bytes,
            "audio_waveform_dtype": "float32",
            "audio_waveform_shape": [8],
            "sample_rate": 44100,
        },
    }
    chunk: GenerateChunk = Client._default_result_builder("req-x", merged)
    assert chunk.text == "Hello world."
    assert chunk.modality == "audio"
    assert chunk.audio_data is not None
    assert int(chunk.audio_data.shape[0]) == 8
    assert chunk.sample_rate == 44100
