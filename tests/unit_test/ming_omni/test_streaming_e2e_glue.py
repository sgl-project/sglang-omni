# SPDX-License-Identifier: Apache-2.0
"""Glue tests for streaming TTS: thinker emits + client merges."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.ming_omni.bootstrap import make_thinker_stream_output_builder
from sglang_omni.models.ming_omni.stages import MingStreamingDecodeScheduler
from sglang_omni.proto import OmniRequest, StagePayload


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


def _make_req_data(req, *, stage_payload=None):
    return SimpleNamespace(req=req, stage_payload=stage_payload)


def _make_req_output(token_id):
    return SimpleNamespace(data=token_id)


def _make_stage_payload(*, stream: bool, output_modalities=None):
    metadata = {}
    if output_modalities is not None:
        metadata["output_modalities"] = output_modalities
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=[],
            params={"stream": stream},
            metadata=metadata,
        ),
        data={
            "prompt": {"input_ids": [1, 2]},
            "engine_outputs": {
                "thinker": {
                    "output_ids": [5, 6, 7],
                    "step": 3,
                    "is_final": True,
                    "extra_model_outputs": {},
                    "finish_reason": "stop",
                }
            },
        },
    )


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


def test_thinker_stream_builder_emits_decode_only_for_client_streaming():
    builder = make_thinker_stream_output_builder(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
        target_stages=["decode", "segmenter"],
        client_stream_target_stages=("decode",),
    )

    req_data = _make_req_data(
        _make_req(),
        stage_payload=_make_stage_payload(stream=True, output_modalities=["text"]),
    )
    msgs = builder("req-1", req_data, _make_req_output(5))
    assert [msg.target for msg in msgs] == ["decode", "segmenter"]

    non_stream_req_data = _make_req_data(
        _make_req(),
        stage_payload=_make_stage_payload(stream=False, output_modalities=["text"]),
    )
    msgs = builder("req-1", non_stream_req_data, _make_req_output(5))
    assert [msg.target for msg in msgs] == ["segmenter"]

    audio_only_req_data = _make_req_data(
        _make_req(),
        stage_payload=_make_stage_payload(stream=True, output_modalities=["audio"]),
    )
    msgs = builder("req-1", audio_only_req_data, _make_req_output(5))
    assert [msg.target for msg in msgs] == ["segmenter"]


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


def test_streaming_decode_scheduler_forwards_deltas_and_slims_final():
    scheduler = MingStreamingDecodeScheduler(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
    )
    stream_messages = []
    for text in ("Hello", " world", "."):
        scheduler._on_stream_chunk(
            "req-1",
            SimpleNamespace(
                data=torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
            ),
        )
        stream_messages.append(scheduler.outbox.get_nowait())

    assert [msg.type for msg in stream_messages] == ["stream", "stream", "stream"]
    assert [msg.target for msg in stream_messages] == [None, None, None]
    assert [msg.data["text"] for msg in stream_messages] == ["Hello", " world", "."]
    assert stream_messages[0].data == {
        "text": "Hello",
        "modality": "text",
        "stage_name": "decode",
    }

    scheduler._on_stream_done("req-1")
    scheduler._on_new_request("req-1", _make_stage_payload(stream=True))

    result_msg = scheduler.outbox.get_nowait()
    assert result_msg.type == "result"
    assert result_msg.data.data["finish_reason"] == "stop"
    assert result_msg.data.data["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert "text" not in result_msg.data.data


def test_streaming_decode_scheduler_emits_final_suffix_if_stream_missed_tail():
    scheduler = MingStreamingDecodeScheduler(
        tokenizer=_FakeTokenizer(),
        eos_token_id=None,
    )
    scheduler._on_stream_chunk(
        "req-1",
        SimpleNamespace(
            data=torch.tensor(list("Hello".encode("utf-8")), dtype=torch.uint8)
        ),
    )
    scheduler.outbox.get_nowait()

    scheduler._on_stream_done("req-1")
    scheduler._on_new_request("req-1", _make_stage_payload(stream=True))

    result_msg = scheduler.outbox.get_nowait()
    assert result_msg.type == "result"
    assert result_msg.data.data["text"] == " world."
    assert result_msg.data.data["finish_reason"] == "stop"


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
