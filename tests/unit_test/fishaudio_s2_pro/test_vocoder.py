# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro import stages
from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import S2ProVocoderScheduler
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import CompleteMessage
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.fish_fakes import FakeFishCodec, make_s2pro_payload
from tests.unit_test.pipeline.helpers import run_scheduler


def test_fish_vocoder_batches_and_trims_audio_by_code_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves batched vocoder decode and per-request trim by code length."""
    codec = FakeFishCodec(frame_length=4)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "_load_codec", lambda checkpoint, device: codec)
    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=50,
    )
    assert scheduler._stream_stride == 40
    assert scheduler._stream_followup_stride == 45

    usage_override = {
        "prompt_tokens": 7,
        "completion_tokens": 11,
        "total_tokens": 18,
        "engine_time_s": 0.75,
    }

    def payload(request_id: str, code_len: int) -> object:
        item = make_s2pro_payload(
            S2ProState(
                output_codes=torch.arange(3 * code_len).reshape(3, code_len),
                prompt_tokens=4,
                completion_tokens=code_len,
                engine_time_s=0.5,
                finish_reason="length" if request_id == "req-short" else None,
            ),
            request_id=request_id,
        )
        if request_id == "req-short":
            item.data["usage"] = usage_override
        return item

    first, second = run_scheduler(
        scheduler,
        [
            IncomingMessage("req-short", "new_request", payload("req-short", 2)),
            IncomingMessage("req-long", "new_request", payload("req-long", 3)),
        ],
        output_count=2,
    )
    outputs = {first.request_id: first.data, second.request_id: second.data}
    short_data = outputs["req-short"].data
    long_data = outputs["req-long"].data

    restored = deserialize_message(
        serialize_message(
            CompleteMessage(
                request_id="req-short",
                from_stage="vocoder",
                success=True,
                result=short_data,
            )
        )
    )
    assert restored.result == short_data

    assert codec.calls == [(2, 2, 3)]
    short_audio = np.frombuffer(short_data["audio_waveform"], dtype=np.float32).reshape(
        short_data["audio_waveform_shape"]
    )
    long_audio = np.frombuffer(long_data["audio_waveform"], dtype=np.float32).reshape(
        long_data["audio_waveform_shape"]
    )
    np.testing.assert_array_equal(short_audio, np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(long_audio, np.full(12, 2.0, dtype=np.float32))
    assert short_data["audio_waveform_shape"] == [8]
    assert long_data["audio_waveform_shape"] == [12]
    assert short_data["audio_waveform_dtype"] == "float32"
    assert long_data["audio_waveform_dtype"] == "float32"
    assert short_data["sample_rate"] == long_data["sample_rate"] == 44100
    assert short_data["modality"] == long_data["modality"] == "audio"
    assert short_data["usage"] == usage_override
    assert long_data["usage"]["total_tokens"] == 7
    assert short_data["finish_reason"] == "length"
    assert "finish_reason" not in long_data
    assert set(short_data) == {
        "audio_waveform",
        "audio_waveform_shape",
        "audio_waveform_dtype",
        "sample_rate",
        "modality",
        "usage",
        "finish_reason",
    }
    assert set(long_data) == {
        "audio_waveform",
        "audio_waveform_shape",
        "audio_waveform_dtype",
        "sample_rate",
        "modality",
        "usage",
    }


def test_fish_terminal_payload_does_not_mutate_input() -> None:
    scheduler = S2ProVocoderScheduler(
        FakeFishCodec(frame_length=4),
        device="cpu",
    )
    state = S2ProState(
        output_codes=torch.arange(3 * 2).reshape(3, 2),
        prompt_tokens=1,
        completion_tokens=2,
    )
    payload = make_s2pro_payload(state, request_id="req-terminal-ownership")
    original_data = payload.data

    result = scheduler._store_audio(
        payload,
        state,
        torch.ones(8, dtype=torch.float32),
    )

    assert result is not payload
    assert payload.data is original_data
    assert "output_codes" in payload.data
    assert "audio_waveform" not in payload.data
    assert result.data["audio_waveform_shape"] == [8]


def test_fish_stream_fallback_preserves_terminal_metadata() -> None:
    usage = {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    scheduler = S2ProVocoderScheduler(
        FakeFishCodec(frame_length=4),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    payload = make_s2pro_payload(
        S2ProState(
            output_codes=torch.arange(3 * 2).reshape(3, 2),
            prompt_tokens=1,
            completion_tokens=2,
            finish_reason="length",
        ),
        request_id="req-stream-fallback",
        params={"stream": True},
    )
    payload.data["usage"] = usage
    scheduler._on_streaming_new_request("req-stream-fallback", payload)

    stream, final = scheduler.on_stream_done("req-stream-fallback")

    assert stream.type == "stream"
    np.testing.assert_array_equal(
        stream.data["audio_data"], np.ones(8, dtype=np.float32)
    )
    assert stream.data["sample_rate"] == 44100
    assert final.type == "result"
    assert final.data.data == {
        "modality": "audio",
        "sample_rate": 44100,
        "usage": usage,
        "finish_reason": "length",
    }
