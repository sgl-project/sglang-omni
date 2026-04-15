# SPDX-License-Identifier: Apache-2.0
"""Regression tests for S2-Pro streaming helpers."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_s2_pro.pipeline import stages
from sglang_omni.models.fishaudio_s2_pro.pipeline.streaming_vocoder import (
    STREAM_VOCODER_STATE_KEY,
    build_stream_vocoder_chunk,
    flush_stream_vocoder_chunk,
    load_stream_vocoder_state,
    resolve_stream_overlap_tokens,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeCodec:
    sample_rate = 24000
    delay = 3
    frame_length = 2

    def from_indices(self, codes):
        num_tokens = int(codes.shape[-1])
        audio = torch.arange(num_tokens * 2, dtype=torch.float32)
        return audio.reshape(1, 1, -1)


class _FactoryCodec:
    sample_rate = 24000
    delay = 4
    frame_length = 2


def test_resolve_stream_overlap_tokens_uses_codec_delay_math() -> None:
    assert resolve_stream_overlap_tokens(_FakeCodec(), None) == 2


def test_build_incremental_audio_chunk_emits_delta_audio() -> None:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs="hello"),
        data={
            STREAM_VOCODER_STATE_KEY: {
                "codes": [
                    torch.tensor([[1, 2], [3, 4]]),
                    torch.tensor([[5], [6]]),
                ],
                "code_start_token": 0,
                "last_vocode_tokens": 0,
                "next_vocode_tokens": 0,
                "pending_tail": None,
                "total_tokens": 3,
            }
        },
    )

    first_chunk = flush_stream_vocoder_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
        stream_crossfade_samples=0,
    )
    assert first_chunk is not None
    assert first_chunk["modality"] == "audio"
    assert len(first_chunk["audio_data"]) == 6
    state = load_stream_vocoder_state(payload)
    assert state is None

    payload.data = {}
    second_chunk = build_stream_vocoder_chunk(
        payload,
        torch.tensor([[7], [8]]),
        codec=_FakeCodec(),
        device="cpu",
        stream_stride=1,
        stream_followup_stride=1,
        stream_overlap_tokens=2,
        stream_crossfade_samples=0,
    )
    assert second_chunk is not None
    assert second_chunk["audio_data"] == [0.0, 1.0]


def test_build_incremental_audio_chunk_crossfades_chunk_boundaries() -> None:
    payload = StagePayload(
        request_id="req-xfade",
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={},
    )

    first_chunk = build_stream_vocoder_chunk(
        payload,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        codec=_FakeCodec(),
        device="cpu",
        stream_stride=1,
        stream_followup_stride=1,
        stream_overlap_tokens=2,
        stream_crossfade_samples=2,
    )
    assert first_chunk is not None
    assert first_chunk["audio_data"] == [0.0, 1.0, 2.0, 3.0]

    second_chunk = build_stream_vocoder_chunk(
        payload,
        torch.tensor([[7], [8]]),
        codec=_FakeCodec(),
        device="cpu",
        stream_stride=1,
        stream_followup_stride=1,
        stream_overlap_tokens=2,
        stream_crossfade_samples=2,
    )
    assert second_chunk is None

    final_chunk = flush_stream_vocoder_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
        stream_crossfade_samples=2,
    )
    assert final_chunk is not None
    assert final_chunk["audio_data"] == [4.0, 5.0]


def test_incremental_chunk_builder_bounds_retained_code_history() -> None:
    payload = StagePayload(
        request_id="req-bounded",
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={},
    )

    first_codes = torch.tensor([[1, 2], [3, 4]])
    first_chunk = build_stream_vocoder_chunk(
        payload,
        first_codes,
        codec=_FakeCodec(),
        device="cpu",
        stream_stride=2,
        stream_followup_stride=2,
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    assert first_chunk is not None
    state = load_stream_vocoder_state(payload)
    assert state is not None
    assert state["last_vocode_tokens"] == 2
    assert state["code_start_token"] == 1
    assert state["total_tokens"] == 2

    second_codes = torch.tensor([[5, 6], [7, 8]])
    second_chunk = build_stream_vocoder_chunk(
        payload,
        second_codes,
        codec=_FakeCodec(),
        device="cpu",
        stream_stride=2,
        stream_followup_stride=2,
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    assert second_chunk is not None
    assert second_chunk["audio_data"] == [2.0, 3.0, 4.0, 5.0]
    state = load_stream_vocoder_state(payload)
    assert state is not None
    assert state["last_vocode_tokens"] == 4
    assert state["code_start_token"] == 3
    assert state["total_tokens"] == 4
    retained_codes = state["codes"]
    assert len(retained_codes) == 1
    assert retained_codes[0].tolist() == [[6], [8]]


def test_tts_engine_executor_flush_builder_calls_streaming_vocoder_flush(
    monkeypatch,
) -> None:
    class _FakeServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _fake_load_audio_decoder(model_path: str, device: str):
        del model_path, device
        return object(), 10, 4096, object(), "/tmp/fake-model"

    def _fake_load_codec(checkpoint_dir: str, device: str):
        del checkpoint_dir, device
        return _FactoryCodec()

    def _fake_warmup_codec(codec, *, num_codebooks: int, device: str) -> None:
        del codec, num_codebooks, device

    def _fake_patch_config(model_path: str) -> None:
        del model_path

    class _DummyEngine:
        pass

    def _fake_create_engine(**kwargs):
        del kwargs
        return _DummyEngine()

    flush_calls: list[dict] = []

    def _fake_flush_stream_vocoder_chunk(
        payload: StagePayload,
        *,
        codec,
        device: str,
        stream_overlap_tokens: int,
        stream_crossfade_samples: int,
    ) -> dict[str, object]:
        flush_calls.append(
            {
                "payload": payload,
                "codec": codec,
                "device": device,
                "stream_overlap_tokens": stream_overlap_tokens,
                "stream_crossfade_samples": stream_crossfade_samples,
            }
        )
        return {
            "audio_data": [0.0],
            "sample_rate": codec.sample_rate,
            "modality": "audio",
        }

    monkeypatch.setattr(stages, "_load_audio_decoder", _fake_load_audio_decoder)
    monkeypatch.setattr(stages, "_load_codec", _fake_load_codec)
    monkeypatch.setattr(stages, "_warmup_codec", _fake_warmup_codec)
    monkeypatch.setattr(
        stages,
        "flush_stream_vocoder_chunk",
        _fake_flush_stream_vocoder_chunk,
    )
    monkeypatch.setattr(
        stages.torch.cuda,
        "mem_get_info",
        lambda gpu_id: (8 * 1024**3, 0),
    )

    import sglang.srt.server_args as server_args_module

    import sglang_omni.models.fishaudio_s2_pro.factory as factory

    monkeypatch.setattr(server_args_module, "ServerArgs", _FakeServerArgs)
    monkeypatch.setattr(factory, "_patch_fish_config_for_sglang", _fake_patch_config)
    monkeypatch.setattr(factory, "create_s2pro_sglang_engine", _fake_create_engine)

    executor = stages.create_sglang_tts_engine_executor(
        model_path="unused",
        device="cuda:0",
        stream_overlap_tokens=20,
        stream_crossfade_samples=512,
        stream_vocoder_device="cuda:0",
        warmup_stream_codec_on_startup=False,
    )

    payload = StagePayload(
        request_id="req-flush",
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={},
    )
    chunk = executor._stream_builder.flush(payload)

    assert chunk is not None
    assert chunk["audio_data"] == [0.0]
    assert len(flush_calls) == 1
    assert flush_calls[0]["payload"] is payload
    assert isinstance(flush_calls[0]["codec"], _FactoryCodec)
    assert flush_calls[0]["device"] == "cuda:0"
    assert flush_calls[0]["stream_overlap_tokens"] == 20
    assert flush_calls[0]["stream_crossfade_samples"] == 512
