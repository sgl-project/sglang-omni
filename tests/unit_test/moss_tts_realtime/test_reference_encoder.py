# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import concurrent.futures
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_tts_realtime import reference_encoder
from sglang_omni.models.moss_tts_realtime.reference_encoder import (
    BatchedMossTTSRealtimeAudioEncoder,
    MossTTSRealtimeAudioEncoder,
    MossTTSRealtimeReferenceEncoder,
)

N_VQ = 16


class _FakeCodec:
    config = SimpleNamespace(sampling_rate=24000, downsample_rate=1)

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], int | None]] = []

    def batch_encode(
        self,
        waveforms: list[torch.Tensor],
        *,
        num_quantizers: int | None = None,
    ) -> Any:
        lengths = [int(waveform.numel()) for waveform in waveforms]
        self.calls.append((lengths, num_quantizers))
        quantizers = int(num_quantizers or 32)
        codes = torch.zeros(
            quantizers,
            len(waveforms),
            max(lengths),
            dtype=torch.long,
        )
        for index, length in enumerate(lengths):
            codes[:, index, :length] = length
        return SimpleNamespace(
            audio_codes=codes,
            audio_codes_lengths=torch.tensor(lengths, dtype=torch.long),
        )


def test_audio_encoder_uses_codec_tensor_contract() -> None:
    calls: list[tuple[torch.Tensor, dict[str, Any]]] = []
    output = SimpleNamespace(audio_codes=torch.zeros((32, 1, 4), dtype=torch.long))

    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000, downsample_rate=8)

        def encode(self, values: torch.Tensor, **kwargs: Any) -> Any:
            calls.append((values.detach().clone(), kwargs))
            return output

    encoder = MossTTSRealtimeAudioEncoder(FakeCodec(), device="cpu")
    stereo = np.stack([np.ones(32, dtype=np.float32), np.zeros(32, dtype=np.float32)])

    result = encoder.encode(stereo)

    assert result.shape == (4, 32)
    assert torch.equal(result, torch.zeros((4, 32), dtype=torch.long))
    assert len(calls) == 1
    values, kwargs = calls[0]
    assert values.shape == (1, 32)
    assert values.dtype == torch.float32
    assert torch.equal(values, torch.full((1, 32), 0.5))
    assert kwargs == {"return_dict": True}


def test_audio_encoder_normalizes_base64_mapping(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_load_audio(source: Any, **kwargs: Any) -> np.ndarray:
        seen["source"] = source
        seen["kwargs"] = kwargs
        return np.arange(12, dtype=np.float32)

    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000, downsample_rate=12)

        def encode(self, values: torch.Tensor, **kwargs: Any) -> Any:
            seen["values"] = values.detach().clone()
            seen["encode_kwargs"] = kwargs
            return SimpleNamespace(
                audio_codes=torch.zeros((32, 1, 1), dtype=torch.long)
            )

    monkeypatch.setattr(reference_encoder, "load_audio", fake_load_audio)
    encoder = MossTTSRealtimeAudioEncoder(FakeCodec(), device="cpu")

    encoder.encode({"base64": "ZmFrZQ==", "media_type": "audio/flac"})

    assert seen["source"] == "data:audio/flac;base64,ZmFrZQ=="
    assert seen["kwargs"] == {
        "source_name": "MOSS-TTS-Realtime audio",
        "target_sample_rate": 24000,
        "mono": True,
    }
    assert seen["values"].shape == (1, 12)
    assert seen["encode_kwargs"] == {"return_dict": True}


@pytest.mark.parametrize(
    ("waveform_samples", "raw_frames", "reported_frames"),
    [
        (8, 2, 2),
        (9, 3, 2),
    ],
)
def test_audio_encoder_uses_upstream_ceil_prompt_length(
    waveform_samples: int,
    raw_frames: int,
    reported_frames: int,
) -> None:
    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000, downsample_rate=4)

        def encode(self, values: torch.Tensor, **kwargs: Any) -> Any:
            assert values.shape == (1, waveform_samples)
            assert kwargs == {"return_dict": True, "num_quantizers": N_VQ}
            codes = torch.arange(raw_frames, dtype=torch.long).view(1, 1, -1)
            return SimpleNamespace(
                audio_codes=codes.expand(N_VQ, 1, -1).contiguous(),
                audio_codes_lengths=torch.tensor([reported_frames]),
            )

    encoder = MossTTSRealtimeAudioEncoder(
        FakeCodec(),
        device="cpu",
        num_quantizers=N_VQ,
    )

    result = encoder.encode(torch.ones(waveform_samples))

    assert result.shape == (raw_frames, N_VQ)
    assert torch.equal(result[:, 0], torch.arange(raw_frames))


def test_audio_encoder_uses_per_item_ceil_lengths_for_mixed_batch() -> None:
    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000, downsample_rate=4)

        def batch_encode(
            self,
            waveforms: list[torch.Tensor],
            *,
            num_quantizers: int | None = None,
        ) -> Any:
            assert [int(waveform.numel()) for waveform in waveforms] == [8, 9, 17]
            assert num_quantizers == N_VQ
            codes = torch.zeros(N_VQ, 3, 5, dtype=torch.long)
            for batch_index in range(3):
                codes[:, batch_index] = 100 * batch_index + torch.arange(
                    5, dtype=torch.long
                )
            return SimpleNamespace(
                audio_codes=codes,
                audio_codes_lengths=torch.tensor([2, 2, 4]),
            )

    encoder = MossTTSRealtimeAudioEncoder(
        FakeCodec(),
        device="cpu",
        num_quantizers=N_VQ,
    )

    results = encoder.encode_waveforms(
        [
            torch.ones(8),
            torch.ones(9),
            torch.ones(17),
        ]
    )

    assert [tuple(result.shape) for result in results] == [
        (2, N_VQ),
        (3, N_VQ),
        (5, N_VQ),
    ]
    assert [int(result[-1, 0]) for result in results] == [1, 102, 204]


def test_audio_encoder_rejects_codec_with_insufficient_raw_frames() -> None:
    class FakeCodec:
        config = SimpleNamespace(sampling_rate=24000, downsample_rate=4)

        def encode(self, values: torch.Tensor, **kwargs: Any) -> Any:
            del values, kwargs
            return SimpleNamespace(
                audio_codes=torch.zeros(N_VQ, 1, 2, dtype=torch.long),
                audio_codes_lengths=torch.tensor([2]),
            )

    encoder = MossTTSRealtimeAudioEncoder(
        FakeCodec(),
        device="cpu",
        num_quantizers=N_VQ,
    )

    with pytest.raises(RuntimeError, match="fewer raw frames"):
        encoder.encode(torch.ones(9))


def test_codec_adapter_batch_encode_uses_requested_quantizers_and_lengths() -> None:
    codec = _FakeCodec()
    encoder = MossTTSRealtimeAudioEncoder(
        codec,
        device="cpu",
        num_quantizers=N_VQ,
    )

    results = encoder.encode_waveforms([torch.ones(3), torch.ones(5)])

    assert codec.calls == [([3, 5], N_VQ)]
    assert [tuple(result.shape) for result in results] == [(3, N_VQ), (5, N_VQ)]
    assert torch.all(results[0] == 3)
    assert torch.all(results[1] == 5)


def test_batched_encoder_coalesces_concurrent_misses() -> None:
    codec = _FakeCodec()
    encoder = BatchedMossTTSRealtimeAudioEncoder(
        MossTTSRealtimeAudioEncoder(
            codec,
            device="cpu",
            num_quantizers=N_VQ,
        ),
        max_batch_size=3,
        max_batch_wait_ms=50,
    )
    barrier = threading.Barrier(3)
    results: dict[int, torch.Tensor] = {}

    def run(length: int) -> None:
        barrier.wait(timeout=5)
        results[length] = encoder.encode(np.ones(length, dtype=np.float32))

    threads = [threading.Thread(target=run, args=(length,)) for length in (3, 5, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert len(codec.calls) == 1
    assert sorted(codec.calls[0][0]) == [3, 5, 7]
    assert codec.calls[0][1] == N_VQ
    for length, result in results.items():
        assert result.shape == (length, N_VQ)
        assert torch.all(result == length)


def test_batched_encoder_uses_one_collection_deadline(monkeypatch) -> None:
    jobs = [(torch.ones(length), concurrent.futures.Future()) for length in (3, 5, 7)]

    class _FakeQueue:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def get(self, timeout: float | None = None):
            self.timeouts.append(timeout)
            return jobs.pop(0)

    encoder = object.__new__(BatchedMossTTSRealtimeAudioEncoder)
    encoder._queue = _FakeQueue()
    encoder._max_batch_size = 4
    encoder._max_wait_s = 0.004
    timestamps = iter((10.0, 10.001, 10.003, 10.004))
    monkeypatch.setattr(reference_encoder.time, "monotonic", lambda: next(timestamps))

    batch = encoder._drain_batch()

    assert len(batch) == 3
    assert encoder._queue.timeouts[0] is None
    assert encoder._queue.timeouts[1:] == pytest.approx([0.003, 0.001])


def test_batched_encoder_retries_per_item_after_batch_failure() -> None:
    class _FailingCodec(_FakeCodec):
        def batch_encode(
            self,
            waveforms: list[torch.Tensor],
            *,
            num_quantizers: int | None = None,
        ) -> Any:
            lengths = [int(waveform.numel()) for waveform in waveforms]
            self.calls.append((lengths, num_quantizers))
            if len(waveforms) > 1:
                raise RuntimeError("batch failed")
            if lengths[0] == 4:
                raise RuntimeError("bad reference")
            return super().batch_encode(
                waveforms,
                num_quantizers=num_quantizers,
            )

    codec = _FailingCodec()
    encoder = BatchedMossTTSRealtimeAudioEncoder(
        MossTTSRealtimeAudioEncoder(
            codec,
            device="cpu",
            num_quantizers=N_VQ,
        ),
        max_batch_size=2,
        max_batch_wait_ms=50,
    )
    barrier = threading.Barrier(2)
    results: dict[int, torch.Tensor | BaseException] = {}

    def run(length: int) -> None:
        try:
            barrier.wait(timeout=5)
            results[length] = encoder.encode(np.ones(length, dtype=np.float32))
        except BaseException as exc:
            results[length] = exc

    threads = [threading.Thread(target=run, args=(length,)) for length in (3, 4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert isinstance(results[4], RuntimeError)
    assert isinstance(results[3], torch.Tensor)
    assert results[3].shape == (3, N_VQ)
    assert any(len(lengths) == 2 for lengths, _ in codec.calls)


class _RecordingBatchedEncoder:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def encode(self, value: Any) -> torch.Tensor:
        self.calls.append(value)
        return torch.full((4, N_VQ), 17, dtype=torch.long)


def _cached_encoder(
    encoder: _RecordingBatchedEncoder,
) -> MossTTSRealtimeReferenceEncoder:
    return MossTTSRealtimeReferenceEncoder(
        encoder,
        model_revision="codec-revision",
        num_quantizers=N_VQ,
        max_items=8,
        max_bytes=1 << 20,
    )


def test_reference_cache_hit_isolated_return_and_file_invalidation(tmp_path) -> None:
    path = tmp_path / "voice.wav"
    path.write_bytes(b"voice-v1")
    underlying = _RecordingBatchedEncoder()
    encoder = _cached_encoder(underlying)

    miss = encoder.encode(str(path))
    hit = encoder.encode(str(path))
    hit.fill_(-1)
    isolated_hit = encoder.encode(str(path))

    assert len(underlying.calls) == 1
    assert torch.equal(miss, torch.full((4, N_VQ), 17, dtype=torch.long))
    assert torch.equal(isolated_hit, torch.full((4, N_VQ), 17, dtype=torch.long))
    assert encoder.stats()["hits"] == 2

    path.write_bytes(b"voice-v2-with-different-content")
    encoder.encode(str(path))

    assert len(underlying.calls) == 2
    assert encoder.stats()["misses"] == 2


def test_reference_cache_merges_same_key_concurrent_encode(tmp_path) -> None:
    path = tmp_path / "voice.wav"
    path.write_bytes(b"same voice")
    started = threading.Event()
    release = threading.Event()

    class _BlockingEncoder(_RecordingBatchedEncoder):
        def encode(self, value: Any) -> torch.Tensor:
            self.calls.append(value)
            started.set()
            assert release.wait(timeout=5)
            return torch.full((4, N_VQ), 23, dtype=torch.long)

    underlying = _BlockingEncoder()
    encoder = _cached_encoder(underlying)
    results: list[torch.Tensor] = []

    first = threading.Thread(target=lambda: results.append(encoder.encode(str(path))))
    second = threading.Thread(target=lambda: results.append(encoder.encode(str(path))))
    first.start()
    assert started.wait(timeout=5)
    second.start()
    deadline = time.monotonic() + 5
    while encoder.stats()["merged"] != 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert len(underlying.calls) == 1
    assert len(results) == 2
    assert encoder.stats()["merged"] == 1
    assert all(torch.all(result == 23) for result in results)


def test_reference_cache_data_uri_uses_decoded_bytes_identity() -> None:
    raw = b"RIFF-fake-audio-bytes"
    data_uri = f"data:audio/wav;base64,{base64.b64encode(raw).decode()}"
    underlying = _RecordingBatchedEncoder()
    encoder = _cached_encoder(underlying)

    encoder.encode(data_uri)
    encoder.encode(data_uri)

    assert underlying.calls == [raw]
    assert encoder.stats()["hits"] == 1
    assert encoder.stats()["misses"] == 1


def test_reference_cache_does_not_cache_remote_urls() -> None:
    underlying = _RecordingBatchedEncoder()
    encoder = _cached_encoder(underlying)

    encoder.encode("https://example.com/voice.wav")
    encoder.encode("https://example.com/voice.wav")

    assert len(underlying.calls) == 2
    assert encoder.stats()["uncacheable"] == 2


@pytest.mark.parametrize(
    ("source", "expected_note", "redacted"),
    [
        (
            "data:audio/wav;base64,U0VDUkVULUFVRElP",
            "Reference encode context: data-URI",
            "U0VDUkVULUFVRElP",
        ),
        (
            "https://example.com/voice.wav?token=secret#fragment",
            "Reference encode context: 'https://example.com/voice.wav'",
            "secret",
        ),
    ],
)
def test_reference_encode_error_context_redacts_payload_secrets(
    source: str,
    expected_note: str,
    redacted: str,
) -> None:
    class _FailingEncoder(_RecordingBatchedEncoder):
        def encode(self, value: Any) -> torch.Tensor:
            del value
            raise RuntimeError("encode failed")

    encoder = _cached_encoder(_FailingEncoder())

    with pytest.raises(RuntimeError, match="encode failed") as exc_info:
        encoder.encode(source)

    notes = getattr(exc_info.value, "__notes__", [])
    assert notes == [expected_note]
    assert redacted not in " ".join(notes)


def test_audio_encoder_rejects_oversized_waveform_before_codec() -> None:
    codec = _FakeCodec()
    encoder = MossTTSRealtimeAudioEncoder(
        codec,
        device="cpu",
        num_quantizers=N_VQ,
    )

    with pytest.raises(ValueError, match="100"):
        encoder.prepare_waveform(torch.zeros(100 * 24000 + 1))

    assert codec.calls == []
