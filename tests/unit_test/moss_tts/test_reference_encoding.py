# SPDX-License-Identifier: Apache-2.0
"""Reference encoding, caching, and lifecycle tests for MOSS-TTS."""

from __future__ import annotations

import base64
import concurrent.futures
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_moss_tts_reference_encoder_uses_shared_audio_loader(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    encoded = torch.tensor([[7, 8]], dtype=torch.long)
    reference_bytes = b"reference audio"
    reference_path = tmp_path / "voice.wav"
    reference_path.write_bytes(reference_bytes)
    load_calls: list[tuple[str, str, int, bool]] = []
    encode_calls: list[tuple[list[tuple[torch.Tensor, int]], int]] = []
    load_threads: list[str] = []
    encode_threads: list[str] = []

    class FakeAudioTokenizer:
        sample_rate = 24000

        def encode_waveforms(self, waveforms, *, num_quantizers):
            encode_threads.append(threading.current_thread().name)
            encode_calls.append((waveforms, num_quantizers))
            return [encoded]

    def fake_load_audio(
        audio_source,
        *,
        source_name,
        target_sample_rate,
        mono,
    ):
        load_threads.append(threading.current_thread().name)
        load_calls.append((audio_source, source_name, target_sample_rate, mono))
        return np.asarray([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    encoder = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=2,
        max_batch_wait_ms=0,
    )
    request.addfinalizer(encoder.close)

    result = encoder.encode(reference_path)

    assert result is encoded
    assert load_calls == [(str(reference_path), "MOSS-TTS reference", 24000, True)]
    assert len(encode_calls) == 1
    waveforms, num_quantizers = encode_calls[0]
    assert num_quantizers == 2
    assert len(waveforms) == 1
    waveform, sample_rate = waveforms[0]
    assert waveform.tolist() == pytest.approx([0.1, 0.2])
    assert sample_rate == 24000
    assert load_threads == [threading.current_thread().name]
    assert encode_threads == ["moss-tts-ref-encode"]


def test_moss_tts_slow_reference_load_does_not_block_ready_reference(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    from sglang_omni.models.moss_tts import stages

    slow_started = threading.Event()
    release_slow = threading.Event()
    results: dict[str, torch.Tensor | BaseException] = {}
    slow_source = "https://slow.test/ref.wav"
    fast_source = "https://fast.test/ref.wav"

    class FakeAudioTokenizer:
        sample_rate = 24000

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == 2
            return [
                torch.tensor([[int(waveform[0].item())]], dtype=torch.long)
                for waveform, _ in waveforms
            ]

    def fake_load_audio(source, **kwargs):
        del kwargs
        if source == slow_source:
            slow_started.set()
            assert release_slow.wait(timeout=5)
            return np.asarray([1.0], dtype=np.float32)
        return np.asarray([2.0], dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    encoder = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=2,
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    request.addfinalizer(encoder.close)

    def run(path: str) -> None:
        try:
            results[path] = encoder.encode(path)
        except BaseException as exc:
            results[path] = exc

    slow = threading.Thread(target=run, args=(slow_source,))
    fast = threading.Thread(target=run, args=(fast_source,))
    slow.start()
    assert slow_started.wait(timeout=5)
    fast.start()
    try:
        fast.join(timeout=2)
        assert not fast.is_alive()
        assert int(results[fast_source][0, 0]) == 2
    finally:
        release_slow.set()
        slow.join(timeout=5)
        fast.join(timeout=5)

    assert int(results[slow_source][0, 0]) == 1


def test_moss_tts_batch_wait_uses_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    entries = [(object(), object()), (object(), object())]

    class FakeQueue:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def get(self, timeout=None):
            self.timeouts.append(timeout)
            if entries:
                return entries.pop(0)
            raise stages.queue.Empty

    fake_queue = FakeQueue()
    encoder = object.__new__(stages._BatchedReferenceEncoder)
    encoder._max_batch_size = 8
    encoder._max_wait_s = 0.004
    encoder._queue = fake_queue
    clock = iter((10.0, 10.001, 10.003))
    monkeypatch.setattr(stages.time, "monotonic", lambda: next(clock))

    batch, shutdown = encoder._drain_batch()

    assert len(batch) == 2
    assert shutdown is False
    assert fake_queue.timeouts[0] is None
    assert fake_queue.timeouts[1:] == pytest.approx([0.003, 0.001])


def test_moss_tts_batched_reference_encoder_close_is_idempotent() -> None:
    from sglang_omni.models.moss_tts import stages

    codec = SimpleNamespace(sample_rate=24000)
    encoder = stages._BatchedReferenceEncoder(codec, n_vq=2)
    worker = encoder._thread

    encoder.close()
    encoder.close()

    assert not worker.is_alive()
    with pytest.raises(RuntimeError, match="reference encoder is closed"):
        encoder.encode("unused.wav")


def test_moss_tts_preprocessing_context_closes_replaced_encoder() -> None:
    from sglang_omni.models.moss_tts import request_builders as rb
    from sglang_omni.models.moss_tts import stages

    codec = SimpleNamespace(sample_rate=24000, device="cuda:0", model=None)
    first = stages._BatchedReferenceEncoder(codec, n_vq=2)
    second_batcher = stages._BatchedReferenceEncoder(codec, n_vq=2)
    second = stages._MossTTSReferenceEncoder(
        second_batcher,
        codec_model_path="codec",
        n_vq=2,
    )

    try:
        rb.set_moss_tts_preprocessing_context(
            processor=object(), reference_encoder=first
        )
        rb.set_moss_tts_preprocessing_context(
            processor=object(), reference_encoder=second
        )
        assert not first._thread.is_alive()
        assert second_batcher._thread.is_alive()
    finally:
        rb.clear_moss_tts_preprocessing_context()

    assert not second_batcher._thread.is_alive()


def _make_moss_tts_wav_data_uri(
    n_samples: int = 100,
    sample_rate: int = 16000,
) -> tuple[str, bytes]:
    samples = b"\x00\x00" * n_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(samples),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(samples),
    )
    raw = header + samples
    return f"data:audio/wav;base64,{base64.b64encode(raw).decode()}", raw


def test_moss_tts_path_file_uri_and_data_uri_use_shared_audio_loader(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    data_uri, raw = _make_moss_tts_wav_data_uri()
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(raw)
    encoded_waveforms: list[tuple[torch.Tensor, int]] = []

    class FakeAudioTokenizer:
        sample_rate = 16000

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == 2
            encoded_waveforms.extend(waveforms)
            return [torch.full((2, 2), 7, dtype=torch.long) for _ in waveforms]

    encoder = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=2,
        max_batch_wait_ms=0,
    )
    request.addfinalizer(encoder.close)

    path_result = encoder.encode(reference_path)
    file_uri_result = encoder.encode(reference_path.as_uri())
    data_uri_result = encoder.encode(data_uri)

    expected = torch.full((2, 2), 7, dtype=torch.long)
    assert torch.equal(path_result, expected)
    assert torch.equal(file_uri_result, expected)
    assert torch.equal(data_uri_result, expected)
    assert len(encoded_waveforms) == 3
    for waveform, sample_rate in encoded_waveforms:
        assert sample_rate == 16000
        assert waveform.shape == (100,)
        assert torch.count_nonzero(waveform) == 0


def test_moss_tts_batched_reference_encoder_coalesces_and_isolates_errors(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    calls: list[list[int]] = []
    loaded_sources: list[str] = []
    paths = {name: tmp_path / f"{name}.wav" for name in ("aa", "bbb", "bad1")}
    for name, path in paths.items():
        path.write_bytes(name.encode())

    class FakeAudioTokenizer:
        sample_rate = 24000
        device = "cuda:0"

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == 2
            lengths = [int(wav.shape[-1]) for wav, _ in waveforms]
            calls.append(lengths)
            if 4 in lengths:
                raise RuntimeError("cannot encode bad reference")
            return [torch.full((length, 2), length) for length in lengths]

    def fake_load_audio(source, **kwargs):
        del kwargs
        name = Path(source).read_text()
        loaded_sources.append(name)
        length = {"aa": 2, "bbb": 3, "bad1": 4}[name]
        return np.full(length, length, dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    encoder = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=2,
        max_batch_size=4,
        max_batch_wait_ms=20,
    )
    request.addfinalizer(encoder.close)
    results: dict[str, torch.Tensor | BaseException] = {}

    def run(path: str) -> None:
        try:
            results[path] = encoder.encode(paths[path])
        except BaseException as exc:
            results[path] = exc

    threads = [
        threading.Thread(target=run, args=(path,)) for path in ("aa", "bbb", "bad1")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert isinstance(results["bad1"], BaseException)
    assert int(results["aa"][0, 0]) == 2
    assert int(results["bbb"][0, 0]) == 3
    assert sorted(loaded_sources) == ["aa", "bad1", "bbb"]
    assert any(len(call) > 1 for call in calls)


def test_moss_tts_batched_reference_encoder_deduplicates_same_batch(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    reference_path = tmp_path / "same.wav"
    reference_path.write_bytes(b"same reference")
    loaded_sources: list[str] = []
    encode_batch_sizes: list[int] = []

    class FakeAudioTokenizer:
        sample_rate = 24000
        device = "cuda:0"

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == 2
            encode_batch_sizes.append(len(waveforms))
            return [torch.full((3, 2), 7) for _ in waveforms]

    def fake_load_audio(source, **kwargs):
        del kwargs
        loaded_sources.append(str(source))
        return np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    encoder = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=2,
        max_batch_size=2,
        max_batch_wait_ms=100,
    )
    request.addfinalizer(encoder.close)
    barrier = threading.Barrier(2)

    def run() -> torch.Tensor:
        barrier.wait(timeout=5)
        return encoder.encode(reference_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(timeout=10) for future in (pool.submit(run), pool.submit(run))
        ]

    assert all(torch.equal(result, torch.full((3, 2), 7)) for result in results)
    assert loaded_sources == [str(reference_path), str(reference_path)]
    assert encode_batch_sizes == [1]


def test_moss_tts_path_replacement_keeps_content_versions_isolated(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"1")
    old_load_started = threading.Event()
    release_old_load = threading.Event()
    load_calls: list[int] = []

    class FakeAudioTokenizer:
        sample_rate = 1
        device = "cuda:0"
        model = None

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == 1
            return [
                torch.tensor([[int(waveform[0].item())]], dtype=torch.long)
                for waveform, _ in waveforms
            ]

    def fake_load_audio(source, **kwargs):
        del kwargs
        value = int(Path(source).read_text())
        load_calls.append(value)
        if value == 1:
            old_load_started.set()
            assert release_old_load.wait(timeout=5)
        return np.asarray([value], dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    batcher = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=1,
        max_batch_size=2,
        max_batch_wait_ms=0,
    )
    encoder = stages._MossTTSReferenceEncoder(
        batcher,
        codec_model_path="codec",
        n_vq=1,
        max_items=8,
        max_bytes=4096,
    )
    request.addfinalizer(encoder.close)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        old_result = pool.submit(encoder.encode, reference_path)
        assert old_load_started.wait(timeout=5)
        reference_path.write_bytes(b"2")
        new_result = pool.submit(encoder.encode, reference_path)
        try:
            assert int(new_result.result(timeout=2)[0, 0]) == 2
        finally:
            release_old_load.set()
        assert int(old_result.result(timeout=5)[0, 0]) == 1

    assert int(encoder.encode(reference_path)[0, 0]) == 2
    assert load_calls == [1, 2, 2]
    assert encoder.stats()["hits"] == 1
    assert encoder.stats()["misses"] == 2
    assert encoder.stats()["entries"] == 2


def test_moss_tts_url_cache_tracks_fetched_waveform(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    from sglang_omni.models.moss_tts import stages

    loaded_values = iter((1.0, 2.0, 2.0))
    encode_count = 0

    class FakeAudioTokenizer:
        sample_rate = 1
        device = "cuda:0"
        model = None

        def encode_waveforms(self, waveforms, *, num_quantizers):
            nonlocal encode_count
            assert num_quantizers == 1
            encode_count += 1
            return [
                torch.tensor([[int(waveform[0].item())]], dtype=torch.long)
                for waveform, _ in waveforms
            ]

    def fake_load_audio(source, **kwargs):
        del kwargs
        assert source == "https://example.test/reference.wav"
        return np.asarray([next(loaded_values)], dtype=np.float32)

    monkeypatch.setattr(stages, "load_audio", fake_load_audio)
    batcher = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=1,
        max_batch_wait_ms=0,
    )
    encoder = stages._MossTTSReferenceEncoder(
        batcher,
        codec_model_path="codec",
        n_vq=1,
        max_items=8,
        max_bytes=4096,
    )
    request.addfinalizer(encoder.close)

    assert int(encoder.encode("https://example.test/reference.wav")[0, 0]) == 1
    assert int(encoder.encode("https://example.test/reference.wav")[0, 0]) == 2
    assert int(encoder.encode("https://example.test/reference.wav")[0, 0]) == 2
    assert encode_count == 2
    assert encoder.stats()["hits"] == 1
    assert encoder.stats()["misses"] == 2
    assert encoder.stats()["entries"] == 2


def test_moss_tts_cached_reference_encoder_uses_loaded_waveform_identity(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    from sglang_omni.models.moss_tts import stages

    data_uri, raw = _make_moss_tts_wav_data_uri()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(raw)
    load_calls: list[str] = []
    encode_calls: list[list[tuple[torch.Tensor, int]]] = []
    real_load_audio = stages.load_audio

    class FakeAudioTokenizer:
        sample_rate = 16000
        device = "cuda:0"
        model = None

        def encode_waveforms(self, waveforms, *, num_quantizers):
            assert num_quantizers == 2
            encode_calls.append(waveforms)
            return [torch.full((4, 2), 99, dtype=torch.long)]

    def counting_load_audio(source, **kwargs):
        load_calls.append(str(source))
        return real_load_audio(source, **kwargs)

    monkeypatch.setattr(stages, "load_audio", counting_load_audio)

    batcher = stages._BatchedReferenceEncoder(
        FakeAudioTokenizer(),
        n_vq=2,
        max_batch_wait_ms=0,
    )
    encoder = stages._MossTTSReferenceEncoder(
        batcher,
        codec_model_path="codec",
        n_vq=2,
        max_items=8,
        max_bytes=4096,
    )
    request.addfinalizer(encoder.close)

    first_path = encoder.encode(ref)
    first_path.fill_(-1)
    second_path = encoder.encode(ref)
    first_uri = encoder.encode(data_uri)
    first_uri.fill_(-1)
    second_uri = encoder.encode(data_uri)

    assert torch.all(second_path == 99)
    assert torch.all(second_uri == 99)
    assert load_calls == [str(ref), str(ref), data_uri, data_uri]
    assert len(encode_calls) == 1
    for waveforms in encode_calls:
        assert len(waveforms) == 1
        waveform, sample_rate = waveforms[0]
        assert sample_rate == 16000
        assert waveform.shape == (100,)
    assert encoder.stats()["hits"] == 3
    assert encoder.stats()["misses"] == 1
    assert encoder.stats()["entries"] == 1


def test_moss_tts_cached_reference_encoder_merges_inflight(tmp_path: Path) -> None:
    from sglang_omni.models.moss_tts import stages

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"reference bytes")
    entered = threading.Event()
    release = threading.Event()
    encode_count = 0

    class FakeBatched:
        class AudioTokenizer:
            sample_rate = 24000
            device = "cuda:0"
            model = None

        _audio_encoder = AudioTokenizer()

        @staticmethod
        def load(source):
            del source
            return stages._LoadedReferenceWaveform(
                torch.zeros(4, dtype=torch.float32),
                24000,
                "waveform:shared",
            )

        def encode_input(self, item):
            nonlocal encode_count
            del item
            encode_count += 1
            entered.set()
            assert release.wait(timeout=5)
            return torch.ones((4, 2), dtype=torch.long)

    encoder = stages._MossTTSReferenceEncoder(
        FakeBatched(),
        codec_model_path="codec",
        n_vq=2,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(encoder.encode, ref)
        assert entered.wait(timeout=5)
        second = pool.submit(encoder.encode, ref)
        deadline = time.monotonic() + 5
        while encoder.stats()["merged"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            assert encoder.stats()["merged"] == 1
        finally:
            release.set()
        assert torch.equal(first.result(timeout=5), second.result(timeout=5))

    assert encode_count == 1
    assert encoder.stats()["merged"] == 1
