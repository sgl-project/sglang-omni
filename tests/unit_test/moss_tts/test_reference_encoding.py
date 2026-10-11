# SPDX-License-Identifier: Apache-2.0
"""Shared MOSS reference input, caching, execution, and lifecycle contracts."""

from __future__ import annotations

import base64
import concurrent.futures
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.models.moss_tts import reference_encoder
from sglang_omni.models.moss_tts import request_builders as delay_requests
from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioEncoder
from sglang_omni.models.moss_tts.reference_encoder import MossReferenceEncoder
from sglang_omni.models.moss_tts_local import request_builders as local_requests


class RecordingEncoder:
    def __init__(self, channels: int = 1) -> None:
        self.sample_rate = 8000
        self.number_channels = channels
        self.device = "cpu"
        self.model = SimpleNamespace(encoder_dtype=torch.float32)
        self.calls: list[list[torch.Tensor]] = []
        self.threads: list[int] = []
        self.entered = threading.Event()
        self.release: threading.Event | None = None
        self.fail_negative = False

    def encode_waveforms(
        self, waveforms: list[tuple[torch.Tensor, int]], *, num_quantizers: int
    ) -> list[torch.Tensor]:
        self.calls.append([waveform.clone() for waveform, _ in waveforms])
        self.threads.append(threading.get_ident())
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        else:
            pass
        outputs = []
        for waveform, sample_rate in waveforms:
            assert sample_rate == self.sample_rate
            if self.fail_negative and waveform.flatten()[0] < 0:
                raise ValueError("bad reference")
            else:
                pass
            outputs.append(
                torch.full((4, num_quantizers), int(waveform.flatten()[0] * 1000))
            )
        return outputs


def make_encoder(
    request: pytest.FixtureRequest,
    codec: RecordingEncoder | MossAudioEncoder,
    *,
    cache_enabled: bool = True,
    batch_size: int = 8,
    wait_ms: int = 4,
) -> MossReferenceEncoder:
    encoder = MossReferenceEncoder(
        codec,
        codec_model_path="test-codec",
        n_vq=2,
        max_batch_size=batch_size,
        max_batch_wait_ms=wait_ms,
        cache_enabled=cache_enabled,
        max_items=8,
        max_bytes=4096,
    )
    request.addfinalizer(encoder.close)
    return encoder


def audio_source(
    path: Path, *, amplitude: float = 0.25, channels: int = 1, samples: int = 80
) -> str:
    waveform = np.full((samples, channels), amplitude, dtype=np.float32)
    if channels == 2:
        waveform[:, 1] *= -1
    else:
        pass
    sf.write(path, waveform, 8000, subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("channels", [1, 2])
def test_reference_sources_preserve_channels_and_use_worker(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    cache_enabled: bool,
    channels: int,
) -> None:
    path = tmp_path / "reference.wav"
    uri = audio_source(path, channels=channels)
    codec = RecordingEncoder(channels)
    encoder = make_encoder(request, codec, cache_enabled=cache_enabled, batch_size=1)
    for source in [path, path.as_uri(), uri]:
        codes = encoder.encode(source)
        assert codes.dtype == torch.long
        assert codes.shape == (4, 2)
        assert torch.all(codes == 250)
    assert len(codec.calls) == (2 if cache_enabled else 3)
    assert all(thread != threading.get_ident() for thread in codec.threads)
    for call in codec.calls:
        waveform = call[0]
        assert waveform.shape == ((80,) if channels == 1 else (2, 80))
        if channels == 2:
            torch.testing.assert_close(waveform[0], -waveform[1])
        else:
            pass


def test_cache_hit_skips_audio_loading_and_returns_owned_codes(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.wav"
    uri = audio_source(first)
    second = tmp_path / "second.wav"
    second.write_bytes(first.read_bytes())
    codec = RecordingEncoder()
    encoder = make_encoder(request, codec)
    load_count = 0
    original = reference_encoder.load_audio

    def load(
        source: str | bytes, *, source_name: str, target_sample_rate: int, mono: bool
    ) -> np.ndarray:
        nonlocal load_count
        load_count += 1
        return original(
            source,
            source_name=source_name,
            target_sample_rate=target_sample_rate,
            mono=mono,
        )

    monkeypatch.setattr(reference_encoder, "load_audio", load)
    encoder.encode(first).fill_(-1)
    assert torch.all(encoder.encode(second) == 250)
    assert torch.all(encoder.encode(first) == 250)
    assert load_count == 1
    assert len(codec.calls) == 1
    encoder.encode(uri)
    encoder.encode(uri)
    assert load_count == 2
    assert encoder.stats()["hits"] == 3


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_reference_preparation_runs_on_encoding_worker(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_enabled: bool,
    batch_size: int,
) -> None:
    path = tmp_path / "reference.wav"
    audio_source(path)
    prepared_threads: list[int] = []
    original = reference_encoder.load_audio

    def load(
        source: str | bytes, *, source_name: str, target_sample_rate: int, mono: bool
    ) -> np.ndarray:
        prepared_threads.append(threading.get_ident())
        return original(
            source,
            source_name=source_name,
            target_sample_rate=target_sample_rate,
            mono=mono,
        )

    monkeypatch.setattr(reference_encoder, "load_audio", load)
    codec = RecordingEncoder()
    encoder = make_encoder(
        request, codec, cache_enabled=cache_enabled, batch_size=batch_size
    )
    assert torch.all(encoder.encode(path) == 250)
    assert torch.all(encoder.encode(path) == 250)
    assert len(prepared_threads) == (1 if cache_enabled else 2)
    assert prepared_threads == codec.threads
    assert all(thread != threading.get_ident() for thread in prepared_threads)


def test_file_changes_during_encoding_are_not_cached_under_old_identity(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    path = tmp_path / "reference.wav"
    audio_source(path)
    codec = RecordingEncoder()
    codec.release = threading.Event()
    encoder = make_encoder(request, codec, batch_size=1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(encoder.encode, path)
        assert codec.entered.wait(timeout=5)
        audio_source(path, amplitude=0.5)
        codec.release.set()
        assert torch.all(result.result(timeout=5) == 250)
    assert encoder.stats()["entries"] == 0
    assert torch.all(encoder.encode(path) == 500)
    assert torch.all(encoder.encode(path) == 500)
    assert len(codec.calls) == 2


def test_url_cache_uses_fetched_content(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reference.wav"
    audio_source(path)
    content = path.read_bytes()
    url = "https://reference.test/audio.wav"

    def fetch(source: str, *, timeout: int, follow_redirects: bool) -> httpx.Response:
        assert source == url
        assert timeout > 0 and follow_redirects
        return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(reference_encoder.httpx, "get", fetch)
    codec = RecordingEncoder()
    encoder = make_encoder(request, codec)
    assert torch.all(encoder.encode(url) == 250)
    audio_source(path, amplitude=0.5)
    content = path.read_bytes()
    assert torch.all(encoder.encode(url) == 500)
    assert torch.all(encoder.encode(url) == 500)
    assert len(codec.calls) == 2


def test_identical_concurrent_requests_share_one_encode(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    path = tmp_path / "reference.wav"
    audio_source(path)
    codec = RecordingEncoder()
    codec.release = threading.Event()
    encoder = make_encoder(request, codec)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(encoder.encode, path) for _ in range(4)]
        assert codec.entered.wait(timeout=5)
        codec.release.set()
        for future in futures:
            assert torch.all(future.result(timeout=5) == 250)
    assert len(codec.calls) == 1


@pytest.mark.parametrize("failure_stage", ["preparation", "encoding"])
def test_batch_failure_isolates_bad_input(
    request: pytest.FixtureRequest, tmp_path: Path, failure_stage: str
) -> None:
    sources = []
    for index, amplitude in enumerate([0.25, -0.25, 0.5]):
        path = tmp_path / f"reference{index}.wav"
        samples = 101 * 8000 if index == 1 and failure_stage == "preparation" else 80
        sources.append(audio_source(path, amplitude=amplitude, samples=samples))
    codec = RecordingEncoder()
    codec.fail_negative = True
    encoder = make_encoder(request, codec, wait_ms=30)
    barrier = threading.Barrier(3)

    def run(source: str) -> torch.Tensor:
        barrier.wait(timeout=5)
        return encoder.encode(source)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, source) for source in sources]
        assert torch.all(futures[0].result(timeout=5) == 250)
        message = "limit is 100s" if failure_stage == "preparation" else "bad reference"
        with pytest.raises(ValueError, match=message):
            futures[1].result(timeout=5)
        assert torch.all(futures[2].result(timeout=5) == 500)
    assert encoder.stats()["failed"] == 1


@pytest.mark.parametrize("source_kind", ["path", "data_uri"])
def test_reference_duration_limit(
    request: pytest.FixtureRequest, tmp_path: Path, source_kind: str
) -> None:
    path = tmp_path / "long.wav"
    uri = audio_source(path, samples=101 * 8000)
    codec = RecordingEncoder()
    encoder = make_encoder(request, codec)
    with pytest.raises(ValueError, match="limit is 100s"):
        encoder.encode(path if source_kind == "path" else uri)
    assert not codec.calls
    assert encoder.stats()["entries"] == 0


def test_close_is_idempotent_and_rejects_new_work(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    path = tmp_path / "reference.wav"
    audio_source(path)
    encoder = make_encoder(request, RecordingEncoder())
    encoder.encode(path)
    encoder.close()
    encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.encode(path)


@pytest.mark.parametrize("variant", ["delay", "local"])
def test_context_replacement_closes_old_encoder(
    request: pytest.FixtureRequest, tmp_path: Path, variant: str
) -> None:
    path = tmp_path / "reference.wav"
    audio_source(path)
    first = make_encoder(request, RecordingEncoder())
    second = make_encoder(request, RecordingEncoder())
    if variant == "delay":
        setter = delay_requests.set_moss_tts_preprocessing_context
        clear = delay_requests.clear_moss_tts_preprocessing_context
    else:
        setter = local_requests.set_moss_tts_local_preprocessing_context
        clear = local_requests.clear_moss_tts_local_preprocessing_context
    try:
        setter(processor=None, reference_encoder=first)
        setter(processor=None, reference_encoder=second)
        with pytest.raises(RuntimeError, match="closed"):
            first.encode(path)
        assert torch.all(second.encode(path) == 250)
    finally:
        clear()
    with pytest.raises(RuntimeError, match="closed"):
        second.encode(path)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_reference_encoding_uses_dedicated_cuda_stream(
    request: pytest.FixtureRequest, tmp_path: Path, cache_enabled: bool
) -> None:
    calls: list[tuple[int, torch.cuda.Stream]] = []
    device = torch.device("cuda", torch.cuda.current_device())

    class CudaCodec:
        config = SimpleNamespace(sampling_rate=8000, number_channels=1)
        encoder_dtype = torch.float32

        def batch_encode(
            self, waveforms: list[torch.Tensor], *, num_quantizers: int
        ) -> SimpleNamespace:
            calls.append((threading.get_ident(), torch.cuda.current_stream(device)))
            assert waveforms[0].device == device
            return SimpleNamespace(
                audio_codes=waveforms[0]
                .gt(0)
                .long()
                .view(1, 1, -1)
                .repeat(num_quantizers, 1, 1),
                audio_codes_lengths=torch.tensor([waveforms[0].numel()], device=device),
            )

    path = tmp_path / "reference.wav"
    audio_source(path)
    encoder = make_encoder(
        request,
        MossAudioEncoder(CudaCodec(), device=str(device)),
        cache_enabled=cache_enabled,
        batch_size=1,
    )
    result = encoder.encode(path)
    assert torch.equal(result, torch.ones(80, 2, dtype=torch.long))
    thread, stream = calls[0]
    assert thread != threading.get_ident()
    assert stream.device == device
    assert stream != torch.cuda.default_stream(device)
