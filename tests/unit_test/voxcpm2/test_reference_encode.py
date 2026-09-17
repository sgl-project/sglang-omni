# SPDX-License-Identifier: Apache-2.0
"""Reference-cache integration using generated WAVs and a CPU recording VAE."""

import base64
import io
import os
import wave

import librosa
import numpy as np
import pytest
import torch

from sglang_omni.models.voxcpm2.reference_encode import VoxCPM2ReferenceEncoder


class _RecordingVAE:
    sample_rate = 16000
    latent_dim = 2
    hop_length = 2

    def __init__(self):
        self.inputs = []

    def parameters(self):
        yield torch.zeros(1)

    def encode(self, waveform, sample_rate):
        assert waveform.device.type == "cpu"
        assert waveform.dtype == torch.float32
        assert sample_rate == self.sample_rate
        self.inputs.append(waveform.clone())
        frames = waveform[:, :: self.hop_length]
        return frames.unsqueeze(1).expand(-1, self.latent_dim, -1).clone()


def _write_wave(path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def make_encoder():
    vae = _RecordingVAE()
    return vae, VoxCPM2ReferenceEncoder(
        vae, patch_size=4, cache_model_identity="test-voxcpm2"
    )


def test_reference_cache_hit_returns_independent_tensors(tmp_path):
    path = tmp_path / "ref.wav"
    _write_wave(path, [1000] * 16)
    vae, encoder = make_encoder()
    original = encoder.encode_cached(str(path), "right")
    expected = original.clone()
    original.zero_()
    cached = encoder.encode_cached(str(path), "right")
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    assert len(vae.inputs) == 1
    assert encoder.service.stats()["hits"] == 1


def test_same_path_same_size_rewrite_invalidates_reference_cache(tmp_path):
    path = tmp_path / "ref.wav"
    _write_wave(path, [1000] * 20000)
    before_stat = path.stat()
    vae, encoder = make_encoder()
    first = encoder.encode_cached(str(path), "right")
    changed = [1000] * 20000
    changed[10000] = -1000
    _write_wave(path, changed)
    os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    assert path.stat().st_size == before_stat.st_size
    assert path.stat().st_mtime_ns == before_stat.st_mtime_ns
    second = encoder.encode_cached(str(path), "right")
    assert len(vae.inputs) == 2
    assert not torch.equal(first, second)
    assert encoder.service.stats()["misses"] == 2


def test_equal_sized_references_do_not_collide(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wave(a, [1000] * 16)
    _write_wave(b, [-1000] * 16)
    assert a.stat().st_size == b.stat().st_size
    vae, encoder = make_encoder()
    first = encoder.encode_cached(str(a), "right")
    second = encoder.encode_cached(str(b), "right")
    assert not torch.equal(first, second)
    assert len(vae.inputs) == 2


def test_padding_sides_use_separate_reference_cache_entries(tmp_path):
    path = tmp_path / "ref.wav"
    _write_wave(path, [1000] * 10)
    vae, encoder = make_encoder()
    right = encoder.encode_cached(str(path), "right")
    left = encoder.encode_cached(str(path), "left")
    assert not torch.equal(right, left)
    assert len(vae.inputs) == 2
    assert torch.count_nonzero(vae.inputs[0][0, 10:]) == 0
    assert torch.count_nonzero(vae.inputs[1][0, :6]) == 0
    torch.testing.assert_close(encoder.encode_cached(str(path), "right"), right)
    torch.testing.assert_close(encoder.encode_cached(str(path), "left"), left)
    assert len(vae.inputs) == 2


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize(
    "source_kind", ["path", "file_uri", "data_uri", "bytes", "http"]
)
def test_reference_resampling_matches_upstream(
    tmp_path, monkeypatch, padding_side, source_kind
):
    # A 24 kHz signal with high frequencies detects replacing the upstream
    # resampler even when both implementations return the same sample count.
    path = tmp_path / "reference audio.wav"
    t = np.arange(7919) / 24000
    pcm = (12000 * np.sin(2 * np.pi * 7300 * t)).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(pcm.tobytes())
    raw = path.read_bytes()
    sources = {
        "path": str(path),
        "file_uri": path.as_uri(),
        "data_uri": "data:audio/wav;base64," + base64.b64encode(raw).decode(),
        "bytes": raw,
        "http": "https://example.test/ref.wav",
    }
    if source_kind == "http":
        import httpx

        from sglang_omni.models.voxcpm2 import reference_encode

        def fetch(url, **kwargs):
            assert url == sources["http"]
            return httpx.Response(200, content=raw, request=httpx.Request("GET", url))

        monkeypatch.setattr(reference_encode.httpx, "get", fetch)
    expected, _ = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    pad = (-len(expected)) % 8
    expected = np.pad(expected, (pad, 0) if padding_side == "left" else (0, pad))
    vae, encoder = make_encoder()
    encoder.encode_audio(sources[source_kind], padding_side=padding_side)
    np.testing.assert_array_equal(vae.inputs[-1].numpy()[0], expected)
