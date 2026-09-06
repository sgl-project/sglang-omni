# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from sglang_omni.models.qwen3_asr.gpu_mel import (
    bind_audio_frontend,
    log_mel_spectrogram,
)


def _extractor() -> WhisperFeatureExtractor:
    return WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )


def _reference_mel(
    extractor: WhisperFeatureExtractor, audio: np.ndarray
) -> torch.Tensor:
    extracted = extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",
        truncation=False,
    )
    return extracted.input_features[0]


@pytest.mark.parametrize("num_samples", [16000, 31000, 1600])
def test_gpu_mel_matches_whisper_feature_extractor(num_samples: int) -> None:
    extractor = _extractor()
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(num_samples).astype(np.float32)
    reference = _reference_mel(extractor, audio)

    model = type("M", (), {})()
    bind_audio_frontend(model, extractor)
    waveform = torch.from_numpy(audio)
    got = log_mel_spectrogram(waveform, model._audio_frontend)

    assert got.shape == reference.shape
    assert torch.allclose(got, reference, atol=1e-4, rtol=1e-4)


def test_gpu_mel_max8_clamp_matches_whisper_feature_extractor() -> None:
    extractor = _extractor()
    audio = np.zeros(16000, dtype=np.float32)
    audio[:400] = 1.0
    reference = _reference_mel(extractor, audio)

    model = type("M", (), {})()
    bind_audio_frontend(model, extractor)
    got = log_mel_spectrogram(torch.from_numpy(audio), model._audio_frontend)

    assert got.shape == reference.shape
    assert torch.allclose(got, reference, atol=1e-4, rtol=1e-4)
    log_spec = got * 4.0 - 4.0
    span = float(log_spec.max() - log_spec.min())
    assert span == pytest.approx(8.0, abs=1e-4)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_mel_cuda_matches_whisper_feature_extractor() -> None:
    extractor = _extractor()
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(16000).astype(np.float32)
    reference = _reference_mel(extractor, audio)
    model = type("M", (), {})()
    bind_audio_frontend(model, extractor)
    gpu = log_mel_spectrogram(torch.from_numpy(audio).cuda(), model._audio_frontend)
    assert gpu.device.type == "cuda"
    assert torch.allclose(gpu.cpu(), reference, atol=1e-4, rtol=1e-4)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_mel_matches_cpu_on_cuda() -> None:
    extractor = _extractor()
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(16000).astype(np.float32)
    model = type("M", (), {})()
    bind_audio_frontend(model, extractor)
    cpu = log_mel_spectrogram(torch.from_numpy(audio), model._audio_frontend)
    gpu = log_mel_spectrogram(torch.from_numpy(audio).cuda(), model._audio_frontend)
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-4, rtol=1e-4)


def test_bind_audio_frontend_requires_extractor() -> None:
    with pytest.raises(ValueError, match="requires a feature extractor"):
        bind_audio_frontend(type("M", (), {})(), None)


def test_bind_audio_frontend_requires_hop_and_filters() -> None:
    model = type("M", (), {})()
    with pytest.raises(ValueError, match="hop_length or mel_filters"):
        bind_audio_frontend(model, object())
    with pytest.raises(ValueError, match="invalid hop length"):
        bind_audio_frontend(model, SimpleNamespace(hop_length=0, mel_filters=[[1.0]]))
    with pytest.raises(ValueError, match="no mel_filters"):
        bind_audio_frontend(
            model, SimpleNamespace(hop_length=160, n_fft=400, mel_filters=None)
        )
