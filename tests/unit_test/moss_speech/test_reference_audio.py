# SPDX-License-Identifier: Apache-2.0
"""Reference transport and resampling contracts for voice conditioning."""

import base64
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pytest
import soundfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from test_pipeline import make_codec

from sglang_omni.models.moss_speech.codec import load_reference_audio
from sglang_omni.models.moss_speech.request_builders import RequestValidationError


@pytest.mark.parametrize("sample_rate", [16000, 24000, 44100, 48000])
@pytest.mark.parametrize("transport", ["path", "file-uri", "data-uri", "https"])
def test_conditioning_uses_independent_token_and_speaker_resampling(
    tmp_path: Path, sample_rate: int, transport: str
) -> None:
    seconds = np.arange(sample_rate // 3) / sample_rate
    samples = 0.1 * np.sin(2 * np.pi * 7311 * seconds)
    path = tmp_path / "reference voice.wav"
    soundfile.write(path, samples, sample_rate, subtype="PCM_16")
    sources = {
        "path": str(path),
        "file-uri": path.as_uri(),
        "data-uri": "data:audio/wav;base64,"
        + base64.b64encode(path.read_bytes()).decode(),
        "https": "https://voice.example/reference.wav",
    }
    waveform, _ = torchaudio.load(path)
    expected_token_waveform = torchaudio.transforms.Resample(sample_rate, 16000)(
        waveform
    )
    expected_mel_waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
    expected_speaker_waveform = torchaudio.transforms.Resample(24000, 16000)(
        expected_mel_waveform
    )
    codec = make_codec()
    codec.feature_extractor = Mock(wraps=codec.feature_extractor)
    codec.feature_extractor.hop_length = 160
    codec.feature_extractor.sampling_rate = 16000
    expected_mel = codec.decoder.feat_extractor(expected_mel_waveform)
    expected_speaker_mel = kaldi.fbank(
        expected_speaker_waveform,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    expected_speaker_mel -= expected_speaker_mel.mean(dim=0, keepdim=True)
    response = httpx.Response(
        200, content=path.read_bytes(), request=httpx.Request("GET", sources["https"])
    )
    with patch("sglang_omni.models.moss_speech.codec.httpx.get", return_value=response):
        conditioning = codec.encode_reference(sources[transport])
    np.testing.assert_array_equal(
        codec.feature_extractor.call_args.args[0][0], expected_token_waveform[0].numpy()
    )
    expected_mel = expected_mel.transpose(1, 2)
    assert torch.equal(
        conditioning["prompt_feat"],
        expected_mel[:, : conditioning["prompt_feat"].shape[1]],
    )
    np.testing.assert_array_equal(
        codec.decoder.campplus_session.run.call_args.args[1]["speech"],
        expected_speaker_mel.unsqueeze(0).numpy(),
    )


def test_stereo_reference_downmix_preserves_native_sample_rate(tmp_path: Path) -> None:
    samples = np.random.default_rng(42).uniform(-0.2, 0.2, (4410, 2)).astype(np.float32)
    path = tmp_path / "stereo.wav"
    soundfile.write(path, samples, 44100, subtype="FLOAT")
    waveform, sample_rate = load_reference_audio(str(path))
    assert sample_rate == 44100
    assert torch.equal(waveform, torch.from_numpy(samples.T).mean(dim=0, keepdim=True))


@pytest.mark.parametrize("sample_count", [1, 31 * 24000])
def test_reference_duration_is_validated(tmp_path: Path, sample_count: int) -> None:
    path = tmp_path / "invalid-duration.wav"
    soundfile.write(path, np.zeros(sample_count), 24000)
    with pytest.raises(RequestValidationError):
        make_codec().encode_reference(str(path))


def test_nonfinite_reference_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.wav"
    soundfile.write(path, np.full(24000, np.nan), 24000, subtype="FLOAT")
    with pytest.raises(RequestValidationError, match="finite"):
        make_codec().encode_reference(str(path))
