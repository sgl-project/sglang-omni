# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torchaudio

from sglang_omni.models.mimo_audio.preprocessor import (
    MiMoAudioPreprocessor,
    MiMoAudioTokenizerSettings,
)


class _FakeEncoder:
    def __init__(self, frames: int):
        self.frames = frames
        self.features_shape: tuple[int, ...] | None = None
        self.lengths: torch.Tensor | None = None

    def encode(
        self,
        *,
        input_features: torch.Tensor,
        input_lens: torch.Tensor,
        return_codes_only: bool,
    ):
        self.features_shape = tuple(input_features.shape)
        self.lengths = input_lens.detach().cpu()
        assert return_codes_only is True
        codes = torch.arange(20 * self.frames, device=input_features.device).reshape(
            20, self.frames
        )
        return codes, torch.tensor([self.frames], device=input_features.device)


def test_tokenizer_settings_match_official_checkpoint() -> None:
    settings = MiMoAudioTokenizerSettings.from_config(
        {
            "sampling_rate": 24_000,
            "nfft": 960,
            "hop_length": 240,
            "window_size": 960,
            "n_mels": 128,
            "fmin": 0,
            "fmax": None,
        }
    )

    assert settings == MiMoAudioTokenizerSettings()


def test_log_mel_is_exact_official_torchaudio_contract() -> None:
    settings = MiMoAudioTokenizerSettings()
    preprocessor = MiMoAudioPreprocessor(settings)
    waveform = torch.linspace(-0.5, 0.5, 24_000)

    actual = preprocessor.waveform_to_log_mel(waveform)
    official_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24_000,
        n_fft=960,
        hop_length=240,
        win_length=960,
        f_min=0,
        f_max=None,
        n_mels=128,
        power=1.0,
        center=True,
    )
    expected = (
        torch.log(torch.clip(official_transform(waveform[None, :]), min=1.0e-7))
        .squeeze()
        .transpose(0, 1)
    )

    assert actual.shape == (101, 128)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_encode_uses_first_eight_channels_and_pads_last_frame() -> None:
    preprocessor = MiMoAudioPreprocessor()
    log_mel = torch.zeros((6010, 128), dtype=torch.float32)
    encoder = _FakeEncoder(frames=5)

    codes = preprocessor.encode_log_mel(log_mel, encoder)

    assert encoder.features_shape == (6010, 128)
    assert encoder.lengths.tolist() == [6000, 10]
    assert codes.shape == (8, 8)
    assert torch.equal(codes[:5], torch.arange(20 * 5).reshape(20, 5)[:8].T)
    assert torch.equal(codes[5:], codes[4:5].expand(3, -1))


def test_real_official_audio_fixture_loads_and_produces_log_mel() -> None:
    path = (
        "/workspace/references/MiMo-Audio/examples/spoken_dialogue_assistant_turn_1.wav"
    )
    preprocessor = MiMoAudioPreprocessor()

    waveform = preprocessor.load_waveform(path)
    log_mel = preprocessor.waveform_to_log_mel(waveform)

    assert waveform.ndim == 1
    assert waveform.dtype == torch.float32
    assert waveform.numel() > 0
    assert log_mel.ndim == 2
    assert log_mel.shape[0] > 0
    assert log_mel.shape[1] == 128
    assert torch.isfinite(log_mel).all()
