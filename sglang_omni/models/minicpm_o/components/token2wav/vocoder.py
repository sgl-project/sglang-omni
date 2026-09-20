# SPDX-License-Identifier: Apache-2.0
# Adapted from Step-Audio2; see THIRD_PARTY_NOTICES.md.
# Modifications: explicit device ownership and restricted checkpoint loading.
"""Load the MiniCPM-o vocoder and prepare speaker conditioning."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
import yaml
from librosa.filters import mel as librosa_mel

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    CausalConditionalCFM,
    CausalMaskedDiffWithXvec,
)
from sglang_omni.models.minicpm_o.components.token2wav.hift import HiFTGenerator
from sglang_omni.models.minicpm_o.components.token2wav.speech_tokenizer import (
    S3TokenizerV2,
)

SpeakerPrompt = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
FLOW_TYPES = {
    "!new:cosyvoice2.flow.flow.CausalMaskedDiffWithXvec": CausalMaskedDiffWithXvec,
    "!new:cosyvoice2.transformer.upsample_encoder_v2.UpsampleConformerEncoderV2": UpsampleConformerEncoderV2,
    "!new:cosyvoice2.flow.flow_matching.CausalConditionalCFM": CausalConditionalCFM,
    "!new:cosyvoice2.flow.decoder_dit.DiT": DiT,
}


def load_flow(path: Path) -> CausalMaskedDiffWithXvec:
    """Read only the four component tags used by the checkpoint's flow.yaml."""

    class FlowLoader(yaml.SafeLoader):
        pass

    def construct_component(
        loader: FlowLoader, node: yaml.MappingNode
    ) -> torch.nn.Module:
        return FLOW_TYPES[node.tag](**loader.construct_mapping(node, deep=True))

    for tag in FLOW_TYPES:
        FlowLoader.add_constructor(tag, construct_component)
    with path.open() as stream:
        config = yaml.load(stream, Loader=FlowLoader)
    if not isinstance(config, dict) or not isinstance(
        config.get("flow"), CausalMaskedDiffWithXvec
    ):
        raise ValueError("flow.yaml must define a MiniCPM-o flow model")
    return config["flow"]


@lru_cache(maxsize=1)
def prompt_mel_filters() -> tuple[torch.Tensor, torch.Tensor]:
    mel = librosa_mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)
    return torch.from_numpy(mel).float(), torch.hann_window(1920)


def prompt_mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    """Compute the checkpoint's 24 kHz, 80-bin, 50 Hz conditioning features."""
    mel, window = prompt_mel_filters()
    audio = torch.nn.functional.pad(
        audio.unsqueeze(1), (720, 720), mode="reflect"
    ).squeeze(1)
    spectrum = torch.view_as_real(
        torch.stft(
            audio,
            1920,
            hop_length=480,
            win_length=1920,
            window=window.to(audio.device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spectrum = torch.sqrt(spectrum.pow(2).sum(-1) + 1e-9)
    return torch.log(
        torch.clamp(torch.matmul(mel.to(audio.device), spectrum), min=1e-5)
    )


class Token2Wav(torch.nn.Module):
    def __init__(
        self,
        model_path: Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        n_timesteps: int = 10,
    ) -> None:
        super().__init__()
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        self.device = device
        self.dtype = dtype
        self.n_timesteps = n_timesteps
        self.audio_tokenizer = (
            S3TokenizerV2(model_path / "speech_tokenizer_v2_25hz.onnx")
            .to(device)
            .eval()
        )
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            str(model_path / "campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.flow = load_flow(model_path / "flow.yaml")
        if dtype != torch.float32:
            self.flow.to(dtype)
        self.flow.load_state_dict(
            torch.load(model_path / "flow.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.flow.to(device).eval()
        self.hift = HiFTGenerator()
        weights = torch.load(
            model_path / "hift.pt", map_location="cpu", weights_only=True
        )
        self.hift.load_state_dict(
            {key.removeprefix("generator."): value for key, value in weights.items()},
            strict=True,
        )
        self.hift.to(device).eval()
        self.cache: SpeakerPrompt | None = None

    @torch.inference_mode()
    def prepare_prompt(self, path: str) -> SpeakerPrompt:
        audio, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        else:
            speech = audio
        # note (MayDomine): tokenizer/voice embedding use channel zero; mel uses mono.
        speech = speech[0]
        mel = whisper.log_mel_spectrogram(speech, n_mels=128).unsqueeze(0)
        lengths = torch.tensor([mel.shape[2]], dtype=torch.int32, device=self.device)
        tokens, token_lengths = self.audio_tokenizer(mel.to(self.device), lengths)
        features = kaldi.fbank(
            speech.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        features = features - features.mean(dim=0, keepdim=True)
        embedding = torch.tensor(
            self.spk_model.run(
                None,
                {self.spk_model.get_inputs()[0].name: features.unsqueeze(0).numpy()},
            )[0],
            device=self.device,
        )
        audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(sample_rate, 24000)(audio)
        prompt_mel = prompt_mel_spectrogram(audio).transpose(1, 2).to(self.device)
        prompt_mel = torch.nn.functional.pad(
            prompt_mel,
            (0, 0, 0, tokens.shape[1] * self.flow.up_rate - prompt_mel.shape[1]),
            mode="replicate",
        )
        return tokens, token_lengths, embedding, prompt_mel
