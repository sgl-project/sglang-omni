# SPDX-License-Identifier: Apache-2.0
# Adapted from Step-Audio2; see THIRD_PARTY_NOTICES.md.
# Modifications: explicit device ownership and restricted checkpoint loading.
"""Load the MiniCPM-o vocoder and prepare speaker conditioning."""

from __future__ import annotations

import io
import json
import os
from contextlib import nullcontext
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
from sglang_omni.models.minicpm_o.payload_types import SpeakerPromptInputs
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.scheduling.reference_encoder import KeyedReferenceEncodeHook

PROMPT_AUDIO_SAMPLE_RATE = 16000
FLOW_AUDIO_SAMPLE_RATE = 24000

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
    else:
        pass
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


def resolve_token2wav_assets(model_path: str) -> tuple[Path, Path | None]:
    """Locate required Token2Wav assets and the optional default speaker reference."""
    model_dir = Path(resolve_model_path(model_path))
    asset_dir = model_dir / "assets" / "token2wav"
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"token2wav assets not found at {asset_dir}; copy the "
            "checkpoint's assets/token2wav directory next to the weights"
        )
    else:
        pass
    default_reference = model_dir / "assets" / "HT_ref_audio.wav"
    if default_reference.is_file():
        return asset_dir, default_reference
    else:
        return asset_dir, None


class MiniCPMOReferenceEncodeHook(
    KeyedReferenceEncodeHook[bytes | None, SpeakerPromptInputs, SpeakerPromptInputs]
):
    """Extract cacheable CPU speaker conditioning from inline or default audio."""

    model_id = "minicpm_o"
    encoder_id = "s3tokenizer_v2_25hz+campplus+prompt_mel_24k"
    artifact_kind = "token2wav_speaker_conditioning"

    def __init__(
        self,
        model_path: Path,
        *,
        device: torch.device,
        default_reference: Path | None,
        onnx_intra_op_threads: int,
    ) -> None:
        self.device = device
        self.default_reference = default_reference
        self.model_revision = str(model_path.resolve())
        self.encoder_config_hash = hash_bytes(
            json.dumps(
                {
                    "tokenizer_audio_sr": PROMPT_AUDIO_SAMPLE_RATE,
                    "tokenizer_mels": 128,
                    "fbank_mels": 80,
                    "prompt_mel": {
                        "sr": FLOW_AUDIO_SAMPLE_RATE,
                        "n_fft": 1920,
                        "hop": 480,
                        "mels": 80,
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        # note(liuqihao): a private stream avoids waiting for the thinker's default stream.
        if device.type == "cpu":
            self.device_module = None
        else:
            self.device_module = torch.get_device_module(device)
        if self.device_module is None:
            self.stream = None
        else:
            self.stream = self.device_module.Stream(device=device)
        self.audio_tokenizer = S3TokenizerV2(
            model_path / "speech_tokenizer_v2_25hz.onnx"
        )
        self.audio_tokenizer.to(device)
        self.audio_tokenizer.eval()
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.intra_op_num_threads = max(
            1, min(onnx_intra_op_threads, os.cpu_count() or 1)
        )
        self.spk_model = onnxruntime.InferenceSession(
            str(model_path / "campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def input_key(self, item: bytes | None) -> str | None:
        if item is not None:
            key = f"bytes:{hash_bytes(item)}"
        elif self.default_reference is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            key = reference_path_cache_key(str(self.default_reference))
        return key

    def encode_one(self, item: bytes | None) -> SpeakerPromptInputs:
        # note(liuqihao): decode inline references in memory to avoid temporary audio files.
        if item is not None:
            source: str | io.BytesIO = io.BytesIO(item)
        elif self.default_reference is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            source = str(self.default_reference)
        return self.extract(source)

    def store_artifact(self, artifact: SpeakerPromptInputs) -> SpeakerPromptInputs:
        return clone_speaker_prompt(artifact)

    def load_artifact(self, stored: SpeakerPromptInputs) -> SpeakerPromptInputs:
        return clone_speaker_prompt(stored)

    @torch.inference_mode()
    def extract(self, source: str | io.BytesIO) -> SpeakerPromptInputs:
        audio, sample_rate = torchaudio.load(source)
        if sample_rate != PROMPT_AUDIO_SAMPLE_RATE:
            speech = torchaudio.transforms.Resample(
                sample_rate, PROMPT_AUDIO_SAMPLE_RATE
            )(audio)
        else:
            speech = audio
        # note(liuqihao): preserve channel-zero tokenizer input and mono mel conditioning.
        speech = speech[0]
        mel = whisper.log_mel_spectrogram(speech, n_mels=128)
        mel = mel.unsqueeze(0)
        if self.device_module is None:
            stream_context = nullcontext()
        else:
            stream_context = self.device_module.stream(self.stream)
        with stream_context:
            lengths = torch.tensor(
                [mel.shape[2]], dtype=torch.int32, device=self.device
            )
            device_mel = mel.to(self.device)
            tokens, token_lengths = self.audio_tokenizer(device_mel, lengths)
            tokens = tokens.cpu()
            token_lengths = token_lengths.cpu()
        features = kaldi.fbank(
            speech.unsqueeze(0),
            num_mel_bins=80,
            dither=0,
            sample_frequency=PROMPT_AUDIO_SAMPLE_RATE,
        )
        features = features - features.mean(dim=0, keepdim=True)
        speaker_inputs = self.spk_model.get_inputs()
        speaker_input_name = speaker_inputs[0].name
        speaker_features = features.unsqueeze(0)
        speaker_features = speaker_features.numpy()
        speaker_outputs = self.spk_model.run(
            None, {speaker_input_name: speaker_features}
        )
        embedding = torch.from_numpy(speaker_outputs[0])
        audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != FLOW_AUDIO_SAMPLE_RATE:
            audio = torchaudio.transforms.Resample(sample_rate, FLOW_AUDIO_SAMPLE_RATE)(
                audio
            )
        else:
            pass
        # note(liuqihao): keep conditioning on CPU for caching and cross-process transport.
        prompt_mel = prompt_mel_spectrogram(audio)
        prompt_mel = prompt_mel.transpose(1, 2)
        return {
            "speech_tokens": tokens,
            "speech_token_len": token_lengths,
            "speaker_embedding": embedding,
            "prompt_mel": prompt_mel,
        }


def clone_speaker_prompt(prompt: SpeakerPromptInputs) -> SpeakerPromptInputs:
    speech_tokens = prompt["speech_tokens"].detach()
    speech_tokens = speech_tokens.cpu()
    speech_token_len = prompt["speech_token_len"].detach()
    speech_token_len = speech_token_len.cpu()
    speaker_embedding = prompt["speaker_embedding"].detach()
    speaker_embedding = speaker_embedding.cpu()
    prompt_mel = prompt["prompt_mel"].detach()
    prompt_mel = prompt_mel.cpu()
    return {
        "speech_tokens": speech_tokens.clone(),
        "speech_token_len": speech_token_len.clone(),
        "speaker_embedding": speaker_embedding.clone(),
        "prompt_mel": prompt_mel.clone(),
    }


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
        else:
            pass
        self.device = device
        self.dtype = dtype
        self.n_timesteps = n_timesteps
        self.flow = load_flow(model_path / "flow.yaml")
        if dtype != torch.float32:
            self.flow.to(dtype)
        else:
            pass
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
