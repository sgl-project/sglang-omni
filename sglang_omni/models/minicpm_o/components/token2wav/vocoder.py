# SPDX-License-Identifier: Apache-2.0
# Adapted from Step-Audio2; see THIRD_PARTY_NOTICES.md.
# Modifications: explicit device ownership and restricted checkpoint loading.
"""Load the MiniCPM-o vocoder and prepare speaker conditioning."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
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
StreamCaches = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
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
        self.cache: SpeakerPrompt | None = None
        self.mel_cache_len = 8
        self.source_cache_len = self.mel_cache_len * 480
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(
            device
        )

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
        else:
            pass
        prompt_mel = prompt_mel_spectrogram(audio).transpose(1, 2).to(self.device)
        prompt_mel = torch.nn.functional.pad(
            prompt_mel,
            (0, 0, 0, tokens.shape[1] * self.flow.up_rate - prompt_mel.shape[1]),
            mode="replicate",
        )
        return tokens, token_lengths, embedding, prompt_mel

    @torch.inference_mode()
    def open_stream(self, prompt: SpeakerPrompt) -> StreamCaches:
        """Return the flow and HiFT caches that start a streaming decode."""
        prompt_speech_tokens, _, speaker_embedding, prompt_mels = prompt
        right_pad_speech_tokens = torch.full(
            (1, 3),
            4218,
            device=prompt_speech_tokens.device,
            dtype=prompt_speech_tokens.dtype,
        )
        with torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            flow_cache = self.flow.setup_cache(
                torch.cat([prompt_speech_tokens, right_pad_speech_tokens], dim=1),
                prompt_mels,
                speaker_embedding,
                n_timesteps=self.n_timesteps,
            )
        hift_cache = dict(
            mel=torch.zeros(1, prompt_mels.shape[2], 0, device=self.device),
            source=torch.zeros(1, 1, 0, device=self.device),
            speech=torch.zeros(1, 0, device=self.device),
        )
        return flow_cache, hift_cache

    @torch.inference_mode()
    def stream(
        self,
        generated_speech_tokens: list[int],
        prompt: SpeakerPrompt,
        caches: StreamCaches,
        last_chunk: bool = False,
    ) -> tuple[bytes, StreamCaches]:
        """Decode one token chunk; the caller owns the caches and receives new ones."""
        _, _, speaker_embedding, prompt_mels = prompt
        flow_cache, hift_cache = caches
        tokens = torch.tensor(
            [generated_speech_tokens], dtype=torch.int32, device=self.device
        )
        with torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            chunk_mel, flow_cache = self.flow.inference_chunk(
                token=tokens,
                spk=speaker_embedding,
                cache=flow_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )
        prompt_len = prompt_mels.shape[1]
        if flow_cache["estimator_att_cache"].shape[4] > prompt_len + 100:
            flow_cache["estimator_att_cache"] = torch.cat(
                [
                    flow_cache["estimator_att_cache"][:, :, :, :, :prompt_len],
                    flow_cache["estimator_att_cache"][:, :, :, :, -100:],
                ],
                dim=4,
            )
        else:
            pass
        if flow_cache["conformer_att_cache"].shape[3] > prompt_len + 100:
            flow_cache["conformer_att_cache"] = torch.cat(
                [
                    flow_cache["conformer_att_cache"][:, :, :, :prompt_len, :],
                    flow_cache["conformer_att_cache"][:, :, :, -100:, :],
                ],
                dim=3,
            )
        else:
            pass
        hift_cache_speech = hift_cache["speech"]
        mel = torch.concat([hift_cache["mel"], chunk_mel], dim=2)
        speech, source = self.hift(mel.float(), hift_cache["source"])
        if hift_cache_speech.shape[-1] > 0:
            overlap = min(
                self.source_cache_len, speech.shape[-1], hift_cache_speech.shape[-1]
            )
            speech = speech.clone()
            speech[..., :overlap] = (
                speech[..., :overlap] * self.speech_window[:overlap]
                + hift_cache_speech[..., -overlap:]
                * self.speech_window[
                    self.source_cache_len : self.source_cache_len + overlap
                ]
            )
        else:
            pass
        is_first_chunk = hift_cache_speech.shape[-1] == 0
        hift_cache = dict(
            mel=mel[..., -self.mel_cache_len :].clone(),
            source=source[:, :, -self.source_cache_len :].clone(),
            speech=speech[:, -self.source_cache_len :].clone(),
        )
        if not last_chunk:
            if is_first_chunk:
                silence_padding = torch.zeros(
                    1, self.source_cache_len, device=speech.device
                )
                speech = torch.cat(
                    [silence_padding, speech[:, : -self.source_cache_len]], dim=1
                )
            else:
                speech = speech[:, : -self.source_cache_len]
        else:
            pass
        wav_np = np.clip(speech.cpu().numpy(), -1.0, 1.0)
        return (wav_np * 32767.0).astype("<i2").tobytes(), (flow_cache, hift_cache)
