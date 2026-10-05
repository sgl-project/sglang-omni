# SPDX-License-Identifier: Apache-2.0
# Adapted from Step-Audio2; see THIRD_PARTY_NOTICES.md.
# Modifications: explicit device ownership and restricted checkpoint loading.
"""Load the MiniCPM-o vocoder and prepare speaker conditioning."""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
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

# note (Junnan Li): stepaudio2 Token2wav keeps the prompt plus this many frames so positions stay in range.
FLOW_CACHE_TAIL_FRAMES = 100
SILENCE_TOKEN_ID = 4218
MEL_CACHE_FRAMES = 8
SAMPLES_PER_MEL_FRAME = 480
# note (Junnan Li): cuDNN builds a convolution plan per input shape, so HiFT batches are padded to a few repeating sizes.
HIFT_BATCH_GRANULE = 8


def hift_batch_rows(rows: int) -> int:
    if rows <= HIFT_BATCH_GRANULE:
        return 1 << (rows - 1).bit_length()
    else:
        return -(-rows // HIFT_BATCH_GRANULE) * HIFT_BATCH_GRANULE


@dataclass(kw_only=True, frozen=True)
class SpeakerPrompt:
    """Prompt tokens, their lengths, the speaker embedding, and the prompt mel."""

    prompt_tokens: torch.Tensor
    prompt_token_lengths: torch.Tensor
    speaker_embedding: torch.Tensor
    prompt_mel: torch.Tensor


StreamCaches = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
HiftBatchKey = tuple[int, bool, int, int, int]


@dataclass(kw_only=True, frozen=True)
class StreamChunk:
    token_ids: list[int]
    prompt: SpeakerPrompt
    caches: StreamCaches
    is_last_chunk: bool


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
        self.speaker_model = onnxruntime.InferenceSession(
            str(model_path / "campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.flow = load_flow(model_path / "flow.yaml")
        if dtype != torch.float32:
            self.flow.to(dtype)
        else:
            pass
        flow_weights = torch.load(
            model_path / "flow.pt", map_location="cpu", weights_only=True
        )
        checkpoint_speaker_projection = "spk_embed_affine_layer."
        self.flow.load_state_dict(
            {
                (
                    key.replace(
                        checkpoint_speaker_projection,
                        "speaker_embedding_projection.",
                        1,
                    )
                    if key.startswith(checkpoint_speaker_projection)
                    else key
                ): value
                for key, value in flow_weights.items()
            },
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
        self.mel_cache_len = MEL_CACHE_FRAMES
        self.source_cache_len = MEL_CACHE_FRAMES * SAMPLES_PER_MEL_FRAME
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(
            device
        )

    @torch.inference_mode()
    def prepare_prompt(self, source: str | io.BytesIO) -> SpeakerPrompt:
        audio, sample_rate = torchaudio.load(source)
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        else:
            speech = audio
        # note (MayDomine): tokenizer/voice embedding use channel zero; mel uses mono.
        speech = speech[0]
        mel = whisper.log_mel_spectrogram(speech, n_mels=128).unsqueeze(0)
        mel_lengths = torch.tensor(
            [mel.shape[2]], dtype=torch.int32, device=self.device
        )
        prompt_tokens, prompt_token_lengths = self.audio_tokenizer(
            mel.to(self.device), mel_lengths
        )
        fbank_features = kaldi.fbank(
            speech.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
        )
        fbank_features = fbank_features - fbank_features.mean(dim=0, keepdim=True)
        speaker_embedding = torch.tensor(
            self.speaker_model.run(
                None,
                {
                    self.speaker_model.get_inputs()[0]
                    .name: fbank_features.unsqueeze(0)
                    .numpy()
                },
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
            (
                0,
                0,
                0,
                prompt_tokens.shape[1] * self.flow.up_rate - prompt_mel.shape[1],
            ),
            mode="replicate",
        )
        return SpeakerPrompt(
            prompt_tokens=prompt_tokens,
            prompt_token_lengths=prompt_token_lengths,
            speaker_embedding=speaker_embedding,
            prompt_mel=prompt_mel,
        )

    @torch.inference_mode()
    def open_stream(self, prompt: SpeakerPrompt) -> StreamCaches:
        """Return the flow and HiFT caches that start a streaming decode."""
        prompt_speech_tokens = prompt.prompt_tokens
        speaker_embedding = prompt.speaker_embedding
        prompt_mels = prompt.prompt_mel
        right_pad_speech_tokens = torch.full(
            (1, self.flow.pre_lookahead_len),
            SILENCE_TOKEN_ID,
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
    def warm_up_vocoder(self, max_rows: int, max_chunk_frames: int) -> None:
        for rows in sorted({hift_batch_rows(rows) for rows in range(1, max_rows + 1)}):
            for frames in range(
                self.flow.up_rate,
                self.mel_cache_len + max_chunk_frames + 1,
                self.flow.up_rate,
            ):
                self.hift(
                    torch.zeros(
                        rows, self.flow.output_size, frames, device=self.device
                    ),
                    torch.zeros(rows, 1, 0, device=self.device),
                )

    @torch.inference_mode()
    def stream_batch(
        self, chunks: list[StreamChunk]
    ) -> list[tuple[bytes, StreamCaches]]:
        """Decode one token chunk of each of several streams; each caller owns its caches and receives new ones."""
        with torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            decoded = self.flow.inference_chunks(
                token_ids=[
                    torch.tensor(
                        [chunk.token_ids], dtype=torch.int32, device=self.device
                    )
                    for chunk in chunks
                ],
                speaker_embeddings=torch.cat(
                    [chunk.prompt.speaker_embedding for chunk in chunks]
                ),
                caches=[chunk.caches[0] for chunk in chunks],
                is_last_chunk=[chunk.is_last_chunk for chunk in chunks],
                n_timesteps=self.n_timesteps,
            )
        groups: dict[HiftBatchKey, list[int]] = defaultdict(list)
        for index, (chunk, (predicted_mel, flow_cache)) in enumerate(
            zip(chunks, decoded, strict=True)
        ):
            prompt_mel_frames = chunk.prompt.prompt_mel.shape[1]
            if (
                flow_cache["estimator_attention_cache"].shape[4]
                > prompt_mel_frames + FLOW_CACHE_TAIL_FRAMES
            ):
                flow_cache["estimator_attention_cache"] = torch.cat(
                    [
                        flow_cache["estimator_attention_cache"][
                            :, :, :, :, :prompt_mel_frames
                        ],
                        flow_cache["estimator_attention_cache"][
                            :, :, :, :, -FLOW_CACHE_TAIL_FRAMES:
                        ],
                    ],
                    dim=4,
                )
            else:
                pass
            if (
                flow_cache["conformer_attention_cache"].shape[3]
                > prompt_mel_frames + FLOW_CACHE_TAIL_FRAMES
            ):
                flow_cache["conformer_attention_cache"] = torch.cat(
                    [
                        flow_cache["conformer_attention_cache"][
                            :, :, :, :prompt_mel_frames, :
                        ],
                        flow_cache["conformer_attention_cache"][
                            :, :, :, -FLOW_CACHE_TAIL_FRAMES:, :
                        ],
                    ],
                    dim=3,
                )
            else:
                pass
            hift_cache = chunk.caches[1]
            groups[
                (
                    predicted_mel.shape[2],
                    chunk.is_last_chunk,
                    hift_cache["mel"].shape[2],
                    hift_cache["source"].shape[2],
                    hift_cache["speech"].shape[1],
                )
            ].append(index)
        results: dict[int, tuple[bytes, StreamCaches]] = {}
        for indices in groups.values():
            is_last_chunk = chunks[indices[0]].is_last_chunk
            hift_cache = {
                name: torch.cat([chunks[index].caches[1][name] for index in indices])
                for name in ("mel", "source", "speech")
            }
            predicted_mel = torch.cat([decoded[index][0] for index in indices])
            hift_cache_speech = hift_cache["speech"]
            mel = torch.concat([hift_cache["mel"], predicted_mel], dim=2)
            padding_rows = hift_batch_rows(len(indices)) - len(indices)
            if padding_rows:
                # note (Junnan Li): Padding rows repeat the last row; HiFT treats rows independently and their output is dropped.
                speech, source = self.hift(
                    torch.cat((mel, mel[-1:].expand(padding_rows, -1, -1))).float(),
                    torch.cat(
                        (
                            hift_cache["source"],
                            hift_cache["source"][-1:].expand(padding_rows, -1, -1),
                        )
                    ),
                )
                speech, source = speech[: len(indices)], source[: len(indices)]
            else:
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
            next_hift_cache = dict(
                mel=mel[..., -self.mel_cache_len :],
                source=source[:, :, -self.source_cache_len :],
                speech=speech[:, -self.source_cache_len :],
            )
            if not is_last_chunk:
                if is_first_chunk:
                    silence_padding = torch.zeros(
                        len(indices), self.source_cache_len, device=speech.device
                    )
                    speech = torch.cat(
                        [silence_padding, speech[:, : -self.source_cache_len]], dim=1
                    )
                else:
                    speech = speech[:, : -self.source_cache_len]
            else:
                pass
            waveform = np.clip(speech.cpu().numpy(), -1.0, 1.0)
            pcm = (waveform * 32767.0).astype("<i2")
            for position, index in enumerate(indices):
                results[index] = (
                    pcm[position].tobytes(),
                    (
                        decoded[index][1],
                        {
                            name: value[position : position + 1].clone()
                            for name, value in next_hift_cache.items()
                        },
                    ),
                )
        return [results[index] for index in range(len(chunks))]
