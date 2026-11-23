# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Stable Audio 3 loading and batched text-to-audio generation with ping-pong sampling."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import PreTrainedTokenizerBase, T5GemmaEncoderModel

from sglang_omni.models.stable_audio3.autoencoder import AudioDecoder
from sglang_omni.models.stable_audio3.conditioning import Conditioner, load_text_encoder
from sglang_omni.models.stable_audio3.dit import DiffusionTransformer
from sglang_omni.models.stable_audio3.model_config import (
    MAX_DURATION_SECONDS,
    StableAudio3Config,
)

SAMPLING_STEPS = 8
HEADROOM_SECONDS = 6.0
# note (ruilin): the checkpoint's sampling shift has rate 0, so the schedule does not depend on length.
LOGSNR_START = -6.2
LOGSNR_END = 2.0
CHECKPOINT_FILENAME = "model.safetensors"
COMPONENT_PREFIXES = {
    "dit": "model.model.",
    "conditioner": "conditioner.",
    "decoder": "pretransform.model.",
}
# The encoder and the bottleneck's encode-side tensors serve audio-to-audio only.
UNUSED_PREFIXES = (
    "pretransform.model.encoder.",
    "pretransform.model.bottleneck.scaling_factor",
    "pretransform.model.bottleneck.bias",
    "pretransform.model.bottleneck.noise_scaling_factor",
)


@dataclass(frozen=True, kw_only=True)
class GenerationRequest:
    prompt: str
    seconds: float
    seed: int


def sampling_schedule(device: torch.device) -> torch.Tensor:
    """Noise levels from 1 to 0, evenly spaced in log-SNR between the checkpoint's bounds."""
    timesteps = torch.linspace(1, 0, SAMPLING_STEPS + 1, device=device)
    logsnr_start = torch.tensor(LOGSNR_START, device=device)
    logsnr = LOGSNR_END - timesteps * (LOGSNR_END - logsnr_start)
    schedule = torch.sigmoid(-logsnr)
    schedule = torch.where(timesteps <= 0, torch.zeros_like(schedule), schedule)
    return torch.where(timesteps >= 1, torch.ones_like(schedule), schedule)


def load_component(module: nn.Module, archive_path: Path, prefix: str) -> None:
    with safe_open(archive_path, framework="pt", device="cpu") as archive:
        weights = {
            name.removeprefix(prefix): archive.get_tensor(name)
            for name in archive.keys()
            if name.startswith(prefix) and not name.startswith(UNUSED_PREFIXES)
        }
    module.load_state_dict(weights, strict=True, assign=True)


class StableAudio3Runtime:
    def __init__(
        self,
        *,
        config: StableAudio3Config,
        dit: DiffusionTransformer,
        conditioner: Conditioner,
        decoder: AudioDecoder,
        tokenizer: PreTrainedTokenizerBase,
        text_encoder: T5GemmaEncoderModel,
        device: torch.device,
    ) -> None:
        assert config.sample_size >= MAX_DURATION_SECONDS * config.sample_rate
        self.config = config
        self.dit = dit
        self.conditioner = conditioner
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = next(dit.parameters()).dtype
        self.schedule = sampling_schedule(device)
        encoder = config.model.pretransform.config.encoder.config
        self.latent_alignment = encoder.chunk_size // encoder.strides[0]
        self.latent_channels = config.model.diffusion.config.io_channels

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: Path, device: torch.device, dtype: torch.dtype
    ) -> "StableAudio3Runtime":
        config = StableAudio3Config.from_checkpoint(checkpoint_dir)
        with torch.device("meta"):
            dit = DiffusionTransformer(config.model.diffusion.config)
            conditioner = Conditioner(config.model.conditioning)
            decoder = AudioDecoder(config.model.pretransform.config)
        modules = {"dit": dit, "conditioner": conditioner, "decoder": decoder}
        for name, module in modules.items():
            load_component(
                module, checkpoint_dir / CHECKPOINT_FILENAME, COMPONENT_PREFIXES[name]
            )
            module.to(device=device, dtype=dtype).eval().requires_grad_(False)
        tokenizer, text_encoder = load_text_encoder(
            checkpoint_dir / config.model.conditioning.configs[0].config.subfolder,
            device,
        )
        return cls(
            config=config,
            dit=dit,
            conditioner=conditioner,
            decoder=decoder,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
        )

    def latent_length(self, seconds: float) -> int:
        """Latents for the request plus headroom, aligned to the autoencoder and capped at the model's sample size."""
        ratio = self.decoder.downsampling_ratio
        alignment = ratio * self.latent_alignment
        samples = int((seconds + HEADROOM_SECONDS) * self.config.sample_rate)
        samples = math.ceil(math.ceil(samples / ratio) * ratio / alignment) * alignment
        return min(samples, self.config.sample_size) // ratio

    def valid_length(self, seconds: float, latent_length: int) -> int:
        ratio = self.decoder.downsampling_ratio
        headroom = int(HEADROOM_SECONDS * self.config.sample_rate / ratio)
        content = math.ceil(int(seconds * self.config.sample_rate) / ratio)
        return min(content + headroom, latent_length)

    def noise(
        self, generator: torch.Generator, latent_length: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.randn(
            (1, self.latent_channels, latent_length),
            generator=generator,
            device=self.device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def generate(self, requests: list[GenerationRequest]) -> list[torch.Tensor]:
        """Stereo float32 audio per request; each request draws its noise from its own seed, so batching does not change it."""
        seconds_embedding = self.conditioner.embed_seconds(
            [request.seconds for request in requests]
        )
        condition_tokens = torch.cat(
            [
                self.conditioner.embed_prompts(
                    [request.prompt for request in requests],
                    self.tokenizer,
                    self.text_encoder,
                ),
                seconds_embedding[:, None],
            ],
            dim=1,
        )
        latent_lengths = [self.latent_length(request.seconds) for request in requests]
        valid_lengths = torch.tensor(
            [
                self.valid_length(request.seconds, length)
                for request, length in zip(requests, latent_lengths, strict=True)
            ],
            device=self.device,
        )
        generators = [
            torch.Generator(self.device).manual_seed(request.seed)
            for request in requests
        ]
        latents = torch.zeros(
            len(requests),
            self.latent_channels,
            max(latent_lengths),
            device=self.device,
            dtype=self.dtype,
        )
        for row, (generator, length) in enumerate(
            zip(generators, latent_lengths, strict=True)
        ):
            latents[row, :, :length] = self.noise(generator, length, torch.float32)
        for step in range(SAMPLING_STEPS):
            timestep = self.schedule[step].to(self.dtype)
            next_timestep = self.schedule[step + 1].to(self.dtype)
            velocity = self.dit(
                latents,
                timestep.expand(len(requests)),
                condition_tokens=condition_tokens,
                seconds_embedding=seconds_embedding,
                valid_lengths=valid_lengths,
            )
            denoised = latents - timestep * velocity
            noise = torch.zeros_like(latents)
            for row, (generator, length) in enumerate(
                zip(generators, latent_lengths, strict=True)
            ):
                noise[row, :, :length] = self.noise(generator, length, self.dtype)
            latents = (1 - next_timestep) * denoised + next_timestep * noise
        audio: list[torch.Tensor] = []
        for row, (request, generator, length) in enumerate(
            zip(requests, generators, latent_lengths, strict=True)
        ):
            decoded = self.decoder(latents[row : row + 1, :, :length], generator)
            samples = int(request.seconds * self.config.sample_rate)
            audio.append(decoded[0, :, :samples].float().clamp(-1, 1))
        return audio
