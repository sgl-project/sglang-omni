# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""SAME-L decoder: latents to 44.1 kHz stereo audio, decoded in overlapping chunks."""

import torch
from torch import nn
from torch.nn.utils import weight_norm

from sglang_omni.models.stable_audio3.layers import (
    AttentionBackend,
    DifferentialAttention,
    DynamicTanh,
    GatedFeedForward,
    RotaryEmbedding,
)
from sglang_omni.models.stable_audio3.model_config import AutoencoderOptions

FEED_FORWARD_MULTIPLIER = 3
BOTTLENECK_NOISE_SCALE = 1e-3
CHUNK_LATENTS = 128
CHUNK_OVERLAP_LATENTS = 32


class DecoderBlock(nn.Module):
    def __init__(
        self, dim: int, *, head_dim: int, sinusoidal: bool, backend: AttentionBackend
    ) -> None:
        super().__init__()
        self.pre_norm = DynamicTanh(dim)
        self.self_attn = DifferentialAttention(
            dim, head_dim=head_dim, context_dim=None, qk_norm="dyt", backend=backend
        )
        self.ff_norm = DynamicTanh(dim)
        self.ff = GatedFeedForward(
            dim, multiplier=FEED_FORWARD_MULTIPLIER, sinusoidal=sinusoidal
        )
        self.rope = RotaryEmbedding(head_dim // 2)

    def forward(self, hidden_states: torch.Tensor, window_tokens: int) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.pre_norm(hidden_states),
            rotary_angles=self.rope(hidden_states.shape[1]),
            window_tokens=window_tokens,
        )
        return hidden_states + self.ff(self.ff_norm(hidden_states))


class TransformerResamplingBlock(nn.Module):
    """Each latent is followed by stride noised query tokens that become its audio patches."""

    def __init__(
        self, options: AutoencoderOptions, *, backend: AttentionBackend
    ) -> None:
        super().__init__()
        decoder = options.decoder.config
        self.dim: int = decoder.channels * decoder.c_mults[0]
        depth = decoder.transformer_depths[0]
        self.stride: int = decoder.strides[0]
        self.mask_noise: float = decoder.mask_noise
        # note (ruilin): each query sees one latent's tokens on either side, one sub-chunk of stride + 1.
        self.window_tokens: int = decoder.sliding_window[0] * (self.stride + 1)
        self.mapping = weight_norm(nn.Conv1d(self.dim, decoder.out_channels, 1))
        self.new_tokens = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.transformers = nn.ModuleList(
            DecoderBlock(
                self.dim,
                head_dim=decoder.dim_heads,
                sinusoidal=depth - index < decoder.sinusoidal_blocks[0],
                backend=backend,
            )
            for index in range(depth)
        )

    def forward(
        self, latent_tokens: torch.Tensor, query_noise: torch.Tensor
    ) -> torch.Tensor:
        batch_size, latent_length, dim = latent_tokens.shape
        new_tokens = self.new_tokens.expand(
            batch_size * latent_length, self.stride, dim
        )
        tokens = torch.cat(
            [
                latent_tokens.reshape(batch_size * latent_length, 1, dim),
                new_tokens + query_noise * self.mask_noise,
            ],
            dim=1,
        ).reshape(batch_size, latent_length * (self.stride + 1), dim)
        for block in self.transformers:
            tokens = block(tokens, self.window_tokens)
        patches = tokens.reshape(batch_size * latent_length, self.stride + 1, dim)[
            :, 1:
        ]
        return self.mapping(
            patches.reshape(batch_size, latent_length * self.stride, dim).transpose(
                1, 2
            )
        )


class SoftNormBottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.running_std = nn.Parameter(torch.ones(1))

    def forward(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        latents = latents * self.running_std
        return latents + noise * self.running_std * BOTTLENECK_NOISE_SCALE


class SameDecoder(nn.Module):
    def __init__(
        self, options: AutoencoderOptions, *, backend: AttentionBackend
    ) -> None:
        super().__init__()
        decoder = options.decoder.config
        # note (ruilin): the identities stand for the checkpoint's transposes, keeping the layers.1 / layers.3 names.
        self.layers = nn.ModuleList(
            [
                nn.Identity(),
                nn.Linear(decoder.latent_dim, decoder.channels * decoder.c_mults[0]),
                nn.Identity(),
                TransformerResamplingBlock(options, backend=backend),
            ]
        )

    @property
    def latent_projection(self) -> nn.Linear:
        return self.layers[1]

    @property
    def resampling(self) -> TransformerResamplingBlock:
        return self.layers[3]

    def forward(self, latents: torch.Tensor, query_noise: torch.Tensor) -> torch.Tensor:
        return self.resampling(
            self.latent_projection(latents.transpose(1, 2)), query_noise
        )


class AudioDecoder(nn.Module):
    """Checkpoint tensors under pretransform.model, without the encoder."""

    def __init__(
        self, options: AutoencoderOptions, *, backend: AttentionBackend
    ) -> None:
        super().__init__()
        self.bottleneck = SoftNormBottleneck()
        self.decoder = SameDecoder(options, backend=backend)
        self.patch_size: int = options.pretransform.config.patch_size
        self.downsampling_ratio: int = options.downsampling_ratio

    def forward(
        self, latents: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """One request's latents to audio: overlapping chunks, the last anchored at the end, decoded as one batch and stitched from their inner halves."""
        latent_length = latents.shape[-1]
        if latent_length < CHUNK_LATENTS:
            chunk_start_latents, chunk_length_latents = [0], latent_length
        else:
            hop_latents = CHUNK_LATENTS - CHUNK_OVERLAP_LATENTS
            chunk_start_latents = list(
                range(0, latent_length - CHUNK_LATENTS + 1, hop_latents)
            )
            if chunk_start_latents[-1] != latent_length - CHUNK_LATENTS:
                chunk_start_latents.append(latent_length - CHUNK_LATENTS)
            else:
                pass
            chunk_length_latents = CHUNK_LATENTS
        chunks = [
            latents[..., start_latent : start_latent + chunk_length_latents]
            for start_latent in chunk_start_latents
        ]
        resampling = self.decoder.resampling
        bottleneck_noise: list[torch.Tensor] = []
        query_noise: list[torch.Tensor] = []
        # note (ruilin): the checkpoint noises the latents and the query tokens at inference too; each chunk draws both in turn, as when chunks are decoded one by one.
        for chunk in chunks:
            bottleneck_noise.append(
                torch.randn(
                    chunk.shape,
                    generator=generator,
                    device=chunk.device,
                    dtype=chunk.dtype,
                )
            )
            query_noise.append(
                torch.randn(
                    (chunk_length_latents, resampling.stride, resampling.dim),
                    generator=generator,
                    device=chunk.device,
                    dtype=resampling.new_tokens.dtype,
                )
            )
        patches = self.decoder(
            self.bottleneck(torch.cat(chunks), torch.cat(bottleneck_noise)),
            torch.cat(query_noise),
        )
        chunk_count, channels_by_patch, patch_count = patches.shape
        decoded = (
            patches.view(chunk_count, -1, self.patch_size, patch_count)
            .transpose(2, 3)
            .reshape(chunk_count, channels_by_patch // self.patch_size, -1)
        )
        if chunk_count == 1:
            return decoded
        else:
            pass
        samples_per_latent = self.downsampling_ratio
        chunk_samples = CHUNK_LATENTS * samples_per_latent
        trim_samples = CHUNK_OVERLAP_LATENTS // 2 * samples_per_latent
        audio = latents.new_zeros(
            1, decoded.shape[1], latent_length * samples_per_latent
        )
        for index, start_latent in enumerate(chunk_start_latents):
            is_last = index == chunk_count - 1
            output_start_sample = (
                audio.shape[-1] - chunk_samples
                if is_last
                else start_latent * samples_per_latent
            )
            keep_start_sample = 0 if index == 0 else trim_samples
            keep_end_sample = chunk_samples if is_last else chunk_samples - trim_samples
            write_start_sample = output_start_sample + keep_start_sample
            write_end_sample = output_start_sample + keep_end_sample
            audio[..., write_start_sample:write_end_sample] = decoded[
                index, :, keep_start_sample:keep_end_sample
            ]
        return audio
