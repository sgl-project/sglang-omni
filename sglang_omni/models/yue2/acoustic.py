# SPDX-License-Identifier: Apache-2.0
"""Chunked midpoint flow matching with reusable AR conditioning."""

import torch
from torch.nn import functional as F

from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    CODEC_OFFSET,
    MUSIC_END,
    CancellationCheck,
    check_cancelled,
)


class AcousticChunk:
    def __init__(
        self, model: Yue2Model, prefix: list[int], noise: torch.Tensor
    ) -> None:
        self.model = model
        self.noise = noise
        self.device = model.model.embed_tokens.weight.device
        self.dtype = model.model.embed_tokens.weight.dtype
        self.frame_count = noise.shape[0]
        self.prefix_length = len(prefix)
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        positions = torch.arange(self.prefix_length, device=self.device)[None]
        cosine, sine = model.model.rotary_emb(positions)
        hidden_states = model.model.embed_tokens(
            torch.tensor([prefix], device=self.device)
        )
        for layer in model.model.layers:
            query, key, value = layer.self_attn.project_qkv(
                layer.input_layernorm(hidden_states), cosine, sine
            )
            self.cache.append((key, value))
            hidden_states = hidden_states + layer.self_attn.attend(
                query, key, value, causal=True
            )
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
        positions = torch.arange(
            self.prefix_length,
            self.prefix_length + self.frame_count + 2,
            device=self.device,
        )[None]
        self.cosine, self.sine = model.model.rotary_emb(positions)
        local_positions = torch.arange(self.frame_count + 2, device=self.device).clamp(
            max=model.config.max_latent_frames - 1
        )
        self.position_embedding = model.latent_pos_embed(local_positions)[None]

    def velocity(self, latent: torch.Tensor, time: float) -> torch.Tensor:
        model = self.model
        raw_time = (
            torch.logit(torch.tensor(time, dtype=torch.float64)).clamp(-20, 20).item()
        )
        timestep = torch.sigmoid(
            torch.tensor(raw_time, dtype=self.dtype, device=self.device)
        )
        shift = model.config.timestep_shift
        timestep = shift * timestep / (1 + (shift - 1) * timestep)
        hidden_states = model.vae2llm(F.pad(latent, (0, 0, 1, 1))[None])
        hidden_states = (
            hidden_states
            + model.time_embedder(timestep.expand(self.frame_count + 2))[None]
        )
        hidden_states = hidden_states + self.position_embedding
        for layer, (prefix_key, prefix_value) in zip(model.model.layers, self.cache):
            query, key, value = layer.nar_self_attn.project_qkv(
                layer.nar_input_layernorm(hidden_states), self.cosine, self.sine
            )
            key = torch.cat((prefix_key, key), dim=2)
            value = torch.cat((prefix_value, value), dim=2)
            hidden_states = hidden_states + layer.nar_self_attn.attend(
                query, key, value, causal=False
            )
            hidden_states = hidden_states + layer.nar_mlp(
                layer.nar_pre_mlp_layernorm(hidden_states)
            )
        return model.llm2vae(model.model.norm(hidden_states))[0, 1:-1]

    def solve(self, steps: int, cancelled: CancellationCheck) -> torch.Tensor:
        latent = self.noise.to(device=self.device, dtype=self.dtype)
        step_size = 1.0 / steps
        for step in range(steps):
            check_cancelled(cancelled)
            time = 1.0 - step * step_size
            midpoint = latent - self.velocity(latent, time) * (step_size / 2)
            check_cancelled(cancelled)
            latent = latent - self.velocity(midpoint, time - step_size / 2) * step_size
        return latent.float().cpu()


@torch.inference_mode()
def synthesize_latents(
    model: Yue2Model,
    prefix: list[int],
    codec_tokens: list[int],
    seed: int,
    steps: int,
    cancelled: CancellationCheck,
) -> torch.Tensor:
    chunk_frames = (model.config.max_position_embeddings - len(prefix) - 3) // 2
    if not codec_tokens or chunk_frames < 1:
        raise ValueError("YuE2 has no semantic tokens or no remaining acoustic context")
    else:
        pass
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (len(codec_tokens), model.config.latent_dim),
        dtype=torch.float32,
        generator=generator,
    )
    chunks: list[torch.Tensor] = []
    for start in range(0, len(codec_tokens), chunk_frames):
        check_cancelled(cancelled)
        end = min(start + chunk_frames, len(codec_tokens))
        conditioning = [
            *prefix,
            *(token + CODEC_OFFSET for token in codec_tokens[start:end]),
            MUSIC_END,
        ]
        chunk = AcousticChunk(model, conditioning, noise[start:end])
        try:
            chunks.append(chunk.solve(steps, cancelled))
        finally:
            chunk.cache.clear()
    latents = torch.cat(chunks)
    if not torch.isfinite(latents).all():
        raise FloatingPointError("YuE2 produced non-finite acoustic latents")
    else:
        return latents
