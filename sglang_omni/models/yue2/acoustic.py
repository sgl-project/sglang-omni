# SPDX-License-Identifier: Apache-2.0
"""Chunked midpoint flow matching with reusable AR conditioning."""

import torch
from torch.nn import functional as F

from sglang_omni.models.yue2.cuda_graph import CapturedGraph
from sglang_omni.models.yue2.layers import TokenCache
from sglang_omni.models.yue2.model import Yue2Model
from sglang_omni.models.yue2.request import (
    CODEC_OFFSET,
    MUSIC_END,
    CancellationCheck,
    check_cancelled,
)


class AcousticChunk:
    def __init__(
        self,
        model: Yue2Model,
        prefix: list[int],
        noise: torch.Tensor,
    ) -> None:
        self.model = model
        self.noise = noise
        self.device = model.model.embed_tokens.weight.device
        self.dtype = model.model.embed_tokens.weight.dtype
        self.frame_count = noise.shape[0]
        self.prefix_length = len(prefix)
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        cache = TokenCache(capacity=len(prefix))
        model(torch.tensor([prefix], device=self.device), cache)
        for key, value in zip(cache.keys, cache.values):
            shape = (
                *key.shape[:2],
                self.prefix_length + self.frame_count + 2,
                key.shape[-1],
            )
            combined_key, combined_value = key.new_empty(shape), value.new_empty(shape)
            combined_key[:, :, : self.prefix_length].copy_(key[:, :, : cache.length])
            combined_value[:, :, : self.prefix_length].copy_(
                value[:, :, : cache.length]
            )
            self.cache.append((combined_key, combined_value))
        positions = torch.arange(
            self.prefix_length,
            self.prefix_length + self.frame_count + 2,
            device=self.device,
            dtype=torch.int32,
        )[None]
        self.cosine, self.sine = model.model.rotary_emb(positions)
        self.positions = positions
        local_positions = torch.arange(self.frame_count + 2, device=self.device).clamp(
            max=model.config.max_latent_frames - 1
        )
        self.position_embedding = model.latent_pos_embed(local_positions)[None]

    def velocity(self, latent: torch.Tensor, time: float) -> torch.Tensor:
        raw_time = (
            torch.logit(torch.tensor(time, dtype=torch.float64)).clamp(-20, 20).item()
        )
        return self.predict_velocity(
            latent, torch.tensor(raw_time, dtype=self.dtype, device=self.device)
        )

    def predict_velocity(
        self, latent: torch.Tensor, raw_time: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        timestep = torch.sigmoid(raw_time)
        shift = model.config.timestep_shift
        timestep = shift * timestep / (1 + (shift - 1) * timestep)
        hidden_states = model.vae2llm(F.pad(latent, (0, 0, 1, 1))[None])
        hidden_states = (
            hidden_states
            + model.time_embedder(timestep.expand(self.frame_count + 2))[None]
        )
        residual = hidden_states + self.position_embedding
        hidden_states = model.model.layers[0].nar_input_layernorm(residual)
        for layer_index, (layer, (combined_key, combined_value)) in enumerate(
            zip(model.model.layers, self.cache)
        ):
            query, key, value = layer.nar_self_attn.project_qkv(
                hidden_states, self.cosine, self.sine, self.positions
            )
            combined_key[:, :, self.prefix_length :].copy_(key)
            combined_value[:, :, self.prefix_length :].copy_(value)
            attended = layer.nar_self_attn.attend(
                query, combined_key, combined_value, causal=False
            )
            hidden_states, residual = layer.nar_pre_mlp_layernorm.add(
                attended, residual
            )
            next_norm = (
                model.model.layers[layer_index + 1].nar_input_layernorm
                if layer_index + 1 < len(model.model.layers)
                else model.model.norm
            )
            hidden_states, residual = next_norm.add(
                layer.nar_mlp(hidden_states), residual
            )
        return model.llm2vae(hidden_states)[0, 1:-1]

    def solve(
        self, steps: int, cancelled: CancellationCheck, *, cuda_graph: bool = False
    ) -> torch.Tensor:
        latent = self.noise.to(device=self.device, dtype=self.dtype)
        step_size = 1.0 / steps
        graph = None
        if cuda_graph:
            graph = CapturedGraph(self.device)
            graph_latent = latent.clone()
            graph_time = torch.zeros((), device=self.device, dtype=self.dtype)
            stream = graph.stream
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.predict_velocity(graph_latent, graph_time)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            check_cancelled(cancelled)
            with torch.cuda.graph(graph.graph, stream=stream):
                graph_velocity = self.predict_velocity(graph_latent, graph_time)
            raw_times = (
                torch.tensor(
                    [
                        time
                        for step in range(steps)
                        for time in (
                            1.0 - step * step_size,
                            1.0 - step * step_size - step_size / 2,
                        )
                    ],
                    dtype=torch.float64,
                )
                .logit()
                .clamp(-20, 20)
                .to(device=self.device, dtype=self.dtype)
            )
        else:
            pass
        for step in range(steps):
            check_cancelled(cancelled)
            time = 1.0 - step * step_size
            if graph is None:
                velocity = self.velocity(latent, time)
            else:
                graph_latent.copy_(latent)
                graph_time.copy_(raw_times[2 * step])
                graph.replay()
                velocity = graph_velocity
            midpoint = latent - velocity * (step_size / 2)
            check_cancelled(cancelled)
            if graph is None:
                velocity = self.velocity(midpoint, time - step_size / 2)
            else:
                graph_latent.copy_(midpoint)
                graph_time.copy_(raw_times[2 * step + 1])
                graph.replay()
                velocity = graph_velocity
            latent = latent - velocity * step_size
        return latent.float().cpu()


@torch.inference_mode()
def synthesize_latents(
    model: Yue2Model,
    prefix: list[int],
    codec_tokens: list[int],
    seed: int,
    steps: int,
    cancelled: CancellationCheck,
    *,
    cuda_graph: bool = False,
) -> torch.Tensor:
    chunk_frames = (model.config.max_position_embeddings - len(prefix) - 3) // 2
    if not codec_tokens:
        raise ValueError("YuE2 has no semantic tokens")
    elif chunk_frames < 1:
        raise ValueError("YuE2 request: prefix leaves no remaining acoustic context")
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
            chunks.append(chunk.solve(steps, cancelled, cuda_graph=cuda_graph))
        finally:
            chunk.cache.clear()
    latents = torch.cat(chunks)
    if not torch.isfinite(latents).all():
        raise FloatingPointError("YuE2 produced non-finite acoustic latents")
    else:
        return latents
