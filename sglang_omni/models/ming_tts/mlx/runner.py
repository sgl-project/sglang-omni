# SPDX-License-Identifier: Apache-2.0
"""Synchronous Ming latent recurrence shared by component tests and the worker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

from .flow_matching import build_cfm_timesteps
from .model import MingTTSModel


@dataclass
class RequestState:
    cache: list[Any]
    hidden: mx.array
    history: mx.array
    max_steps: int
    steps: int = 0
    feedback: mx.array | None = None


@dataclass
class LatentStep:
    latent: mx.array
    stop_prob: float
    finish_reason: Literal["stop", "length"] | None


class MingTTSMlxRunner:
    def __init__(
        self, model: MingTTSModel, *, cache_factory: Callable[[], list[Any]] | None = None
    ) -> None:
        self.model = model
        self.model.eval()
        self.states: dict[str, RequestState] = {}
        self._cache_factory = cache_factory or model.make_cache

    def start(
        self,
        request_id: str,
        input_ids: mx.array,
        *,
        max_steps: int = 256,
        reference_latents: mx.array | None = None,
        reference_start: int | None = None,
        speaker_embedding: mx.array | None = None,
        speaker_positions: Sequence[int] = (),
    ) -> None:
        if self.states:
            raise ValueError("Ming MLX latent runner supports one active request")
        if input_ids.ndim != 1 or input_ids.size == 0:
            raise ValueError("input_ids must be a nonempty one-dimensional token sequence")
        config = self.model.config
        if (max_steps < 1
                or input_ids.size + max_steps - 1 > config.llm_config.max_position_embeddings):
            raise ValueError("Requested generation exceeds the model context or has no steps")
        embeds = self.model.model.word_embeddings(input_ids[None])
        history = mx.zeros(
            (1, config.history_patch_size, config.latent_dim), dtype=mx.float32
        )
        if speaker_embedding is not None:
            if speaker_embedding.shape != (len(speaker_positions), 192):
                raise ValueError("Speaker embeddings must match speaker placeholder positions")
            if any(p < 0 or p >= input_ids.size for p in speaker_positions):
                raise ValueError("Speaker placeholder is outside the prompt")
            projected = self.model.spk_head(
                speaker_embedding.astype(self.model.spk_head.weight.dtype)
            )
            for i, position in enumerate(speaker_positions):
                embeds[0, position] = projected[i].astype(embeds.dtype)
        if reference_latents is not None:
            if (
                reference_latents.ndim != 2
                or reference_latents.shape[1] != config.latent_dim
                or reference_latents.shape[0] == 0
                or reference_latents.shape[0] % config.patch_size
            ):
                raise ValueError(
                    "Reference latents must contain complete [frames, latent_dim] patches"
                )
            count = reference_latents.shape[0] // config.patch_size
            if (reference_start is None or reference_start < 0
                    or reference_start + count > input_ids.size):
                raise ValueError("Reference latent placeholders are outside the prompt")
            projected = self.model.project_reference_latents(reference_latents)
            embeds[0, reference_start:reference_start + count] = projected.astype(embeds.dtype)
            tail = reference_latents[-config.history_patch_size:].astype(mx.float32)
            history[:, -tail.shape[0]:] = tail[None]
        cache = self._cache_factory()
        hidden = self.model(inputs_embeds=embeds, cache=cache)[:, -1:]
        mx.eval(hidden, history, [c.state for c in cache])
        self.states[request_id] = RequestState(cache, hidden, history, max_steps)

    def step(
        self,
        request_id: str,
        *,
        cfg: float | mx.array = 2.0,
        sigma: float | mx.array = 0.25,
        temperature: float | mx.array = 0.0,
        noise: mx.array | None = None,
        timesteps: mx.array | None = None,
        sde_random: mx.array | None = None,
    ) -> LatentStep:
        state = self.states[request_id]
        config = self.model.config
        try:
            if state.feedback is not None:
                state.hidden = self.model(
                    inputs_embeds=state.feedback[:, None], cache=state.cache
                )
            if timesteps is None:
                timesteps = build_cfm_timesteps()
            if noise is None:
                noise = mx.random.normal((1, config.latent_dim, config.patch_size))
            if sde_random is None:
                sde_random = mx.random.normal(
                    (timesteps.shape[0] - 2, 1, config.patch_size, config.latent_dim)
                )
            tail = self.model._compute_tail_step(
                state.hidden, state.history, noise=noise,
                timesteps=timesteps, sde_random=sde_random,
                cfg=cfg, sigma=sigma, temperature=temperature,
            )
            mx.eval(tail.sampled, tail.feedback_embeddings, tail.stop_prob)
            probability = float(tail.stop_prob[0].item())
            # Match Omni: zero-based steps > 3, and retain the terminal patch.
            stop = probability > 0.5 and state.steps > 3
            length = state.steps + 1 >= state.max_steps
            reason: Literal["stop", "length"] | None = (
                "stop" if stop else "length" if length else None
            )
            state.steps += 1
            if reason is not None:
                self.release(request_id)
            else:
                state.history = mx.concatenate((state.history, tail.sampled), axis=1)[
                    :, -config.history_patch_size:
                ]
                state.feedback = tail.feedback_embeddings.astype(
                    self.model.model.word_embeddings.weight.dtype
                )
                mx.eval(state.history, state.feedback)
            return LatentStep(tail.sampled[0], probability, reason)
        except Exception:
            self.release(request_id)
            raise

    def release(self, request_id: str) -> None:
        self.states.pop(request_id, None)
