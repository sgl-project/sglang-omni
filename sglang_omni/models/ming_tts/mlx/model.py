# SPDX-License-Identifier: Apache-2.0
"""Native MLX Ming A3B AR model; AudioVAE is owned by a separate stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .acoustic import Aggregator
from .backbone import BailingMoeTextModel
from .config import AcousticConfig, ModelConfig
from .flow_matching import FlowLoss


@dataclass
class MingTTSTailOutputs:
    sampled: mx.array
    feedback_embeddings: mx.array
    stop_prob: mx.array


class MingTTSModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BailingMoeTextModel(config.llm_config)
        self.linear_proj_audio = Aggregator(
            AcousticConfig.from_dict(config.aggregator_config),
            config.latent_dim, config.llm_config.hidden_size,
        )
        self.flowloss = FlowLoss(
            AcousticConfig.from_dict(config.ditar_config),
            config.latent_dim, config.llm_config.hidden_size,
        )
        self.stop_head = nn.Linear(config.llm_config.hidden_size, 2)
        self.spk_head = nn.Linear(192, config.llm_config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return self.config.llm_config.vocab_size

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        positions: mx.array | None = None,
        cache: Sequence[KVCache | None] | None = None,
    ) -> mx.array:
        return self.model(
            input_ids, inputs_embeds=inputs_embeds, positions=positions, cache=cache
        )

    def make_cache(self) -> list[KVCache]:
        return self.model.make_cache()

    def project_reference_latents(self, latents: mx.array) -> mx.array:
        patches = latents.reshape(-1, self.config.patch_size, self.config.latent_dim)
        return self.linear_proj_audio(patches).reshape(
            -1, self.config.llm_config.hidden_size
        )

    def _compute_tail_step(
        self,
        hidden_states: mx.array,
        latent_history: mx.array,
        *,
        noise: mx.array,
        timesteps: mx.array,
        sde_random: mx.array,
        cfg: float | mx.array = 2.0,
        sigma: float | mx.array = 0.25,
        temperature: float | mx.array = 0.0,
    ) -> MingTTSTailOutputs:
        sampled = self.flowloss.sample(
            hidden_states, latent_history, noise, timesteps, sde_random,
            cfg=cfg, sigma=sigma, temperature=temperature,
        )
        feedback = self.linear_proj_audio(sampled).reshape(
            int(hidden_states.shape[0]),
            -1,
        )
        stop_logits = self.stop_head(hidden_states.astype(self.stop_head.weight.dtype))
        stop = mx.softmax(stop_logits, axis=-1)[:, 0, 1]
        return MingTTSTailOutputs(sampled, feedback, stop)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Map official unquantized AR weights; strict load checks all remaining keys."""
        mapped = {}
        for key, value in weights.items():
            if key.startswith(("audio.", "model.lm_head.")):
                continue
            if key.endswith(".rotary_emb.inv_freq") or key in (
                "linear_proj_audio.rotary_embed.inv_freq",
                "flowloss.cfm.model.rotary_embed.inv_freq",
            ):
                continue
            if key.startswith("model.model."):
                key = key[len("model."):]
            key = key.replace(".mlp.ff.0.0.", ".mlp.fc1.")
            key = key.replace(".mlp.ff.2.", ".mlp.fc2.")
            if key in mapped:
                raise ValueError(f"Duplicate checkpoint mapping: {key}")
            mapped[key] = value
        for i in range(len(self.model.layers)):
            if i < self.config.llm_config.first_k_dense_replace:
                continue
            prefix = f"model.layers.{i}.mlp.experts"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                first = f"{prefix}.0.{proj}.weight"
                if first not in mapped:
                    continue
                target = f"{prefix}.{proj}.weight"
                if target in mapped:
                    raise ValueError(f"Both stacked and individual expert weights: {target}")
                mapped[target] = mx.stack([
                    mapped.pop(f"{prefix}.{e}.{proj}.weight")
                    for e in range(self.config.llm_config.num_experts)
                ])
        return mapped
