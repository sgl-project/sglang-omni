# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for the Stable Audio 3 prompt encoder and DiT step.

Both are launch-bound for short clips: a DiT step runs about 1700 small kernels
with an H200 busy a fifth of the time, and the prompt encoder, at a fixed
prompt length, spends 9 ms launching 2 ms of work. Each bucket records one
graph on first use and later calls replay it.

Rows pad up to the batch bucket and latents up to the length bucket. Every
row's keys stop at its own key length, so FlashAttention never reads a padded
position or row, and padded outputs are dropped. The DiT conditions are copied
into shared buffers once per batch; a step copies only the latents and the
timestep.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from sglang.srt.utils.common import get_available_gpu_memory
from transformers import T5GemmaEncoderModel

from sglang_omni.models.stable_audio3.dit import (
    DiffusionTransformer,
    SamplingConditions,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

logger = logging.getLogger(__name__)

BATCH_BUCKETS = (1, 2, 4, 8)
LATENT_BUCKET_STEP = 32
# note (ruilin): on H200 an eager step stays near 20 ms up to about 2000 tokens, then turns GPU-bound; past that a graph saves about 3 ms and bucket padding can cost more.
MAX_GRAPH_TOKENS = 2048
MIN_FREE_GB = 3.0
WARMUP_ITERATIONS = 2

GraphedStep = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class StepGraph:
    graph: ReplayableGraph
    latents: torch.Tensor
    timesteps: torch.Tensor
    rotary_angles: torch.Tensor
    velocity: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class PromptGraph:
    graph: ReplayableGraph
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class ConditionBuffers:
    """Conditions for the largest batch bucket; smaller buckets read leading rows."""

    seconds_embedding: torch.Tensor
    context_keys_values: list[tuple[torch.Tensor, torch.Tensor]]
    local_embeddings: list[torch.Tensor]
    key_lengths: torch.Tensor


def bucket_batch(batch_size: int, max_batch_size: int) -> int | None:
    for bucket in BATCH_BUCKETS:
        if bucket >= batch_size and bucket <= max_batch_size:
            return bucket
        else:
            pass
    return max_batch_size if batch_size <= max_batch_size else None


def bucket_latents(latent_length: int) -> int:
    return -(-latent_length // LATENT_BUCKET_STEP) * LATENT_BUCKET_STEP


def capture_graph(
    forward: Callable[[], torch.Tensor],
    *,
    graph_backend: DeviceGraphBackend,
    pool: DeviceGraphPool,
    device: torch.device,
    label: str,
) -> tuple[ReplayableGraph, torch.Tensor] | None:
    """Warms forward up on a side stream and records it, or returns None to run eager."""
    free_gb = get_available_gpu_memory(
        device.type, device.index or 0, distributed=False, empty_cache=False
    )
    if free_gb < MIN_FREE_GB:
        logger.warning(
            f"Stable Audio 3 {label} graph: {free_gb:.1f} GB free < {MIN_FREE_GB:.1f} GB, running eager"
        )
        return None
    else:
        pass
    device_module = torch.get_device_module(device)
    try:
        with current_platform.graph_capture_attention():
            stream = device_module.Stream(device=device)
            stream.wait_stream(device_module.current_stream())
            with device_module.stream(stream):
                for _ in range(WARMUP_ITERATIONS):
                    forward()
            device_module.current_stream().wait_stream(stream)
            device_module.synchronize()
            entry_stream = device_module.current_stream(device)
            try:
                with graph_backend.capture(
                    pool=pool, thread_local_errors=True
                ) as graph:
                    output = forward()
            finally:
                device_module.set_stream(entry_stream)
    except Exception as error:
        logger.warning(
            f"Stable Audio 3 {label} graph capture failed: {error}; running eager",
            exc_info=True,
        )
        return None
    logger.info(f"Captured Stable Audio 3 {label} graph")
    return graph, output


class PromptEncoderCudaGraphRunner:
    """Replays the text encoder per batch bucket; prompts pad to one fixed length, so the batch is the only size that varies."""

    def __init__(
        self,
        text_encoder: T5GemmaEncoderModel,
        *,
        graph_backend: DeviceGraphBackend,
        max_batch_size: int,
    ) -> None:
        self.text_encoder = text_encoder
        self.graph_backend = graph_backend
        self.max_batch_size = max_batch_size
        self.device = next(text_encoder.parameters()).device
        self.graphs: dict[int, PromptGraph] = {}
        self.failed: set[int] = set()
        self.pool = graph_backend.graph_pool_handle()

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Hidden states for the prompts; the result is a view the next call overwrites."""
        batch_size = input_ids.shape[0]
        bucket = bucket_batch(batch_size, self.max_batch_size)
        entry = None
        if bucket is None or bucket in self.failed:
            pass
        else:
            entry = self.graphs.get(bucket)
            if entry is None:
                entry = self.capture(bucket, input_ids.shape[1])
            else:
                pass
        if entry is None:
            if bucket is not None:
                self.failed.add(bucket)
            else:
                pass
            return self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        else:
            self.graphs[bucket] = entry
        entry.input_ids[:batch_size].copy_(input_ids)
        entry.attention_mask[:batch_size].copy_(attention_mask)
        entry.graph.replay()
        return entry.hidden_states[:batch_size]

    def capture(self, bucket: int, prompt_length: int) -> PromptGraph | None:
        # note (ruilin): padded rows keep every token visible, so their attention stays finite.
        input_ids = torch.zeros(
            bucket, prompt_length, device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones(
            bucket, prompt_length, device=self.device, dtype=torch.bool
        )
        captured = capture_graph(
            lambda: self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state,
            graph_backend=self.graph_backend,
            pool=self.pool,
            device=self.device,
            label=f"prompt encoder batch={bucket}",
        )
        if captured is None:
            return None
        else:
            graph, hidden_states = captured
            return PromptGraph(
                graph=graph,
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
            )


class DiTCudaGraphRunner:
    """Captures one DiT step per bucket and replays it; buckets past the token budget run eager."""

    def __init__(
        self,
        dit: DiffusionTransformer,
        *,
        graph_backend: DeviceGraphBackend,
        max_batch_size: int,
    ) -> None:
        self.dit = dit
        self.graph_backend = graph_backend
        self.max_batch_size = max_batch_size
        reference = next(dit.parameters())
        self.device = reference.device
        self.dtype = reference.dtype
        self.num_memory_tokens = dit.transformer.memory_tokens.shape[0]
        self.graphs: dict[tuple[int, int], StepGraph] = {}
        self.failed: set[tuple[int, int]] = set()
        self.pool = graph_backend.graph_pool_handle()
        self.conditions: ConditionBuffers | None = None

    def bind(
        self, conditions: SamplingConditions, batch_size: int, latent_length: int
    ) -> GraphedStep | None:
        """Loads this batch's conditions and returns its graphed step, or None to run eager."""
        batch_bucket = bucket_batch(batch_size, self.max_batch_size)
        latent_bucket = bucket_latents(latent_length)
        if (
            batch_bucket is None
            or batch_bucket * (latent_bucket + self.num_memory_tokens)
            > MAX_GRAPH_TOKENS
        ):
            return None
        else:
            pass
        key = (batch_bucket, latent_bucket)
        if key in self.failed:
            return None
        else:
            pass
        self.load_conditions(conditions, batch_size)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self.capture(batch_bucket, latent_bucket)
        else:
            pass
        if entry is None:
            self.failed.add(key)
            return None
        else:
            self.graphs[key] = entry

        def step(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            entry.latents[:batch_size, :, :latent_length].copy_(latents)
            entry.timesteps.copy_(timestep.expand(batch_bucket))
            entry.graph.replay()
            # note (ruilin): the next replay overwrites this view, so the caller consumes it on this stream first.
            return entry.velocity[:batch_size, :, :latent_length]

        return step

    def load_conditions(self, conditions: SamplingConditions, batch_size: int) -> None:
        if self.conditions is None:
            rows = self.max_batch_size
            self.conditions = ConditionBuffers(
                seconds_embedding=conditions.seconds_embedding.new_zeros(
                    rows, *conditions.seconds_embedding.shape[1:]
                ),
                context_keys_values=[
                    (
                        keys.new_zeros(rows, *keys.shape[1:]),
                        values.new_zeros(rows, *values.shape[1:]),
                    )
                    for keys, values in conditions.context_keys_values
                ],
                local_embeddings=[
                    embedding.clone() for embedding in conditions.local_embeddings
                ],
                key_lengths=conditions.key_lengths.new_zeros(rows),
            )
        else:
            pass
        buffers = self.conditions
        buffers.seconds_embedding[:batch_size].copy_(conditions.seconds_embedding)
        for (keys, values), (new_keys, new_values) in zip(
            buffers.context_keys_values, conditions.context_keys_values, strict=True
        ):
            keys[:batch_size].copy_(new_keys)
            values[:batch_size].copy_(new_values)
        for embedding, new_embedding in zip(
            buffers.local_embeddings, conditions.local_embeddings, strict=True
        ):
            embedding.copy_(new_embedding)
        # note (ruilin): rows past the batch keep only the memory tokens as keys, so a smaller bucket never reads past its own length.
        buffers.key_lengths.fill_(self.num_memory_tokens)
        buffers.key_lengths[:batch_size].copy_(conditions.key_lengths)

    def capture(self, batch_bucket: int, latent_bucket: int) -> StepGraph | None:
        buffers = self.conditions
        assert buffers is not None
        latents = torch.zeros(
            batch_bucket,
            self.dit.transformer.project_in.in_features,
            latent_bucket,
            device=self.device,
            dtype=self.dtype,
        )
        timesteps = torch.zeros(batch_bucket, device=self.device, dtype=self.dtype)
        rotary_angles = self.dit.transformer.rotary_pos_emb(
            latent_bucket + self.num_memory_tokens
        )
        conditions = SamplingConditions(
            seconds_embedding=buffers.seconds_embedding[:batch_bucket],
            context_keys_values=[
                (keys[:batch_bucket], values[:batch_bucket])
                for keys, values in buffers.context_keys_values
            ],
            local_embeddings=buffers.local_embeddings,
            rotary_angles=rotary_angles,
            key_lengths=buffers.key_lengths[:batch_bucket],
        )
        captured = capture_graph(
            lambda: self.dit(latents, timesteps, conditions),
            graph_backend=self.graph_backend,
            pool=self.pool,
            device=self.device,
            label=f"DiT batch={batch_bucket} latents={latent_bucket}",
        )
        if captured is None:
            return None
        else:
            graph, velocity = captured
            return StepGraph(
                graph=graph,
                latents=latents,
                timesteps=timesteps,
                rotary_angles=rotary_angles,
                velocity=velocity,
            )
