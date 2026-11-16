# SPDX-License-Identifier: Apache-2.0
"""Blocks of several Sessions generated as one batch, from KV caches kept across steps and blocks.

The reference path re-encodes the whole retained history of both audio
streams and re-runs it through the backbone once per step. Here each Session
keeps, in a slot of two KV pools, the audio encoder's keys and values of both
streams and the backbone's of its prompt. Both stacks are causal, so a
position's keys and values depend only on the positions before it: a new
block recomputes only what its audio changed, from the last encoder group of
the block before (whose frames read past that block's end) and, in the
backbone, from that group's embedding on: about 43 encoder and 44 backbone
positions a block instead of up to 600 and 640. A Session whose history was
cut back to its last blocks starts over. The five steps after a block's first
then each run one position per Session against the backbone cache; on a GPU
they run as one captured CUDA graph per batch size.

A batch is right-padded to its longest row. The codes follow the
reference's in distribution, not bit for bit: the kernels and their rounding
differ.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from sglang_omni.models.ellsa2.fbank import LogMelFilterbank, tail_start
from sglang_omni.models.ellsa2.generator import (
    AUDIO_EMBEDDINGS_PER_BLOCK,
    BLOCK_SAMPLES,
    BLOCK_STEPS,
    CONNECTOR_FRAMES,
    NUM_CODEBOOKS,
    Ellsa2Generator,
    PromptTokens,
)
from sglang_omni.models.ellsa2.rvq import CODEBOOK_SIZE

# Batch sizes with a captured graph; a batch runs in the smallest that holds it.
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128)
# Backbone positions a Session keeps: the 12 s history's longest prompt (639 with the release's 150-step style) and its six steps, rounded up.
DEFAULT_CACHE_POSITIONS = 768
# Encoder positions a stream keeps: 40 a block, 599 for the 12 s history, rounded up.
DEFAULT_ENCODER_POSITIONS = 640
# Log-mel frames a block adds; the subsampling halves them into encoder positions.
FRAMES_PER_BLOCK = BLOCK_SAMPLES // 160
# A block's backbone positions: its two audio streams between their markers, then its motion once completed.
BLOCK_POSITIONS = 2 * (AUDIO_EMBEDDINGS_PER_BLOCK + 2) + BLOCK_STEPS
# Key positions are gathered in multiples of this, so an attention mask row stays aligned.
POSITION_ALIGNMENT = 64


def audio_from(blocks: list[np.ndarray], first_frame: int) -> np.ndarray:
    """A stream's audio from the first sample log-mel frame first_frame reads on: all of it for the first frame, else from within the blocks that hold that sample."""
    if first_frame == 0:
        return np.concatenate(blocks)
    else:
        start = tail_start(first_frame)
        first_block = start // BLOCK_SAMPLES
        return np.concatenate(blocks[first_block:])[
            start - first_block * BLOCK_SAMPLES :
        ]


@dataclass(kw_only=True)
class SessionCache:
    """One Session's slot in the KV and embedding pools, and the blocks of history the slot holds."""

    slot: int
    blocks: int = 0


@dataclass(frozen=True, kw_only=True)
class BlockRequest:
    """One Session's next block: both streams' float32 16 kHz audio of each block of the retained history and of this block, the codes known before it, and the Session's cache."""

    target_audio: list[np.ndarray]
    partner_audio: list[np.ndarray]
    known_codes: torch.Tensor
    style_steps: int
    num_blocks: int
    cache: SessionCache


@dataclass(kw_only=True)
class StepGraph:
    """The five cached steps of one batch size, captured with their static inputs and outputs."""

    graph: torch.cuda.CUDAGraph
    codes: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor
    steps: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class Stack:
    """A causal stack of Llama-style layers and the shape of its attention."""

    layers: nn.ModuleList
    rotary: nn.Module
    norm: nn.Module
    num_heads: int
    num_kv_heads: int
    head_dim: int


class BatchedBlocks:
    """Generates the next block of several Sessions at once on one generator."""

    def __init__(
        self,
        generator: Ellsa2Generator,
        filterbank: LogMelFilterbank,
        *,
        max_batch: int = GRAPH_BATCH_SIZES[-1],
        cache_positions: int = DEFAULT_CACHE_POSITIONS,
        encoder_positions: int = DEFAULT_ENCODER_POSITIONS,
        cuda_graphs: bool = True,
        reuse_history: bool = True,
    ) -> None:
        self.generator = generator
        self.filterbank = filterbank
        backbone = generator.backbone.config
        self.backbone = Stack(
            layers=generator.backbone.layers,
            rotary=generator.backbone.rotary_emb,
            norm=generator.backbone.norm,
            num_heads=backbone.num_attention_heads,
            num_kv_heads=backbone.num_key_value_heads,
            head_dim=backbone.hidden_size // backbone.num_attention_heads,
        )
        encoder = generator.audio_encoder.encoder
        encoder_config = encoder.layers[0].self_attn.config
        self.encoder = Stack(
            layers=encoder.layers,
            rotary=encoder.rotary_emb,
            norm=encoder.norm,
            num_heads=encoder_config.num_attention_heads,
            num_kv_heads=encoder_config.num_key_value_heads,
            head_dim=encoder_config.hidden_size // encoder_config.num_attention_heads,
        )
        self.device = generator.motion_embed_weights.device
        self.dtype = generator.backbone.embed_tokens.weight.dtype
        self.max_batch = self.graph_batch(max_batch)
        self.cuda_graphs = cuda_graphs and self.device.type == "cuda"
        self.reuse_history = reuse_history
        self.cache_positions = cache_positions
        self.encoder_positions = encoder_positions
        # The decode steps' working cache: the batch's backbone keys and values, one row per Session of the batch.
        self.cache = self.new_pool(self.backbone, self.max_batch, cache_positions)
        self.slots = 0
        self.backbone_pool = self.new_pool(self.backbone, 0, cache_positions)
        self.encoder_pool = self.new_pool(self.encoder, 0, encoder_positions)
        # Both streams' audio embeddings of each slot's history: [2 * slots, embeddings, hidden].
        self.embedding_pool = self.new_embeddings(0)
        self.free_slots: list[int] = []
        self.released: list[int] = []
        self.release_lock = threading.Lock()
        self.graphs: dict[int, StepGraph] = {}
        self.layouts: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.graph_pool = None
        # The 18 heads as one projection: [NUM_CODEBOOKS * CODEBOOK_SIZE, hidden].
        self.head_weight = torch.cat([head.weight for head in generator.motion_heads])
        self.head_bias = torch.cat([head.bias for head in generator.motion_heads])

    def graph_batch(self, batch: int) -> int:
        for size in GRAPH_BATCH_SIZES:
            if size >= batch:
                return size
            else:
                pass
        return batch

    def new_pool(self, stack: Stack, rows: int, positions: int) -> torch.Tensor:
        """[layers, key/value, rows, kv heads, positions, head dim]."""
        return torch.zeros(
            len(stack.layers),
            2,
            rows,
            stack.num_kv_heads,
            positions,
            stack.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def new_embeddings(self, rows: int) -> torch.Tensor:
        return torch.zeros(
            rows,
            self.encoder_positions // CONNECTOR_FRAMES,
            self.generator.backbone.config.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

    def acquire(self, in_use: set[int]) -> SessionCache:
        """A free slot for a new Session; slots released while a batch may still use them wait for one that does not."""
        with self.release_lock:
            waiting = [slot for slot in self.released if slot in in_use]
            self.free_slots += [slot for slot in self.released if slot not in in_use]
            self.released = waiting
        if not self.free_slots:
            self.grow(max(self.slots * 2, 8))
        else:
            pass
        return SessionCache(slot=self.free_slots.pop(0))

    def release(self, cache: SessionCache) -> None:
        """Called from any thread once the cache's Session is closed."""
        with self.release_lock:
            self.released.append(cache.slot)

    def grow(self, slots: int) -> None:
        backbone = self.new_pool(self.backbone, slots, self.cache_positions)
        encoder = self.new_pool(self.encoder, 2 * slots, self.encoder_positions)
        embeddings = self.new_embeddings(2 * slots)
        backbone[:, :, : self.slots] = self.backbone_pool
        encoder[:, :, : 2 * self.slots] = self.encoder_pool
        embeddings[: 2 * self.slots] = self.embedding_pool
        self.backbone_pool, self.encoder_pool = backbone, encoder
        self.embedding_pool = embeddings
        self.free_slots += list(range(self.slots, slots))
        self.slots = slots

    def attention(self) -> sdpa_kernel:
        # Note (Junnan Li): cuDNN is left out: it builds an execution plan for every new shape, and batches bring many.
        return sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )

    def motion_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Ellsa2Generator.motion_embeddings outside autocast, which would sum the codebooks in float32: the reference sums them in the weights' dtype."""
        with torch.autocast(self.device.type, enabled=False):
            return self.generator.motion_embeddings(codes)

    def heads(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final hidden states [batch, hidden] → greedy codes [batch, NUM_CODEBOOKS]."""
        logits = F.linear(hidden, self.head_weight, self.head_bias)
        return logits.view(-1, NUM_CODEBOOKS, CODEBOOK_SIZE).float().argmax(dim=-1)

    def layer_inputs(
        self,
        stack: Stack,
        layer: nn.Module,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[batch, length, width] → rotated queries, keys and values [batch, heads, length, head dim]."""
        attention = layer.self_attn
        normed = layer.input_layernorm(hidden)
        shape = (*hidden.shape[:2], -1, stack.head_dim)
        query = attention.q_proj(normed).view(shape).transpose(1, 2)
        key = attention.k_proj(normed).view(shape).transpose(1, 2)
        value = attention.v_proj(normed).view(shape).transpose(1, 2)
        cos, sin = stack.rotary(hidden, positions)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value

    def layer_outputs(
        self, layer: nn.Module, hidden: torch.Tensor, attended: torch.Tensor
    ) -> torch.Tensor:
        """The attention output [batch, length, heads * head dim] through o_proj, the residuals and the MLP."""
        hidden = hidden + layer.self_attn.o_proj(attended)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def extend(
        self,
        stack: Stack,
        pool: torch.Tensor,
        slots: torch.Tensor,
        starts: torch.Tensor,
        counts: torch.Tensor,
        hidden: torch.Tensor,
        gathered: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run hidden [rows, length, width], each row's first counts[row] positions following the starts[row] its slot of pool keeps, through the stack; their keys and values join the pool. With gathered, each row's keys and values end up in its rows too. Returns the stack's normed output [rows, length, width]."""
        rows, length, _ = hidden.shape
        offsets = torch.arange(length, device=hidden.device)
        positions = starts[:, None] + offsets
        real = offsets < counts[:, None]
        row_index, offset_index = real.nonzero(as_tuple=True)
        write_slots = slots[row_index]
        write_positions = positions[row_index, offset_index]
        span = int((starts + counts).max())
        cold = int(starts.max()) == 0
        if not cold:
            span = -(-span // POSITION_ALIGNMENT) * POSITION_ALIGNMENT
            # Each position reads the positions up to its own.
            mask = (
                torch.arange(span, device=hidden.device)[None, None, :]
                <= positions[:, :, None]
            )[:, None]
        else:
            mask = None
        groups = stack.num_heads // stack.num_kv_heads
        for index, layer in enumerate(stack.layers):
            query, key, value = self.layer_inputs(stack, layer, hidden, positions)
            keys = pool[index, 0]
            values = pool[index, 1]
            keys[write_slots, :, write_positions] = key.transpose(1, 2)[real]
            values[write_slots, :, write_positions] = value.transpose(1, 2)[real]
            if cold:
                # No row keeps earlier positions: the new ones, right-padded under a causal mask, are all a row reads.
                row_keys, row_values = key, value
            else:
                row_keys = keys[:, :, :span].index_select(0, slots)
                row_values = values[:, :, :span].index_select(0, slots)
            if gathered is not None:
                width = row_keys.shape[2]
                gathered[index, 0, :rows, :, :width] = row_keys
                gathered[index, 1, :rows, :, :width] = row_values
            else:
                pass
            if groups > 1:
                row_keys = row_keys.repeat_interleave(groups, dim=1)
                row_values = row_values.repeat_interleave(groups, dim=1)
            else:
                pass
            attended = F.scaled_dot_product_attention(
                query, row_keys, row_values, attn_mask=mask, is_causal=cold
            )
            hidden = self.layer_outputs(
                layer, hidden, attended.transpose(1, 2).reshape(rows, length, -1)
            )
        return stack.norm(hidden)

    def encoder_start(self, request: BlockRequest) -> int:
        """The first encoder position the block recomputes: 0 for a Session starting over, else the first of the last group of the history before, whose last frames read past its end."""
        if self.is_cold(request):
            return 0
        else:
            previous = request.num_blocks - 1
            return CONNECTOR_FRAMES * (previous * AUDIO_EMBEDDINGS_PER_BLOCK - 1)

    def is_cold(self, request: BlockRequest) -> bool:
        return (
            not self.reuse_history
            or request.cache.blocks == 0
            or request.cache.blocks != request.num_blocks - 1
        )

    @torch.no_grad()
    def encode_audio(self, requests: list[BlockRequest]) -> None:
        """Bring each request's audio embeddings of both streams in the embedding pool up to its block, recomputing from encoder_start on."""
        generator = self.generator
        streams = [(request, "target") for request in requests] + [
            (request, "partner") for request in requests
        ]
        starts = [self.encoder_start(request) for request, _ in streams]
        # A recomputed position reads log-mel frames from two before its doubled position on: the subsampling's first row there reads one frame short and is dropped.
        first_frames = [max(2 * start - 2, 0) for start in starts]
        features = self.features(
            [
                audio_from(getattr(request, name + "_audio"), first)
                for (request, name), first in zip(streams, first_frames)
            ],
            first_frames,
        )
        frames = [len(item) for item in features]
        batch = torch.stack(
            [F.pad(item, (0, 0, 0, max(frames) - len(item))) for item in features]
        )
        totals = [
            request.num_blocks * FRAMES_PER_BLOCK // 2 - 1 for request, _ in streams
        ]
        counts = torch.tensor(
            [total - start for total, start in zip(totals, starts)],
            device=self.device,
        )
        drop = torch.tensor([int(start > 0) for start in starts], device=self.device)
        width = int(counts.max())
        with generator.autocast(), self.attention():
            subsampled = generator.audio_encoder.encoder_embed(batch)
            index = (
                torch.arange(width, device=self.device)[None] + drop[:, None]
            ).clamp(max=subsampled.shape[1] - 1)
            hidden = subsampled.gather(
                1, index[:, :, None].expand(-1, -1, subsampled.shape[2])
            )
            slots = torch.tensor(
                [
                    2 * request.cache.slot + int(name == "partner")
                    for request, name in streams
                ],
                device=self.device,
            )
            hidden = self.extend(
                self.encoder,
                self.encoder_pool,
                slots,
                torch.tensor(starts, device=self.device),
                counts,
                hidden,
            )
            hidden = generator.ln_audio(hidden)
            valid = torch.arange(width, device=self.device) < counts[:, None]
            hidden = hidden * valid[:, :, None].to(hidden.dtype)
            hidden = F.pad(hidden, (0, 0, 0, -width % CONNECTOR_FRAMES))
            embeddings = generator.connector(
                hidden.reshape(
                    len(streams), -1, generator.encoder_width * CONNECTOR_FRAMES
                )
            )
        fresh = (
            torch.arange(embeddings.shape[1], device=self.device)
            < (counts[:, None] + CONNECTOR_FRAMES - 1) // CONNECTOR_FRAMES
        )
        rows, offsets = fresh.nonzero(as_tuple=True)
        first = torch.tensor(starts, device=self.device) // CONNECTOR_FRAMES
        self.embedding_pool[slots[rows], first[rows] + offsets] = embeddings[
            rows, offsets
        ]

    def features(
        self, streams: list[np.ndarray], first_frames: list[int]
    ) -> list[torch.Tensor]:
        """Each stream's log-mel features from its first frame on, on the generator's device; streams alike are computed there as one batch."""
        alike: dict[tuple[int, int], list[int]] = {}
        for index, (stream, first) in enumerate(zip(streams, first_frames)):
            alike.setdefault((len(stream), first), []).append(index)
        features: list[torch.Tensor | None] = [None] * len(streams)
        for (_, first), indices in alike.items():
            computed = self.filterbank.many(
                torch.from_numpy(np.stack([streams[index] for index in indices])).to(
                    self.device
                ),
                first,
            )
            for row, index in enumerate(indices):
                # Under autocast the subsampling reads them in the generator's dtype anyway.
                features[index] = computed[row].to(self.dtype)
        return features

    def backbone_start(self, prompt: PromptTokens, request: BlockRequest) -> int:
        """The first prompt position the block recomputes: 0 for a Session starting over, else the last target embedding of the block before, the first one encode_audio changed."""
        if self.is_cold(request):
            return 0
        else:
            return (
                len(prompt.head)
                + request.style_steps
                + (request.num_blocks - 2) * BLOCK_POSITIONS
                + AUDIO_EMBEDDINGS_PER_BLOCK
            )

    def prompts(
        self, prompt: PromptTokens, requests: list[BlockRequest], starts: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each request's backbone inputs from its start on, laid out as Ellsa2Generator.block_inputs lays them out: right-padded [len(requests), length, hidden] and each row's length."""
        generator = self.generator
        head = len(prompt.head)
        known = [len(request.known_codes) for request in requests]
        with generator.autocast():
            fixed = generator.backbone.embed_tokens(
                torch.cat(
                    [
                        prompt.head,
                        torch.tensor(
                            [prompt.audio_start, prompt.audio_end],
                            device=prompt.head.device,
                        ),
                    ]
                )
            )
        # One source for the head, the markers and every request's motion; the audio comes from the embedding pool.
        small = torch.cat(
            [
                fixed,
                self.motion_embeddings(
                    torch.cat([request.known_codes for request in requests])[None]
                )[0],
            ]
        )
        per_stream = self.embedding_pool.shape[1]
        rows = []
        motion_offset = len(fixed)
        for request, start, count in zip(requests, starts, known):
            layout = self.layout(head, count, request.style_steps, request.num_blocks)[
                start:
            ]
            target = head + 2 + count
            partner = target + request.num_blocks * AUDIO_EMBEDDINGS_PER_BLOCK
            audio = layout >= target
            small_index = np.where(
                layout < head + 2, layout, layout - (head + 2) + motion_offset
            )
            pool_index = np.where(
                layout < partner,
                2 * request.cache.slot * per_stream + layout - target,
                (2 * request.cache.slot + 1) * per_stream + layout - partner,
            )
            rows.append(
                np.stack(
                    [
                        np.where(audio, 0, small_index),
                        np.where(audio, pool_index, 0),
                        audio,
                    ]
                )
            )
            motion_offset += count
        length = max(row.shape[1] for row in rows)
        index = np.zeros((len(rows), 3, length), dtype=np.int64)
        for row, item in enumerate(rows):
            index[row, :, : item.shape[1]] = item
        index = torch.from_numpy(index).to(self.device)
        embeds = torch.where(
            index[:, 2, :, None].bool(),
            self.embedding_pool.flatten(0, 1)[index[:, 1]],
            small[index[:, 0]],
        )
        counts = torch.tensor([row.shape[1] for row in rows], device=self.device)
        return embeds, counts

    def layout(
        self, head: int, known: int, style_steps: int, num_blocks: int
    ) -> np.ndarray:
        """Where each prompt position reads from a source of the head's tokens, the two audio markers, the known codes' motion, then both streams' audio."""
        key = (head, known, style_steps, num_blocks)
        cached = self.layouts.get(key)
        if cached is not None:
            return cached
        else:
            pass
        start, end = head, head + 1
        motion = head + 2
        target = motion + known
        partner = target + num_blocks * AUDIO_EMBEDDINGS_PER_BLOCK
        index = list(range(head)) + list(range(motion, motion + style_steps))
        for block in range(num_blocks):
            audio = range(
                block * AUDIO_EMBEDDINGS_PER_BLOCK,
                (block + 1) * AUDIO_EMBEDDINGS_PER_BLOCK,
            )
            index += [start] + [target + position for position in audio] + [end]
            index += [start] + [partner + position for position in audio] + [end]
            if block < num_blocks - 1:
                first = motion + style_steps + block * BLOCK_STEPS
                index += list(range(first, first + BLOCK_STEPS))
            else:
                pass
        layout = np.asarray(index, dtype=np.int64)
        self.layouts[key] = layout
        return layout

    def decode_steps(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        valid: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """The five cached steps after the first: codes [batch, NUM_CODEBOOKS] of the step before, each row's prompt length [batch] and key mask [batch, positions] over the working cache; writes the steps' codes into steps [batch, BLOCK_STEPS - 1, NUM_CODEBOOKS] and updates valid in place."""
        stack = self.backbone
        batch = codes.shape[0]
        rows = torch.arange(batch, device=codes.device)
        groups = stack.num_heads // stack.num_kv_heads
        for step in range(BLOCK_STEPS - 1):
            positions = lengths + step
            valid.scatter_(1, positions[:, None], True)
            mask = valid[:, None, None, :]
            hidden = self.motion_embeddings(codes[:, None, :])
            for index, layer in enumerate(stack.layers):
                query, key, value = self.layer_inputs(
                    stack, layer, hidden, positions[:, None]
                )
                keys = self.cache[index, 0, :batch]
                values = self.cache[index, 1, :batch]
                keys[rows, :, positions] = key[:, :, 0]
                values[rows, :, positions] = value[:, :, 0]
                # One query position per row: the heads that share a key/value head stand in as its query rows.
                attended = F.scaled_dot_product_attention(
                    query.reshape(batch, stack.num_kv_heads, groups, stack.head_dim),
                    keys,
                    values,
                    attn_mask=mask,
                )
                hidden = self.layer_outputs(
                    layer, hidden, attended.reshape(batch, 1, -1)
                )
            codes = self.heads(stack.norm(hidden[:, 0]))
            steps[:, step] = codes

    def step_graph(self, batch: int) -> StepGraph:
        graph = self.graphs.get(batch)
        if graph is not None:
            return graph
        else:
            pass
        positions = self.cache.shape[4]
        codes = torch.zeros(batch, NUM_CODEBOOKS, dtype=torch.long, device=self.device)
        lengths = torch.ones(batch, dtype=torch.long, device=self.device)
        valid = torch.zeros(batch, positions, dtype=torch.bool, device=self.device)
        steps = torch.zeros(
            batch,
            BLOCK_STEPS - 1,
            NUM_CODEBOOKS,
            dtype=torch.long,
            device=self.device,
        )
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                valid.zero_()
                valid[:, 0] = True
                self.run_steps(codes, lengths, valid, steps)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            self.run_steps(codes, lengths, valid, steps)
        self.graph_pool = graph.pool()
        captured = StepGraph(
            graph=graph, codes=codes, lengths=lengths, valid=valid, steps=steps
        )
        self.graphs[batch] = captured
        return captured

    def run_steps(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        valid: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        with self.generator.autocast(), self.attention():
            self.decode_steps(codes, lengths, valid, steps)

    @torch.no_grad()
    def generate(
        self, prompt: PromptTokens, requests: list[BlockRequest]
    ) -> torch.Tensor:
        """Greedy codes [len(requests), BLOCK_STEPS, NUM_CODEBOOKS] of each request's block; each request's cache then holds its history through this block."""
        count = len(requests)
        batch = self.graph_batch(count) if self.cuda_graphs else count
        if batch > self.max_batch:
            self.max_batch = batch
            self.cache = self.new_pool(self.backbone, batch, self.cache_positions)
            self.graphs.clear()
        else:
            pass
        starts = [self.backbone_start(prompt, request) for request in requests]
        self.encode_audio(requests)
        embeds, counts = self.prompts(prompt, requests, starts)
        lengths = torch.tensor(starts, device=self.device) + counts
        if int(lengths.max()) + BLOCK_STEPS > self.cache_positions:
            raise ValueError(
                f"an ELLSA 2 prompt of {int(lengths.max())} positions and its steps exceed the {self.cache_positions} the cache keeps"
            )
        else:
            pass
        with self.generator.autocast(), self.attention():
            hidden = self.extend(
                self.backbone,
                self.backbone_pool,
                torch.tensor(
                    [request.cache.slot for request in requests], device=self.device
                ),
                torch.tensor(starts, device=self.device),
                counts,
                embeds,
                gathered=self.cache,
            )
            first = self.heads(
                hidden[torch.arange(count, device=self.device), counts - 1]
            )
        for request in requests:
            request.cache.blocks = request.num_blocks
        # Rows past the requests pad the batch to its graph's size: one position each.
        first = F.pad(first, (0, 0, 0, batch - count))
        lengths = F.pad(lengths, (0, batch - count), value=1)
        valid = (
            torch.arange(self.cache_positions, device=self.device) < lengths[:, None]
        )
        if self.cuda_graphs:
            graph = self.step_graph(batch)
            graph.codes.copy_(first)
            graph.lengths.copy_(lengths)
            graph.valid.copy_(valid)
            graph.graph.replay()
            later = graph.steps
        else:
            later = torch.zeros(
                batch,
                BLOCK_STEPS - 1,
                NUM_CODEBOOKS,
                dtype=torch.long,
                device=self.device,
            )
            self.run_steps(first, lengths, valid, later)
        return torch.cat([first[:, None], later], dim=1)[:count].clone()


__all__ = [
    "BatchedBlocks",
    "BlockRequest",
    "GRAPH_BATCH_SIZES",
    "SessionCache",
]
