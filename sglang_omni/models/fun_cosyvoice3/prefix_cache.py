# SPDX-License-Identifier: Apache-2.0
"""Prefix K/V cache for the chunk-causal streaming DiT.

A causal hop re-solves the whole utterance so far. Under the chunk-causal mask
a frame only attends to its own chunk and the chunks before it, the causal
positional convs only look left and every hop restarts from the same noise, so
a frame whose chunk is complete produces the same K and V at every Euler step
and layer on every later hop. This keeps those in a paged pool and runs a hop
over the frames past them, attending to the cached prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    FA3_PAGE_SIZE,
    PACKED_INDUCTOR_OPTIONS,
    PackedDiT,
    PackedRows,
    gather_rows,
    pack_rows,
    packed_fa3,
    ragged_fa3,
    rotate_in_place,
    rotated,
)

BLOCK_FRAMES = 64
# Note (Jiaxin Deng): each positional conv has kernel 31, so it reads the 30
# frames before its input frame.
CONV_CONTEXT_FRAMES = 30


class PrefixForward(Protocol):
    def __call__(
        self,
        estimator: PackedDiT,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        x: torch.Tensor,
        mu: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: PrefixRowAttention,
        rope: tuple[torch.Tensor, torch.Tensor],
        first_context: torch.Tensor,
        second_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class PrefixKVPool:
    """K and V for every (Euler step, layer) in blocks of BLOCK_FRAMES pages;
    keys[euler_step][layer] is a (pages, 1, head_num, head_dim) tensor of its
    own."""

    def __init__(
        self,
        *,
        layer_num: int,
        euler_steps: int,
        head_num: int,
        head_dim: int,
        capacity_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        block_count = max(int(capacity_frames) // BLOCK_FRAMES, 0)
        shape = (block_count * BLOCK_FRAMES, FA3_PAGE_SIZE, head_num, head_dim)
        # Note (Jiaxin Deng): separate storages, not views of one slab: the
        # compiled hop mutates them in place only when its inputs don't alias.
        self.keys = [
            [torch.empty(shape, device=device, dtype=dtype) for _ in range(layer_num)]
            for _ in range(euler_steps)
        ]
        self.values = [
            [torch.empty(shape, device=device, dtype=dtype) for _ in range(layer_num)]
            for _ in range(euler_steps)
        ]
        self.free_blocks: list[int] = list(range(block_count))
        self.device = device
        # Note (Jiaxin Deng): the compile warmup installs the compiled contract.
        self.forward: PrefixForward = forward_prefix

    @property
    def free_frames(self) -> int:
        return len(self.free_blocks) * BLOCK_FRAMES

    def allocate(self, block_count: int) -> list[int] | None:
        if block_count > len(self.free_blocks):
            return None
        else:
            taken = self.free_blocks[-block_count:] if block_count else []
            del self.free_blocks[len(self.free_blocks) - block_count :]
            return taken

    def release(self, blocks: list[int]) -> None:
        self.free_blocks.extend(blocks)

    @staticmethod
    def bytes_per_frame(
        *,
        layer_num: int,
        euler_steps: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        return (
            2
            * layer_num
            * euler_steps
            * head_num
            * head_dim
            * torch.tensor([], dtype=dtype).element_size()
        )


@dataclass
class PrefixCacheRow:
    """One row's (one CFG twin's) cached frames."""

    blocks: list[int] = field(default_factory=list)
    committed_frames: int = 0
    # (euler_steps, 2, CONV_CONTEXT_FRAMES, hidden_size): each positional conv's
    # input over the last committed frames at each Euler step.
    conv_context: torch.Tensor | None = None

    @property
    def allocated_frames(self) -> int:
        return len(self.blocks) * BLOCK_FRAMES

    def pages(self, device: torch.device) -> torch.Tensor:
        blocks = torch.tensor(self.blocks, dtype=torch.int32, device=device)
        return (
            blocks.unsqueeze(1) * BLOCK_FRAMES
            + torch.arange(BLOCK_FRAMES, dtype=torch.int32, device=device)
        ).reshape(-1)


def grow_rows(
    pool: PrefixKVPool, rows: list[PrefixCacheRow], total_frames: list[int]
) -> bool:
    """Give every row enough blocks for its total frames; on a shortfall
    nothing is taken and False is returned."""
    needed_blocks = [
        max((frames + BLOCK_FRAMES - 1) // BLOCK_FRAMES - len(row.blocks), 0)
        for row, frames in zip(rows, total_frames, strict=True)
    ]
    if sum(needed_blocks) > len(pool.free_blocks):
        return False
    else:
        for row, block_count in zip(rows, needed_blocks, strict=True):
            taken = pool.allocate(block_count)
            assert taken is not None
            row.blocks.extend(taken)
        return True


def release_rows(pool: PrefixKVPool, rows: list[PrefixCacheRow]) -> None:
    for row in rows:
        pool.release(row.blocks)
        row.blocks = []
        row.committed_frames = 0
        row.conv_context = None


class PrefixRowAttention:
    """Queries are each row's new frames in chunk segments; keys are the row's
    cached prefix plus its new frames, all addressed through pool pages."""

    def __init__(
        self,
        *,
        prefix_frames: list[int],
        new_frames: list[int],
        pages: list[torch.Tensor],
        chunk_size: int,
        device: torch.device,
    ) -> None:
        segment_rows: list[int] = []
        segment_ends: list[int] = []
        offsets: list[int] = [0]
        for row, (start_frame, new_frame_count) in enumerate(
            zip(prefix_frames, new_frames, strict=True)
        ):
            end_frame = start_frame + new_frame_count
            segment_start = start_frame
            while segment_start < end_frame:
                segment_end = min(
                    (segment_start // chunk_size + 1) * chunk_size, end_frame
                )
                segment_rows.append(row)
                segment_ends.append(segment_end)
                offsets.append(offsets[-1] + segment_end - segment_start)
                segment_start = segment_end
        self.cache_seqlens = torch.tensor(
            segment_ends, dtype=torch.int32, device=device
        )
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.max_seqlen_q = max(b - a for a, b in pairwise(offsets))
        widest = max(segment_ends)
        table = torch.zeros(len(segment_ends), widest, dtype=torch.int32, device=device)
        for segment, (row, segment_end) in enumerate(
            zip(segment_rows, segment_ends, strict=True)
        ):
            assert (
                pages[row].numel() >= segment_end
            ), "row holds fewer pages than frames"
            table[segment, :segment_end] = pages[row][:segment_end]
        self.page_table = table
        # note(ratish): a frame's K and V are final once its whole chunk exists,
        # so a row keeps whole chunks and recomputes the rest on its next hop.
        self.committed_frames = [
            (start_frame + new_frame_count) // chunk_size * chunk_size
            for start_frame, new_frame_count in zip(
                prefix_frames, new_frames, strict=True
            )
        ]
        self.tail_index = torch.tensor(
            [
                committed - start_frame
                for committed, start_frame in zip(
                    self.committed_frames, prefix_frames, strict=True
                )
            ],
            device=device,
        ).unsqueeze(1) + torch.arange(CONV_CONTEXT_FRAMES, device=device)
        # every new frame's page, in packed order: where this hop writes K and V
        self.write_index = torch.cat(
            [
                pages[row][start_frame : start_frame + new_frame_count]
                for row, (start_frame, new_frame_count) in enumerate(
                    zip(prefix_frames, new_frames, strict=True)
                )
            ]
        ).to(torch.int64)
        # Note (Jiaxin Deng): the row count and width reach the compiled graph
        # only as this tensor's shape, which is marked dynamic, not as guards.
        slots = torch.full((len(new_frames), max(new_frames)), -1, dtype=torch.int64)
        offset = 0
        for row, new_frame_count in enumerate(new_frames):
            slots[row, :new_frame_count] = torch.arange(
                offset, offset + new_frame_count
            )
            offset += new_frame_count
        self.slots = slots.to(device)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        head_num: int,
        head_dim: int,
    ) -> torch.Tensor:
        # Note (Jiaxin Deng): head_num and head_dim come from the module so the
        # dynamic graph keeps them constant and the reshapes vectorize.
        page_shape = (-1, FA3_PAGE_SIZE, head_num, head_dim)
        key_pool.index_copy_(0, self.write_index, key[0].reshape(page_shape))
        value_pool.index_copy_(0, self.write_index, value[0].reshape(page_shape))
        if torch.compiler.is_compiling():
            fa3 = packed_fa3
        else:
            fa3 = ragged_fa3
        out = fa3(
            query[0].reshape(-1, head_num, head_dim),
            key_pool,
            value_pool,
            self.cache_seqlens,
            self.page_table,
            self.cu_seqlens_q,
            self.max_seqlen_q,
        )
        return out.reshape(1, -1, head_num * head_dim)

    def mark_dynamic(self, rows: PackedRows, absolute_positions: torch.Tensor) -> None:
        dynamo.mark_dynamic(self.page_table, (0, 1))
        dynamo.mark_dynamic(self.cu_seqlens_q, 0)
        dynamo.mark_dynamic(self.cache_seqlens, 0)
        dynamo.mark_dynamic(self.write_index, 0)
        dynamo.mark_dynamic(self.slots, (0, 1))
        dynamo.mark_dynamic(rows.starts_host, 0)
        dynamo.mark_dynamic(rows.row_ids, 0)
        dynamo.mark_dynamic(rows.positions, 0)
        dynamo.mark_dynamic(absolute_positions, 0)


def conv_pos_embed_prefix(
    estimator: PackedDiT,
    hidden_states: torch.Tensor,
    rows: PackedRows,
    first_context: torch.Tensor,
    second_context: torch.Tensor,
    attention: PrefixRowAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two causal positional convs over each row's new frames, each fed
    the last CONV_CONTEXT_FRAMES of its own input from the prefix (zeros for
    an empty prefix, the padding the whole-sequence call uses); each context
    is (rows, CONV_CONTEXT_FRAMES, hidden_size). Returns the new frames'
    embedding and the two next contexts."""
    # Note (Jiaxin Deng): the whole-sequence call zero-pads conv2's input,
    # not conv1's output, so the second conv needs its own cached tail.
    conv_pos_embed = estimator.dit.input_embed.conv_pos_embed
    slots = attention.slots
    padded = torch.where(
        (slots >= 0).unsqueeze(-1), hidden_states[0][slots.clamp(min=0)], 0.0
    )  # (rows, width, hidden_size)
    first_input = torch.cat((first_context.to(padded.dtype), padded), dim=1)
    first_output = conv_pos_embed.conv1(first_input.permute(0, 2, 1)).permute(0, 2, 1)
    second_input = torch.cat(
        (second_context.to(first_output.dtype), first_output), dim=1
    )
    second_output = conv_pos_embed.conv2(second_input.permute(0, 2, 1)).permute(0, 2, 1)
    tail_index = attention.tail_index.unsqueeze(-1).expand(-1, -1, first_input.shape[2])
    return (
        gather_rows(second_output, rows),
        first_input.gather(1, tail_index),
        second_input.gather(1, tail_index),
    )


def forward_prefix(
    estimator: PackedDiT,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    x: torch.Tensor,
    mu: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    mel_conditioning: torch.Tensor,
    t: torch.Tensor,
    rows: PackedRows,
    attention: PrefixRowAttention,
    rope: tuple[torch.Tensor, torch.Tensor],
    first_context: torch.Tensor,
    second_context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DiT.forward over the new frames only. keys, values: each layer's
    (pages, 1, head_num, head_dim) pool for this Euler step; x, mu,
    mel_conditioning, speaker_embeddings: (1, total new frames, channels);
    rope covers the rows' absolute positions. Returns the vector field and
    the two next positional-conv contexts."""
    dit = estimator.dit
    t = dit.time_embed(t)
    hidden_states = dit.input_embed.proj(
        torch.cat((x, mel_conditioning, mu, speaker_embeddings), dim=-1)
    )
    embedded, first_tail, second_tail = conv_pos_embed_prefix(
        estimator, hidden_states, rows, first_context, second_context, attention
    )
    hidden_states = embedded + hidden_states
    residual = hidden_states
    for layer, block in enumerate(dit.transformer_blocks):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(
            hidden_states, emb=t
        )
        attn = block.attn
        query, key, value = F.linear(
            norm, estimator.qkv_weights[layer], estimator.qkv_biases[layer]
        ).chunk(3, dim=-1)
        if torch.compiler.is_compiling():
            query = rotated(query, *rope)
            key = rotated(key, *rope)
        else:
            rotate_in_place(query, *rope)
            rotate_in_place(key, *rope)
        head_num = attn.heads
        out = attention(
            query,
            key,
            value,
            keys[layer],
            values[layer],
            head_num,
            attn.inner_dim // head_num,
        ).to(query.dtype)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn.to_out[1](
            attn.to_out[0](out)
        )
        ff_norm = (
            block.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
    if dit.long_skip_connection is not None:
        hidden_states = dit.long_skip_connection(
            torch.cat((hidden_states, residual), dim=-1)
        )
    else:
        pass
    hidden_states = dit.norm_out(hidden_states, t)
    return dit.proj_out(hidden_states), first_tail, second_tail


def compile_forward_prefix() -> PrefixForward:
    """forward_prefix under Inductor with the packed contracts' precision
    options; dynamic=True, so row counts and widths stay symbolic."""
    return torch.compile(
        forward_prefix,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
        options=dict(PACKED_INDUCTOR_OPTIONS),
    )


def solve_flow_euler_prefix(
    estimator: PackedDiT,
    pool: PrefixKVPool,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    mel_conditioning: torch.Tensor,
    new_frames: list[int],
    caches: list[tuple[PrefixCacheRow, PrefixCacheRow]],
    *,
    cfg_rate: float,
) -> torch.Tensor:
    """Euler steps over the new frames of each row with classifier free
    guidance; the conditional rows and their unconditional twins each keep
    their own cached prefix. noise, mu, mel_conditioning: (1, total new frames,
    channels) in row order; speaker_embeddings: (rows, channels). Commits every
    cache up to its last whole chunk."""
    device = noise.device
    dtype = speaker_embeddings.dtype
    total_new_frames = noise.shape[1]
    twin_caches = [pair[0] for pair in caches] + [pair[1] for pair in caches]
    prefix_frames = [row.committed_frames for row in twin_caches]
    twin_new_frames = list(new_frames) * 2
    twin_rows = pack_rows(twin_new_frames, device)
    # Note (Jiaxin Deng): the rows stay local for scatter/gather; RoPE alone
    # sees each frame's absolute position in its row.
    absolute_positions = (
        twin_rows.positions
        + torch.tensor(prefix_frames, device=device)[twin_rows.row_ids]
    )
    attention = PrefixRowAttention(
        prefix_frames=prefix_frames,
        new_frames=twin_new_frames,
        pages=[row.pages(device) for row in twin_caches],
        chunk_size=estimator.chunk_size,
        device=device,
    )
    angles, scale = estimator.dit.rotary_embed.forward_from_seq_len(
        max(
            start_frame + new_frame_count
            for start_frame, new_frame_count in zip(
                prefix_frames, twin_new_frames, strict=True
            )
        )
    )
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
    angles = angles[:, absolute_positions]
    rope = (angles.cos(), angles.sin())
    euler_steps = len(time_span) - 1
    hidden_size = int(estimator.dit.input_embed.proj.out_features)
    contexts: list[torch.Tensor] = []
    for row in twin_caches:
        if row.conv_context is None:
            contexts.append(
                torch.zeros(
                    euler_steps,
                    2,
                    CONV_CONTEXT_FRAMES,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            contexts.append(row.conv_context)
    # (euler_steps, twin rows, 2, CONV_CONTEXT_FRAMES, hidden_size)
    context = torch.stack(contexts, dim=1)
    next_context = torch.empty_like(context)
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    mel_conditioning_cfg = torch.cat(
        (mel_conditioning, torch.zeros_like(mel_conditioning)), dim=1
    )
    speaker_embeddings_cfg = torch.cat(
        (speaker_embeddings, torch.zeros_like(speaker_embeddings)), dim=0
    )
    speaker_embeddings_cfg = speaker_embeddings_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=device, dtype=dtype)
    forward = pool.forward
    if forward is not forward_prefix:
        attention.mark_dynamic(twin_rows, absolute_positions)
    else:
        pass
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for euler_step in range(euler_steps):
        flow_time[:] = t
        (
            vector_field,
            next_context[euler_step, :, 0],
            next_context[euler_step, :, 1],
        ) = forward(
            estimator,
            pool.keys[euler_step],
            pool.values[euler_step],
            torch.cat((x, x), dim=1),
            mu_cfg,
            speaker_embeddings_cfg,
            mel_conditioning_cfg,
            flow_time,
            twin_rows,
            attention,
            rope,
            context[euler_step, :, 0],
            context[euler_step, :, 1],
        )
        conditional = vector_field[:, :total_new_frames]
        unconditional = vector_field[:, total_new_frames:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if euler_step < euler_steps - 1:
            dt = time_span[euler_step + 2] - t
        else:
            pass
    for index, row in enumerate(twin_caches):
        row.committed_frames = attention.committed_frames[index]
        row.conv_context = next_context[:, index].clone()
    return x.float()
