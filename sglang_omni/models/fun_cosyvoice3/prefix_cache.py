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
    CONV_CONTEXT_FRAMES,
    FA3_PAGE_SIZE,
    PACKED_INDUCTOR_OPTIONS,
    PackedDiT,
    packed_fa3,
    ragged_fa3,
    rotate_in_place,
    rotated,
)

BLOCK_FRAMES = 64
# note(ratish): FA3's pick for these segments with a tight page table; pinned, since a
# graph's wider table would change the pick and the result.
PREFIX_FA3_SPLITS = 1


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
        self.padding_block = block_count
        shape = ((block_count + 1) * BLOCK_FRAMES, FA3_PAGE_SIZE, head_num, head_dim)
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


@dataclass(frozen=True, kw_only=True)
class PrefixStepLayout:
    """Frames and row slots are per CFG half; frames past a step's own are padding."""

    half_frames: int
    row_slots: int
    segment_count: int
    page_table_width: int


class PrefixRowAttention:
    """Queries are each row's new frames in chunk segments; keys are the row's
    cached prefix plus its new frames, all addressed through pool pages. The
    convs read one packed sequence: unused slots' contexts, then per CFG half
    each row's context and new frames, then the half's padding frames."""

    def __init__(
        self,
        *,
        rows: list[PrefixCacheRow],
        new_frames: list[int],
        layout: PrefixStepLayout,
        padding_block: int,
        chunk_size: int,
        device: torch.device,
    ) -> None:
        assert chunk_size <= BLOCK_FRAMES, "a padding segment reads one block"
        row_count = len(rows) // 2
        row_slots = layout.row_slots
        padding_frames = layout.half_frames - sum(new_frames[:row_count])
        assert padding_frames >= 0 and row_count <= row_slots
        prefix_frames = [row.committed_frames for row in rows]
        padding_row = 2 * row_count
        query_lengths: list[int] = []
        query_prefixes: list[int] = []
        query_block_rows: list[int] = []
        query_slots: list[int] = []
        for half in range(2):
            for index in range(row_count):
                twin_row = half * row_count + index
                query_lengths.append(new_frames[twin_row])
                query_prefixes.append(prefix_frames[twin_row])
                query_block_rows.append(twin_row)
                query_slots.append(half * row_slots + index)
            query_lengths.append(padding_frames)
            query_prefixes.append(0)
            query_block_rows.append(padding_row)
            query_slots.append(row_slots)
        context_starts: list[int] = []
        frame_starts: list[int] = []
        segment_rows: list[int] = []
        segment_ends: list[int] = []
        offsets: list[int] = [0]
        extended_frames = 2 * (row_slots - row_count) * CONV_CONTEXT_FRAMES
        for length, start_frame, block_row in zip(
            query_lengths, query_prefixes, query_block_rows, strict=True
        ):
            if block_row == padding_row:
                pass
            else:
                assert (
                    rows[block_row].allocated_frames >= start_frame + length
                ), "row holds fewer pages than frames"
                context_starts.append(extended_frames)
                extended_frames += CONV_CONTEXT_FRAMES
            frame_starts.append(extended_frames)
            extended_frames += length
            segment_start = start_frame
            while segment_start < start_frame + length:
                segment_end = min(
                    (segment_start // chunk_size + 1) * chunk_size,
                    start_frame + length,
                )
                segment_rows.append(block_row)
                if block_row == padding_row:
                    segment_ends.append(segment_end - segment_start)
                else:
                    segment_ends.append(segment_end)
                offsets.append(offsets[-1] + segment_end - segment_start)
                segment_start = segment_end
        self.max_seqlen_q = max(end - start for start, end in pairwise(offsets))
        unused_segments = layout.segment_count - len(segment_ends)
        assert unused_segments >= 0, "the layout holds fewer segments than the step"
        segment_rows += [padding_row] * unused_segments
        segment_ends += [0] * unused_segments
        offsets += [offsets[-1]] * unused_segments
        # note(ratish): a frame's K and V are final once its whole chunk exists,
        # so a row keeps whole chunks and recomputes the rest on its next hop.
        self.committed_frames = [
            (start_frame + new_frame_count) // chunk_size * chunk_size
            for start_frame, new_frame_count in zip(
                prefix_frames, new_frames, strict=True
            )
        ]

        lengths = torch.tensor(query_lengths)
        query_of_frame = torch.repeat_interleave(
            torch.arange(len(query_lengths)), lengths
        )
        local_frames = (
            torch.arange(len(query_of_frame))
            - (lengths.cumsum(0) - lengths)[query_of_frame]
        )
        frame_block_rows = torch.tensor(query_block_rows)[query_of_frame]
        positions = torch.where(
            frame_block_rows == padding_row,
            local_frames % chunk_size,
            torch.tensor(query_prefixes)[query_of_frame] + local_frames,
        )
        extended_positions = torch.tensor(frame_starts)[query_of_frame] + local_frames
        context_frames = torch.arange(CONV_CONTEXT_FRAMES)
        is_row_query = torch.tensor(query_block_rows) != padding_row
        row_query_slots = torch.tensor(query_slots)[is_row_query]
        extended_index = torch.zeros(extended_frames, dtype=torch.int64)
        extended_index[extended_positions] = 2 * row_slots * CONV_CONTEXT_FRAMES + (
            torch.arange(len(query_of_frame))
        )
        row_context_starts = torch.tensor(context_starts, dtype=torch.int64)
        extended_index[row_context_starts.unsqueeze(1) + context_frames] = (
            row_query_slots.unsqueeze(1) * CONV_CONTEXT_FRAMES + context_frames
        )
        tail_index = torch.zeros(2 * row_slots, CONV_CONTEXT_FRAMES, dtype=torch.int64)
        tail_index[row_query_slots] = (
            row_context_starts
            + torch.tensor(self.committed_frames, dtype=torch.int64)
            - torch.tensor(prefix_frames, dtype=torch.int64)
        ).unsqueeze(1) + context_frames
        block_lists = [row.blocks for row in rows] + [[padding_block]]
        block_width = max(len(blocks) for blocks in block_lists)
        block_table = torch.tensor(
            [blocks + [0] * (block_width - len(blocks)) for blocks in block_lists]
        )
        host_parts = (
            positions,
            extended_positions - CONV_CONTEXT_FRAMES,
            torch.tensor(query_slots)[query_of_frame],
            frame_block_rows,
            extended_index,
            tail_index.view(-1),
            torch.tensor(segment_rows),
            torch.tensor(segment_ends),
            torch.tensor(offsets),
            block_table.view(-1),
        )
        (
            self.positions,
            self.conv_output_index,
            self.speaker_index,
            frame_block_rows,
            self.extended_index,
            tail_index,
            segment_block_rows,
            segment_ends_tensor,
            offsets_tensor,
            block_table,
        ) = (
            torch.cat(host_parts).to(device).split([len(part) for part in host_parts])
        )
        self.tail_index = tail_index.view(2 * row_slots, CONV_CONTEXT_FRAMES)
        block_table = block_table.view(len(block_lists), block_width)
        self.cache_seqlens = segment_ends_tensor.to(torch.int32)
        self.cu_seqlens_q = offsets_tensor.to(torch.int32)
        # every new frame's page, in packed order: where this hop writes K and V
        self.write_index = (
            block_table[frame_block_rows, self.positions // BLOCK_FRAMES] * BLOCK_FRAMES
            + self.positions % BLOCK_FRAMES
        )
        page = torch.arange(layout.page_table_width, device=device)
        block_column = (page // BLOCK_FRAMES).clamp(max=block_width - 1)
        pages = (
            block_table[segment_block_rows][:, block_column] * BLOCK_FRAMES
            + page % BLOCK_FRAMES
        )
        self.page_table = torch.where(
            page < segment_ends_tensor.unsqueeze(1), pages, 0
        ).to(torch.int32)

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
            PREFIX_FA3_SPLITS,
        )
        return out.reshape(1, -1, head_num * head_dim)

    def mark_dynamic(self) -> None:
        # Note (Jiaxin Deng): the row count and width reach the compiled graph
        # only as these tensors' shapes, which are marked dynamic, not as guards.
        dynamo.mark_dynamic(self.page_table, (0, 1))
        dynamo.mark_dynamic(self.tail_index, 0)
        for tensor in (
            self.cu_seqlens_q,
            self.cache_seqlens,
            self.write_index,
            self.extended_index,
            self.conv_output_index,
        ):
            dynamo.mark_dynamic(tensor, 0)


def conv_pos_embed_prefix(
    estimator: PackedDiT,
    hidden_states: torch.Tensor,
    first_context: torch.Tensor,
    second_context: torch.Tensor,
    attention: PrefixRowAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two causal positional convs over each row's new frames, each fed
    the last CONV_CONTEXT_FRAMES of its own input from the prefix (zeros for
    an empty prefix, the padding the whole-sequence call uses); each context
    is (twin row slots, CONV_CONTEXT_FRAMES, hidden_size). Returns the new frames'
    embedding and the two next contexts."""
    # Note (Jiaxin Deng): the whole-sequence call zero-pads conv2's input,
    # not conv1's output, so the second conv needs its own cached tail.
    hidden_size = hidden_states.shape[2]
    first_input = torch.cat(
        (
            first_context.reshape(-1, hidden_size).to(hidden_states.dtype),
            hidden_states[0],
        )
    )[attention.extended_index]
    first_output = estimator.positional_conv(first_input, 0)
    second_input = torch.cat(
        (
            second_context.reshape(-1, hidden_size).to(first_output.dtype),
            first_output[attention.conv_output_index],
        )
    )[attention.extended_index]
    second_output = estimator.positional_conv(second_input, 1)
    return (
        second_output[attention.conv_output_index].unsqueeze(0),
        first_input[attention.tail_index],
        second_input[attention.tail_index],
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
        estimator, hidden_states, first_context, second_context, attention
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


def run_prefix_solve(
    estimator: PackedDiT,
    pool: PrefixKVPool,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    speaker_embeddings: torch.Tensor,
    mel_conditioning: torch.Tensor,
    attention: PrefixRowAttention,
    angles: torch.Tensor,
    context: torch.Tensor,
    next_context: torch.Tensor,
    *,
    cfg_rate: float,
) -> torch.Tensor:
    """noise, mu, mel_conditioning: (1, half frames, channels); speaker_embeddings:
    (row slots, channels); context, next_context: (euler_steps, twin row slots, 2,
    CONV_CONTEXT_FRAMES, hidden_size)."""
    half_frames = noise.shape[1]
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    mel_conditioning_cfg = torch.cat(
        (mel_conditioning, torch.zeros_like(mel_conditioning)), dim=1
    )
    speaker_embeddings_cfg = torch.cat(
        (speaker_embeddings, torch.zeros_like(speaker_embeddings)), dim=0
    )[attention.speaker_index].unsqueeze(0)
    angles = angles[:, attention.positions]
    rope = (angles.cos(), angles.sin())
    flow_time = torch.zeros(1, device=noise.device, dtype=speaker_embeddings.dtype)
    euler_steps = len(time_span) - 1
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for euler_step in range(euler_steps):
        flow_time[:] = t
        (
            vector_field,
            next_context[euler_step, :, 0],
            next_context[euler_step, :, 1],
        ) = pool.forward(
            estimator,
            pool.keys[euler_step],
            pool.values[euler_step],
            torch.cat((x, x), dim=1),
            mu_cfg,
            speaker_embeddings_cfg,
            mel_conditioning_cfg,
            flow_time,
            attention,
            rope,
            context[euler_step, :, 0],
            context[euler_step, :, 1],
        )
        conditional = vector_field[:, :half_frames]
        unconditional = vector_field[:, half_frames:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if euler_step < euler_steps - 1:
            dt = time_span[euler_step + 2] - t
        else:
            pass
    return x.float()


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
    chunk_size = estimator.chunk_size
    twin_caches = [pair[0] for pair in caches] + [pair[1] for pair in caches]
    twin_new_frames = list(new_frames) * 2
    end_frames = [
        row.committed_frames + new_frame_count
        for row, new_frame_count in zip(twin_caches, twin_new_frames, strict=True)
    ]
    attention = PrefixRowAttention(
        rows=twin_caches,
        new_frames=twin_new_frames,
        layout=PrefixStepLayout(
            half_frames=sum(new_frames),
            row_slots=len(caches),
            segment_count=sum(
                -(-end_frame // chunk_size) - row.committed_frames // chunk_size
                for row, end_frame in zip(twin_caches, end_frames, strict=True)
            ),
            page_table_width=max(end_frames),
        ),
        padding_block=pool.padding_block,
        chunk_size=chunk_size,
        device=device,
    )
    angles, scale = estimator.dit.rotary_embed.forward_from_seq_len(max(end_frames))
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
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
                    dtype=speaker_embeddings.dtype,
                )
            )
        else:
            contexts.append(row.conv_context)
    # (euler_steps, twin rows, 2, CONV_CONTEXT_FRAMES, hidden_size)
    context = torch.stack(contexts, dim=1)
    next_context = torch.empty_like(context)
    if pool.forward is not forward_prefix:
        attention.mark_dynamic()
    else:
        pass
    x = run_prefix_solve(
        estimator,
        pool,
        noise,
        time_span,
        mu,
        speaker_embeddings,
        mel_conditioning,
        attention,
        angles,
        context,
        next_context,
        cfg_rate=cfg_rate,
    )
    for index, row in enumerate(twin_caches):
        row.committed_frames = attention.committed_frames[index]
        row.conv_context = next_context[:, index].clone()
    return x
