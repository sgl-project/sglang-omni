# SPDX-License-Identifier: Apache-2.0
"""Zonos3 streams decoded as their codes arrive: a slot carries every causal layer's state, so each frame passes the codec trunk once."""

from __future__ import annotations

import bisect
import math
import threading
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sglang_omni.models.zonos3.components.codec import (
    CODEBOOK_COUNT,
    CONVNEXT_DEPTH,
    HEAD_FRAMES_PER_CODE_FRAME,
    STAGE_PATCH_SIZE,
    Zonos3CodecDecoder,
)
from sglang_omni.models.zonos3.components.codec_layers import ConvNeXtBlock, patch_up
from sglang_omni.models.zonos3.components.codec_transformer import rotary_tables
from sglang_omni.platforms import current_platform

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
else:
    pass

# note (zhaochenyang20): FA3 reads a slot's ring through a page table; 16-frame pages keep the tables small, and on one H100 a ring layer's append and window attention took 9-30 us against 35-100 us for index writes, gathers and masked memory-efficient attention.
RING_PAGE_FRAMES = 16
# Eager passes on the capture stream before each capture, so lazily initialized kernels and workspaces exist first.
GRAPH_WARMUP_ITERATIONS = 2


@dataclass(kw_only=True)
class SlotPass:
    """Static inputs (rows, codebooks x frames + 2) of one fixed-shape pass: each row's codes, slot and real frame count.

    Once captured, replaying graph writes the rows' new complex spectra (rows, bins, frames x 16) into spectra.
    """

    inputs: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    spectra: list[torch.Tensor] = field(default_factory=list)


def pass_input_views(
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The codes (rows, codebooks, frames), slots (rows,) and real frame counts (rows,) packed in a pass's inputs."""
    return (
        inputs[:, :-2].unflatten(1, (CODEBOOK_COUNT, -1)),
        inputs[:, -2],
        inputs[:, -1],
    )


@dataclass(kw_only=True)
class Zonos3DecoderStream:
    """One stream's slot, the frames its slot has decoded, and its kept spectra (bins, head frames) per resolution.

    The spectra start at spectra_first_head_frame, early enough to synthesize every sample not yet sent.
    """

    slot: int
    pushed_frame_count: int = 0
    spectra: list[torch.Tensor] = field(default_factory=list)
    spectra_first_head_frame: int = 0


def convnext_with_history(
    block: ConvNeXtBlock,
    hidden_states: torch.Tensor,
    history: torch.Tensor,
    valid_frame_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a causal ConvNeXt block on (B, C, T) states after each row's kept inputs; returns the states and the inputs to keep.

    A row's kept inputs end at its last real frame, valid_frame_counts (B,), so padded frames never enter them.
    """
    extended_states = torch.cat([history, hidden_states], dim=-1)
    convolved = F.conv1d(
        extended_states,
        block.dwconv.weight,
        block.dwconv.bias,
        groups=block.dwconv.groups,
    )
    history_indices = (
        valid_frame_counts[:, None]
        + torch.arange(history.shape[-1], device=extended_states.device)[None, :]
    )
    next_history = extended_states.gather(
        2, history_indices[:, None, :].expand(-1, extended_states.shape[1], -1)
    )
    return block.add_pointwise(hidden_states, convolved), next_history


class Zonos3DecoderSlots:
    """The decoder state of up to stream_slots streams in static buffers, advanced together by fixed-shape passes.

    A slot holds a stream's frame count, its ConvNeXt inputs and per-layer rings of rotated keys and values that FA3 reads through page tables; padding rows of a pass point at a scratch slot past the last, and every (rows, frames) pass is a captured graph.
    """

    def __init__(
        self,
        decoder: Zonos3CodecDecoder,
        *,
        stream_slots: int,
        row_buckets: tuple[int, ...],
        pass_frame_buckets: tuple[int, ...],
        max_stream_frames: int,
    ) -> None:
        self.decoder = decoder
        self.max_stream_frames = max_stream_frames
        self.device = decoder.to_external_kernel.device
        self.scratch_slot = stream_slots
        self.free_slots = list(range(stream_slots - 1, -1, -1))
        self.free_slots_lock = threading.Lock()
        self.row_buckets = row_buckets
        self.pass_frame_buckets = pass_frame_buckets
        buffer_count = stream_slots + 1
        semantic_block = decoder.quantizer.semantic_head[0]
        head_block = decoder.head.blocks[0]
        with torch.inference_mode():
            self.seen_frame_counts = torch.zeros(
                buffer_count, dtype=torch.long, device=self.device
            )
            self.quantizer_histories = torch.zeros(
                CONVNEXT_DEPTH,
                buffer_count,
                semantic_block.dwconv.in_channels,
                semantic_block.dwconv.left_padding,
                dtype=semantic_block.dwconv.weight.dtype,
                device=self.device,
            )
            self.head_histories = torch.zeros(
                len(decoder.head.blocks),
                buffer_count,
                head_block.dwconv.in_channels,
                head_block.dwconv.left_padding,
                dtype=head_block.dwconv.weight.dtype,
                device=self.device,
            )
            self.window_frames: list[int] = []
            self.page_tables: list[torch.Tensor] = []
            self.ring_keys: list[torch.Tensor] = []
            self.ring_values: list[torch.Tensor] = []
            for stage_index, stage in enumerate(decoder.transformer_stages):
                states_per_frame = STAGE_PATCH_SIZE**stage_index
                self.window_frames.append(stage.spec.context_frames - 1)
                # A ring holds the attention window plus the longest pass; a stage's page table maps position p to ring entry p mod the ring length.
                ring_pages = math.ceil(
                    (
                        self.window_frames[-1]
                        + max(pass_frame_buckets) * states_per_frame
                    )
                    / RING_PAGE_FRAMES
                )
                self.page_tables.append(
                    (
                        torch.arange(buffer_count, device=self.device)[:, None]
                        * ring_pages
                        + torch.arange(
                            # A pass appends its whole frame bucket, padded frames included, past the last real frame.
                            math.ceil(
                                (max_stream_frames + max(pass_frame_buckets))
                                * states_per_frame
                                / RING_PAGE_FRAMES
                            ),
                            device=self.device,
                        )[None, :]
                        % ring_pages
                    ).to(torch.int32)
                )
                for rings in (self.ring_keys, self.ring_values):
                    rings.append(
                        torch.zeros(
                            stage.spec.layer_count,
                            buffer_count * ring_pages,
                            RING_PAGE_FRAMES,
                            stage.spec.head_count,
                            stage.spec.model_dimension // stage.spec.head_count,
                            dtype=stage.input_proj.weight.dtype,
                            device=self.device,
                        )
                    )
            self.passes: dict[tuple[int, int], SlotPass] = {}
            for row_count in row_buckets:
                for frame_count in pass_frame_buckets:
                    inputs = torch.zeros(
                        row_count, CODEBOOK_COUNT * frame_count + 2, dtype=torch.long
                    )
                    pass_input_views(inputs)[1].fill_(self.scratch_slot)
                    self.passes[(row_count, frame_count)] = SlotPass(
                        inputs=inputs.to(self.device)
                    )

    @torch.inference_mode()
    def acquire(self) -> Zonos3DecoderStream | None:
        """A stream in a free slot with empty state, or None when every slot is held; runs on the decode stream."""
        with self.free_slots_lock:
            if self.free_slots:
                slot = self.free_slots.pop()
            else:
                return None
        self.seen_frame_counts[slot].zero_()
        self.quantizer_histories[:, slot].zero_()
        self.head_histories[:, slot].zero_()
        return Zonos3DecoderStream(slot=slot)

    def release(self, decoder_stream: Zonos3DecoderStream) -> None:
        with self.free_slots_lock:
            assert decoder_stream.slot not in self.free_slots, decoder_stream.slot
            self.free_slots.append(decoder_stream.slot)

    @torch.inference_mode()
    def capture_graphs(self) -> None:
        """Capture every pass over the scratch slot, largest first so the shared pool grows once."""
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        graph_pool = torch.cuda.graph_pool_handle()
        for row_count in reversed(self.row_buckets):
            for frame_count in reversed(self.pass_frame_buckets):
                slot_pass = self.passes[(row_count, frame_count)]
                with torch.cuda.stream(capture_stream):
                    for _ in range(GRAPH_WARMUP_ITERATIONS):
                        self.run_pass(slot_pass)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=graph_pool,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    slot_pass.spectra = self.run_pass(slot_pass)
                slot_pass.graph = graph
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)

    def run_pass(self, slot_pass: SlotPass) -> list[torch.Tensor]:
        """Advance the rows' slots by the pass's codes; returns the complex spectra (rows, bins, frames x 16) per resolution.

        Only device tensors of fixed shapes take part, so the same code records a graph and runs eagerly.
        """
        quantizer = self.decoder.quantizer
        codes, row_slots, valid_frame_counts = pass_input_views(slot_pass.inputs)
        hidden_states = quantizer.semantic_quantizer.decode(codes[:, 0])
        quantizer_histories: list[torch.Tensor] = []
        for block_index, block in enumerate(quantizer.semantic_head[:CONVNEXT_DEPTH]):
            hidden_states, history = convnext_with_history(
                block,
                hidden_states,
                self.quantizer_histories[block_index, row_slots],
                valid_frame_counts,
            )
            quantizer_histories.append(history)
        self.quantizer_histories.index_copy_(
            1, row_slots, torch.stack(quantizer_histories)
        )
        hidden_states = quantizer.semantic_head[CONVNEXT_DEPTH](hidden_states)
        for codebook_index, residual_quantizer in enumerate(
            quantizer.residual_quantizers
        ):
            hidden_states = hidden_states + residual_quantizer.decode(
                codes[:, codebook_index + 1]
            )
        seen_frame_counts = self.seen_frame_counts[row_slots]
        for stage_index, stage in enumerate(self.decoder.transformer_stages):
            if stage_index == 0:
                hidden_states = hidden_states.to(stage.input_proj.weight.dtype)
            else:
                hidden_states = patch_up(hidden_states, STAGE_PATCH_SIZE)
            hidden_states = stage.input_proj(hidden_states.transpose(1, 2))
            row_count, frame_count, model_dimension = hidden_states.shape
            first_positions = seen_frame_counts * STAGE_PATCH_SIZE**stage_index
            rotary_cos, rotary_signed_sin = rotary_tables(
                (
                    first_positions[:, None]
                    + torch.arange(frame_count, device=self.device)[None, :]
                )[:, None].float(),
                model_dimension // stage.spec.head_count,
            )
            page_table = self.page_tables[stage_index][row_slots]
            cache_lengths = first_positions.to(torch.int32)
            for layer_index, layer in enumerate(stage.layers):
                # The rings hold keys rotated and rounded to the value dtype, the bits a whole decode's attention reads.
                query, key, value = layer.rotated_states(
                    hidden_states, rotary_cos, rotary_signed_sin, 0, frame_count
                )
                # Each row appends all its frames; those past its real ones are overwritten by its next pass, and no real frame attends them.
                attention = flash_attn_with_kvcache(
                    query.transpose(1, 2),
                    self.ring_keys[stage_index][layer_index],
                    self.ring_values[stage_index][layer_index],
                    k=key.transpose(1, 2),
                    v=value.transpose(1, 2),
                    cache_seqlens=cache_lengths,
                    page_table=page_table,
                    causal=True,
                    window_size=(self.window_frames[stage_index], 0),
                )
                hidden_states = layer.add_attention(
                    hidden_states,
                    attention.reshape(row_count, frame_count, model_dimension),
                )
            if stage.output_proj is None:
                pass
            else:
                hidden_states = stage.output_proj(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)
        head = self.decoder.head
        head_valid_frame_counts = valid_frame_counts * HEAD_FRAMES_PER_CODE_FRAME
        head_histories: list[torch.Tensor] = []
        for block_index, block in enumerate(head.blocks):
            hidden_states, history = convnext_with_history(
                block,
                hidden_states,
                self.head_histories[block_index, row_slots],
                head_valid_frame_counts,
            )
            head_histories.append(history)
        self.head_histories.index_copy_(1, row_slots, torch.stack(head_histories))
        self.seen_frame_counts.index_add_(0, row_slots, valid_frame_counts)
        return head.spectra(hidden_states.transpose(1, 2))

    @torch.inference_mode()
    def advance(
        self, decoder_streams: list[Zonos3DecoderStream], codes: list[torch.Tensor]
    ) -> None:
        """Run each stream's new (codebooks, frames) codes through its slot and append their spectra to its kept spectra.

        Streams go in passes of at most max(row_buckets) rows and max(pass_frame_buckets) frames each, on the caller's stream.
        """
        for decoder_stream, stream_codes in zip(decoder_streams, codes, strict=True):
            if (
                decoder_stream.pushed_frame_count + stream_codes.shape[1]
                > self.max_stream_frames
            ):
                raise ValueError(
                    f"Zonos3 decoder slots hold streams of at most {self.max_stream_frames} frames"
                )
            else:
                pass
        longest_pass_frames = self.pass_frame_buckets[-1]
        consumed_frame_counts = [0] * len(decoder_streams)
        while True:
            pending_indices = [
                index
                for index, stream_codes in enumerate(codes)
                if consumed_frame_counts[index] < stream_codes.shape[1]
            ]
            if not pending_indices:
                return
            else:
                pass
            for group_start in range(0, len(pending_indices), self.row_buckets[-1]):
                group_indices = pending_indices[
                    group_start : group_start + self.row_buckets[-1]
                ]
                group_frame_counts = [
                    min(
                        codes[index].shape[1] - consumed_frame_counts[index],
                        longest_pass_frames,
                    )
                    for index in group_indices
                ]
                slot_pass = self.passes[
                    (
                        self.row_buckets[
                            bisect.bisect_left(self.row_buckets, len(group_indices))
                        ],
                        self.pass_frame_buckets[
                            bisect.bisect_left(
                                self.pass_frame_buckets, max(group_frame_counts)
                            )
                        ],
                    )
                ]
                host_inputs = torch.zeros(slot_pass.inputs.shape, dtype=torch.long)
                host_codes, host_row_slots, host_valid_frame_counts = pass_input_views(
                    host_inputs
                )
                host_row_slots.fill_(self.scratch_slot)
                for row, (index, row_frame_count) in enumerate(
                    zip(group_indices, group_frame_counts, strict=True)
                ):
                    first_frame = consumed_frame_counts[index]
                    host_codes[row, :, :row_frame_count] = codes[index][
                        :, first_frame : first_frame + row_frame_count
                    ]
                    host_row_slots[row] = decoder_streams[index].slot
                    host_valid_frame_counts[row] = row_frame_count
                # The caching host allocator keeps a pinned block until its copy has run.
                slot_pass.inputs.copy_(host_inputs.pin_memory(), non_blocking=True)
                assert slot_pass.graph is not None
                slot_pass.graph.replay()
                for row, (index, row_frame_count) in enumerate(
                    zip(group_indices, group_frame_counts, strict=True)
                ):
                    decoder_stream = decoder_streams[index]
                    new_spectra = [
                        spectrum[row, :, : row_frame_count * HEAD_FRAMES_PER_CODE_FRAME]
                        for spectrum in slot_pass.spectra
                    ]
                    if decoder_stream.spectra:
                        decoder_stream.spectra = [
                            torch.cat([kept, new], dim=-1)
                            for kept, new in zip(
                                decoder_stream.spectra, new_spectra, strict=True
                            )
                        ]
                    else:
                        decoder_stream.spectra = [
                            spectrum.clone() for spectrum in new_spectra
                        ]
                    decoder_stream.pushed_frame_count += row_frame_count
                    consumed_frame_counts[index] += row_frame_count
