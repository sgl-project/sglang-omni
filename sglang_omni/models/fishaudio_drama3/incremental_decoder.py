# SPDX-License-Identifier: Apache-2.0
"""Drama 3 codec decoding as streams: every window decodes only its new frames.

Rows of one decode belong to different streams. A stream owns one slot: its causal
convolutions' left context and transposed convolutions' overlap live in one arena row,
and its post-module keys and values in rows of per-layer rings that FA3 reads through a
page table. The last slot belongs to no stream; rows that only pad a batch use it.
"""

from __future__ import annotations

import bisect

import torch
from dac.nn.layers import Snake1d

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    SNAKE_BLOCK_CHANNELS,
    SNAKE_BLOCK_ROWS,
    ChannelsLastDecoder,
    libdevice,
    tl,
    triton,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    CausalConvNet,
    CausalTransConvNet,
    DecoderBlock,
    ResidualUnit,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.utils.channels_last_conv import channels_last_conv_transpose1d

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
else:
    flash_attn_with_kvcache = None

LATENT_BLOCK_ROWS = 32
LATENT_BLOCK_COLUMNS = 128

if triton is not None:

    @triton.jit(do_not_specialize=["rows", "frames"])
    def codebook_latent_kernel(
        indices_ptr: torch.Tensor,
        table_ptr: torch.Tensor,
        output_ptr: torch.Tensor,
        rows: int,
        frames: int,
        CODEBOOKS: tl.constexpr,
        SEMANTIC_SIZE: tl.constexpr,
        RESIDUAL_SIZE: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLUMNS: tl.constexpr,
    ) -> None:
        """(B, T, DIM) sum over codebooks of each code's projected table row; codes clamp
        to their codebook, the first of which is the semantic one."""
        row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        column = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
        row_mask = row < rows
        mask = row_mask[:, None] & (column < DIM)[None, :]
        batch_row = row // frames
        frame = row % frames
        latent = tl.zeros([BLOCK_ROWS, BLOCK_COLUMNS], dtype=tl.float32)
        for codebook in tl.static_range(CODEBOOKS):
            code = tl.load(
                indices_ptr + (batch_row * CODEBOOKS + codebook) * frames + frame,
                mask=row_mask,
                other=0,
            )
            if codebook == 0:
                table_row = tl.minimum(tl.maximum(code, 0), SEMANTIC_SIZE - 1)
            else:
                table_row = (
                    SEMANTIC_SIZE
                    + (codebook - 1) * RESIDUAL_SIZE
                    + tl.minimum(tl.maximum(code, 0), RESIDUAL_SIZE - 1)
                )
            latent += tl.load(
                table_ptr + table_row[:, None] * DIM + column[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            output_ptr + row[:, None] * DIM + column[None, :],
            latent.to(output_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit(do_not_specialize=["rows", "frames", "batch_frames", "pad_right"])
    def snake_history_kernel(
        x_ptr: torch.Tensor,
        alpha_ptr: torch.Tensor,
        inverse_alpha_ptr: torch.Tensor,
        y_ptr: torch.Tensor,
        history_ptr: torch.Tensor,
        slots_ptr: torch.Tensor,
        rows: int,
        frames: int,
        channels: int,
        batch_frames: int,
        slot_stride: int,
        pad_right: int,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
        HISTORY: tl.constexpr,
    ) -> None:
        """The Snake of x after each row's HISTORY frames of its slot's left context,
        which stay as they are, ready for the causal convolution that follows."""
        row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        row_mask = row < rows
        channel_mask = channel < channels
        mask = row_mask[:, None] & channel_mask[None, :]
        output_frames = frames + HISTORY + pad_right
        batch_row = row // output_frames
        output_frame = row % output_frames
        source_frame = output_frame - HISTORY
        source_mask = (source_frame >= 0) & (source_frame < frames)
        source = (batch_row * batch_frames + source_frame) * channels
        x = tl.load(
            x_ptr + source[:, None] + channel[None, :],
            mask=mask & source_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        alpha = tl.load(alpha_ptr + channel, mask=channel_mask, other=1.0)
        inverse_alpha = tl.load(
            inverse_alpha_ptr + channel, mask=channel_mask, other=1.0
        )
        sine = libdevice.sin(alpha[None, :] * x)
        y = x + inverse_alpha[None, :] * sine * sine
        history_mask = output_frame < HISTORY
        slots = tl.load(slots_ptr + batch_row, mask=row_mask, other=0)
        history_row = slots * slot_stride + output_frame * channels
        history = tl.load(
            history_ptr + history_row[:, None] + channel[None, :],
            mask=mask & history_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            y_ptr + row[:, None] * channels + channel[None, :],
            tl.where(source_mask[:, None], y, history).to(y_ptr.dtype.element_ty),
            mask=mask,
        )

else:
    pass


class IncrementalDecoder:
    """Decodes (B, codebooks, T) new frames of the streams in slots, whose frames so far
    number positions, into (B, 1, T * frame_length) audio."""

    def __init__(
        self,
        decoder: ChannelsLastDecoder,
        *,
        stream_slots: int,
        max_window_frames: int,
        device: torch.device,
    ) -> None:
        self.decoder = decoder
        self.codec = decoder.codec
        self.stream_slots = stream_slots
        self.max_window_frames = max_window_frames
        post_module = self.codec.quantizer.post_module
        attention = post_module.layers[0].attention
        if attention.n_local_heads != attention.n_head:
            raise ValueError("incremental decoding needs one key and value per head")
        else:
            pass
        self.dtype = post_module.norm.weight.dtype
        self.pad_slot = stream_slots
        self.num_heads = attention.n_head
        self.head_dim = attention.head_dim
        self.softmax_scale = 1.0 / self.head_dim**0.5
        self.context_frames = post_module.window_size - 1
        self.ring_frames = self.context_frames + max_window_frames
        ring_shape = (
            (stream_slots + 1) * self.ring_frames,
            1,
            self.num_heads,
            self.head_dim,
        )
        self.key_rings = [
            torch.empty(ring_shape, dtype=self.dtype, device=device)
            for _ in post_module.layers
        ]
        self.value_rings = [
            torch.empty(ring_shape, dtype=self.dtype, device=device)
            for _ in post_module.layers
        ]
        layout: list[tuple[int, tuple[int, int]]] = []
        for block in self.codec.quantizer.upsample:
            convolution = block[1].dwconv
            layout.append(
                (id(convolution), (convolution.conv.in_channels, convolution.padding))
            )
        for module in self.codec.decoder.modules():
            if isinstance(module, CausalConvNet) and module.padding > 0:
                layout.append((id(module), (module.padding, module.conv.in_channels)))
            elif (
                isinstance(module, CausalTransConvNet)
                and module.kernel_size > module.stride
            ):
                layout.append(
                    (
                        id(module),
                        (module.kernel_size - module.stride, module.conv.out_channels),
                    )
                )
            else:
                pass
        row_elements = sum(rows * columns for _, (rows, columns) in layout)
        self.arena = torch.zeros(
            (stream_slots + 1, row_elements), dtype=self.dtype, device=device
        )
        self.states: dict[int, torch.Tensor] = {}
        offset = 0
        for module_id, (rows, columns) in layout:
            elements = rows * columns
            self.states[module_id] = self.arena[:, offset : offset + elements].view(
                stream_slots + 1, rows, columns
            )
            offset += elements
        quantizer = self.codec.quantizer
        self.semantic_size = quantizer.semantic_quantizer.codebook_size
        self.residual_size = quantizer.quantizer.codebook_size
        with torch.no_grad():
            self.projected_codes = torch.cat(
                [
                    codebook.out_proj(codebook.codebook.weight.T[None])[0].T
                    for codebook in (
                        *quantizer.semantic_quantizer.quantizers,
                        *quantizer.quantizer.quantizers,
                    )
                ]
            ).contiguous()

    def reset_slots(self, slots: list[int]) -> None:
        """Start new streams in slots; their rings are read only up to their positions."""
        for slot in slots:
            self.arena[slot].zero_()

    def __call__(
        self, indices: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        latent = self.post_module(self.frame_latents(indices), slots, positions)
        latent = self.upsample(latent, slots)
        hidden_states = latent.transpose(1, 2).contiguous()
        hidden_states = self.apply_sequence(
            self.codec.decoder.model, hidden_states, slots
        )
        return hidden_states.transpose(1, 2)

    def frame_latents(self, indices: torch.Tensor) -> torch.Tensor:
        batch_size, codebook_count, frame_count = indices.shape
        dim = self.projected_codes.shape[1]
        output = torch.empty(
            (batch_size, frame_count, dim),
            dtype=self.projected_codes.dtype,
            device=indices.device,
        )
        rows = batch_size * frame_count
        codebook_latent_kernel[
            (
                triton.cdiv(rows, LATENT_BLOCK_ROWS),
                triton.cdiv(dim, LATENT_BLOCK_COLUMNS),
            )
        ](
            indices,
            self.projected_codes,
            output,
            rows,
            frame_count,
            CODEBOOKS=codebook_count,
            SEMANTIC_SIZE=self.semantic_size,
            RESIDUAL_SIZE=self.residual_size,
            DIM=dim,
            BLOCK_ROWS=LATENT_BLOCK_ROWS,
            BLOCK_COLUMNS=LATENT_BLOCK_COLUMNS,
        )
        return output

    def post_module(
        self, latent: torch.Tensor, slots: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        post_module = self.codec.quantizer.post_module
        hidden_states = post_module.look_ahead_conv(post_module.input_proj(latent))
        frame_count = hidden_states.shape[1]
        device = hidden_states.device
        new_frames = torch.arange(frame_count, device=device)[None, :]
        new_positions = positions[:, None] + new_frames
        held_frames = positions.clamp(max=self.context_frames)
        first_rows = slots[:, None] * self.ring_frames
        new_rows = (first_rows + new_positions % self.ring_frames).reshape(-1)
        window_frames = torch.arange(self.context_frames + frame_count, device=device)[
            None, :
        ]
        page_table = (
            first_rows
            + ((positions - held_frames)[:, None] + window_frames) % self.ring_frames
        ).to(torch.int32)
        window_lengths = (held_frames + frame_count).to(torch.int32)
        freqs_cis = post_module.freqs_cis[new_positions]
        for layer_index, layer in enumerate(post_module.layers):
            attended = self.attend(
                layer.attention,
                layer.attention_norm(hidden_states),
                freqs_cis,
                self.key_rings[layer_index],
                self.value_rings[layer_index],
                new_rows,
                page_table,
                window_lengths,
            )
            hidden_states = hidden_states + layer.attention_layer_scale(attended)
            hidden_states = hidden_states + layer.ffn_layer_scale(
                layer.feed_forward(layer.ffn_norm(hidden_states))
            )
        hidden_states = post_module.output_proj(post_module.norm(hidden_states))
        return hidden_states.transpose(1, 2)

    def attend(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        key_ring: torch.Tensor,
        value_ring: torch.Tensor,
        new_rows: torch.Tensor,
        page_table: torch.Tensor,
        window_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, frame_count, _ = hidden_states.shape
        shape = (batch_size, frame_count, self.num_heads, self.head_dim)
        ring_row_shape = (batch_size * frame_count, 1, self.num_heads, self.head_dim)
        query, key, value = attention.wqkv(hidden_states).split(
            self.num_heads * self.head_dim, dim=-1
        )
        key_ring.index_copy_(
            0,
            new_rows,
            rotated(key.reshape(shape), freqs_cis).reshape(ring_row_shape),
        )
        value_ring.index_copy_(0, new_rows, value.reshape(ring_row_shape))
        output = flash_attn_with_kvcache(
            rotated(query.reshape(shape), freqs_cis),
            key_ring,
            value_ring,
            page_table=page_table,
            cache_seqlens=window_lengths,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(self.context_frames, 0),
        )
        return attention.wo(output.reshape(batch_size, frame_count, -1))

    def upsample(self, latent: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        for block in self.codec.quantizer.upsample:
            latent = block[0](latent)
            convnext = block[1]
            convolution = convnext.dwconv
            state = self.states[id(convolution)]
            padded = torch.cat([state.index_select(0, slots), latent], dim=2)
            state.index_copy_(0, slots, padded[:, :, latent.shape[2] :])
            hidden_states = convolution.conv(padded).transpose(1, 2)
            hidden_states = convnext.pwconv2(
                convnext.act(convnext.pwconv1(convnext.norm(hidden_states)))
            )
            if convnext.gamma is not None:
                hidden_states = convnext.gamma * hidden_states
            else:
                pass
            latent = latent + hidden_states.transpose(1, 2)
        return latent

    def apply_sequence(
        self,
        sequence: torch.nn.Sequential,
        hidden_states: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        modules = iter(sequence)
        module = next(modules, None)
        while module is not None:
            if isinstance(module, Snake1d):
                following = next(modules, None)
                if isinstance(following, CausalConvNet) and following.padding > 0:
                    hidden_states = self.snake_and_conv(
                        module, following, hidden_states, slots
                    )
                    module = next(modules, None)
                elif isinstance(following, CausalConvNet):
                    hidden_states = self.decoder.causal_conv(
                        following, self.decoder.snake(module, hidden_states)
                    )
                    module = next(modules, None)
                else:
                    hidden_states = self.decoder.snake(module, hidden_states)
                    module = following
            else:
                hidden_states = self.apply(module, hidden_states, slots)
                module = next(modules, None)
        return hidden_states

    def apply(
        self, module: torch.nn.Module, hidden_states: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(module, CausalConvNet):
            return self.conv(module, hidden_states, slots)
        elif isinstance(module, CausalTransConvNet):
            return self.transposed_conv(module, hidden_states, slots)
        elif isinstance(module, ResidualUnit):
            residual = self.apply_sequence(module.block, hidden_states, slots)
            return hidden_states + residual
        elif isinstance(module, DecoderBlock):
            return self.apply_sequence(module.block, hidden_states, slots)
        elif isinstance(module, torch.nn.Tanh):
            return torch.tanh(hidden_states)
        else:
            raise TypeError(
                f"no incremental decoding for {type(module).__name__} in the codec"
            )

    def conv(
        self, module: CausalConvNet, hidden_states: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        frame_count = hidden_states.shape[1]
        state = self.states[id(module)]
        padded = torch.cat([state.index_select(0, slots), hidden_states], dim=1)
        state.index_copy_(0, slots, padded[:, frame_count:])
        return self.decoder.causal_conv(
            module, padded, original_frame_count=frame_count
        )

    def snake_and_conv(
        self,
        snake: Snake1d,
        module: CausalConvNet,
        hidden_states: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        frame_count = hidden_states.shape[1]
        history_frames = module.padding
        alignment = (-(history_frames + frame_count)) % module.conv.dilation[0]
        padded = self.snake_with_history(
            snake, hidden_states, self.states[id(module)], slots, alignment
        )
        self.states[id(module)].index_copy_(
            0, slots, padded[:, frame_count : frame_count + history_frames]
        )
        return self.decoder.causal_conv(
            module, padded, original_frame_count=frame_count
        )

    def snake_with_history(
        self,
        module: Snake1d,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        pad_right: int,
    ) -> torch.Tensor:
        alpha, inverse_alpha = self.decoder.snake_alphas[id(module)]
        batch_size, frame_count, channel_count = hidden_states.shape
        history_frames = state.shape[1]
        assert hidden_states.stride(2) == 1
        assert hidden_states.stride(1) == channel_count
        assert batch_size == 1 or hidden_states.stride(0) % channel_count == 0
        assert state.stride(1) == channel_count and state.stride(2) == 1
        output_frame_count = frame_count + history_frames + pad_right
        output = torch.empty(
            (batch_size, output_frame_count, channel_count),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        rows = batch_size * output_frame_count
        snake_history_kernel[
            (
                triton.cdiv(rows, SNAKE_BLOCK_ROWS),
                triton.cdiv(channel_count, SNAKE_BLOCK_CHANNELS),
            )
        ](
            hidden_states,
            alpha,
            inverse_alpha,
            output,
            state,
            slots,
            rows,
            frame_count,
            channel_count,
            hidden_states.stride(0) // channel_count,
            state.stride(0),
            pad_right,
            BLOCK_ROWS=SNAKE_BLOCK_ROWS,
            BLOCK_CHANNELS=SNAKE_BLOCK_CHANNELS,
            HISTORY=history_frames,
        )
        return output

    def transposed_conv(
        self,
        module: CausalTransConvNet,
        hidden_states: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        output = channels_last_conv_transpose1d(
            hidden_states,
            module.conv,
            self.decoder.conv_weights[id(module.conv)],
            module.conv.bias,
        )
        overlap_frames = module.kernel_size - module.stride
        new_frames = output.shape[1] - overlap_frames
        state = self.states[id(module)]
        carry = state.index_select(0, slots)
        state.index_copy_(0, slots, output[:, new_frames:] - module.conv.bias)
        output[:, :overlap_frames] += carry
        return output[:, :new_frames]


def rotated(hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """(B, T, heads, head_dim) rotated by each row's own (B, T, pairs, 2) rotary rows."""
    batch_size, frame_count, heads, _ = hidden_states.shape
    pairs = hidden_states.float().reshape(batch_size, frame_count, heads, -1, 2)
    rotations = freqs_cis.view(batch_size, frame_count, 1, pairs.shape[3], 2)
    return (
        torch.stack(
            [
                pairs[..., 0] * rotations[..., 0] - pairs[..., 1] * rotations[..., 1],
                pairs[..., 1] * rotations[..., 0] + pairs[..., 0] * rotations[..., 1],
            ],
            -1,
        )
        .flatten(3)
        .type_as(hidden_states)
    )


class IncrementalWindowDecoder:
    """Runs one group of stream windows, through a captured graph per (batch, frames)."""

    def __init__(
        self,
        decoder: IncrementalDecoder,
        *,
        num_codebooks: int,
        frame_length: int,
        device: torch.device,
        batch_sizes: tuple[int, ...],
        window_lengths: tuple[int, ...],
        use_graphs: bool,
    ) -> None:
        if not all(length <= decoder.max_window_frames for length in window_lengths):
            raise ValueError(
                f"window lengths {window_lengths} exceed the rings' "
                f"{decoder.max_window_frames} new frames"
            )
        else:
            pass
        self.decoder = decoder
        self.frame_length = frame_length
        self.device = device
        self.batch_sizes = tuple(sorted(batch_sizes))
        self.window_lengths = tuple(sorted(window_lengths))
        self.indices: dict[tuple[int, int], torch.Tensor] = {}
        self.slots: dict[int, torch.Tensor] = {}
        self.positions: dict[int, torch.Tensor] = {}
        # Host copies wait for the GPU, which the engine's process time-slices; pinned
        # sources let them queue up with the replay, so a decode waits for it only once.
        self.staged_indices: dict[tuple[int, int], torch.Tensor] = {}
        self.staged_slots: dict[int, torch.Tensor] = {}
        self.staged_positions: dict[int, torch.Tensor] = {}
        self.graphs: dict[tuple[int, int], ReplayableGraph] = {}
        self.audio: dict[tuple[int, int], torch.Tensor] = {}
        for batch_size in self.batch_sizes:
            self.slots[batch_size] = torch.full(
                (batch_size,), decoder.pad_slot, dtype=torch.long, device=device
            )
            self.positions[batch_size] = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            self.staged_slots[batch_size] = torch.empty(
                batch_size, dtype=torch.long
            ).pin_memory(device)
            self.staged_positions[batch_size] = torch.empty(
                batch_size, dtype=torch.long
            ).pin_memory(device)
            for window_length in self.window_lengths:
                self.indices[(batch_size, window_length)] = torch.zeros(
                    (batch_size, num_codebooks, window_length),
                    dtype=torch.long,
                    device=device,
                )
                self.staged_indices[(batch_size, window_length)] = torch.empty(
                    (batch_size, num_codebooks, window_length), dtype=torch.long
                ).pin_memory(device)
        if use_graphs:
            self.capture_graphs()
        else:
            pass

    def capture_graphs(self) -> None:
        graph_backend = current_platform.get_device_graph_backend(self.device)
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {self.device}; set "
                f"stages.vocoder.factory.window_cuda_graphs to false to decode eagerly"
            )
        else:
            pass
        device_module = torch.get_device_module(self.device)
        current_stream = device_module.current_stream(self.device)
        capture_stream = device_module.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        shapes = sorted(
            self.indices,
            key=lambda shape: (shape[0] * shape[1], shape),
            reverse=True,
        )
        with torch.no_grad(), device_module.device(self.device):
            with device_module.stream(capture_stream):
                for shape in shapes:
                    self.decoder(
                        self.indices[shape],
                        self.slots[shape[0]],
                        self.positions[shape[0]],
                    )
            capture_stream.synchronize()
            graph_pool = device_module.graph_pool_handle()
            for shape in shapes:
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream
                ) as graph:
                    self.audio[shape] = self.decoder(
                        self.indices[shape],
                        self.slots[shape[0]],
                        self.positions[shape[0]],
                    )
                self.graphs[shape] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(self.device)

    def window_length_for(self, frame_count: int) -> int:
        position = bisect.bisect_left(self.window_lengths, frame_count)
        if position == len(self.window_lengths):
            raise ValueError(
                f"a window of {frame_count} frames exceeds the longest of "
                f"{self.window_lengths}"
            )
        else:
            return self.window_lengths[position]

    def batch_size_for(self, row_count: int) -> int:
        return self.batch_sizes[bisect.bisect_left(self.batch_sizes, row_count)]

    def decode(
        self,
        windows: list[torch.Tensor],
        slots: list[int],
        positions: list[int],
        new_slots: list[int],
    ) -> torch.Tensor:
        """CPU (len(windows), samples) audio of (codebooks, T) windows, each padded to the
        longest; only a stream's last window may be shorter than its window length."""
        row_count = len(windows)
        window_length = self.window_length_for(
            max(int(window.shape[1]) for window in windows)
        )
        batch_size = self.batch_size_for(row_count)
        shape = (batch_size, window_length)
        staged_indices = self.staged_indices[shape].zero_()
        staged_slots = self.staged_slots[batch_size].fill_(self.decoder.pad_slot)
        staged_positions = self.staged_positions[batch_size].zero_()
        for row, window in enumerate(windows):
            staged_indices[row, :, : window.shape[1]] = window
            staged_slots[row] = slots[row]
            staged_positions[row] = positions[row]
        with torch.no_grad():
            indices = self.indices[shape]
            indices.copy_(staged_indices, non_blocking=True)
            self.slots[batch_size].copy_(staged_slots, non_blocking=True)
            self.positions[batch_size].copy_(staged_positions, non_blocking=True)
            self.decoder.reset_slots(new_slots)
            if shape in self.graphs:
                self.graphs[shape].replay()
                audio = self.audio[shape]
            else:
                audio = self.decoder(
                    indices, self.slots[batch_size], self.positions[batch_size]
                )
        return audio[:row_count, 0].cpu().float()


def incremental_window_lengths(strides: tuple[int, ...]) -> tuple[int, ...]:
    """The strides, and between far-apart ones a midpoint for the shorter last windows."""
    lengths = sorted(set(strides))
    midpoints = {
        (shorter + longer) // 2
        for shorter, longer in zip(lengths, lengths[1:])
        if longer - shorter >= 8
    }
    return tuple(sorted({*lengths, *midpoints}))


__all__ = [
    "IncrementalDecoder",
    "IncrementalWindowDecoder",
    "incremental_window_lengths",
]
