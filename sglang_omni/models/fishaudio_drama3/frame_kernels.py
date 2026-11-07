# SPDX-License-Identifier: Apache-2.0
"""Single-launch frame embeddings and decode-row staging for Drama 3.

The backbone input and the shared tower input each replace a chain of gathers,
clamps, sums and selects with one kernel that reproduces the PyTorch path bit for
bit. The decode-row kernels replace the per-field gathers and scatters the model
runner issues around every decode step.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

EMBEDDING_BLOCK = 512
MINIMUM_EMBEDDING_BLOCK = 32

__all__ = [
    "EMBEDDING_BLOCK",
    "embedding_block",
    "frame_backbone_input",
    "frame_shared_input",
    "gather_decode_rows",
    "publish_decode_frames",
]


def embedding_block(dim: int) -> int:
    """Columns per program: the largest power of two up to EMBEDDING_BLOCK that
    divides dim; below MINIMUM_EMBEDDING_BLOCK the kernels do not apply."""
    return math.gcd(dim, EMBEDDING_BLOCK)


if HAS_TRITON:

    @triton.jit
    def codebook_embedding_sum(
        frames_row_pointer: tl.tensor,
        codebook_weight_pointer: tl.tensor,
        columns: tl.tensor,
        DIM: tl.constexpr,
        CODEBOOK_SIZE: tl.constexpr,
        FIRST_CODEBOOK: tl.constexpr,
        END_CODEBOOK: tl.constexpr,
    ) -> tl.tensor:
        """embedding_bag(mode="sum") of codes FIRST..END-1: fp32 sum in code order."""
        summed = tl.zeros(columns.shape, dtype=tl.float32)
        for codebook in tl.static_range(FIRST_CODEBOOK, END_CODEBOOK):
            code = tl.load(frames_row_pointer + 1 + codebook)
            code = tl.minimum(tl.maximum(code, 0), CODEBOOK_SIZE - 1)
            summed += tl.load(
                codebook_weight_pointer
                + (code + codebook * CODEBOOK_SIZE) * DIM
                + columns
            ).to(tl.float32)
        return summed

    @triton.jit
    def backbone_input_kernel(
        frames_pointer: tl.tensor,
        frames_row_stride: tl.tensor,
        text_weight_pointer: tl.tensor,
        codebook_weight_pointer: tl.tensor,
        output_pointer: tl.tensor,
        semantic_begin_id: tl.tensor,
        semantic_end_id: tl.tensor,
        scale: tl.tensor,
        DIM: tl.constexpr,
        CODEBOOK_SIZE: tl.constexpr,
        FIRST_CODEBOOK: tl.constexpr,
        END_CODEBOOK: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0).to(tl.int64)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        frames_row_pointer = frames_pointer + row * frames_row_stride
        token = tl.load(frames_row_pointer)
        text = tl.load(text_weight_pointer + token * DIM + columns)
        if (token >= semantic_begin_id) & (token <= semantic_end_id):
            summed = codebook_embedding_sum(
                frames_row_pointer,
                codebook_weight_pointer,
                columns,
                DIM,
                CODEBOOK_SIZE,
                FIRST_CODEBOOK,
                END_CODEBOOK,
            )
            # (text + bf16(sum)) rounds to bf16, then the scale rounds again.
            mixed = (
                (text.to(tl.float32) + summed.to(tl.bfloat16).to(tl.float32))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            text = (mixed * scale).to(tl.bfloat16)
        else:
            pass
        tl.store(output_pointer + row * DIM + columns, text)

    @triton.jit
    def shared_input_kernel(
        frames_pointer: tl.tensor,
        frames_row_stride: tl.tensor,
        codebook_weight_pointer: tl.tensor,
        output_pointer: tl.tensor,
        semantic_begin_id: tl.tensor,
        semantic_end_id: tl.tensor,
        DIM: tl.constexpr,
        CODEBOOK_SIZE: tl.constexpr,
        FIRST_CODEBOOK: tl.constexpr,
        END_CODEBOOK: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(0).to(tl.int64)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        frames_row_pointer = frames_pointer + row * frames_row_stride
        token = tl.load(frames_row_pointer)
        summed = tl.zeros([BLOCK], dtype=tl.float32)
        if (token >= semantic_begin_id) & (token <= semantic_end_id):
            summed = codebook_embedding_sum(
                frames_row_pointer,
                codebook_weight_pointer,
                columns,
                DIM,
                CODEBOOK_SIZE,
                FIRST_CODEBOOK,
                END_CODEBOOK,
            )
        else:
            pass
        tl.store(output_pointer + row * DIM + columns, summed.to(tl.bfloat16))

    @triton.jit
    def gather_decode_rows_kernel(
        row_indices_pointer: tl.tensor,
        state_frames_pointer: tl.tensor,
        state_ras_windows_pointer: tl.tensor,
        state_temperature_pointer: tl.tensor,
        state_top_k_pointer: tl.tensor,
        state_top_p_pointer: tl.tensor,
        state_epsilon_cutoff_pointer: tl.tensor,
        frames_pointer: tl.tensor,
        ras_windows_pointer: tl.tensor,
        temperature_pointer: tl.tensor,
        top_k_pointer: tl.tensor,
        top_p_pointer: tl.tensor,
        epsilon_cutoff_pointer: tl.tensor,
        FRAME_WIDTH: tl.constexpr,
        FRAME_BLOCK: tl.constexpr,
        WINDOW: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        state_row = tl.load(row_indices_pointer + row)
        frame_columns = tl.arange(0, FRAME_BLOCK)
        is_frame_column = frame_columns < FRAME_WIDTH
        tl.store(
            frames_pointer + row * FRAME_WIDTH + frame_columns,
            tl.load(
                state_frames_pointer + state_row * FRAME_WIDTH + frame_columns,
                mask=is_frame_column,
            ),
            mask=is_frame_column,
        )
        window_columns = tl.arange(0, WINDOW)
        tl.store(
            ras_windows_pointer + row * WINDOW + window_columns,
            tl.load(state_ras_windows_pointer + state_row * WINDOW + window_columns),
        )
        tl.store(
            temperature_pointer + row, tl.load(state_temperature_pointer + state_row)
        )
        tl.store(top_k_pointer + row, tl.load(state_top_k_pointer + state_row))
        tl.store(top_p_pointer + row, tl.load(state_top_p_pointer + state_row))
        tl.store(
            epsilon_cutoff_pointer + row,
            tl.load(state_epsilon_cutoff_pointer + state_row),
        )

    @triton.jit
    def publish_decode_frames_kernel(
        row_indices_pointer: tl.tensor,
        sampled_frames_pointer: tl.tensor,
        published_frames_pointer: tl.tensor,
        state_frames_pointer: tl.tensor,
        state_ras_windows_pointer: tl.tensor,
        FRAME_WIDTH: tl.constexpr,
        FRAME_BLOCK: tl.constexpr,
        WINDOW: tl.constexpr,
    ) -> None:
        """Copy out a row's sampled frame, store it as the row's newest frame, and
        shift its main token into the row's RAS window."""
        row = tl.program_id(0)
        state_row = tl.load(row_indices_pointer + row)
        frame_columns = tl.arange(0, FRAME_BLOCK)
        is_frame_column = frame_columns < FRAME_WIDTH
        frame = tl.load(
            sampled_frames_pointer + row * FRAME_WIDTH + frame_columns,
            mask=is_frame_column,
        )
        tl.store(
            published_frames_pointer + row * FRAME_WIDTH + frame_columns,
            frame,
            mask=is_frame_column,
        )
        tl.store(
            state_frames_pointer + state_row * FRAME_WIDTH + frame_columns,
            frame,
            mask=is_frame_column,
        )
        window_columns = tl.arange(0, WINDOW)
        window_row_pointer = state_ras_windows_pointer + state_row * WINDOW
        shifted = tl.load(
            window_row_pointer + window_columns + 1,
            mask=window_columns < WINDOW - 1,
            other=0,
        )
        main_token = tl.load(sampled_frames_pointer + row * FRAME_WIDTH)
        shifted = tl.where(window_columns == WINDOW - 1, main_token, shifted)
        tl.debug_barrier()
        tl.store(window_row_pointer + window_columns, shifted)

else:
    pass


def frame_backbone_input(
    *,
    frames: torch.Tensor,
    text_weight: torch.Tensor,
    codebook_weight: torch.Tensor,
    semantic_begin_id: int,
    semantic_end_id: int,
    codebook_size: int,
    backbone_codebook_count: int,
    scale: float,
) -> torch.Tensor:
    """AudioDecoder.embed_backbone(frames, text embedding of frames[:, 0])."""
    row_count = frames.shape[0]
    dim = text_weight.shape[1]
    output = torch.empty(
        (row_count, dim), dtype=text_weight.dtype, device=frames.device
    )
    if row_count == 0:
        return output
    else:
        pass
    block = embedding_block(dim)
    backbone_input_kernel[(row_count, dim // block)](
        frames,
        frames.stride(0),
        text_weight,
        codebook_weight,
        output,
        semantic_begin_id,
        semantic_end_id,
        scale,
        DIM=dim,
        CODEBOOK_SIZE=codebook_size,
        FIRST_CODEBOOK=1,
        END_CODEBOOK=1 + backbone_codebook_count,
        BLOCK=block,
    )
    return output


def frame_shared_input(
    *,
    frames: torch.Tensor,
    codebook_weight: torch.Tensor,
    semantic_begin_id: int,
    semantic_end_id: int,
    codebook_size: int,
    first_codebook: int,
    end_codebook: int,
) -> torch.Tensor:
    """AudioDecoder.shared_input(frames)."""
    row_count = frames.shape[0]
    dim = codebook_weight.shape[1]
    output = torch.empty(
        (row_count, dim), dtype=codebook_weight.dtype, device=frames.device
    )
    if row_count == 0:
        return output
    else:
        pass
    block = embedding_block(dim)
    shared_input_kernel[(row_count, dim // block)](
        frames,
        frames.stride(0),
        codebook_weight,
        output,
        semantic_begin_id,
        semantic_end_id,
        DIM=dim,
        CODEBOOK_SIZE=codebook_size,
        FIRST_CODEBOOK=first_codebook,
        END_CODEBOOK=end_codebook,
        BLOCK=block,
    )
    return output


def gather_decode_rows(
    *,
    row_indices: torch.Tensor,
    state_frames: torch.Tensor,
    state_ras_windows: torch.Tensor,
    state_temperature: torch.Tensor,
    state_top_k: torch.Tensor,
    state_top_p: torch.Tensor,
    state_epsilon_cutoff: torch.Tensor,
    frames: torch.Tensor,
    ras_windows: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    epsilon_cutoff: torch.Tensor,
) -> None:
    """index_select of every decode-state field into the decode buffers."""
    row_count = row_indices.shape[0]
    if row_count == 0:
        return
    else:
        pass
    frame_width = state_frames.shape[1]
    gather_decode_rows_kernel[(row_count,)](
        row_indices,
        state_frames,
        state_ras_windows,
        state_temperature,
        state_top_k,
        state_top_p,
        state_epsilon_cutoff,
        frames,
        ras_windows,
        temperature,
        top_k,
        top_p,
        epsilon_cutoff,
        FRAME_WIDTH=frame_width,
        FRAME_BLOCK=triton.next_power_of_2(frame_width),
        WINDOW=state_ras_windows.shape[1],
        num_warps=1,
    )


def publish_decode_frames(
    *,
    row_indices: torch.Tensor,
    sampled_frames: torch.Tensor,
    state_frames: torch.Tensor,
    state_ras_windows: torch.Tensor,
) -> torch.Tensor:
    """A copy of the sampled frames; each state row takes its frame and RAS shift."""
    row_count = row_indices.shape[0]
    published = torch.empty_like(sampled_frames)
    if row_count == 0:
        return published
    else:
        pass
    frame_width = state_frames.shape[1]
    publish_decode_frames_kernel[(row_count,)](
        row_indices,
        sampled_frames,
        published,
        state_frames,
        state_ras_windows,
        FRAME_WIDTH=frame_width,
        FRAME_BLOCK=triton.next_power_of_2(frame_width),
        WINDOW=state_ras_windows.shape[1],
        num_warps=1,
    )
    return published
