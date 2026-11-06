# SPDX-License-Identifier: Apache-2.0
"""Codebook head projection fused with the first half of top-k sampling.

The codebook sampler spends most of its time ranking all codebook_size logits of a
row in one program. Here the head projection's epilogue ranks each output tile
instead: every tile keeps its SELECTED best candidate keys, so the draw kernel ranks
TILES * SELECTED keys rather than codebook_size. The global SELECTED best keys are
always among the tiles' best, and the draw kernel recomputes the row maximum and
softmax normalizer from the stored logits exactly as the codebook sampler does, so
for the same logits it draws the same codes.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    SAMPLER_WARPS,
    SCORE_KEY_BITS,
    SELECTED_CANDIDATES,
    CodebookSampler,
    untested_gpu_reason,
)
from sglang_omni.models.fishaudio_drama3.tower_tiles import (
    FUSED_BLOCK_M,
    MAX_FUSED_ROWS,
    split_count,
)

try:
    import triton
    import triton.language as tl
    from sglang.srt.runtime_context import get_exec
    from triton.language.extra import libdevice

    from sglang_omni.platforms import current_platform

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    get_exec = None
    current_platform = None
    HAS_TRITON = False

HEAD_BLOCK_N = 128
HEAD_BLOCK_K = 128
HEAD_WARPS = 8
HEAD_STAGES = 4

__all__ = [
    "HEAD_BLOCK_N",
    "HeadSampling",
    "head_sampling_unsupported_reason",
]

if HAS_TRITON:
    from sglang_omni.models.fishaudio_drama3.sampling_selection import (
        ordered_keys,
        race_selected,
        selected_candidates,
    )
    from sglang_omni.models.fishaudio_drama3.tower_tiles import row_tile, weight_tile

    @triton.jit(do_not_specialize=["row_count"])
    def head_candidates_kernel(
        hidden_pointer: tl.tensor,
        hidden_row_stride: tl.tensor,
        weight_pointer: tl.tensor,
        temperature_pointer: tl.tensor,
        logits_pointer: tl.tensor,
        keys_pointer: tl.tensor,
        partials_pointer: tl.tensor,
        counters_pointer: tl.tensor,
        row_count: tl.tensor,
        DIM: tl.constexpr,
        CODEBOOK_SIZE: tl.constexpr,
        TILES: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
    ) -> None:
        """logits = hidden @ weight.T rounded to bf16, plus each row's SELECTED best
        candidate keys within this output tile."""
        output_tile_index = tl.program_id(0)
        reduction_split_index = tl.program_id(1)
        SPLIT_WIDTH: tl.constexpr = DIM // SPLIT
        row_indices = tl.arange(0, BLOCK_M)
        output_columns = output_tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
        is_row = row_indices < row_count
        product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for first_column_index in range(
            reduction_split_index * SPLIT_WIDTH,
            (reduction_split_index + 1) * SPLIT_WIDTH,
            BLOCK_K,
        ):
            product = tl.dot(
                row_tile(
                    hidden_pointer,
                    hidden_row_stride,
                    is_row,
                    first_column_index,
                    BLOCK_M,
                    BLOCK_K,
                ),
                tl.trans(
                    weight_tile(
                        weight_pointer, output_columns, first_column_index, DIM, BLOCK_K
                    )
                ),
                product,
            )
        partial_rows = (reduction_split_index * BLOCK_M + row_indices)[
            :, None
        ] * CODEBOOK_SIZE
        if SPLIT > 1:
            tl.store(
                partials_pointer + partial_rows + output_columns[None, :],
                product,
                mask=is_row[:, None],
            )
            tl.debug_barrier()
            arrived = tl.atomic_add(
                counters_pointer + output_tile_index, 1, sem="acq_rel", scope="gpu"
            )
            is_last = arrived == SPLIT - 1
        else:
            is_last = reduction_split_index == 0
        if is_last:
            if SPLIT > 1:
                product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for other_split in tl.static_range(SPLIT):
                    partial_rows = (other_split * BLOCK_M + row_indices)[
                        :, None
                    ] * CODEBOOK_SIZE
                    product += tl.load(
                        partials_pointer + partial_rows + output_columns[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(counters_pointer + output_tile_index, 0)
            else:
                pass
            # The plain path's nn.Linear hands the sampler bf16 logits.
            logits = product.to(tl.bfloat16)
            tl.store(
                logits_pointer
                + row_indices[:, None] * CODEBOOK_SIZE
                + output_columns[None, :],
                logits,
                mask=is_row[:, None],
            )
            temperature = tl.maximum(
                tl.load(temperature_pointer + row_indices, mask=is_row, other=1.0),
                MINIMUM_TEMPERATURE,
            )
            scores = tl.math.div_rn(
                logits.to(tl.float32),
                tl.broadcast_to(temperature[:, None], [BLOCK_M, BLOCK_N]),
            )
            keys = ordered_keys(
                scores,
                tl.broadcast_to(output_columns[None, :], [BLOCK_M, BLOCK_N]),
                tl.full([BLOCK_M, BLOCK_N], 1, tl.int1),
                INDEX_BITS,
            )
            top_keys = tl.topk(keys, SELECTED)
            selected = tl.arange(0, SELECTED)
            tl.store(
                keys_pointer
                + (row_indices[:, None] * TILES + output_tile_index) * SELECTED
                + selected[None, :],
                top_keys,
                mask=is_row[:, None],
            )
        else:
            pass

    @triton.jit
    def head_draw_kernel(
        logits_pointer: tl.tensor,
        keys_pointer: tl.tensor,
        temperature_pointer: tl.tensor,
        top_k_pointer: tl.tensor,
        top_p_pointer: tl.tensor,
        step_pointer: tl.tensor,
        sampled_codes_pointer: tl.tensor,
        sampled_codes_row_stride: tl.tensor,
        seed: tl.tensor,
        stream_id: tl.tensor,
        CODEBOOK_SIZE: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        CANDIDATES: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
        TOP_K_LIMIT: tl.constexpr,
    ) -> None:
        """codebook_head_kernel's draw from the tiles' candidate keys. The row maximum
        and weight sum use codebook_head_kernel's block shape and reductions."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_WIDTH)
        is_column = columns < CODEBOOK_SIZE
        logits = tl.load(
            logits_pointer + row * CODEBOOK_SIZE + columns, mask=is_column, other=0.0
        ).to(tl.float32)
        temperature = tl.maximum(
            tl.load(temperature_pointer + row), MINIMUM_TEMPERATURE
        )
        scores = tl.where(
            is_column,
            tl.math.div_rn(logits, tl.broadcast_to(temperature, logits.shape)),
            float("-inf"),
        )
        row_maximum = tl.maximum(tl.max(scores, axis=0), float("-inf"))
        weight_sum = tl.sum(libdevice.exp(scores - row_maximum), axis=0) + 0.0
        candidate_keys = tl.load(
            keys_pointer + row * CANDIDATES + tl.arange(0, CANDIDATES)
        )
        top_keys = tl.topk(candidate_keys, SELECTED)
        top_k = tl.minimum(tl.load(top_k_pointer + row), TOP_K_LIMIT)
        selected_indices, selected_weights, is_kept, _ = selected_candidates(
            top_keys,
            row_maximum,
            weight_sum,
            top_k,
            tl.load(top_p_pointer + row),
            INDEX_BITS,
            False,
        )
        sampled_code = race_selected(
            selected_indices,
            selected_weights,
            is_kept,
            seed,
            tl.load(step_pointer),
            row,
            stream_id,
        )
        tl.store(
            sampled_codes_pointer + row * sampled_codes_row_stride,
            sampled_code.to(tl.int64),
        )

else:
    pass


def head_sampling_unsupported_reason(
    *,
    heads: list[nn.Linear],
    top_k_limit: int,
    device: torch.device,
) -> str | None:
    if not HAS_TRITON:
        return "Triton is not installed"
    elif device.type != "cuda" or not current_platform.is_cuda():
        return f"device {device} is not a CUDA device of the CUDA platform"
    elif (gpu_reason := untested_gpu_reason(device)) is not None:
        return gpu_reason
    elif get_exec().deterministic.enable_deterministic_inference:
        return (
            f"deterministic inference is on, and passes above {MAX_FUSED_ROWS} rows "
            f"keep the plain heads"
        )
    elif not 0 < top_k_limit <= SELECTED_CANDIDATES:
        return (
            f"the codebook top-k cap {top_k_limit} is outside "
            f"(0, {SELECTED_CANDIDATES}]"
        )
    else:
        pass
    for head_index, head in enumerate(heads):
        weight = head.weight
        codebook_size, dim = weight.shape
        if type(head) is not nn.Linear or head.bias is not None:
            return f"codebook head {head_index} is not a bias-free torch Linear"
        elif (
            weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.device.type != device.type
            or (device.index is not None and weight.device.index != device.index)
        ):
            return f"codebook head {head_index} weight is not contiguous bfloat16"
        elif (
            codebook_size % HEAD_BLOCK_N != 0
            or dim % HEAD_BLOCK_K != 0
            or codebook_size & (codebook_size - 1) != 0
            or codebook_size // HEAD_BLOCK_N * SELECTED_CANDIDATES > codebook_size
        ):
            return (
                f"codebook head {head_index} of {codebook_size} x {dim} does not fit "
                f"tiles of {HEAD_BLOCK_N} x {HEAD_BLOCK_K}"
            )
        else:
            pass
    return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class HeadSampling:
    """Scratch for the fused codebook heads of up to MAX_FUSED_ROWS rows."""

    logits: torch.Tensor
    keys: torch.Tensor
    partials: torch.Tensor
    counters: torch.Tensor
    split: int
    tiles: int
    top_k_limit: int
    minimum_temperature: float

    @classmethod
    def allocate(
        cls,
        *,
        codebook_size: int,
        dim: int,
        top_k_limit: int,
        minimum_temperature: float,
        device: torch.device,
    ) -> HeadSampling:
        tiles = codebook_size // HEAD_BLOCK_N
        split = split_count(
            tiles,
            dim // HEAD_BLOCK_K,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        return cls(
            logits=torch.zeros(
                (MAX_FUSED_ROWS, codebook_size), dtype=torch.bfloat16, device=device
            ),
            keys=torch.zeros(
                (MAX_FUSED_ROWS, tiles * SELECTED_CANDIDATES),
                dtype=torch.int64,
                device=device,
            ),
            partials=torch.zeros(
                split * FUSED_BLOCK_M * codebook_size,
                dtype=torch.float32,
                device=device,
            ),
            counters=torch.zeros(tiles, dtype=torch.int32, device=device),
            split=split,
            tiles=tiles,
            top_k_limit=top_k_limit,
            minimum_temperature=minimum_temperature,
        )

    def covers(self, row_count: int) -> bool:
        return row_count <= MAX_FUSED_ROWS

    def sample(
        self,
        *,
        head: nn.Linear,
        hidden_states: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        sampler: CodebookSampler,
        stream_id: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Write one code per row of hidden_states into out; returns the logits."""
        row_count = hidden_states.shape[0]
        weight = head.weight
        codebook_size, dim = weight.shape
        index_bits = (codebook_size - 1).bit_length()
        logits = self.logits[:row_count]
        temperature = temperature.float().contiguous()
        assert hidden_states.stride(1) == 1 and out.dtype == torch.int64
        head_candidates_kernel[(self.tiles, self.split)](
            hidden_states,
            hidden_states.stride(0),
            weight,
            temperature,
            logits,
            self.keys,
            self.partials,
            self.counters,
            row_count,
            DIM=dim,
            CODEBOOK_SIZE=codebook_size,
            TILES=self.tiles,
            SPLIT=self.split,
            BLOCK_M=FUSED_BLOCK_M,
            BLOCK_N=HEAD_BLOCK_N,
            BLOCK_K=HEAD_BLOCK_K,
            INDEX_BITS=index_bits,
            SELECTED=SELECTED_CANDIDATES,
            MINIMUM_TEMPERATURE=self.minimum_temperature,
            num_warps=HEAD_WARPS,
            num_stages=HEAD_STAGES,
        )
        head_draw_kernel[(row_count,)](
            logits,
            self.keys,
            temperature,
            top_k.long().contiguous(),
            top_p.float().contiguous(),
            sampler.step,
            out,
            out.stride(0),
            sampler.seed,
            stream_id,
            CODEBOOK_SIZE=codebook_size,
            BLOCK_WIDTH=triton.next_power_of_2(codebook_size),
            CANDIDATES=self.tiles * SELECTED_CANDIDATES,
            INDEX_BITS=index_bits,
            SELECTED=SELECTED_CANDIDATES,
            MINIMUM_TEMPERATURE=self.minimum_temperature,
            TOP_K_LIMIT=self.top_k_limit,
            num_warps=SAMPLER_WARPS,
        )
        return logits
