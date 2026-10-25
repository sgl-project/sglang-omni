# SPDX-License-Identifier: Apache-2.0
"""One-launch kernels for the talker's per-slot sampling state under speculative decoding.

A speculative step reads each running request's sampling state out of the runner's
slots and, after accept, appends the committed block back. Done with torch indexing
that is about thirty small kernels a step, a host cost the step pays in launches; these
kernels do each side in one launch, one program per request.
"""

from __future__ import annotations

import torch

from sglang_omni.models.minicpm_o.talker_dspark_sampling import TalkerSamplingState
from sglang_omni.models.minicpm_o.talker_model_runner import (
    REP_PENALTY_WINDOW,
    TalkerSlotState,
)
from sglang_omni.platforms import current_platform

if not current_platform.is_npu():
    try:
        import triton
        import triton.language as tl
    except ImportError:  # pragma: no cover - depends on runtime image
        triton = None
        tl = None
else:
    triton = None
    tl = None

SUPPRESS_BLOCK = 1024


def has_slot_kernels(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


if triton is not None:

    @triton.jit
    def gather_slot_rows_kernel(
        rows,
        windows,
        generated,
        min_new_tokens,
        penalties,
        suppress,
        temperatures,
        top_ks,
        top_ps,
        out_recent_tokens,
        out_generated,
        out_min_new_tokens,
        out_penalties,
        out_suppress,
        out_temperatures,
        out_top_ks,
        out_top_ps,
        out_greedy,
        vocab,
        WINDOW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        slot = tl.load(rows + row).to(tl.int64)
        count = tl.load(generated + slot)
        offsets = tl.arange(0, WINDOW)
        window = tl.load(windows + slot * WINDOW + (count + offsets) % WINDOW)
        tl.store(out_recent_tokens + row * WINDOW + offsets, window)
        tl.store(out_generated + row, count)
        tl.store(out_min_new_tokens + row, tl.load(min_new_tokens + slot))
        tl.store(out_penalties + row, tl.load(penalties + slot))
        tl.store(out_temperatures + row, tl.load(temperatures + slot))
        top_k = tl.load(top_ks + slot)
        tl.store(out_top_ks + row, top_k.to(tl.int64))
        tl.store(out_greedy + row, (top_k == 1).to(tl.uint8))
        tl.store(out_top_ps + row, tl.load(top_ps + slot))
        for start in range(0, vocab, BLOCK):
            columns = start + tl.arange(0, BLOCK)
            inside = columns < vocab
            values = tl.load(suppress + slot * vocab + columns, mask=inside)
            tl.store(out_suppress + row * vocab + columns, values, mask=inside)

    @triton.jit
    def commit_slot_rows_kernel(
        rows,
        committed,
        counts,
        windows,
        generated,
        block_width,
        WINDOW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        slot = tl.load(rows + row).to(tl.int64)
        start = tl.load(generated + slot)
        count = tl.load(counts + row).to(tl.int64)
        offsets = tl.arange(0, BLOCK)
        kept = offsets < count
        tokens = tl.load(committed + row * block_width + offsets, mask=kept)
        tl.store(
            windows + slot * WINDOW + (start + offsets) % WINDOW,
            tokens.to(tl.int64),
            mask=kept,
        )
        tl.store(generated + slot, start + count)

else:
    gather_slot_rows_kernel = None
    commit_slot_rows_kernel = None


def gather_slot_rows(
    slots: TalkerSlotState, rows: torch.Tensor, out: TalkerSamplingState
) -> None:
    """Write the sampling state of rows into the leading rows of out, as state_from_slots
    would return it."""
    count = int(rows.shape[0])
    vocab = int(slots.suppress.shape[1])
    gather_slot_rows_kernel[(count,)](
        rows,
        slots.windows,
        slots.generated,
        slots.min_new_tokens,
        slots.penalties,
        slots.suppress.view(torch.uint8),
        slots.temperatures,
        slots.top_ks,
        slots.top_ps,
        out.recent_tokens,
        out.generated,
        out.min_new_tokens,
        out.penalties,
        out.suppress.view(torch.uint8),
        out.temperatures,
        out.top_ks,
        out.top_ps,
        out.greedy.view(torch.uint8),
        vocab,
        WINDOW=REP_PENALTY_WINDOW,
        BLOCK=SUPPRESS_BLOCK,
    )


def commit_slot_rows(
    slots: TalkerSlotState,
    rows: torch.Tensor,
    committed: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Append each row's first counts committed tokens, as commit_to_slots does."""
    block_width = int(committed.shape[1])
    if block_width > REP_PENALTY_WINDOW:
        raise ValueError(
            f"a committed block of {block_width} tokens overruns the "
            f"{REP_PENALTY_WINDOW}-token penalty window"
        )
    else:
        pass
    commit_slot_rows_kernel[(int(rows.shape[0]),)](
        rows,
        committed.contiguous(),
        counts,
        slots.windows,
        slots.generated,
        block_width,
        WINDOW=REP_PENALTY_WINDOW,
        BLOCK=triton.next_power_of_2(block_width),
    )
