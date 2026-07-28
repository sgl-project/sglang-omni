# SPDX-License-Identifier: Apache-2.0
"""On-device per-request decode state for ZONOS2.

Each in-flight request holds a fixed row; per-step feedback, the EOS state
machine and the generation step live in pre-allocated GPU tensors indexed by
row, so the decode tail can update state as device math. Modelled on
``moss_tts_local.state_pool``. A reserved padding row (``P-1``) is never handed
out, giving graph replay a stable index for padded batch slots.
"""

from __future__ import annotations

from typing import Any

import torch


class Zonos2DecodeStatePool:
    """Fixed-capacity pool of per-request decode state on the model's device."""

    def __init__(self, model: Any) -> None:
        weight = model._decode_input_embedding.weight
        self.device = weight.device
        self.dim = int(weight.shape[1])
        # The decode buffer caps decode bs (= weight.shape[0]); the pool, though,
        # tracks every in-flight request (prefilled-but-not-finished), which
        # transiently exceeds that during prefill/decode overlap. Size generously
        # (the scheduler's max_running is the real concurrency limit, not the pool)
        # + reserve a padding row; ~4x decode bs of [dim] bf16 is negligible VRAM.
        self.num_rows = 4 * int(weight.shape[0]) + 1
        self.padding_row = self.num_rows - 1

        # Next-step feedback embedding, gathered into the decode input buffer.
        self.feedback_embeds = torch.zeros(
            self.num_rows, self.dim, device=self.device, dtype=weight.dtype
        )
        # EOS state machine (mirrors data.eos_frame / eos_countdown).
        self.eos_frame_set = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.bool
        )
        self.eos_frame_val = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.eos_countdown = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.generation_step = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        # Rep-penalty / loop-break history ring: the last rep_ring frames of [n]
        # codes per request, left-padded with -1 (matches the scalar rep_hist's
        # left-pad). Sized >= the default repetition_window (50) with headroom.
        self.rep_ring = 64
        self.rep_hist = torch.full(
            (self.num_rows, self.rep_ring, int(model.n_codebooks)),
            -1,
            device=self.device,
            dtype=torch.int64,
        )
        self.rep_len = torch.zeros(self.num_rows, device=self.device, dtype=torch.int64)

        # Real rows 0..P-2 are assignable; the padding row stays out of the free
        # list so it is never handed to a request.
        self._free_rows: list[int] = list(range(self.padding_row))
        self._rid_to_row: dict[str, int] = {}

        # Memoized active-row index tensor (see prepare_active_rows): in steady
        # decode the in-flight request set is unchanged for hundreds of frames,
        # so rebuilding the tensor + its H2D copy every step is pure overhead
        # (~60% of scheduler-thread wall time, py-spy). Keyed by the request-id
        # tuple, so any membership/order change misses and rebuilds.
        self._active_ids: tuple[str, ...] | None = None
        self._active_rows: torch.Tensor | None = None

    def acquire_row(self, rid: str) -> int:
        """Assign (or return the existing) row for ``rid``. Idempotent by rid."""
        existing = self._rid_to_row.get(rid)
        if existing is not None:
            return existing
        if not self._free_rows:
            raise RuntimeError(
                "ZONOS2 decode-state pool exhausted "
                f"({self.padding_row} rows, all held); raise max_running_requests"
            )
        row_idx = self._free_rows.pop()
        self._rid_to_row[rid] = row_idx
        return row_idx

    def release_row(self, rid: str) -> None:
        """Free ``rid``'s row and reset it. No-op if ``rid`` holds no row."""
        row_idx = self._rid_to_row.pop(rid, None)
        if row_idx is None:
            return
        # The cached tensor is valid only while every request-to-row mapping
        # used to build it is still owned. Invalidate it before recycling a row
        # so an immediately reused request id cannot bypass acquire_row().
        self._active_ids = None
        self._active_rows = None
        self.reset_row(row_idx)
        self._free_rows.append(row_idx)

    def reset_row(self, row_idx: int) -> None:
        """Clear every field of ``row_idx`` (stranded feedback/EOS state)."""
        self.feedback_embeds[row_idx].zero_()
        self.eos_frame_set[row_idx] = False
        self.eos_frame_val[row_idx] = 0
        self.eos_countdown[row_idx] = 0
        self.generation_step[row_idx] = 0
        self.rep_hist[row_idx].fill_(-1)
        self.rep_len[row_idx] = 0

    def row_for(self, rid: str) -> int:
        return self._rid_to_row[rid]

    def prepare_active_rows(self, requests: list) -> torch.Tensor:
        """Acquire (idempotent) a row per request and return the row-index tensor
        aligned to ``requests`` order, for a single batched gather/scatter.

        Memoized by the request-id tuple: in steady decode the set (and order)
        is unchanged for hundreds of frames, so the per-step torch.tensor(...)
        + H2D copy is skipped. Any membership/order change yields a different
        tuple -> cache miss -> rebuild (acquire_row stays idempotent, so an
        unchanged tuple guarantees an unchanged row mapping)."""
        ids = tuple(sr.request_id for sr in requests)
        if ids == self._active_ids and self._active_rows is not None:
            return self._active_rows
        idx = [self.acquire_row(rid) for rid in ids]
        rows = torch.tensor(idx, device=self.device, dtype=torch.long)
        self._active_ids = ids
        self._active_rows = rows
        return rows

    def release_inactive(self, active_rids) -> None:
        """Reconcile row ownership with the current decode batch.

        Terminal result and abort adapters are the primary cleanup paths. This
        fallback handles batch removal such as retraction and recovers any
        stranded older row before the next decode. A request that comes back
        after retraction/re-prefill acquires a freshly reset row.
        """
        stale = [rid for rid in self._rid_to_row if rid not in active_rids]
        for rid in stale:
            self.release_row(rid)
