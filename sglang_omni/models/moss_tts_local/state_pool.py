# SPDX-License-Identifier: Apache-2.0
"""Row-indexed decode-state pool for MOSS-TTS Local (v1.5).

Next-step-critical per-request decode state (the next frame's feedback
embedding, request-static sampling parameters/seed, repetition penalty state,
and generation step) lives in stable, process-lifetime GPU buffers indexed by a
per-request row.
Output-only frame collection moves to a per-step
:class:`MossTTSLocalDecodeJournal`.

``P = max_running_requests + 1`` rows indexed by a per-request row. The last
row (``padding_row = P - 1``) is reserved — never acquired — as a stable
routing target for non-real/done rows under a future CUDA graph (#736). Buffer
addresses are fixed for the process lifetime.
"""

from __future__ import annotations

from typing import Any

import torch


class MossTTSLocalDecodeStatePool:
    """Row-indexed pool of next-step-critical decode state.

    Sizing and placement are derived from ``model._decode_input_embedding``
    (itself sized at runtime from ``max_running_requests``) so the pool tracks
    the configured concurrency cap without any literal row count.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        weight = model._decode_input_embedding.weight
        # P = max_running_requests + 1; the +1 is the reserved padding row.
        self.num_rows = int(weight.shape[0]) + 1
        self.padding_row = self.num_rows - 1
        self.hidden_size = int(weight.shape[1])
        self.device = weight.device
        self.dtype = weight.dtype
        try:
            config = model.config
        except AttributeError:
            config = None
        try:
            n_vq = config.n_vq
        except AttributeError:
            n_vq = 12
        try:
            audio_vocab_size = config.audio_vocab_size
        except AttributeError:
            audio_vocab_size = 1024
        self.n_vq = int(n_vq or 12)
        self.audio_vocab_size = int(audio_vocab_size or 1024)

        # Feedback embedding for the next decode step; bf16 matches the staging
        # table dtype so before_decode's gather is a plain copy (#736).
        self.feedback_embeds = torch.zeros(
            self.num_rows,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        # Request-static sampling parameters / seed (written once at acquire).
        self.text_temp = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.float32
        )
        self.text_top_p = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.float32
        )
        self.audio_temp = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.float32
        )
        self.audio_top_p = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.float32
        )
        self.text_top_k = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.audio_top_k = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.seeds = torch.zeros(self.num_rows, device=self.device, dtype=torch.int64)
        self.generation_steps = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.sampling_steps = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.audio_repetition_penalty = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.float32
        )
        self.audio_token_presence = torch.zeros(
            self.num_rows,
            self.n_vq,
            self.audio_vocab_size,
            device=self.device,
            dtype=torch.bool,
        )

        self._rid_to_row: dict[str, int] = {}
        self._params_written_rids: set[str] = set()
        self._audio_repetition_penalty_rows: set[int] = set()
        # Real rows 0..P-2 are assignable; the padding row stays out of the
        # free list so it is never handed to a request.
        self._free_rows: list[int] = list(range(self.padding_row))

    def acquire_row(self, rid: str) -> int:
        """Assign (or return the existing) row for ``rid``.

        Idempotent by rid: a request that already holds a row keeps it (the
        first-collect call site invokes this defensively every step). Raises
        ``RuntimeError`` when the pool is exhausted.
        """
        existing = self._rid_to_row.get(rid)
        if existing is not None:
            return existing
        if not self._free_rows:
            raise RuntimeError(
                "MOSS-TTS Local decode-state pool exhausted "
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
        self._params_written_rids.discard(rid)
        self.reset_row(row_idx)
        self._free_rows.append(row_idx)

    def reset_row(self, row_idx: int) -> None:
        """Zero every field of ``row_idx`` (clears stranded feedback/params)."""
        self.feedback_embeds[row_idx].zero_()
        self.text_temp[row_idx] = 0.0
        self.text_top_p[row_idx] = 0.0
        self.audio_temp[row_idx] = 0.0
        self.audio_top_p[row_idx] = 0.0
        self.text_top_k[row_idx] = 0
        self.audio_top_k[row_idx] = 0
        self.seeds[row_idx] = 0
        self.generation_steps[row_idx] = 0
        self.sampling_steps[row_idx] = 0
        self.audio_repetition_penalty[row_idx] = 0.0
        self.audio_token_presence[row_idx].zero_()
        self._audio_repetition_penalty_rows.discard(int(row_idx))

    def write_params(self, row_idx: int, data: Any) -> None:
        """Write the seven request-static sampling fields into ``row_idx``.

        Routed through the same ``float(...)``/``int(...)`` host casts the
        previous per-composition ``_param_cache`` used so the rounded values
        are bit-identical.
        """
        self.text_temp[row_idx] = float(data.text_temperature)
        self.text_top_p[row_idx] = float(data.text_top_p)
        self.audio_temp[row_idx] = float(data.audio_temperature)
        self.audio_top_p[row_idx] = float(data.audio_top_p)
        self.text_top_k[row_idx] = int(data.text_top_k)
        self.audio_top_k[row_idx] = int(data.audio_top_k)
        self.seeds[row_idx] = int(data.sampling_seed)
        try:
            audio_repetition_penalty = float(data.audio_repetition_penalty)
        except AttributeError:
            audio_repetition_penalty = 1.0
        self.audio_repetition_penalty[row_idx] = audio_repetition_penalty
        if audio_repetition_penalty == 1.0:
            self._audio_repetition_penalty_rows.discard(int(row_idx))
        else:
            self._audio_repetition_penalty_rows.add(int(row_idx))

    def ensure_params(self, row_idx: int, rid: str, data: Any) -> None:
        """Write request-static params once for the current row acquisition."""
        if rid not in self._params_written_rids:
            self.write_params(row_idx, data)
            self._params_written_rids.add(rid)

    def invalidate_params(self, rid: str) -> None:
        """Force params to be rewritten on the next ``ensure_params`` call."""
        self._params_written_rids.discard(rid)

    def commit_generation_step(self, rid: str, generation_steps: int) -> None:
        """Mirror the request's committed generation step into its pool row."""
        row_idx = self.row_for(rid)
        if row_idx is None:
            return
        step = int(generation_steps)
        self.generation_steps[row_idx] = step
        self.sampling_steps[row_idx] = torch.maximum(
            self.sampling_steps[row_idx],
            torch.tensor(step, device=self.device, dtype=torch.int64),
        )

    def commit_generation_steps(
        self, row_t: torch.Tensor, generation_steps: torch.Tensor
    ) -> None:
        """Mirror committed generation steps into active pool rows in one write."""
        if row_t.numel() == 0:
            return
        steps = generation_steps.to(device=self.device, dtype=torch.int64)
        row_t = row_t.to(device=self.device, dtype=torch.long)
        self.generation_steps[row_t] = steps
        self.sampling_steps[row_t] = torch.maximum(self.sampling_steps[row_t], steps)

    def reset_for_refill(self, rid: str, generation_steps: int = 0) -> bool:
        """Invalidate params and zero ``rid``'s row for a retraction re-prefill.

        Returns ``False`` (no-op) when ``rid`` holds no row.
        """
        row_idx = self.row_for(rid)
        if row_idx is None:
            return False
        self.invalidate_params(rid)
        self.reset_row(row_idx)
        self.generation_steps[row_idx] = int(generation_steps)
        self.sampling_steps[row_idx] = int(generation_steps)
        return True

    def rows_have_audio_repetition_penalty(self, pool_rows: list[int]) -> bool:
        """Return whether any host row has a non-default audio penalty."""
        return any(int(row) in self._audio_repetition_penalty_rows for row in pool_rows)

    def update_audio_history(self, row_t: torch.Tensor, rows: torch.Tensor) -> None:
        """Mark generated audio codes as present for future repetition penalty."""
        if row_t.numel() == 0:
            return
        if rows.ndim != 2 or int(rows.shape[1]) != self.n_vq + 1:
            raise RuntimeError(
                "MOSS-TTS Local audio history rows must have shape "
                f"[B, {self.n_vq + 1}], got {tuple(rows.shape)}"
            )
        codes = rows[:, 1:].to(device=self.device, dtype=torch.long)
        row_t = row_t.to(device=self.device, dtype=torch.long)
        if int(row_t.numel()) != int(codes.shape[0]):
            raise RuntimeError(
                "MOSS-TTS Local audio history row index mismatch: "
                f"{int(row_t.numel())} rows for {int(codes.shape[0])} code rows"
            )
        valid = (codes >= 0) & (codes < self.audio_vocab_size)
        row_idx = row_t.view(-1, 1).expand_as(codes)
        channel_idx = torch.arange(self.n_vq, device=self.device).view(1, -1)
        channel_idx = channel_idx.expand_as(codes)
        self.audio_token_presence[row_idx[valid], channel_idx[valid], codes[valid]] = (
            True
        )

    def rebuild_audio_history(self, rid: str, output_rows: list[torch.Tensor]) -> bool:
        """Rebuild ``rid``'s pool-resident history after retraction re-prefill."""
        row_idx = self.row_for(rid)
        if row_idx is None:
            return False
        self.audio_token_presence[row_idx].zero_()
        if not output_rows:
            return True
        rows = torch.stack(output_rows, dim=0)
        row_t = torch.full(
            (int(rows.shape[0]),),
            int(row_idx),
            dtype=torch.long,
            device=self.device,
        )
        self.update_audio_history(row_t, rows)
        return True

    def row_for(self, rid: str) -> int | None:
        """Return ``rid``'s row, or ``None`` if it holds no row."""
        return self._rid_to_row.get(rid)

    def prepare_active_rows(
        self, requests: list[Any]
    ) -> tuple[torch.Tensor, list[int], bool]:
        """Acquire active rows and write request-static params for this batch."""
        pool_rows = []
        has_audio_repetition_penalty = False
        for sched_req in requests:
            rid = sched_req.request_id
            row_idx = self.acquire_row(rid)
            pool_rows.append(row_idx)
            self.ensure_params(row_idx, rid, sched_req.data)
            if int(row_idx) in self._audio_repetition_penalty_rows:
                has_audio_repetition_penalty = True
        return (
            torch.tensor(pool_rows, dtype=torch.long, device=self.device),
            pool_rows,
            has_audio_repetition_penalty,
        )


class MossTTSLocalDecodeJournal:
    """Step-private record carrying the frame this step produced to collection.

    Pool rows are overwritten by the same request every step, so they cannot
    carry the "consume one step later" output data. The journal pins the
    per-step ``rows`` tensor together with the request ids (for an alignment
    assertion at apply time) and the pool rows the step touched; it is attached
    to the step's ``batch_result`` so the async-decode lookahead window (#734)
    keeps it alive until resolve.
    """

    def __init__(
        self,
        rids: list[str],
        pool_rows: list[int],
        rows: torch.Tensor,
    ) -> None:
        self.rids = rids
        self.pool_rows = pool_rows
        self.rows = rows
