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

from typing import TYPE_CHECKING

import torch

from sglang_omni.scheduling.types import SchedulerRequest

# Device-side generated-codes pool capacity per slot, in frames. Matches the
# per-request host staging bound it replaces (max_new_tokens default 4096);
# requests that generate more frames fall back to the stack+D2H finish path.
CODES_CAPACITY_FRAMES = 4096

if TYPE_CHECKING:
    from sglang_omni.models.moss_tts_local.request_builders import (
        MossTTSLocalSGLangRequestData,
    )
    from sglang_omni.models.moss_tts_local.sglang_model import MossTTSLocalSGLangModel
else:
    pass


class MossTTSLocalDecodeStatePool:
    """Row-indexed pool of next-step-critical decode state.

    Sizing and placement are derived from ``model._decode_input_embedding``
    (itself sized at runtime from ``max_running_requests``) so the pool tracks
    the configured concurrency cap without any literal row count.
    """

    def __init__(self, model: "MossTTSLocalSGLangModel") -> None:
        self.model = model
        weight = model.decode_input_embedding.weight
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

        self.rid_to_row: dict[str, int] = {}
        self.params_written_rids: set[str] = set()
        self.audio_repetition_penalty_rows: set[int] = set()
        # Real rows 0..P-2 are assignable; the padding row stays out of the
        # free list so it is never handed to a request.
        self.free_rows: list[int] = list(range(self.padding_row))
        # Zero-row templates for reset_row: slice-copies from these are
        # device-to-device kernels, unlike ``t[i] = python_scalar`` whose
        # pageable H2D blocks the scheduler thread until the in-flight
        # forward drains.
        self.zero_row_float32 = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.zero_row_int64 = torch.zeros(1, device=self.device, dtype=torch.int64)
        # Device-side accumulation of generated audio codes (the Drama3
        # slot-pool pattern): each emit scatters its rows into
        # generated_codes[row, pos] with pos from frame_counts, so the
        # terminal payload's whole-utterance D2H becomes one side-stream
        # pinned copy at finish instead of per-step staging + host scatter.
        self.codes_capacity = CODES_CAPACITY_FRAMES
        self.frame_counts = torch.zeros(
            self.num_rows, device=self.device, dtype=torch.int64
        )
        self.generated_codes = torch.zeros(
            self.num_rows,
            self.codes_capacity + 1,
            self.n_vq,
            device=self.device,
            dtype=torch.int64,
        )
        self.codes_copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.pinned_row_buffers: list[tuple[torch.Tensor, torch.cuda.Event | None]] = []
        self.pinned_parameter_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.parameter_buffer_events: list[torch.cuda.Event | None] = []

    def acquire_pinned_row_buffer(
        self, row_count: int
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Pinned staging buffer for the per-step row-index H2D, plus the
        event guarding its last enqueue.

        ``torch.tensor(list, device="cuda")`` is a pageable H2D copy: it waits
        for the current stream to drain, so on the async-decode path (where
        launch N overlaps forward N-1) it serialized the scheduler thread
        against the in-flight forward. Pinned ``copy_(non_blocking=True)``
        returns immediately and the copy overlaps the forward.

        A buffer is reused only once its guarding event reports complete, so
        the host write can never overwrite a copy the GPU has not executed.
        The classic ping-pong pair is only safe at lookahead depth 1 (the
        previous same-slot copy is always covered by a waited event); deeper
        pipelines can hold several un-waited launches, and the pool self-sizes
        to the realized depth instead of guessing it.
        """
        for buf, event in self.pinned_row_buffers:
            if buf.numel() >= row_count and (event is None or event.query()):
                return buf, event
            else:
                pass
        size = max(row_count, 128)
        buf = torch.empty(size, dtype=torch.long, pin_memory=True)
        self.pinned_row_buffers.append((buf, None))
        return buf, None

    def acquire_row(self, rid: str) -> int:
        """Assign (or return the existing) row for ``rid``.

        Idempotent by rid: a request that already holds a row keeps it (the
        first-collect call site invokes this defensively every step). Raises
        ``RuntimeError`` when the pool is exhausted.
        """
        existing = self.rid_to_row.get(rid)
        if existing is not None:
            return existing
        else:
            pass
        if not self.free_rows:
            raise RuntimeError(
                "MOSS-TTS Local decode-state pool exhausted "
                f"({self.padding_row} rows, all held); raise max_running_requests"
            )
        else:
            pass
        row_idx = self.free_rows.pop()
        self.rid_to_row[rid] = row_idx
        return row_idx

    def release_row(self, rid: str) -> None:
        """Free ``rid``'s row and reset it. No-op if ``rid`` holds no row."""
        row_idx = self.rid_to_row.pop(rid, None)
        if row_idx is None:
            return
        else:
            pass
        self.params_written_rids.discard(rid)
        self.reset_row(row_idx)
        self.free_rows.append(row_idx)

    def reset_row(self, row_idx: int) -> None:
        """Zero every field of ``row_idx`` (clears stranded feedback/params).

        Every write is an async device kernel: scalar assignments like
        ``self.text_temp[row_idx] = 0.0`` are pageable H2D copies that block
        the scheduler thread until the in-flight forward drains (~4.6 ms per
        finished request at c=128 once no earlier sync drains the stream).
        """
        self.feedback_embeds[row_idx].zero_()
        z32 = self.zero_row_float32
        z64 = self.zero_row_int64
        self.text_temp[row_idx : row_idx + 1] = z32
        self.text_top_p[row_idx : row_idx + 1] = z32
        self.audio_temp[row_idx : row_idx + 1] = z32
        self.audio_top_p[row_idx : row_idx + 1] = z32
        self.audio_repetition_penalty[row_idx : row_idx + 1] = z32
        self.text_top_k[row_idx : row_idx + 1] = z64
        self.audio_top_k[row_idx : row_idx + 1] = z64
        self.seeds[row_idx : row_idx + 1] = z64
        self.generation_steps[row_idx : row_idx + 1] = z64
        self.sampling_steps[row_idx : row_idx + 1] = z64
        # Codes accumulation restarts at position 0 for the row's next owner;
        # the codes pool itself is overwritten lazily (reads are [:count]).
        self.frame_counts[row_idx : row_idx + 1] = z64
        self.audio_token_presence[row_idx].zero_()
        self.audio_repetition_penalty_rows.discard(int(row_idx))

    def write_params(self, row_idx: int, data: "MossTTSLocalSGLangRequestData") -> None:
        """Write the seven request-static sampling fields into ``row_idx``.

        The values travel as ONE pinned ring upload plus device-side row
        writes: eight per-row scalar assignments are pageable H2D copies, and
        each one blocks the scheduler thread until every in-flight forward
        drains — on the lookahead pipeline that stall scales with the queue
        depth (measured 15% of the scheduler thread at depth 2).
        """
        try:
            audio_repetition_penalty = float(data.audio_repetition_penalty)
        except AttributeError:
            audio_repetition_penalty = 1.0
        if audio_repetition_penalty == 1.0:
            self.audio_repetition_penalty_rows.discard(int(row_idx))
        else:
            self.audio_repetition_penalty_rows.add(int(row_idx))
        if self.device.type == "cuda":
            fbuf, ibuf, slot = self.acquire_pinned_parameter_buffers()
            fbuf[0, 0] = float(data.text_temperature)
            fbuf[0, 1] = float(data.text_top_p)
            fbuf[0, 2] = float(data.audio_temperature)
            fbuf[0, 3] = float(data.audio_top_p)
            fbuf[0, 4] = audio_repetition_penalty
            ibuf[0, 0] = int(data.text_top_k)
            ibuf[0, 1] = int(data.audio_top_k)
            ibuf[0, 2] = int(data.sampling_seed)
            staged_f = fbuf[:1].to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            staged_i = ibuf[:1].to(
                device=self.device, dtype=torch.long, non_blocking=True
            )
            lo, hi = int(row_idx), int(row_idx) + 1
            self.text_temp[lo:hi] = staged_f[:, 0]
            self.text_top_p[lo:hi] = staged_f[:, 1]
            self.text_top_k[lo:hi] = staged_i[:, 0]
            self.audio_temp[lo:hi] = staged_f[:, 2]
            self.audio_top_p[lo:hi] = staged_f[:, 3]
            self.audio_top_k[lo:hi] = staged_i[:, 1]
            self.seeds[lo:hi] = staged_i[:, 2]
            self.audio_repetition_penalty[lo:hi] = staged_f[:, 4]
            if self.parameter_buffer_events[slot] is None:
                self.parameter_buffer_events[slot] = torch.cuda.Event()
            else:
                pass
            self.parameter_buffer_events[slot].record()
        else:
            self.text_temp[row_idx] = float(data.text_temperature)
            self.text_top_p[row_idx] = float(data.text_top_p)
            self.audio_temp[row_idx] = float(data.audio_temperature)
            self.audio_top_p[row_idx] = float(data.audio_top_p)
            self.text_top_k[row_idx] = int(data.text_top_k)
            self.audio_top_k[row_idx] = int(data.audio_top_k)
            self.seeds[row_idx] = int(data.sampling_seed)
            self.audio_repetition_penalty[row_idx] = audio_repetition_penalty

    def ensure_params(
        self, row_idx: int, rid: str, data: "MossTTSLocalSGLangRequestData"
    ) -> None:
        """Write request-static params once for the current row acquisition."""
        if rid not in self.params_written_rids:
            self.write_params(row_idx, data)
            self.params_written_rids.add(rid)
        else:
            pass

    def acquire_pinned_parameter_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Reuse completed parameter uploads, growing while all sources remain in flight."""
        for slot, (fbuf, ibuf) in enumerate(self.pinned_parameter_buffers):
            event = self.parameter_buffer_events[slot]
            if event is None or event.query():
                return fbuf, ibuf, slot
            else:
                pass
        fbuf = torch.empty(1, 5, dtype=torch.float32, pin_memory=True)
        ibuf = torch.empty(1, 3, dtype=torch.long, pin_memory=True)
        slot = len(self.pinned_parameter_buffers)
        self.pinned_parameter_buffers.append((fbuf, ibuf))
        self.parameter_buffer_events.append(None)
        return fbuf, ibuf, slot

    def invalidate_params(self, rid: str) -> None:
        """Force params to be rewritten on the next ``ensure_params`` call."""
        self.params_written_rids.discard(rid)

    def accumulate_codes(self, row_t: torch.Tensor, codes: torch.Tensor) -> None:
        """Scatter one step's emitted code rows into the per-slot pool.

        The extra frame at codes_capacity absorbs EOS and lookahead overruns
        without overwriting a committed frame or compacting device tensors.
        """
        k = int(row_t.numel())
        if k == 0:
            return
        else:
            pass
        pos_t = self.frame_counts[row_t].clamp(max=self.codes_capacity)
        self.generated_codes[row_t, pos_t] = codes
        self.frame_counts[row_t] += 1

    def copy_codes_to_host(self, rid: str, frame_count: int) -> torch.Tensor | None:
        """One side-stream pinned D2H of ``rid``'s accumulated codes.

        Called from the finish path after resolve waited on the launch
        completion event, so the last emit's scatter has already completed
        (host barrier) and the row is still held — a fresh side-stream copy
        races nothing. Returns ``None`` when the pool cannot serve the
        request (no row, or more frames than capacity) and the caller falls
        back to the synchronous stack+D2H.
        """
        row_idx = self.row_for(rid)
        if row_idx is None or frame_count <= 0 or frame_count > self.codes_capacity:
            return None
        else:
            pass
        codes_dev = self.generated_codes[row_idx, :frame_count]
        if self.codes_copy_stream is None:
            return codes_dev.detach().to("cpu", dtype=torch.long, copy=True)
        else:
            pass
        with torch.cuda.stream(self.codes_copy_stream):
            host = torch.empty(
                (frame_count, self.n_vq),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
            host.copy_(codes_dev, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.codes_copy_stream)
        event.synchronize()
        return host

    def reset_for_refill(self, rid: str, generation_steps: int = 0) -> bool:
        """Invalidate params and zero ``rid``'s row for a retraction re-prefill.

        Returns ``False`` (no-op) when ``rid`` holds no row.
        """
        row_idx = self.row_for(rid)
        if row_idx is None:
            return False
        else:
            pass
        self.invalidate_params(rid)
        self.reset_row(row_idx)
        self.generation_steps[row_idx] = int(generation_steps)
        self.sampling_steps[row_idx] = int(generation_steps)
        return True

    def rows_have_audio_repetition_penalty(self, pool_rows: list[int]) -> bool:
        """Return whether any host row has a non-default audio penalty."""
        return any(int(row) in self.audio_repetition_penalty_rows for row in pool_rows)

    def update_audio_history(self, row_t: torch.Tensor, rows: torch.Tensor) -> None:
        """Mark generated audio codes as present for future repetition penalty."""
        if row_t.numel() == 0:
            return
        else:
            pass
        if rows.ndim != 2 or int(rows.shape[1]) != self.n_vq + 1:
            raise RuntimeError(
                "MOSS-TTS Local audio history rows must have shape "
                f"[B, {self.n_vq + 1}], got {tuple(rows.shape)}"
            )
        else:
            pass
        codes = rows[:, 1:].to(device=self.device, dtype=torch.long)
        row_t = row_t.to(device=self.device, dtype=torch.long)
        if int(row_t.numel()) != int(codes.shape[0]):
            raise RuntimeError(
                "MOSS-TTS Local audio history row index mismatch: "
                f"{int(row_t.numel())} rows for {int(codes.shape[0])} code rows"
            )
        else:
            pass
        valid = (codes >= 0) & (codes < self.audio_vocab_size)
        row_idx = row_t.view(-1, 1).expand_as(codes)
        channel_idx = torch.arange(self.n_vq, device=self.device).view(1, -1)
        channel_idx = channel_idx.expand_as(codes)
        self.audio_token_presence[row_idx[valid], channel_idx[valid], codes[valid]] = (
            True
        )

    def rebuild_audio_history(self, rid: str, output_rows: list[torch.Tensor]) -> bool:
        """Restore committed codes and repetition history after retraction re-prefill."""
        row_idx = self.row_for(rid)
        if row_idx is None:
            return False
        else:
            pass
        self.audio_token_presence[row_idx].zero_()
        if not output_rows:
            return True
        else:
            pass
        rows = torch.stack(output_rows, dim=0).to(device=self.device, dtype=torch.long)
        retained_frames = min(len(output_rows), self.codes_capacity)
        self.generated_codes[row_idx, :retained_frames].copy_(
            rows[:retained_frames, 1:]
        )
        self.frame_counts[row_idx].fill_(len(output_rows))
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
        return self.rid_to_row.get(rid)

    def prepare_active_rows(
        self, requests: list[SchedulerRequest]
    ) -> tuple[torch.Tensor, list[int], bool]:
        """Acquire active rows and write request-static params for this batch."""
        pool_rows = []
        has_audio_repetition_penalty = False
        for sched_req in requests:
            rid = sched_req.request_id
            row_idx = self.acquire_row(rid)
            pool_rows.append(row_idx)
            self.ensure_params(row_idx, rid, sched_req.data)
            if int(row_idx) in self.audio_repetition_penalty_rows:
                has_audio_repetition_penalty = True
            else:
                pass
        if self.device.type == "cuda":
            buf, event = self.acquire_pinned_row_buffer(len(pool_rows))
            buf[: len(pool_rows)] = torch.tensor(pool_rows, dtype=torch.long)
            row_tensor = torch.empty(
                len(pool_rows), dtype=torch.long, device=self.device
            )
            row_tensor.copy_(buf[: len(pool_rows)], non_blocking=True)
            if event is None:
                event = torch.cuda.Event()
            else:
                pass
            event.record()
            for i, (entry_buf, _) in enumerate(self.pinned_row_buffers):
                if entry_buf is buf:
                    self.pinned_row_buffers[i] = (buf, event)
                    break
                else:
                    pass
        else:
            row_tensor = torch.tensor(pool_rows, dtype=torch.long, device=self.device)
        return (
            row_tensor,
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
