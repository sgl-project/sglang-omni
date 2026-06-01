# SPDX-License-Identifier: Apache-2.0
"""Base model runner — shared execute() pipeline for all AR models.

Handles: ForwardBatch construction, phase-aware pre/post hooks, forward
pass, sampling, logit post-processing, and output extraction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.scheduling.types import ModelRunnerOutput, RequestOutput

logger = logging.getLogger(__name__)


@dataclass
class _PendingStep:
    """One decode step launched on the GPU but not yet consumed on the host.

    Async-decode (one-step lookahead) bookkeeping: a launched step has its
    forward + on-GPU sample enqueued, its collect-staging buffer async-copied
    (D2H) into ``host_buf``, and ``event`` recorded right after that copy.
    ``execute_resolve`` later waits on ``event`` and reads ``host_buf``.

    Invariant: at most one ``_PendingStep`` is live at a time (see
    ``ModelRunner._pending``). ``host_buf`` is pinned and ping-ponged between
    two buffers so resolve(N) can read one while launch(N+1)'s D2H writes the
    other (a CPU-read vs GPU-write race not covered by stream ordering —
    design.md §1.4).
    """

    event: Any  # torch.cuda.Event, recorded right after the async D2H copy
    host_buf: Any  # pinned host tensor holding this step's staging snapshot
    scheduler_output: Any  # this step's SchedulerOutput (routing + output proc)
    forward_batch: Any  # for resolve-time finalize sampling
    schedule_batch: Any  # to set .output_ids during resolve
    model_worker_batch: Any  # for the prefill-only finalize branch (unused in decode)
    batch_result: Any  # carries logits_output (device of next_token_ids)
    n_real: int  # number of real (non-padding) rows this step


class ModelRunner:
    """Base AR model runner.

    Subclasses provide phase-specific behavior:
      - prefill hooks for extend/prompt processing
      - decode hooks for single-step autoregressive decode processing
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        self.tp_worker = tp_worker
        self.output_processor = output_processor
        self.device = torch.device(f"cuda:{tp_worker.gpu_id}")
        self.model = tp_worker.model_runner.model

        # Async decode (one-step lookahead). Inert unless ``_async_enabled`` is set.
        self._async_enabled: bool = False
        self._staging_slot: int = 0
        self._host_staging_buffers: list[torch.Tensor] = []
        # Observability: how often resolve found the event already done
        # (overlap worked) vs had to block on synchronize().
        self._async_query_hit: int = 0
        self._async_query_miss: int = 0

    def _next_host_staging(self, device_staging: torch.Tensor) -> torch.Tensor:
        """Return a pinned host buffer mirroring ``device_staging``'s full
        shape, ping-ponging between two buffers on each call.

        Two buffers are required: resolve(N) reads one on the host while
        launch(N+1)'s async D2H writes the other. That CPU-read vs GPU-write
        overlap is not protected by single-stream ordering (design.md §1.4).
        Buffers are allocated lazily on first use (the base runner does not
        know the model-specific staging shape at construction time).
        """
        if not self._host_staging_buffers:
            self._host_staging_buffers = [
                torch.empty(
                    device_staging.shape,
                    dtype=device_staging.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        buf = self._host_staging_buffers[self._staging_slot]
        self._staging_slot ^= 1
        return buf

    def execute(self, scheduler_output: Any) -> ModelRunnerOutput:
        """Full synchronous pipeline: build → prepare → forward → post →
        sample → output.

        Used when async decode is disabled. Behavior is byte-identical to the
        pre-async implementation: it is a pure extraction over the same shared
        sub-steps (``_build_forward_batch`` / ``_prepare_and_forward`` /
        ``_finalize``) that ``execute_launch`` + ``execute_resolve`` also use,
        in the same order. Async decode splits this at the post-decode boundary.
        """
        built = self._build_forward_batch(scheduler_output)
        if built is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})
        forward_batch, schedule_batch, model_worker_batch, is_prefill = built
        batch_result = self._prepare_and_forward(
            forward_batch, schedule_batch, scheduler_output.requests, is_prefill
        )
        if is_prefill:
            self.post_prefill(
                batch_result, forward_batch, schedule_batch, scheduler_output.requests
            )
        else:
            self.post_decode(
                batch_result, forward_batch, schedule_batch, scheduler_output.requests
            )
        return self._finalize(
            batch_result,
            forward_batch,
            schedule_batch,
            model_worker_batch,
            scheduler_output,
        )

    def execute_launch(self, scheduler_output: Any) -> "_PendingStep | None":
        """Enqueue a decode step's forward + on-GPU sample, snapshot its
        collect state into a pinned host buffer (``post_decode_launch``), and
        record a CUDA event right after that async D2H. Does NOT wait on the
        GPU. Decode batches only.

        Returns the ``_PendingStep`` handle (or None if there was no batch).
        The CALLER owns the handle and passes it to ``execute_resolve`` later.
        Ownership lives with the caller (not on ``self``) because launch-first
        scheduling has two steps momentarily in flight: the just-launched step
        N and the not-yet-resolved step N-1.
        """
        built = self._build_forward_batch(scheduler_output)
        if built is None:
            return None
        forward_batch, schedule_batch, model_worker_batch, is_prefill = built
        assert not is_prefill, "async lookahead launch is decode-only"
        batch_result = self._prepare_and_forward(
            forward_batch,
            schedule_batch,
            scheduler_output.requests,
            is_prefill,
            is_lookahead=True,
        )
        host_buf = self.post_decode_launch(
            batch_result, forward_batch, scheduler_output.requests
        )
        # Publish this step's output token ids now (post_decode_launch set them
        # from GPU state without a host sync) so the NEXT decode step's
        # get_next_batch_to_run / prepare_for_decode can build its input_ids —
        # under lookahead the host collect (resolve) lags by one step.
        if batch_result.next_token_ids is not None:
            schedule_batch.output_ids = batch_result.next_token_ids
        event = torch.cuda.Event()
        # Recorded AFTER the async D2H enqueued by post_decode_launch, so
        # event.query()==True means the host buffer is ready (design.md §3).
        event.record()
        return _PendingStep(
            event=event,
            host_buf=host_buf,
            scheduler_output=scheduler_output,
            forward_batch=forward_batch,
            schedule_batch=schedule_batch,
            model_worker_batch=model_worker_batch,
            batch_result=batch_result,
            n_real=len(scheduler_output.requests),
        )

    def execute_resolve(
        self, pending: "_PendingStep | None"
    ) -> ModelRunnerOutput | None:
        """Consume a launched decode step: wait on its event (non-blocking
        ``query()``, else ``synchronize()``), read the pinned host buffer and
        run the per-request collect loop (``post_decode_resolve``), then
        finalize sampling/output. Returns that step's ``ModelRunnerOutput``,
        or None if ``pending`` is None (first iteration / after a drain).
        """
        if pending is None:
            return None
        if pending.event.query():
            self._async_query_hit += 1
        else:
            pending.event.synchronize()
            self._async_query_miss += 1
        skip_rids = {
            req.request_id
            for req in pending.scheduler_output.requests
            if req.data.req.finished()
        }
        self.post_decode_resolve(
            pending.host_buf,
            pending.batch_result,
            pending.forward_batch,
            pending.schedule_batch,
            pending.scheduler_output.requests,
        )
        return self._finalize(
            pending.batch_result,
            pending.forward_batch,
            pending.schedule_batch,
            pending.model_worker_batch,
            pending.scheduler_output,
            set_output_ids=False,
            skip_rids=skip_rids,
        )

    def _build_forward_batch(self, scheduler_output: Any):
        """Build the ForwardBatch + capture-hidden mode. Returns
        ``(forward_batch, schedule_batch, model_worker_batch, is_prefill)``, or
        None when there is no batch to run."""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return None

        model_worker_batch = schedule_batch.get_model_worker_batch()
        is_prefill = bool(schedule_batch.forward_mode.is_extend())

        capture_hidden_mode = (
            self.requested_capture_hidden_mode_prefill(
                schedule_batch, scheduler_output.requests
            )
            if is_prefill
            else self.requested_capture_hidden_mode_decode(
                schedule_batch, scheduler_output.requests
            )
        )
        if capture_hidden_mode is not None:
            model_worker_batch.capture_hidden_mode = capture_hidden_mode
        elif self.output_processor._capture_hidden:
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.tp_worker.model_runner
        )
        return forward_batch, schedule_batch, model_worker_batch, is_prefill

    def _prepare_and_forward(
        self,
        forward_batch,
        schedule_batch,
        requests,
        is_prefill,
        *,
        is_lookahead: bool = False,
    ):
        """Prepare hook → standard forward (if not custom) → sample-before-post
        block. Returns ``batch_result``."""
        if is_prefill:
            self.before_prefill(forward_batch, schedule_batch, requests)
            batch_result = self.custom_prefill_forward(
                forward_batch, schedule_batch, requests
            )
        else:
            self.before_decode(
                forward_batch,
                schedule_batch,
                requests,
                is_lookahead=is_lookahead,
            )
            batch_result = self.custom_decode_forward(
                forward_batch, schedule_batch, requests
            )
        if batch_result is None:
            batch_result = self.tp_worker.forward_batch_generation(forward_batch)

        if (
            not schedule_batch.is_prefill_only
            and batch_result.next_token_ids is None
            and (
                self.sample_before_post_prefill(forward_batch, schedule_batch, requests)
                if is_prefill
                else self.sample_before_post_decode(
                    forward_batch, schedule_batch, requests
                )
            )
        ):
            batch_result.next_token_ids = self._sample_next_token_ids(
                batch_result.logits_output, forward_batch, schedule_batch, requests
            )
            schedule_batch.output_ids = batch_result.next_token_ids
        return batch_result

    def _finalize(
        self,
        batch_result,
        forward_batch,
        schedule_batch,
        model_worker_batch,
        scheduler_output,
        set_output_ids: bool = True,
        skip_rids: set[str] | None = None,
    ) -> ModelRunnerOutput:
        """Final sampling (if still needed) + output extraction + per-request
        bookkeeping. Shared tail of both the sync and async paths.

        ``set_output_ids`` publishes this step's tokens onto
        ``schedule_batch.output_ids`` so the NEXT step's ``prepare_for_decode``
        can build its input_ids. The synchronous path needs this. The async
        RESOLVE path must NOT do it: under launch-first the resolve runs one
        step behind, and ``schedule_batch`` here is the *live* running batch
        whose output_ids was already published by the (current) launch at the
        right length — re-stamping the lagged step's next_token_ids would leave
        a stale-length output_ids on the running batch, which the next
        prepare_for_decode turns into an input_ids that mismatches seq_lens once
        a request finishes mid-batch (the bs>1 replay size mismatch)."""
        if schedule_batch.is_prefill_only:
            if batch_result.next_token_ids is None:
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
        elif batch_result.next_token_ids is None:
            batch_result.next_token_ids = self._sample_next_token_ids(
                batch_result.logits_output,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
        if set_output_ids:
            schedule_batch.output_ids = batch_result.next_token_ids

        outputs = self.output_processor.process(batch_result, scheduler_output)
        self.post_process_outputs(batch_result, scheduler_output, outputs)
        skip_rids = skip_rids or set()
        for sched_req in scheduler_output.requests:
            if sched_req.request_id in skip_rids:
                continue
            data = sched_req.data
            data.generation_steps = int(data.generation_steps) + 1
            req_output = outputs[sched_req.request_id]
            extra = req_output.extra
            if isinstance(extra, dict) and extra:
                data.extra_model_outputs.update(extra)
        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses
    # ------------------------------------------------------------------

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Mutate state before the standard or custom prefill forward."""

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Mutate state before the standard or custom decode forward."""
        del is_lookahead

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Run a model-specific prefill forward.

        Return a batch result when the subclass owns the forward path for this
        batch, or None to use the standard tp_worker forward path.
        """
        return None

    def custom_decode_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> Any | None:
        """Run a model-specific decode forward.

        Return a batch result when the subclass owns the forward path for this
        batch, or None to use the standard tp_worker forward path.
        """
        return None

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after prefill forward."""

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        """Called after decode forward."""

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Called after output tokens are materialized into RequestOutput."""

    def post_decode_launch(
        self, result: Any, forward_batch: Any, requests: list
    ) -> Any:
        """Async-decode GPU half of ``post_decode``: scatter GPU state, pack
        the collect tensors, enqueue a non-blocking D2H into a pinned host
        buffer (obtained via ``self._next_host_staging``), and return that
        buffer. The caller records a CUDA event immediately after.

        Default raises: a model must implement this together with
        ``post_decode_resolve`` to be async-decode-safe. The synchronous
        ``post_decode`` reads live GPU buffers that the next launch would
        overwrite, so it cannot simply be deferred (design.md §1.6).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support async decode: implement "
            "post_decode_launch / post_decode_resolve"
        )

    def post_decode_resolve(
        self,
        host_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Async-decode host half of ``post_decode``: read the pinned
        ``host_buf`` (populated by the launch-time D2H) and run the
        per-request collect loop, setting ``result.next_token_ids``.
        Default raises (see ``post_decode_launch``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support async decode: implement "
            "post_decode_launch / post_decode_resolve"
        )

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return False

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        return False

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any | None:
        return None

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any | None:
        return None

    # ------------------------------------------------------------------
    # Shared logit processing
    # ------------------------------------------------------------------

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._apply_repetition_penalty(logits_output, requests)
        self._apply_codec_suppress_tokens(logits_output, requests)
        return self.tp_worker.model_runner.sample(logits_output, forward_batch)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        vocab = logits.shape[1]
        device = logits.device
        rep_rows: list[int] = []
        rep_toks: list[int] = []
        rep_penalties: list[float] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            penalty = req.sampling_params.repetition_penalty
            if penalty == 1.0:
                continue
            output_ids = req.output_ids
            if not output_ids:
                continue
            unique = {int(t) for t in output_ids if 0 <= int(t) < vocab}
            if not unique:
                continue
            rep_rows.extend([row_idx] * len(unique))
            rep_toks.extend(unique)
            rep_penalties.extend([float(penalty)] * len(unique))
        if rep_rows:
            orig_dtype = logits.dtype
            rows_t = torch.tensor(rep_rows, dtype=torch.long, device=device)
            toks_t = torch.tensor(rep_toks, dtype=torch.long, device=device)
            pens_t = torch.tensor(rep_penalties, dtype=torch.float32, device=device)
            scores = logits[rows_t, toks_t].to(torch.float32)
            scores = torch.where(scores > 0, scores / pens_t, scores * pens_t)
            logits[rows_t, toks_t] = scores.to(orig_dtype)

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        vocab = logits.shape[1]
        device = logits.device
        sup_rows: list[int] = []
        sup_toks: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            suppress_tokens = data.suppress_tokens
            if not suppress_tokens:
                req = data.req
                suppress_tokens = getattr(req, "_codec_suppress_tokens", None)
            if not suppress_tokens:
                continue
            for token_id in suppress_tokens:
                tok = int(token_id)
                if 0 <= tok < vocab:
                    sup_rows.append(row_idx)
                    sup_toks.append(tok)
        if sup_rows:
            logits[
                torch.tensor(sup_rows, dtype=torch.long, device=device),
                torch.tensor(sup_toks, dtype=torch.long, device=device),
            ] = float("-inf")
