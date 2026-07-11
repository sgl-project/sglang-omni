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

from sglang_omni.sampling.seed import (
    SAMPLING_SEED_MASK,
    derive_sampling_seed,
    resolve_row_seed,
)
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerRequest,
    sampled_logprobs_to_list,
)

logger = logging.getLogger(__name__)


def _current_sglang_sampling_backend() -> str | None:
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args().sampling_backend
    except ValueError:
        return None


def _rank_shared_unseeded_sampling_seed(request: SchedulerRequest, row_idx: int) -> int:
    request_id = request.request_id or f"row-{row_idx}"
    return derive_sampling_seed("sglang-omni-unseeded-row", request_id)


@dataclass
class _PendingStep:
    """One decode step launched on the GPU but not yet consumed on the host.

    Async-decode (one-step lookahead) bookkeeping: a launched step has its
    forward + on-GPU sample + collect enqueued and ``event`` recorded right
    after, so ``event.query()`` true means the launched step's GPU work is
    published. ``launch_buf`` is whatever ``post_decode_launch`` returns for
    resolve to consume: a device-side correctness snapshot of the published ids
    (MOSS-TTS-Local, no host copy), or a pinned host staging buffer an async host
    copy filled (Higgs); only the latter provides host-D2H overlap.
    ``execute_resolve`` later waits on ``event`` and reads ``launch_buf``.

    Invariant: at most one ``_PendingStep`` is live at a time (see
    ``ModelRunner._pending``). When the launch uses host staging it is pinned
    and ping-ponged between two buffers so resolve(N) reads one while
    launch(N+1) writes the other (design.md section 1.4).
    """

    event: Any  # torch.cuda.Event, recorded after post_decode_launch publishes
    launch_buf: Any  # post_decode_launch return: device snapshot or host staging
    scheduler_output: Any  # this step's SchedulerOutput (routing + output proc)
    forward_batch: Any  # for resolve-time finalize sampling
    schedule_batch: Any  # to set .output_ids during resolve
    model_worker_batch: Any  # for the prefill-only finalize branch (unused in decode)
    batch_result: Any  # carries logits_output (device of next_token_ids)


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
        # Observability: how often resolve found the launched step's event
        # already done (no blocking) vs had to block on synchronize(). This
        # counts whether the launched step's GPU work was published in time; it
        # does NOT measure host-D2H overlap (only host-staging runners like Higgs
        # overlap a host copy; the device-snapshot path does not).
        self._async_query_hit: int = 0
        self._async_query_miss: int = 0

    def _next_host_staging(
        self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a pinned host staging buffer covering ``shape``/``dtype``,
        ping-ponging between two buffers on each call. Runners that stage the
        collect to host use this: Higgs passes its fixed CG staging shape, the
        base plain-LM launch passes the step's ``(batch_size,)`` ids shape.
        Device-snapshot runners (MOSS-TTS-Local) never do.

        Two buffers are required: resolve(N) reads one on the host while
        launch(N+1)'s async host copy writes the other. That CPU-read vs
        GPU-write overlap is not protected by single-stream ordering.
        Buffers are allocated lazily on first use and replaced when the
        requested dim 0 outgrows them (or trailing dims / dtype change); a
        resolve still holding the previous buffer keeps it alive, so
        replacement cannot alias an in-flight snapshot.
        """
        bufs = self._host_staging_buffers
        if (
            not bufs
            or bufs[0].dtype != dtype
            or bufs[0].shape[1:] != tuple(shape[1:])
            or bufs[0].shape[0] < shape[0]
        ):
            self._host_staging_buffers = bufs = [
                torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)
                for _ in range(2)
            ]
            self._staging_slot = 0
        buf = bufs[self._staging_slot]
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
        """Enqueue a decode step's forward + on-GPU sample, call
        ``post_decode_launch`` to publish a model-specific resolve payload
        (returned as ``launch_buf``), and record a CUDA event right after
        publication. Does NOT wait on the GPU. Decode batches only. ``launch_buf``
        is a device-side correctness snapshot (MOSS-TTS-Local) or pinned host
        staging (Higgs); only the latter overlaps a host copy with the next
        forward, and ``event.query()`` proves the launched step's GPU work is
        done, not that any host overlap happened.

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
        launch_buf = self.post_decode_launch(
            batch_result, forward_batch, scheduler_output.requests
        )
        # Publish this step's output token ids now (post_decode_launch set them
        # from GPU state without a host sync) so the NEXT decode step's
        # get_next_batch_to_run / prepare_for_decode can build its input_ids;
        # under lookahead the host collect (resolve) lags by one step.
        if batch_result.next_token_ids is not None:
            schedule_batch.output_ids = batch_result.next_token_ids
        event = torch.cuda.Event()
        # Recorded after post_decode_launch publishes this step, so
        # event.query()==True means the launched step's GPU work is done and
        # launch_buf is ready (design.md section 3).
        event.record()
        return _PendingStep(
            event=event,
            launch_buf=launch_buf,
            scheduler_output=scheduler_output,
            forward_batch=forward_batch,
            schedule_batch=schedule_batch,
            model_worker_batch=model_worker_batch,
            batch_result=batch_result,
        )

    def execute_resolve(
        self, pending: "_PendingStep | None"
    ) -> ModelRunnerOutput | None:
        """Consume a launched decode step: wait on its event (non-blocking
        ``query()``, else ``synchronize()``), read its ``launch_buf`` (a device
        snapshot or pinned host staging) and run the per-request collect loop
        (``post_decode_resolve``), then
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
        # Skip reqs finished or retracted in a prior (lagged) step so _finalize
        # neither re-emits nor re-frees their KV (mirrors _resolve_and_process).
        skip_rids = {
            req.request_id
            for req in pending.scheduler_output.requests
            if req.data.req.finished() or self._req_is_retracted(req.data.req)
        }
        self.post_decode_resolve(
            pending.launch_buf,
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

    def finalize_skip_rids(self, scheduler_output) -> set[str]:
        """Request ids whose ``generation_steps`` must NOT advance this step.

        Default empty. A model overrides this when a batch contains rows that
        are sampled but must not count as a generated step — e.g. non-final
        chunked-prefill rows, whose spurious step would shift the final chunk's
        sampling position off the no-chunk path. Unioned into ``skip_rids``
        inside ``_finalize`` so it covers the sync, async-resolve, and
        prefill-only paths alike. Additive and behaviour-neutral for any model
        that does not override it.
        """
        return set()

    def on_generation_step_advanced(
        self, sched_req: Any, generation_steps: int
    ) -> None:
        """Hook after ``generation_steps`` is committed on request data."""
        return None

    def on_generation_steps_advanced(
        self, advanced_steps: list[tuple[Any, int]], forward_batch: Any
    ) -> None:
        """Batch hook after ``generation_steps`` are committed on request data."""
        del forward_batch
        for sched_req, generation_steps in advanced_steps:
            self.on_generation_step_advanced(sched_req, generation_steps)

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
        skip_rids = (skip_rids or set()) | self.finalize_skip_rids(scheduler_output)
        advanced_steps = []
        for sched_req in scheduler_output.requests:
            if sched_req.request_id in skip_rids:
                continue
            data = sched_req.data
            data.generation_steps = int(data.generation_steps) + 1
            advanced_steps.append((sched_req, data.generation_steps))
            req_output = outputs[sched_req.request_id]
            extra = req_output.extra
            if isinstance(extra, dict) and extra:
                data.extra_model_outputs.update(extra)
        if advanced_steps:
            self.on_generation_steps_advanced(advanced_steps, forward_batch)
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

    def lookahead_eligible(self, batch: Any) -> bool:
        """Whether this batch may use one-step async-decode lookahead.

        The default launch samples one step before resolve appends the previous
        token to ``req.output_ids`` / the sglang penalizer state, so any sampling
        term scored by output history (repetition / frequency / presence penalty,
        ``min_new_tokens``) would read a one-token-stale view and diverge from
        sync — gate those batches to sync. Over-gating only costs throughput,
        under-gating is a silent divergence. Runners override for other fallbacks.
        """

        def _history_free(sp: Any) -> bool:
            return (
                sp.repetition_penalty == 1.0
                and sp.frequency_penalty == 0.0
                and sp.presence_penalty == 0.0
                and sp.min_new_tokens == 0
            )

        return all(_history_free(req.sampling_params) for req in batch.reqs)

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
        """Async-decode GPU half of ``post_decode``: sample now, publish
        ``result.next_token_ids``, and return the resolve payload (``launch_buf``);
        the caller records a CUDA event right after.

        Default (plain-LM): sample via ``_sample_next_token_ids`` and snapshot the
        ids into a pinned ping-pong host buffer (required — the sampler output can
        alias step-reused buffers the next launch overwrites). Codec runners whose
        collect is more than next_token_ids override this with ``post_decode_resolve``.
        """
        if not requests:
            return None
        if result.next_token_ids is None:
            result.next_token_ids = self._sample_next_token_ids(
                result.logits_output, forward_batch, None, requests
            )
        n = len(requests)
        ids = result.next_token_ids
        host_buf = self._next_host_staging((n,), ids.dtype)
        host_buf[:n].copy_(ids[:n], non_blocking=True)
        return host_buf

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Async-decode host half of ``post_decode``: read ``launch_buf`` and set
        ``result.next_token_ids``. Default (plain-LM): point it at the pinned host
        snapshot — the caller already waited on the launch event, so ``_finalize``
        reads it without a GPU sync.
        """
        del forward_batch, schedule_batch
        if launch_buf is None or not requests:
            return
        result.next_token_ids = launch_buf[: len(requests)]

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
        self._install_sampling_seeds(forward_batch, requests)
        wants_rollout_logprob = any(sr.data.return_logprob for sr in requests)
        if wants_rollout_logprob:
            self._enable_sampler_logprobs(forward_batch, len(requests))
        next_token_ids = self.tp_worker.model_runner.sample(
            logits_output, forward_batch
        )
        if wants_rollout_logprob:
            try:
                next_token_logprobs = logits_output.next_token_logprobs
            except AttributeError as exc:
                raise RuntimeError(
                    "Sampler did not populate next_token_logprobs when "
                    "return_logprob is enabled"
                ) from exc
            if next_token_logprobs is None:
                raise RuntimeError(
                    "Sampler did not populate next_token_logprobs when "
                    "return_logprob is enabled"
                )
            self._record_rollout_logprobs(
                next_token_logprobs,
                next_token_ids,
                requests,
            )
        return next_token_ids

    def _install_sampling_seeds(self, forward_batch: Any, requests: list) -> None:
        """Install per-row ``seed``s onto ``sampling_info`` so SGLang routes to
        ``multinomial_with_seed``. No-op when no request set a seed, or when a
        subclass already installed its own (e.g. Qwen3-TTS).

        Runs once per decode step. User-provided seeds are resolved once and
        cached back onto ``sampling_params.sampling_seed``. In a mixed
        seeded/unseeded batch the SGLang sampler is batch-wide, so unseeded rows
        receive a request-id-derived fallback seed instead of a rank-local random
        seed; this keeps TP ranks in sync without mutating the public request seed.
        """
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_seed is not None:
            self._validate_seeded_sampling_supported(sampling_info)
            return
        sampling_params = [sr.data.req.sampling_params for sr in requests]
        if all(sp.sampling_seed is None for sp in sampling_params):
            return
        self._validate_seeded_sampling_supported(sampling_info)
        row_seeds: list[int] = []
        for row_idx, (sp, request) in enumerate(zip(sampling_params, requests)):
            seed = sp.sampling_seed
            if seed is None:
                seed = _rank_shared_unseeded_sampling_seed(request, row_idx)
            elif not (0 <= seed <= SAMPLING_SEED_MASK):
                seed = resolve_row_seed(seed)  # mask and cache user seed
                sp.sampling_seed = seed
            row_seeds.append(seed)
        sampling_info.sampling_seed = torch.tensor(
            row_seeds, dtype=torch.long, device=sampling_info.device
        )

    @staticmethod
    def _validate_seeded_sampling_supported(sampling_info: Any) -> None:
        if getattr(sampling_info, "need_min_p_sampling", False):
            raise ValueError(
                "SGLang seeded sampling does not support min_p yet; set min_p=0 "
                "or omit request seed"
            )
        need_top_p_sampling = getattr(sampling_info, "need_top_p_sampling", False)
        need_top_k_sampling = getattr(sampling_info, "need_top_k_sampling", False)
        if not (need_top_p_sampling or need_top_k_sampling):
            return
        if _current_sglang_sampling_backend() == "flashinfer":
            raise ValueError(
                "SGLang flashinfer sampling backend does not support request seed "
                "with top_p/top_k filtering; configure sampling_backend='pytorch' "
                "or avoid top_p/top_k with seed"
            )

    @staticmethod
    def _enable_sampler_logprobs(forward_batch: Any, batch_size: int) -> None:
        forward_batch.return_logprob = True
        try:
            top_logprobs_nums = forward_batch.top_logprobs_nums
        except AttributeError:
            top_logprobs_nums = None
        if top_logprobs_nums is None:
            forward_batch.top_logprobs_nums = [0] * batch_size
        try:
            token_ids_logprobs = forward_batch.token_ids_logprobs
        except AttributeError:
            token_ids_logprobs = None
        if token_ids_logprobs is None:
            forward_batch.token_ids_logprobs = [None] * batch_size

    def _record_rollout_logprobs(
        self, next_token_logprobs, next_token_ids, requests
    ) -> None:
        """Append each rollout request's sampled-token logprob (one per step)."""
        logprobs = sampled_logprobs_to_list(next_token_logprobs)
        if logprobs is None:
            try:
                shape = next_token_logprobs.shape
            except AttributeError:
                shape = None
            raise RuntimeError(
                "Failed to convert sampler next_token_logprobs "
                f"type={type(next_token_logprobs).__name__} shape={shape}"
            )
        if next_token_ids is None:
            raise RuntimeError("Sampler did not return next_token_ids")
        try:
            token_id_values = next_token_ids.tolist()
        except AttributeError:
            token_id_values = next_token_ids
        token_ids = [int(t) for t in token_id_values]
        if len(logprobs) != len(token_ids) or len(logprobs) != len(requests):
            raise RuntimeError(
                "rollout logprob batch-size mismatch: "
                f"logprobs={len(logprobs)} token_ids={len(token_ids)} "
                f"requests={len(requests)}"
            )
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            if data.return_logprob:
                data.output_token_logprobs.append(
                    [logprobs[row_idx], token_ids[row_idx]]
                )

    @staticmethod
    def _req_is_retracted(req: Any) -> bool:
        try:
            return bool(req.is_retracted)
        except AttributeError:
            return False

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
                try:
                    suppress_tokens = req._codec_suppress_tokens
                except AttributeError:
                    suppress_tokens = None
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
