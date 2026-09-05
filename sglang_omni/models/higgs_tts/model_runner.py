# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS model runner — phase-aware AR base-runner subclass.

Decode-mode hooks gather sampler-pool state into ``_cg_active_*`` shadow
buffers before the captured forward and scatter results back after, so
the graph itself only ever does ``_cg_active_*[:bs]`` slicing — no
``pool[row_indices]`` gather/scatter under capture (capture-time
``row_indices`` are all-zero placeholders → duplicate-index UB).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.higgs_tts.model import _TRAJECTORY_DELTA_LAMBDA
from sglang_omni.models.higgs_tts.sampler import K_MAX, selected_token_logprobs
from sglang_omni.models.higgs_tts.text_tokenizer import AUDIO_PLACEHOLDER_ID
from sglang_omni.models.higgs_tts.utils import EOC_ID
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_HIGGS_STREAM_STRIDE,
    HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA,
    HIGGS_STREAM_STRIDE_METADATA,
)
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM

logger = logging.getLogger(__name__)


def _syncfree_launch_enabled() -> bool:
    """Parse SGLANG_OMNI_SYNCFREE_LAUNCH (default ON; 0/false/no disable)."""
    value = os.environ.get("SGLANG_OMNI_SYNCFREE_LAUNCH", "1").lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise ValueError(
        f'"{value}" is not a valid SGLANG_OMNI_SYNCFREE_LAUNCH boolean value'
    )


class HiggsTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`HiggsTTSModel`."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"
        # Ping-pong pinned host buffers for the async-decode rollout-logprob D2H.
        self._logprob_host_buffers: list[torch.Tensor] | None = None
        self._logprob_slot = 0
        # Note (Yueying Li): sync-free decode launch (see _populate_cg_buffers): skip the
        # composition-invariant sampling-param extraction/upload when the batch
        # composition is unchanged since the previous decode step.
        self._syncfree_launch: bool = _syncfree_launch_enabled()
        self._cg_launch_key: tuple | None = None
        # True iff the model's ``_cg_fusion_group``/``_cg_fusion_weight``
        # buffers currently hold any non-default (non-singleton) values from a
        # past decode step. Lets ``_populate_fusion_buffers`` skip its dict
        # scan + list build + host->device copy entirely once traffic goes
        # back to all-singleton, instead of redoing (harmless but wasteful)
        # identity writes forever after the first fusion request of a
        # server's lifetime.
        self._fusion_buffers_dirty = False

    def _next_logprob_host_staging(self, device_buf: torch.Tensor) -> torch.Tensor:
        return self._pinned_pingpong(
            "_logprob_host_buffers",
            "_logprob_slot",
            device_buf.shape,
            device_buf.dtype,
            realloc_on_grow=True,
        )

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        """Flush a partial streaming-code window before terminal output."""
        self._flush_code_chunks(request_id, req_data, force=True)

    def lookahead_eligible(self, batch: Any) -> bool:
        """Route to sync whenever any voice-fusion request is registered.

        ``_populate_fusion_buffers``'s split-group detection sets
        ``FINISH_ABORT`` during the populate/launch phase — i.e. at launch of
        step S+1, before step S has even been resolved. Under async-decode
        lookahead the scheduler launches S+1 before resolving S, and upstream
        ``process_batch_result_decode`` skips any row that's already
        ``req.finished()`` by the time it runs (treating it as a stale
        one-step overrun): the just-aborted row's KV is never released and it
        never reaches ``stream_output``, so
        ``OmniScheduler._cascade_abort_split_fusion_group`` never fires — the
        exact zombie-sibling failure mode that mechanism exists to prevent.
        Sync-only for fusion traffic sidesteps the whole class of problem:
        this only disables the async optimization while fusion requests are
        live, not correctness.
        """
        del batch
        return not self.model.has_any_fusion()

    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        for req in requests:
            self.model.set_request_seed(
                req.request_id, req.data.req.sampling_params.sampling_seed
            )
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self._build_prefill_input_embeds(forward_batch, requests),
            ),
        )

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._collect_step_outputs(result, requests, forward_batch)

    def before_decode(
        self,
        forward_batch,
        schedule_batch,
        requests,
        *,
        is_lookahead: bool = False,
    ):
        del schedule_batch
        self._populate_cg_buffers(forward_batch, requests, is_lookahead=is_lookahead)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._collect_step_outputs_cg(result, forward_batch, requests)

    def post_decode_launch(self, result, forward_batch, requests):
        """Async-decode GPU half: scatter + pack (GPU->GPU), then a
        non-blocking copy of the staging snapshot into a pinned host staging buffer.
        Returns the buffer; the base runner records the event right after, so
        ``event.query()`` then means "this snapshot is on the host".
        """
        if len(requests) == 0:
            return None
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )
        staging = self._decode_pack_gpu(n_real)
        collect_staging = self.model._cg_collect_staging
        host_buf = self._next_host_staging(collect_staging.shape, collect_staging.dtype)
        host_buf[:n_real].copy_(staging[:n_real], non_blocking=True)
        logprob_host = None
        if self._should_capture_rollout_logprobs(requests):
            logprobs_BN = self._decode_step_logprobs(result, n_real)
            logprob_host = self._next_logprob_host_staging(logprobs_BN)
            logprob_host[:n_real].copy_(logprobs_BN[:n_real], non_blocking=True)
        # Set next_token_ids (cb0) from GPU state now, with NO host sync, so the
        # AR input chain (next step's input_ids = this step's output_ids) is
        # available at launch — the host collect (post_decode_resolve) lags by
        # one step under lookahead. For Higgs the decode input_ids is masked by
        # _decode_step_embeds_cg (rows with codes use _cg_active_last_codes), so
        # this only feeds the upstream bookkeeping. clamp>=0 keeps STOP_CODE(-1)
        # rows in embed_tokens range; the host collect later overwrites with the
        # skip-aware cb0 for output reporting.
        result.next_token_ids = self.next_input_token_ids(
            result, forward_batch, requests
        )
        return host_buf, logprob_host

    def next_input_token_ids(self, result, forward_batch, requests):
        """Return Higgs' device-native cb0 rail for the next decode step.

        Reporting tokens are reconstructed from the combined CPU collect, but
        the autoregressive input already exists in the CUDA-graph output. Keep
        those two contracts separate so sync execution does not perform an
        H2D followed by two D2H copies for the same eight integers.
        """
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_decode():
            return super().next_input_token_ids(result, forward_batch, requests)
        n_real = len(requests)
        if n_real == 0:
            return None
        return self.model._cg_codes_BN[:n_real, 0].clamp_min(0).to(torch.long)

    def post_decode_resolve(
        self, host_buf, result, forward_batch, schedule_batch, requests
    ):
        """Async-decode host half: read the already-copied pinned snapshot and
        run the per-request collect loop. Mirrors the tail of
        ``_collect_step_outputs_cg`` (shares ``_decode_collect_host``).
        """
        del forward_batch, schedule_batch
        if len(requests) == 0:
            return
        n_real = len(requests)
        host_buf, logprob_host = host_buf
        logprobs_cpu = None if logprob_host is None else logprob_host[:n_real]
        # fusion_ell_cpu=None: fusion traffic always forces lookahead_eligible
        # to False (see that method's docstring), so this async-decode path
        # never actually carries fusion rows in practice — nothing to update.
        self._decode_collect_host(
            host_buf[:n_real],
            logprobs_cpu,
            result,
            requests,
            next_token_device=None,
            fusion_ell_cpu=None,
        )

    def _populate_cg_buffers(
        self, forward_batch, requests, *, is_lookahead: bool = False
    ) -> None:
        """Fill the model's CG buffers for one decode step.

        Padding rows (``batch_size > len(requests)``) point at the
        reserved padding row, which is reset every step so it can't
        leak state into real rows.
        """
        model = self.model
        bs = int(forward_batch.batch_size)
        n_real = len(requests)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )

        model._sampler_pool.reset_row(model._padding_row)

        # Note (Yueying Li): a rid absent from the pool map is a fresh acquisition: the rid may be
        # a client-supplied reuse of a finished request's id, and LIFO row
        # recycling makes (rid, row, bs) collide with the stale key — force a
        # rebuild so the new request cannot inherit cached params/redirects.
        if any(req.request_id not in model._rid_to_row for req in requests):
            self._cg_launch_key = None

        rows_py: list[int] = [model.acquire_row(req.request_id) for req in requests]
        # Note (Yueying Li): sync-free launch cache: temperature/top_p/top_k/row-index are
        # per-request constants, so the four H2D uploads below need to
        # rerun only when the batch composition changes (the extract itself is
        # host-side in this adaptation). The key is
        # order-sensitive and includes the pool rows (recomputed from
        # acquire_row every step), so admission / finish / retract / abort /
        # reorder / row re-allocation all miss; bs covers the padding-row
        # count. On a hit the buffers still hold last step's values: the only
        # other in-place writer is the lookahead done-row guard below, whose
        # padding redirect must persist exactly as long as the finished
        # request stays in the batch — i.e. until the key changes.
        launch_key = (tuple(req.request_id for req in requests), tuple(rows_py), bs)
        if not (self._syncfree_launch and launch_key == self._cg_launch_key):
            # Note (Yueying Li): a failed rebuild must not leave a valid key
            self._cg_launch_key = None
            rows_full = rows_py + [model._padding_row] * (bs - n_real)
            model._cg_row_indices[:bs] = torch.tensor(
                rows_full, dtype=torch.long, device=model._cg_row_indices.device
            )

            temps, top_ps, top_ks = self._extract_decode_sampling_params(requests)
            temps.extend([1.0] * (bs - n_real))
            top_ps.extend([1.0] * (bs - n_real))
            model._cg_temperature[:bs] = torch.tensor(
                temps, dtype=torch.float32, device=model._cg_temperature.device
            )
            model._cg_top_p[:bs] = torch.tensor(
                top_ps, dtype=torch.float32, device=model._cg_top_p.device
            )

            top_k_vals = [
                (tk if (tk is not None and tk > 0) else K_MAX) for tk in top_ks
            ]
            top_k_vals.extend([K_MAX] * (bs - n_real))
            model._cg_top_k_buf[:bs] = torch.tensor(
                top_k_vals, dtype=torch.long, device=model._cg_top_k_buf.device
            )
            self._cg_launch_key = launch_key

        if self._async_enabled and is_lookahead and n_real > 0:
            # Async-lookahead overrun guard (GPU-side, no host sync): a request
            # that finished via EOC at the prior step is still in this batch
            # with pool.generation_done=True. Running the normal decode forward
            # for such a done row trips a device-side gather assert, so route it
            # to the reset padding row — its overrun output is discarded by the
            # collect's finished()/was_done skip anyway. Length-finish rows have
            # generation_done=False and are untouched.
            #
            # Only the lookahead launch path can carry such an overrun (the
            # 1-wasted-step lag). On a fast-path (sync) decode step finished reqs
            # are filtered out before the step, so no generation_done row is ever
            # present and this gather+torch.where would be pure wasted GPU work.
            rows_t_real = model._cg_row_indices[:n_real]
            done = model._sampler_pool.generation_done[rows_t_real]
            model._cg_row_indices[:n_real] = torch.where(
                done, torch.full_like(rows_t_real, model._padding_row), rows_t_real
            )

        rows_t = model._cg_row_indices[:bs]
        pool = model._sampler_pool
        model._cg_active_delay_count[:bs] = pool.delay_count[rows_t]
        model._cg_active_eoc_countdown[:bs] = pool.eoc_countdown[rows_t]
        model._cg_active_generation_done[:bs] = pool.generation_done[rows_t]
        model._cg_active_last_codes[:bs] = pool.last_codes[rows_t]
        model._cg_active_seeds[:bs] = pool.seeds[rows_t]
        model._cg_active_step_count[:bs] = pool.step_count[rows_t]

        self._populate_fusion_buffers(requests, bs, n_real)

    def _populate_fusion_buffers(self, requests, bs: int, n_real: int) -> None:
        """Fill the model's voice-fusion CG buffers for one decode step.

        Maps each real row's ``fusion_group_id`` to a *batch-local* group index
        (the row index of the group's first member, always in ``[0, n_real)``)
        and its blend weight; rows with no fusion id — and all padding rows —
        become singleton groups (group = own slot, weight = 1.0), which makes
        :func:`fusion.fuse_group_logits` an identity no-op. Resetting every slot
        each step prevents a stale group id from a prior (larger) batch leaking
        an out-of-range scatter index into the captured graph.

        A fusion group split across this decode step (e.g. a KV-pressure
        retract dropped some members, or they just haven't all reached decode
        yet) is handled by isolating just that group's *present* rows: they're
        demoted to singletons (skip the blend entirely, rather than average an
        incomplete — and therefore wrong — set of distributions) and their
        requests are marked aborted so only they fail, not the rest of this
        decode batch. A group ``HiggsTTSModel._batch_local_fusion`` (the
        prefill-side counterpart) previously caught split and poisoned is
        aborted here too even if presence now looks complete — a prefill-time
        split already sampled an unblended frame per isolated row before
        anyone could stop it, so "looks complete now" does not mean "safe to
        resume fusing" (see ``fusion.py::FusionRegistry._poisoned``).
        """
        model = self.model
        if not model.has_any_fusion() and not self._fusion_buffers_dirty:
            # Hot path: no fusion request has ever been registered (or the
            # last one that was has fully cleared and its dirty writes were
            # already reset to identity below). Skip the dict scan, list
            # build, and host->device copies below entirely — the buffers
            # already hold their singleton-default values from either
            # __init__ or the previous call's reset.
            return

        rids = [req.request_id for req in requests[:n_real]]
        # One locked snapshot of membership for the whole step (Bug D): the
        # decode thread sees a consistent view even if a register/clear races.
        group_of, weight_of, delta_of = model.fusion_membership_snapshot(rids)

        group = list(range(bs))  # default: every slot its own singleton group
        weight = [1.0] * bs
        seen: dict[str, int] = {}
        present: dict[str, int] = {}
        rows_by_group: dict[str, list[int]] = {}
        for b, rid in enumerate(rids):
            gid = group_of.get(rid)
            if gid is None:
                continue
            if gid not in seen:
                seen[gid] = b  # first member's row anchors the batch-local id
            group[b] = seen[gid]
            # Effective weight = nominal weight * exp(trajectory-feedback
            # tilt) -- see FusionRegistry's delta docstring. The blend this
            # step actually uses this value; _decode_collect_host's delta
            # UPDATE math re-derives the group's target from the nominal
            # weight_of instead, not this already-tilted value.
            weight[b] = weight_of.get(rid, 1.0) * math.exp(delta_of.get(rid, 0.0))
            present[gid] = present.get(gid, 0) + 1
            rows_by_group.setdefault(gid, []).append(b)

        for gid, n_present in present.items():
            expected = model.expected_fusion_group_size(gid)
            if not expected:
                continue
            poisoned = model.is_fusion_group_poisoned(gid)
            if n_present == expected and not poisoned:
                continue
            # Split group, or a group that "healed" (looks complete now) but
            # was poisoned by an earlier prefill-time split: isolate the
            # blast radius to this group's present rows only. Demote them to
            # independent singletons for this step (no blend — averaging an
            # incomplete or already-corrupted group would silently produce
            # wrong audio) and abort their requests so the rest of the batch
            # decodes unaffected.
            if poisoned and n_present == expected:
                logger.warning(
                    "voice-fusion group %r reached decode with all %d "
                    "sibling rows present, but was poisoned by an earlier "
                    "prefill-time split — it already sampled an unblended "
                    "frame before that split was caught, so its KV contexts "
                    "are permanently desynced. Aborting rather than "
                    "resuming fused decode on a group that can never be "
                    "correct again.",
                    gid,
                    expected,
                )
            else:
                logger.warning(
                    "voice-fusion group %r split across a decode step: "
                    "%d/%d sibling rows present in the batch. The serving "
                    "engine scheduled the group's rows apart (likely a "
                    "KV-pressure retract); aborting this group's requests. "
                    "Raise max_running_requests or reduce concurrency so "
                    "fusion siblings stay co-batched.",
                    gid,
                    n_present,
                    expected,
                )
            for b in rows_by_group[gid]:
                group[b] = b
                weight[b] = 1.0
                req = requests[b].data.req
                # Sets ``finished_reason`` directly rather than upstream's
                # ``to_finish`` handshake (schedule_batch.py: "if we want to
                # abort ... set to_finish instead of directly setting
                # finished_reason ... the req will get filtered and never
                # respond"). Upstream DOES call ``req.check_finished()`` every
                # decode step (``process_batch_result_decode``) — that
                # handshake is real, not dead code here. This matches
                # ``_mark_sampler_finished`` (right below), the
                # already-working precedent for Higgs TTS's normal EOC
                # completion, which also sets ``finished_reason`` directly at
                # this same point in the per-step collect loop — not the
                # abort-only one-arg-construction call site in
                # ``OmniScheduler``. FINISH_ABORT() itself is the same
                # no-arg construction used there; the "why" for this specific
                # abort goes to the log above, not a message argument, since
                # this repo has never exercised any other FINISH_ABORT
                # signature. Whether setting ``finished_reason`` directly at
                # this exact point is safe against upstream's warning is
                # still an open, real-engine-only verification item — see
                # the design doc.
                if req.finished_reason is None:
                    req.finished_reason = FINISH_ABORT()

        model._cg_fusion_group[:bs] = torch.tensor(
            group, dtype=torch.long, device=model._cg_fusion_group.device
        )
        model._cg_fusion_weight[:bs] = torch.tensor(
            weight, dtype=torch.float32, device=model._cg_fusion_weight.device
        )
        # Record whether the buffer just written can still contain non-default
        # (grouped) entries, so the next call is safe to skip via the
        # early-return above once this goes False.
        still_dirty = model.has_any_fusion()
        if not still_dirty:
            # Clean transition. The write above only covers [:bs] of THIS
            # step, but an earlier, larger dirty step could have left
            # non-identity group/weight values at slots >= bs that this call
            # never reaches (these buffers are fixed pool_size, not resized
            # per step). If bs shrinks back to non-fusion traffic before this
            # transition and later grows again while a smaller step's
            # dirty=False write left those slots stale, a future batch could
            # silently reuse rows [bs, prior_bs) and get blended together as
            # if they were still that finished group — two unrelated ordinary
            # requests corrupted into each other's distributions. Reset the
            # WHOLE buffer here, once, rather than tracking a high-water mark:
            # this only runs on the rare dirty-to-clean transition (the
            # early-return above already skips every call after), and the
            # cost is one bounded, pool_size-sized write — negligible next to
            # the dict scan this path already routes around.
            pool_size = model._cg_fusion_group.shape[0]
            model._cg_fusion_group[:] = torch.arange(
                pool_size, dtype=torch.long, device=model._cg_fusion_group.device
            )
            model._cg_fusion_weight[:] = 1.0
        self._fusion_buffers_dirty = still_dirty

    @staticmethod
    def _extract_decode_sampling_params(requests):
        """Read per-row sampling parameters from request-side host state.

        SGLang's ``sampling_info`` stores these values on CUDA. Copying its
        three tensors back to Python on every decode step forces three stream
        synchronizations even though the values are immutable for a request.
        ``top_k`` values outside ``(0, K_MAX)`` are normalized to ``None``;
        the downstream buffer maps that to ``K_MAX`` = no-op filtering.
        """
        temps: list[float] = []
        top_ps: list[float] = []
        top_ks: list[int | None] = []
        for request in requests:
            params = request.data.req.sampling_params
            temps.append(float(params.temperature))
            top_ps.append(float(params.top_p))
            top_k = params.top_k
            top_ks.append(
                int(top_k) if top_k is not None and 0 < int(top_k) < K_MAX else None
            )
        return temps, top_ps, top_ks

    def _collect_step_outputs_cg(
        self, result: Any, forward_batch: Any, requests: list
    ) -> None:
        """Synchronous collect: scatter + pack (GPU->GPU), one blocking D2H,
        then the host collect loop. Used when async decode is off; behavior is
        identical to the pre-split implementation (now factored into
        ``_decode_pack_gpu`` + ``_decode_collect_host``, which the async
        ``post_decode_launch`` / ``post_decode_resolve`` also reuse).
        """
        if len(requests) == 0:
            return
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )
        staging = self._decode_pack_gpu(n_real)
        combined_cpu = staging[:n_real].cpu()  # one blocking D2H (sync path)
        logprobs_cpu = None
        if self._should_capture_rollout_logprobs(requests):
            logprobs_cpu = self._decode_step_logprobs(result, n_real)[:n_real].cpu()
        # Trajectory-feedback observation D2H: only when fusion traffic is
        # actually present (has_any_fusion), matching the same zero-cost-hot-
        # path gating _populate_fusion_buffers already uses for its own dict
        # scan — an extra small D2H here would otherwise be pure overhead on
        # every decode step of every non-fusion server.
        fusion_ell_cpu = (
            self.model._cg_fusion_ell[:n_real].cpu()
            if self.model.has_any_fusion()
            else None
        )
        self._decode_collect_host(
            combined_cpu,
            logprobs_cpu,
            result,
            requests,
            next_token_device=result.logits_output.next_token_logits.device,
            fusion_ell_cpu=fusion_ell_cpu,
        )

    def _decode_pack_gpu(self, n_real: int) -> torch.Tensor:
        """Scatter shadow sampler state back into the pool and pack the three
        collect tensors (codes / was_done / generation_done) into the staging
        buffer. All GPU->GPU; returns the device staging buffer.
        """
        model = self.model
        rows_t = model._cg_row_indices[:n_real]
        pool = model._sampler_pool
        pool.delay_count[rows_t] = model._cg_active_delay_count[:n_real]
        pool.eoc_countdown[rows_t] = model._cg_active_eoc_countdown[:n_real]
        pool.generation_done[rows_t] = model._cg_active_generation_done[:n_real]
        pool.last_codes[rows_t] = model._cg_active_last_codes[:n_real]
        pool.step_count[rows_t] = model._cg_active_step_count[:n_real]

        # Note(Jiaxin): pack the 3 tensors so a single D2H pulls them all back.
        num_codebooks = model._cg_codes_BN.shape[1]
        staging = model._cg_collect_staging
        staging[:n_real, :num_codebooks] = model._cg_codes_BN[:n_real]
        staging[:n_real, num_codebooks] = model._cg_was_done[:n_real]
        staging[:n_real, num_codebooks + 1] = model._cg_active_generation_done[:n_real]
        return staging

    def _decode_collect_host(
        self,
        combined_cpu: torch.Tensor,
        logprobs_cpu: torch.Tensor | None,
        result: Any,
        requests: list,
        *,
        next_token_device: torch.device | None,
        fusion_ell_cpu: torch.Tensor | None = None,
    ) -> None:
        """Host-side collect loop over an already-D2H'd staging snapshot:
        append per-request codes, mark finishes, build ``result.next_token_ids``.
        Skips chunked and already-done rows, making the one-step-lookahead
        overrun harmless.

        These are reporting tokens for CPU-side output processing. The next
        forward's GPU token rail is published separately by
        :meth:`next_input_token_ids`.

        ``next_token_device`` is set for synchronous decode because those ids
        feed the next step. Async resolve passes ``None``: launch already
        published GPU codebook-0, and resolve only needs a CPU tensor for
        output processing.

        ``fusion_ell_cpu`` (present only on the sync path — see
        ``_collect_step_outputs_cg``) is each row's own, still per-member
        (not group-pooled) entropy-matched log-likelihood of this step's
        sampled frame; when given, drives the trajectory-feedback delta
        update (see ``_update_fusion_deltas``) after the main per-row loop
        below.
        """
        model = self.model
        num_codebooks = model._cg_codes_BN.shape[1]
        codes_BN_cpu = combined_cpu[:, :num_codebooks]
        was_done_cpu = combined_cpu[:, num_codebooks].bool().tolist()
        gen_done_after_cpu = combined_cpu[:, num_codebooks + 1].bool().tolist()
        cb0_per_row: list[int] = []
        # Read once, outside the per-row loop: on a server with zero fusion
        # traffic this is the only fusion-related work this hot loop does —
        # no per-row ``is_fusion_follower`` (lock-guarded dict lookup) call.
        any_fusion = model.has_any_fusion()
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if req.inflight_middle_chunks > 0:
                cb0_per_row.append(0)
                continue
            # Already finished in an earlier step? Skip its append. Under async
            # lookahead the finished req gets one extra (wasted) forward before
            # being dropped; this prevents leaking that overrun token. Catches
            # length finishes too (which `_cg_was_done`, an EOC-only flag, does
            # not). No-op for the sync path: a req is never finished() at its
            # own collect (finish is set later, in process_batch_result).
            if req.finished():
                cb0_per_row.append(0)
                continue
            if was_done_cpu[b]:
                cb0_per_row.append(0)
                continue
            # Voice-fusion followers decode the same frame as their group leader
            # (shared fused distribution + shared seed); only the leader is
            # emitted as audio. See ``_finish_fusion_follower`` for why they
            # still need bridging to a finish state here.
            if any_fusion and self.model.is_fusion_follower(sched_req.request_id):
                self._finish_fusion_follower(data, req, bool(gen_done_after_cpu[b]))
                cb0_per_row.append(0)
                continue
            codes_N = self._append_output_code(data, codes_BN_cpu[b])
            if logprobs_cpu is not None and self._request_captures_rollout_logprobs(
                sched_req
            ):
                data.output_logprobs.append(logprobs_cpu[b].to(torch.float32).clone())
            data.generation_done = bool(gen_done_after_cpu[b])
            self._queue_or_emit_code_chunk(
                sched_req,
                codes_N,
                force=self._is_final_code_step(data),
            )
            self._mark_sampler_finished(req, data.generation_done)
            cb0_per_row.append(int(codes_N[0].item()))

        if fusion_ell_cpu is not None:
            self._update_fusion_deltas(requests, fusion_ell_cpu)

        if next_token_device is None:
            result.next_token_ids = torch.tensor(cb0_per_row, dtype=torch.long)
        else:
            result.next_token_ids = torch.tensor(
                cb0_per_row,
                dtype=torch.long,
                device=next_token_device,
            )

    def _update_fusion_deltas(
        self, requests: list, fusion_ell_cpu: torch.Tensor
    ) -> None:
        """Trajectory-feedback update, CG-decode path — the counterpart of
        ``HiggsTTSModel._update_fusion_deltas`` (eager path). See
        ``FusionRegistry``'s ``_delta`` docstring and
        ``docs/voice_fusion_design.md``'s "AR 滞后" section for what this
        corrects and why entropy matching (a per-step marginal correction)
        alone can't reach it.

        Runs entirely host-side, after ``fusion_ell_cpu`` (this step's
        per-row, still per-member (not group-pooled) entropy-matched
        log-likelihood of the sampled frame, written inside the captured
        graph by ``HiggsTTSModel.decode_codebooks_batch_cg`` — see that
        method's comment for why this must be measured on the
        entropy-matched logits, not the true raw per-member ones) has
        already been D2H'd by the caller — this method itself does no GPU
        work, just Python dict aggregation over however many rows are
        actually fusion members this step (typically a small fraction of the
        batch, if any).
        """
        model = self.model
        rids = [req.request_id for req in requests]
        group_of, weight_of, _delta_of = model.fusion_membership_snapshot(rids)
        if not group_of:
            return

        ell_list = fusion_ell_cpu.tolist()
        members_by_group: dict[str, list[int]] = {}
        for b, rid in enumerate(rids):
            gid = group_of.get(rid)
            if gid is None:
                continue
            # A row `_populate_fusion_buffers` isolated-and-aborted THIS step
            # (a split/poisoned group demoted to independent singletons, see
            # that method) decoded its own, unfused distribution this step —
            # its `ell` reflects "how much did this row like its own solo
            # frame", not a real fusion-group opinion, and folding it into an
            # aggregate here would pollute the delta of a group that's being
            # torn down anyway (harmless in practice today, since the abort
            # + registry-clear that follows in this same scheduler iteration
            # discards the polluted delta with the group — but excluding it
            # keeps this path's behavior identical to the eager path, which
            # groups by the POST-isolation batch-local ids instead of a
            # membership snapshot, so isolated rows are singletons there and
            # never reach this aggregation at all).
            if requests[b].data.req.finished():
                continue
            members_by_group.setdefault(gid, []).append(b)

        for members in members_by_group.values():
            if len(members) < 2:
                # Only one member of this group present in the current
                # batch this step -- not a real co-batched fusion pair right
                # now (e.g. the rest haven't reached decode yet, or this is
                # mid-split before the group-completeness guard aborts it).
                # Nothing meaningful to aggregate against; leave delta as-is.
                continue
            raw_weights = [weight_of.get(rids[b], 1.0) for b in members]
            total_w = sum(raw_weights) or 1.0
            norm_weights = [w / total_w for w in raw_weights]
            weighted_avg_ell = sum(
                w * ell_list[b] for w, b in zip(norm_weights, members)
            )
            for b in members:
                rid = rids[b]
                current_delta = model.get_fusion_delta(rid)
                new_delta = current_delta - _TRAJECTORY_DELTA_LAMBDA * (
                    ell_list[b] - weighted_avg_ell
                )
                model.update_fusion_delta(rid, new_delta)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        device = input_ids.device
        embed_tokens = self.model.backbone.model.embed_tokens
        fused_embed = self.model.multimodal_embedding.modality_embedding_0

        placeholder_mask = input_ids == AUDIO_PLACEHOLDER_ID
        safe_ids = torch.where(placeholder_mask, torch.zeros_like(input_ids), input_ids)
        text_embeds = embed_tokens(safe_ids)

        offset = 0
        for sched_req in requests:
            data = sched_req.data
            end = offset + int(data.req.extend_range.length)
            codes_rows = data.reference_codes_delayed
            if not codes_rows:
                offset = end
                continue

            full_mask = placeholder_mask[offset:end]
            n_placeholders = int(full_mask.sum().item())
            if n_placeholders == 0:
                offset = end
                continue

            codes = torch.tensor(codes_rows, dtype=torch.long, device=device)
            # Radix reuse can begin this request's live extension after a
            # cached reference-audio prefix. Derive the code row from the
            # absolute prompt position instead of request-local progress.
            consumed = sum(
                token == AUDIO_PLACEHOLDER_ID
                for token in data.req.origin_input_ids[: data.req.extend_range.start]
            )
            if consumed + n_placeholders > len(codes_rows):
                raise RuntimeError(
                    "Higgs reference-code rows do not cover the live audio "
                    f"placeholders: consumed={consumed}, "
                    f"live={n_placeholders}, available={len(codes_rows)}"
                )
            with torch.no_grad():
                embed = fused_embed(codes[consumed : consumed + n_placeholders])
            mask_idx = full_mask.nonzero(as_tuple=True)[0] + offset
            text_embeds[mask_idx] = embed.to(text_embeds.dtype)
            offset = end

        return text_embeds

    def _collect_step_outputs(
        self, result: Any, requests: list, forward_batch: Any | None = None
    ) -> None:
        """Pull per-request newly emitted codes from the model into
        ``data.output_codes`` and overwrite ``result.next_token_ids``
        with codebook-0 so the base runner skips its text-vocab sampler.
        """
        batch_size = len(requests)
        if batch_size == 0:
            return

        model = self.model
        logprobs_BN = None
        if self._should_capture_rollout_logprobs(requests):
            logprobs_BN = self._prefill_step_logprobs(result, requests, forward_batch)
        cb0_per_row: list[int] = []
        any_fusion = model.has_any_fusion()
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            rid = sched_req.request_id
            row = model._rid_to_row.get(rid)
            codes_log = model._output_codes.get(rid)
            # Fusion follower: see ``_finish_fusion_follower`` for why it's
            # bridged to a finish state here despite never being emitted.
            if row is not None and any_fusion and model.is_fusion_follower(rid):
                gen_done = bool(model._sampler_pool.generation_done[row].item())
                self._finish_fusion_follower(data, req, gen_done)
                cb0_per_row.append(0)
                continue
            if (
                req.inflight_middle_chunks > 0
                or row is None
                or not codes_log
                or req.finished()
            ):
                cb0_per_row.append(0)
                continue
            codes_N = codes_log[-1]
            codes_cpu = self._append_output_code(data, codes_N)
            if logprobs_BN is not None and self._request_captures_rollout_logprobs(
                sched_req
            ):
                data.output_logprobs.append(logprobs_BN[b].detach().cpu().clone())
            data.generation_done = bool(model._sampler_pool.generation_done[row].item())
            self._queue_or_emit_code_chunk(
                sched_req,
                codes_cpu,
                force=self._is_final_code_step(data),
            )
            self._mark_sampler_finished(req, data.generation_done)
            cb0_per_row.append(int(codes_cpu[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    @staticmethod
    def _request_captures_rollout_logprobs(sched_req: Any) -> bool:
        data = sched_req.data
        return bool(data.return_omni_rollout and data.return_logprob)

    def _should_capture_rollout_logprobs(self, requests: list) -> bool:
        return any(self._request_captures_rollout_logprobs(req) for req in requests)

    @staticmethod
    def _append_output_code(data: Any, codes_N: torch.Tensor) -> torch.Tensor:
        try:
            max_new_tokens = int(data.max_new_tokens)
            num_codebooks = int(data.num_codebooks)
            count = int(data.output_code_count)
            buffer = data.output_code_buffer
        except AttributeError:
            codes_cpu = codes_N.detach().cpu().to(torch.long).clone()
            data.output_codes.append(codes_cpu)
            return codes_cpu

        if buffer is None:
            buffer = torch.empty(
                (max(1, max_new_tokens), num_codebooks),
                dtype=torch.long,
                device="cpu",
            )
            data.output_code_buffer = buffer
        elif count >= buffer.shape[0]:
            new_cap = max(count + 1, buffer.shape[0] * 2)
            new_buffer = torch.empty(
                (new_cap, num_codebooks),
                dtype=torch.long,
                device="cpu",
            )
            new_buffer[:count].copy_(buffer[:count])
            buffer = new_buffer
            data.output_code_buffer = buffer

        row = buffer[count]
        if codes_N.device.type == "cpu" and codes_N.dtype == torch.long:
            row.copy_(codes_N)
        else:
            row.copy_(codes_N.detach().to(device="cpu", dtype=torch.long))
        data.output_code_count = count + 1
        return row

    def _decode_step_logprobs(self, result: Any, n_real: int) -> torch.Tensor:
        model = self.model
        hidden_states = result.logits_output.hidden_states
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        logits_BNV = model.modality_head.generate(hidden_states[:n_real]).to(
            torch.float32
        )
        codes_BN = model._cg_codes_BN[:n_real].clamp_min(0)
        return selected_token_logprobs(
            logits_BNV,
            codes_BN,
            temperature=model._cg_temperature[:n_real],
            top_k_buf=model._cg_top_k_buf[:n_real],
        )

    def _prefill_step_logprobs(
        self, result: Any, requests: list, forward_batch: Any | None
    ) -> torch.Tensor:
        del forward_batch
        model = self.model
        hidden_states = result.logits_output.hidden_states
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        logits_BNV = model.modality_head.generate(hidden_states[: len(requests)]).to(
            torch.float32
        )
        codes = []
        temps = []
        top_ks = []
        for sched_req in requests:
            rid = sched_req.request_id
            codes_log = model._output_codes.get(rid)
            if codes_log:
                codes.append(codes_log[-1])
            else:
                codes.append(
                    torch.zeros(
                        model._num_codebooks,
                        dtype=torch.long,
                        device=logits_BNV.device,
                    )
                )
            sp = sched_req.data.req.sampling_params
            temps.append(float(getattr(sp, "temperature", 1.0)))
            top_k = getattr(sp, "top_k", None)
            top_ks.append(
                int(top_k) if (top_k is not None and int(top_k) > 0) else K_MAX
            )
        codes_BN = torch.stack(
            [c.to(device=logits_BNV.device, dtype=torch.long) for c in codes]
        )
        temperature = torch.tensor(temps, dtype=torch.float32, device=logits_BNV.device)
        top_k_buf = torch.tensor(top_ks, dtype=torch.long, device=logits_BNV.device)
        return selected_token_logprobs(
            logits_BNV,
            codes_BN.clamp_min(0),
            temperature=temperature,
            top_k_buf=top_k_buf,
        )

    @staticmethod
    def _mark_sampler_finished(req: Any, generation_done: bool) -> None:
        """Bridge Higgs sampler completion into upstream SGLang finish state."""
        if generation_done and req.finished_reason is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(EOC_ID)

    @staticmethod
    def _is_final_code_step(data: Any) -> bool:
        if bool(data.generation_done):
            return True
        try:
            max_new_tokens = int(data.max_new_tokens or 0)
        except AttributeError:
            return False
        try:
            output_count = int(data.output_code_count)
        except AttributeError:
            output_count = len(data.output_codes)
        return max_new_tokens > 0 and output_count >= max_new_tokens

    def _finish_fusion_follower(
        self, data: Any, req: Any, generation_done: bool
    ) -> None:
        """Bridge a voice-fusion follower row to a finish state without
        emitting its (duplicate) codes.

        A follower decodes the same frame as its group leader (shared fused
        distribution + shared seed), so its codes must never be appended or
        emitted to the vocoder. But it MUST still be marked finished on the
        step its (barrier-synced) ``generation_done`` flips — the group
        barrier flips all members together, so if the follower isn't bridged
        here it lingers in the batch after the leader retires, the group
        "splits", and the next step's completeness check aborts the batch.
        """
        data.generation_done = generation_done
        self._mark_sampler_finished(req, generation_done)

    def _queue_or_emit_code_chunk(
        self, sched_req: Any, codes_N: torch.Tensor, *, force: bool = False
    ) -> None:
        if self._outbox is None:
            return
        data = sched_req.data
        if data.stream_metadata is None:
            return

        data.stream_code_buffer.append(codes_N)
        data.stream_code_seen_rows += 1
        if int(data.stream_code_next_flush_rows) <= 0:
            data.stream_code_next_flush_rows = self._initial_stream_flush_rows(data)
        if force or data.stream_code_seen_rows >= data.stream_code_next_flush_rows:
            self._flush_code_chunks(
                sched_req.request_id,
                data,
                force=force,
            )

    @staticmethod
    def _stream_params(data: Any) -> tuple[int, int, int, int]:
        num_codebooks = max(int(data.num_codebooks or 1), 1)
        metadata = data.stream_metadata or {}
        stride = max(
            int(
                metadata.get(HIGGS_STREAM_STRIDE_METADATA, DEFAULT_HIGGS_STREAM_STRIDE)
            ),
            1,
        )
        followup = max(
            int(
                metadata.get(
                    HIGGS_STREAM_FOLLOWUP_STRIDE_METADATA,
                    DEFAULT_HIGGS_STREAM_FOLLOWUP_STRIDE,
                )
            ),
            1,
        )
        initial_frames = metadata.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM)
        if initial_frames is None:
            return num_codebooks, stride, followup, 0
        try:
            initial_frames_i = int(initial_frames)
        except (TypeError, ValueError):
            initial_frames_i = 0
        if initial_frames_i <= 0:
            return num_codebooks, stride, followup, 0
        steady_codec_frames = max(1, stride - num_codebooks + 1)
        return (
            num_codebooks,
            stride,
            followup,
            min(initial_frames_i, steady_codec_frames),
        )

    @classmethod
    def _initial_stream_flush_rows(cls, data: Any) -> int:
        num_codebooks, stride, _, initial_frames = cls._stream_params(data)
        steady_codec_frames = max(1, stride - num_codebooks + 1)
        if 0 < initial_frames < steady_codec_frames:
            return max(num_codebooks, initial_frames + num_codebooks - 1)
        return max(num_codebooks, stride)

    @classmethod
    def _next_stream_flush_rows(cls, data: Any, flushed_rows: int) -> int:
        num_codebooks, stride, followup, initial_frames = cls._stream_params(data)
        steady_codec_frames = max(1, stride - num_codebooks + 1)
        emitted_initial_chunk = (
            not bool(data.stream_code_first_flush_done)
            and 0 < initial_frames < steady_codec_frames
        )
        if emitted_initial_chunk:
            return max(num_codebooks, stride) + followup
        return int(flushed_rows) + followup

    def _flush_code_chunks(
        self,
        request_id: str,
        data: Any,
        *,
        force: bool = False,
    ) -> None:
        rows = data.stream_code_buffer
        if not rows:
            return
        if len(rows) == 1:
            payload = rows[0]
        else:
            payload = torch.stack(rows, dim=0)
        flushed_rows = int(data.stream_code_seen_rows)
        if not force:
            data.stream_code_next_flush_rows = self._next_stream_flush_rows(
                data, flushed_rows
            )
        data.stream_code_buffer = []
        data.stream_code_first_flush_done = True
        self._emit_code_chunk(request_id, data, payload)

    def _emit_code_chunk(
        self,
        request_id: str,
        data: Any,
        codes_N: torch.Tensor,
    ) -> None:
        if self._outbox is None:
            return
        metadata = data.stream_metadata
        if metadata is None:
            return
        self._outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=self._vocoder_target,
                data=codes_N,
                metadata=metadata,
            )
        )


__all__ = ["HiggsTTSModelRunner"]
