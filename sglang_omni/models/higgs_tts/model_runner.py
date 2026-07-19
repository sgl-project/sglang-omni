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
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.higgs_tts.model import _flat_sampling_attr
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


class HiggsTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`HiggsTTSModel`."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"
        # Ping-pong pinned host buffers for the async-decode rollout-logprob D2H.
        self._logprob_host_buffers: list[torch.Tensor] | None = None
        self._logprob_slot = 0

    def _next_logprob_host_staging(self, device_buf: torch.Tensor) -> torch.Tensor:
        if self._logprob_host_buffers is None:
            self._logprob_host_buffers = [
                torch.empty(
                    device_buf.shape,
                    dtype=device_buf.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
        buf = self._logprob_host_buffers[self._logprob_slot]
        self._logprob_slot ^= 1
        return buf

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        for req in requests:
            self.model.set_request_seed(
                req.request_id, req.data.req.sampling_params.sampling_seed
            )
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
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
        forward_batch.req_ids = [req.request_id for req in requests]
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
        result.next_token_ids = (
            self.model._cg_codes_BN[:n_real, 0].clamp_min(0).to(torch.long).clone()
        )
        return host_buf, logprob_host

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
        self._decode_collect_host(
            host_buf[:n_real],
            logprobs_cpu,
            result,
            requests,
            next_token_device=None,
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

        rows_py: list[int] = [model.acquire_row(req.request_id) for req in requests]
        rows_py.extend([model._padding_row] * (bs - n_real))
        model._cg_row_indices[:bs] = torch.tensor(
            rows_py, dtype=torch.long, device=model._cg_row_indices.device
        )

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

        temps, top_ps, top_ks = self._extract_decode_sampling_params(
            forward_batch, n_real
        )
        temps.extend([1.0] * (bs - n_real))
        top_ps.extend([1.0] * (bs - n_real))
        model._cg_temperature[:bs] = torch.tensor(
            temps, dtype=torch.float32, device=model._cg_temperature.device
        )
        model._cg_top_p[:bs] = torch.tensor(
            top_ps, dtype=torch.float32, device=model._cg_top_p.device
        )

        top_k_vals = [(tk if (tk is not None and tk > 0) else K_MAX) for tk in top_ks]
        top_k_vals.extend([K_MAX] * (bs - n_real))
        model._cg_top_k_buf[:bs] = torch.tensor(
            top_k_vals, dtype=torch.long, device=model._cg_top_k_buf.device
        )

        rows_t = model._cg_row_indices[:bs]
        pool = model._sampler_pool
        model._cg_active_delay_count[:bs] = pool.delay_count[rows_t]
        model._cg_active_eoc_countdown[:bs] = pool.eoc_countdown[rows_t]
        model._cg_active_generation_done[:bs] = pool.generation_done[rows_t]
        model._cg_active_last_codes[:bs] = pool.last_codes[rows_t]
        model._cg_active_seeds[:bs] = pool.seeds[rows_t]
        model._cg_active_step_count[:bs] = pool.step_count[rows_t]

    @staticmethod
    def _extract_decode_sampling_params(forward_batch, n_real: int):
        """Pull per-row temperature / top_p / top_k off sglang's
        ``sampling_info`` with safe defaults. ``top_k`` values outside
        ``(0, K_MAX)`` (including sglang's ``TOP_K_ALL`` sentinel for
        unspecified top_k) are normalized to ``None`` — the downstream
        buffer maps that to ``K_MAX`` = no-op filter.
        """
        sampling_info = getattr(forward_batch, "sampling_info", None)
        if sampling_info is None or n_real == 0:
            return ([1.0] * n_real, [1.0] * n_real, [None] * n_real)

        temps_raw = _flat_sampling_attr(sampling_info, "temperatures") or [1.0] * n_real
        top_ps_raw = _flat_sampling_attr(sampling_info, "top_ps") or [1.0] * n_real
        top_ks_raw = _flat_sampling_attr(sampling_info, "top_ks")

        temps = [float(t) for t in temps_raw[:n_real]]
        top_ps = [float(t) for t in top_ps_raw[:n_real]]
        if top_ks_raw is None:
            top_ks: list[int | None] = [None] * n_real
        else:
            top_ks = [
                int(t) if (t is not None and 0 < int(t) < K_MAX) else None
                for t in top_ks_raw[:n_real]
            ]
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
        self._decode_collect_host(
            combined_cpu,
            logprobs_cpu,
            result,
            requests,
            next_token_device=result.logits_output.next_token_logits.device,
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
    ) -> None:
        """Host-side collect loop over an already-D2H'd staging snapshot:
        append per-request codes, mark finishes, build ``result.next_token_ids``.
        Skips chunked and already-done rows (the latter is what makes the
        one-step-lookahead overrun harmless — see r1_idempotency_check.md).

        ``next_token_device`` is set for synchronous decode because those ids
        feed the next step. Async resolve passes ``None``: launch already
        published GPU codebook-0, and resolve only needs a CPU tensor for
        output processing.
        """
        model = self.model
        num_codebooks = model._cg_codes_BN.shape[1]
        codes_BN_cpu = combined_cpu[:, :num_codebooks]
        was_done_cpu = combined_cpu[:, num_codebooks].bool().tolist()
        gen_done_after_cpu = combined_cpu[:, num_codebooks + 1].bool().tolist()
        cb0_per_row: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if req.is_chunked > 0:
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

        if next_token_device is None:
            result.next_token_ids = torch.tensor(cb0_per_row, dtype=torch.long)
        else:
            result.next_token_ids = torch.tensor(
                cb0_per_row,
                dtype=torch.long,
                device=next_token_device,
            )

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
            end = offset + int(data.req.extend_input_len)
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
            consumed = data.num_ref_codes_consumed
            with torch.no_grad():
                embed = fused_embed(codes[consumed : consumed + n_placeholders])
            mask_idx = full_mask.nonzero(as_tuple=True)[0] + offset
            text_embeds[mask_idx] = embed.to(text_embeds.dtype)
            data.num_ref_codes_consumed = consumed + n_placeholders
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
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            rid = sched_req.request_id
            row = model._rid_to_row.get(rid)
            codes_log = model._output_codes.get(rid)
            if req.is_chunked > 0 or row is None or not codes_log or req.finished():
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
            self._flush_code_chunks(sched_req, force=force)

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

    def _flush_code_chunks(self, sched_req: Any, *, force: bool = False) -> None:
        data = sched_req.data
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
        self._emit_code_chunk(sched_req, payload)

    def _emit_code_chunk(self, sched_req: Any, codes_N: torch.Tensor) -> None:
        if self._outbox is None:
            return
        metadata = sched_req.data.stream_metadata
        if metadata is None:
            return
        self._outbox.put(
            OutgoingMessage(
                request_id=sched_req.request_id,
                type="stream",
                target=self._vocoder_target,
                data=codes_N,
                metadata=metadata,
            )
        )


__all__ = ["HiggsTTSModelRunner"]
