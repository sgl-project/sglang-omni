# SPDX-License-Identifier: Apache-2.0
# Follows the sglang-omni AR model-runner pattern (cf. Higgs).
"""Frame-step ModelRunner for CSM TTS.

The scheduler sees a normal one-token decode step (1 frame = 1 backbone
position = 1 SGLang "token"); the 31-step depth loop runs inside
``model.forward``'s decode branch. This runner owns everything
OUTSIDE the (future) captured region: CG-buffer population, prefill embed
overlay, the packed one-D2H-per-step collect, stream emission, and the
async launch/resolve halves.

Step anatomy (decode, sync path):

1. ``before_decode → _populate_cg_buffers``: reset padding row; acquire rows;
   lookahead-only GPU-side done-row reroute to the padding row (higgs guard
   #1); fill cb0 sampling buffers from SGLang ``sampling_info`` via
   ``_flat_sampling_attr``; fill DEPTH sampling buffers host-side from
   ``req._omni_data`` (depth params never ride sampling_info); gather
   ``pool.{generation_done, last_codes}[rows] → _cg_active_*[:bs]``.
2. ``model.forward`` (see model.py).
3. ``post_decode → _collect_step_outputs_cg``: ``_decode_pack_gpu`` (scatter
   ``_cg_active_* → pool[rows]``; pack staging ``[P, 34]``) → ONE blocking
   D2H ``staging[:n].cpu()`` → ``_decode_collect_host``.

All three async overrun guards stay verbatim from higgs (padding-row reroute
at launch; ``pre_finished`` snapshot + row drop in resolve; post-drain
``filter_batch()``) — they protect the lookahead overlap, not the wind-down
. Async stays OFF by default (``enable_async_decode=False`` in the
stage factory; ``async_decode_min_batch_size=2`` — the measured bs=1
regression); the launch/resolve halves below are implemented but dormant.


"""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.csm_tts.model import _flat_sampling_attr
from sglang_omni.models.csm_tts.sampler import K_MAX
from sglang_omni.models.csm_tts.utils import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    CODEBOOK_EOS,
    NUM_CODEBOOKS,
)
from sglang_omni.scheduling.messages import OutgoingMessage

__all__ = ["CsmTTSModelRunner"]


class CsmTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`~sglang_omni.models.csm_tts.model.CsmTTSModel`."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"

    def set_stream_outbox(self, outbox: Any) -> None:
        """Wire the OmniScheduler outbox used by :meth:`_emit_code_chunk`."""
        self._outbox = outbox

    # --- prefill --------------------------------------------------------------

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        # Depth sampling params are CSM-private (never on sampling_info);
        # stamp per-row (temperature, top_k) tuples so
        # model._extract_batch_metadata can read them.
        depth_temps, depth_top_ks = self._extract_depth_sampling_params(requests)
        forward_batch.csm_depth_params = list(zip(depth_temps, depth_top_ks))
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def _build_prefill_input_embeds(
        self, forward_batch: Any, requests: list
    ) -> torch.Tensor | None:
        """Prefill embed overlay.

        ``embed_tokens`` of the token ids, then paste context-frame embeds at
        the REAL-id ``AUDIO_TOKEN_ID`` (128002) positions (no -100 sentinel),
        and give each ``AUDIO_EOS_TOKEN_ID`` (128003) position the
        all-zeros-frame embed (HF ``_merge_input_ids_with_input_values``
        parity, modeling_csm.py:877-885). ``num_ctx_codes_consumed`` keeps
        the paste chunked-prefill-correct across ``chunked_prefill_size``
        boundaries.

        Returns:
            bf16 ``[T_total, 2048]`` or ``None`` when no request in the batch
            has context audio (every prompt id is then a real text token and
            the backbone's own embedding is exact).
        """
        if not any(
            getattr(sched_req.data, "context_codes", None) for sched_req in requests
        ):
            return None

        input_ids = forward_batch.input_ids
        device = input_ids.device
        embed_tokens = self.model.backbone.model.embed_tokens
        frame_embedding = self.model.frame_embedding

        # All prompt ids are real in-vocab tokens (128002/128003 included) —
        # no placeholder zeroing needed, the overlay simply overwrites rows.
        text_embeds = embed_tokens(input_ids)
        eos_frame_embed: torch.Tensor | None = None

        offset = 0
        for sched_req in requests:
            data = sched_req.data
            end = offset + int(data.req.extend_input_len)
            chunk_ids = input_ids[offset:end]

            audio_idx = (chunk_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
            n_placeholders = int(audio_idx.numel())
            if n_placeholders > 0:
                codes = self._concat_context_codes(data, device)
                if codes is None:
                    raise ValueError(
                        f"request {sched_req.request_id}: prompt has "
                        f"{n_placeholders} <|AUDIO|> placeholders but no "
                        f"context_codes (prompt/codes mismatch)"
                    )
                consumed = int(data.num_ctx_codes_consumed)
                if consumed + n_placeholders > codes.shape[0]:
                    raise ValueError(
                        f"request {sched_req.request_id}: placeholder count "
                        f"overruns context codes ({consumed} consumed + "
                        f"{n_placeholders} needed > {codes.shape[0]} frames — "
                        f"off-by-one class)"
                    )
                with torch.no_grad():
                    paste = frame_embedding(codes[consumed : consumed + n_placeholders])
                text_embeds[audio_idx + offset] = paste.to(text_embeds.dtype)
                data.num_ctx_codes_consumed = consumed + n_placeholders

            eos_idx = (chunk_ids == AUDIO_EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            if eos_idx.numel() > 0:
                if eos_frame_embed is None:
                    with torch.no_grad():
                        eos_frame_embed = frame_embedding(
                            torch.zeros(
                                1, NUM_CODEBOOKS, dtype=torch.long, device=device
                            )
                        )
                text_embeds[eos_idx + offset] = eos_frame_embed.to(text_embeds.dtype)

            offset = end

        return text_embeds

    @staticmethod
    def _concat_context_codes(data: Any, device: torch.device) -> torch.Tensor | None:
        """Per-request context codes, all segments concatenated in prompt
        order → int64 ``[F_total, 32]`` on ``device`` (placeholder spans
        appear in the same order, so one cursor walks the whole matrix)."""
        segments = getattr(data, "context_codes", None) or []
        mats = []
        for seg in segments:
            t = seg if isinstance(seg, torch.Tensor) else torch.as_tensor(seg)
            mats.append(t.reshape(-1, NUM_CODEBOOKS).to(torch.long))
        if not mats:
            return None
        return torch.cat(mats, dim=0).to(device)

    # --- decode (sync) ----------------------------------------------------------

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        self._populate_cg_buffers(forward_batch, requests, is_lookahead=is_lookahead)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        self._collect_step_outputs_cg(result, forward_batch, requests)

    def _populate_cg_buffers(
        self, forward_batch: Any, requests: list, *, is_lookahead: bool = False
    ) -> None:
        """Fill the model's CG buffers for one decode step (OUTSIDE any
        captured region; step 1).

        Padding rows (``batch_size > len(requests)``) point at the reserved
        padding row, which is reset every step so it can't leak state into
        real rows; their params are neutral (T=1, k=K_MAX).
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
            # that EOS-finished at the prior step is still in this launched
            # batch with pool.generation_done=True; reroute it to the reset
            # padding row — its overrun output is discarded by the collect's
            # finished()/was_done skip anyway. Sync steps never carry done
            # rows, so this gather+where would be pure waste there.
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

        # CSM delta vs higgs: depth-decoder sampling set, host-side from
        # ``req._omni_data`` (= sched_req.data; step 1).
        depth_temps, depth_top_ks = self._extract_depth_sampling_params(requests)
        depth_temps.extend([1.0] * (bs - n_real))
        depth_top_ks.extend([K_MAX] * (bs - n_real))
        model._cg_depth_temperature[:bs] = torch.tensor(
            depth_temps,
            dtype=torch.float32,
            device=model._cg_depth_temperature.device,
        )
        model._cg_depth_top_k_buf[:bs] = torch.tensor(
            depth_top_ks, dtype=torch.long, device=model._cg_depth_top_k_buf.device
        )

        rows_t = model._cg_row_indices[:bs]
        pool = model._sampler_pool
        model._cg_active_generation_done[:bs] = pool.generation_done[rows_t]
        model._cg_active_last_codes[:bs] = pool.last_codes[rows_t]

    @staticmethod
    def _extract_decode_sampling_params(forward_batch: Any, n_real: int) -> Any:
        """cb0 (temperature, top_p, top_k) rows off SGLang's ``sampling_info``
        with safe defaults (one D2H per attribute via ``_flat_sampling_attr``).
        ``top_k`` values outside ``(0, K_MAX)`` (including SGLang's TOP_K_ALL
        sentinel) normalize to ``None`` — the buffer maps that to ``K_MAX``
        = no-op filter.

        Returns:
            ``(temps, top_ps, top_ks)`` host lists of length ``n_real``.
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
                int(t) if (t is not None and 0 < int(t) <= K_MAX) else None
                for t in top_ks_raw[:n_real]
            ]
        return temps, top_ps, top_ks

    @staticmethod
    def _extract_depth_sampling_params(requests: list) -> tuple[list[float], list[int]]:
        """Depth (temperature, top_k) per request from ``sched_req.data``
        (= ``req._omni_data``; defaults (0.9, 50) — HF parity)."""
        depth_temps: list[float] = []
        depth_top_ks: list[int] = []
        for sched_req in requests:
            data = sched_req.data
            depth_temps.append(float(getattr(data, "depth_temperature", 0.9)))
            dk = getattr(data, "depth_top_k", 50)
            try:
                dk = int(dk)
            except (TypeError, ValueError):
                dk = K_MAX
            depth_top_ks.append(dk if 0 < dk <= K_MAX else K_MAX)
        return depth_temps, depth_top_ks

    def _collect_step_outputs_cg(
        self, result: Any, forward_batch: Any, requests: list
    ) -> None:
        """Sync decode tail: ``_decode_pack_gpu(n)`` → ONE blocking D2H
        ``staging[:n].cpu()`` → :meth:`_decode_collect_host` (step 3)."""
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
        self._decode_collect_host(
            combined_cpu,
            result,
            requests,
            next_token_device=result.logits_output.next_token_logits.device,
        )

    def _decode_pack_gpu(self, n_real: int) -> torch.Tensor:
        """Scatter ``_cg_active_* → pool[rows]`` and pack the staging row
        ``[codes_0..31 | was_done | generation_done]``. All GPU→GPU.

        Returns:
            int64 GPU staging buffer (full ``[P, 34]``; callers slice
            ``[:n_real]``).
        """
        model = self.model
        rows_t = model._cg_row_indices[:n_real]
        pool = model._sampler_pool
        pool.generation_done[rows_t] = model._cg_active_generation_done[:n_real]
        pool.last_codes[rows_t] = model._cg_active_last_codes[:n_real]
        # Bookkeeping only (nothing downstream reads it); under lookahead
        # reroute duplicate padding-row indices make this last-write-wins,
        # which is fine — the padding row is reset every step.
        pool.frames_emitted[rows_t] += (~model._cg_was_done[:n_real]).to(torch.int32)

        # Pack so a single D2H pulls codes + both flags (higgs pattern).
        num_codebooks = model._cg_codes_BN.shape[1]
        staging = model._cg_collect_staging
        staging[:n_real, :num_codebooks] = model._cg_codes_BN[:n_real]
        staging[:n_real, num_codebooks] = model._cg_was_done[:n_real]
        staging[:n_real, num_codebooks + 1] = model._cg_active_generation_done[:n_real]
        return staging

    def _decode_collect_host(
        self,
        staging_host: torch.Tensor,
        result: Any,
        requests: list,
        *,
        next_token_device: torch.device | None = None,
    ) -> None:
        """Per-row host collect off the staged snapshot:
        skip if ``req.is_chunked > 0`` / ``req.finished()`` / ``was_done``;
        else append ``codes_32`` to ``data.output_frames``; if
        ``generation_done``: :meth:`_mark_sampler_finished`, NO stream emit
        (EOS frames never reach Mimi — fence #1 of 3); else
        :meth:`_emit_code_chunk`; collect cb0 into ``result.next_token_ids``.

        ``next_token_device`` is set on the sync path (those ids feed the next
        step); the async resolve passes ``None`` (launch already published
        GPU cb0, resolve only needs a CPU tensor for output processing).

        Args:
            staging_host: int64 CPU ``[n_real, 34]``.
        """
        model = self.model
        num_codebooks = model._cg_codes_BN.shape[1]
        codes_BN_cpu = staging_host[:, :num_codebooks]
        was_done_cpu = staging_host[:, num_codebooks].bool().tolist()
        gen_done_after_cpu = staging_host[:, num_codebooks + 1].bool().tolist()
        cb0_per_row: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if req.is_chunked > 0:
                cb0_per_row.append(0)
                continue
            # Already finished in an earlier step? Skip the append. Under
            # async lookahead the finished req gets one extra (wasted) forward
            # before being dropped; this prevents leaking that overrun frame.
            # Catches length finishes too (which `was_done`, an EOS-only flag,
            # does not). No-op on the sync path.
            if req.finished():
                cb0_per_row.append(0)
                continue
            if was_done_cpu[b]:
                cb0_per_row.append(0)
                continue
            codes_32 = codes_BN_cpu[b].to(torch.long).clone()
            data.output_frames.append(codes_32)
            data.generation_done = bool(gen_done_after_cpu[b])
            self._mark_sampler_finished(req, data.generation_done)
            if not data.generation_done:
                # EOS frames are recorded (HF `sequences` parity) but never
                # streamed — fence #1 keeping codes >= 2048 away from Mimi.
                self._emit_code_chunk(sched_req, codes_32)
            cb0_per_row.append(int(codes_32[0].item()))

        if next_token_device is None:
            result.next_token_ids = torch.tensor(cb0_per_row, dtype=torch.long)
        else:
            result.next_token_ids = torch.tensor(
                cb0_per_row, dtype=torch.long, device=next_token_device
            )

    def _collect_step_outputs(self, result: Any, requests: list) -> None:
        """Prefill/eager collect (frame 0): same emission + finish rules as
        the decode collect, including the frame-0-EOS zero-decode lifecycle; overwrites
        ``result.next_token_ids`` with cb0 so the base runner skips its
        text-vocab sampler."""
        if len(requests) == 0:
            return

        model = self.model
        cb0_per_row: list[int] = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rid = sched_req.request_id
            row = model._rid_to_row.get(rid)
            codes_log = model._output_codes.get(rid)
            if req.is_chunked > 0 or row is None or not codes_log or req.finished():
                cb0_per_row.append(0)
                continue
            codes_32 = codes_log[-1].detach().cpu().clone()
            data.output_frames.append(codes_32)
            data.generation_done = bool(model._sampler_pool.generation_done[row].item())
            self._mark_sampler_finished(req, data.generation_done)
            if not data.generation_done:
                self._emit_code_chunk(sched_req, codes_32)
            cb0_per_row.append(int(codes_32[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    # --- decode (async halves; implemented but OFF by default) --------------------

    def post_decode_launch(
        self, result: Any, forward_batch: Any, requests: list
    ) -> Any:
        """Async GPU half: scatter + pack, ``copy_(non_blocking=True)`` into a
        ping-pong pinned host buffer, and publish
        ``result.next_token_ids = _cg_codes_BN[:n, 0].clamp_min(0)`` from GPU
        state with NO host sync (clamp keeps STOP_CODE=-1 in embed range;
        decode embeds read ``last_codes``, so this is bookkeeping only). Dormant by default (``enable_async_decode=False``).

        Returns:
            The pinned host buffer (the base runner records the CUDA event).
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
        host_buf = self._next_host_staging(self.model._cg_collect_staging)
        host_buf[:n_real].copy_(staging[:n_real], non_blocking=True)
        result.next_token_ids = (
            self.model._cg_codes_BN[:n_real, 0].clamp_min(0).to(torch.long).clone()
        )
        return host_buf

    def post_decode_resolve(
        self,
        host_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Async host half: run :meth:`_decode_collect_host` off the pinned
        snapshot (the base runner's ``skip_rids``/``pre_finished`` machinery
        plus the collect's own skips form the overrun guard)."""
        del forward_batch, schedule_batch
        if len(requests) == 0:
            return
        n_real = len(requests)
        self._decode_collect_host(
            host_buf[:n_real],
            result,
            requests,
            next_token_device=None,
        )

    # --- emission / finish ------------------------------------------------------

    def _emit_code_chunk(self, sched_req: Any, codes_32: torch.Tensor) -> None:
        """Stream one frame to the vocoder:
        ``OutgoingMessage(type="stream", target="vocoder", data=codes_32 CPU
        int64 [32], metadata=stream_metadata)`` via the outbox. ``new_done``
        rows are never emitted."""
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
                data=codes_32,
                metadata=metadata,
            )
        )

    @staticmethod
    def _mark_sampler_finished(req: Any, generation_done: bool) -> None:
        """Set ``FINISH_MATCHED_TOKEN(matched=CODEBOOK_EOS)`` (cb0 of the EOS
        frame = 0) — upstream only needs *a* finish reason to fire KV release."""
        if generation_done and req.finished_reason is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(CODEBOOK_EOS)
