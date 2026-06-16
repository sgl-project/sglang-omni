# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts_local.radix_hash import gpu_radix_row_hash
from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeJournal
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput


class MossTTSLocalModelRunner(ModelRunner):
    """Drives the per-frame local-transformer decode and feedback embeddings.

    Per step: the backbone (radix-cached, CUDA-graphed) produces one hidden
    state per request. Prefill collection runs the local micro-decode in the
    runner; decode collection reads the frame sampled inside model forward and
    staged in the row-indexed decode-state pool.
    """

    _outbox: Any | None = None
    _vocoder_target = "vocoder"

    _DYNAMIC_STAGE_FIELDS = (
        ("_cg_active_feedback_embeds", "feedback_embeds"),
        ("_cg_active_sampling_steps", "sampling_steps"),
    )
    _STATIC_STAGE_FIELDS = (
        ("_cg_active_text_temp", "text_temp"),
        ("_cg_active_text_top_p", "text_top_p"),
        ("_cg_active_text_top_k", "text_top_k"),
        ("_cg_active_audio_temp", "audio_temp"),
        ("_cg_active_audio_top_p", "audio_top_p"),
        ("_cg_active_audio_top_k", "audio_top_k"),
        ("_cg_active_seeds", "seeds"),
        ("_cg_active_audio_repetition_penalty", "audio_repetition_penalty"),
    )

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        self._prepare_forward_sample_inputs(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        try:
            is_prefill_only = schedule_batch.is_prefill_only
        except AttributeError:
            is_prefill_only = False
        if bool(is_prefill_only):
            return
        self._collect_frame_in_runner(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not bool(getattr(self, "_forward_sample_native_decode", False)) or bool(
            getattr(forward_batch, "moss_has_audio_repetition_penalty", False)
        ):
            self._collect_frame_in_runner(result, forward_batch, schedule_batch, requests)
            return
        self._collect_frame_from_forward_sample(result, schedule_batch, requests)

    def lookahead_eligible(self, batch: Any) -> bool:
        """Route to sync when the batch cannot take the graphed frame-decode
        path: any request with ``audio_repetition_penalty != 1`` (its eager
        rep-history gather lags one frame under lookahead and would diverge from
        sync) or ``bs > frame_graph_max_bs``.
        """
        try:
            reqs = batch.reqs
        except AttributeError:
            reqs = None
        reqs = reqs or []
        try:
            frame_graph_max_bs = self.model.frame_graph_max_bs
        except AttributeError:
            frame_graph_max_bs = 0
        if len(reqs) > int(frame_graph_max_bs):
            return False
        for req in reqs:
            try:
                data = req._omni_data
            except AttributeError:
                data = None
            if data is None:
                continue
            try:
                audio_repetition_penalty = data.audio_repetition_penalty
            except AttributeError:
                audio_repetition_penalty = 1.0
            if float(audio_repetition_penalty) != 1.0:
                return False
        return True

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS Local prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            pool = self.model._state_pool
            if data.output_rows:
                # KV-pressure retraction re-prefills with an extend region
                # spanning already-generated frames; their rows live in
                # output_rows, not prompt_rows. The resumed prefill samples
                # the next frame itself, superseding any feedback embedding
                # stranded by the retraction.
                generated = torch.stack(data.output_rows, dim=0)
                rows = torch.cat([rows.to(generated.device), generated], dim=0)
            # Realign the launch-side counter and clear the stranded pool row on
            # any retraction re-prefill, including one retracted before it emitted
            # a frame (empty output_rows). Both are no-ops for a fresh prefill:
            # the counters are already aligned and no pool row is held.
            generation_steps = int(data.generation_steps)
            data.sampling_steps = generation_steps
            pool.reset_for_refill(sched_req.request_id, generation_steps)
            if data.output_rows:
                pool.rebuild_audio_history(sched_req.request_id, data.output_rows)
            current_rows = rows[prefix_len : prefix_len + req_len]
            if int(current_rows.shape[0]) != req_len:
                raise RuntimeError(
                    f"MOSS-TTS Local prefill row mismatch for {req.rid}: have "
                    f"{int(current_rows.shape[0])} rows, need {req_len} "
                    f"(prefix={prefix_len}, prompt={int(data.prompt_rows.shape[0])}, "
                    f"generated={len(data.output_rows)})"
                )
            embeds = self.model._prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model.dtype,
        )

    def _prepare_forward_sample_inputs(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        if not requests:
            self._forward_sample_pool_rows = []
            self._forward_sample_pool_row_t = None
            self._forward_sample_rids = []
            self._forward_sample_native_decode = False
            self.model._moss_local_forward_native_decode_active = False
            return
        n_real = len(requests)
        batch_size = int(getattr(forward_batch, "batch_size", n_real) or n_real)
        model = self.model
        max_rows = int(model._cg_pool_rows.shape[0])
        if batch_size < n_real:
            raise RuntimeError(
                "MOSS-TTS Local decode batch is smaller than the "
                f"real batch ({batch_size} < {n_real})"
            )
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-TTS Local decode input_ids must contain one row id per request"
            )
        if batch_size > max_rows:
            raise RuntimeError(
                "MOSS-TTS Local decode batch exceeds staging buffers "
                f"({batch_size} > {max_rows})"
            )

        pool = model._state_pool
        row_tensor, pool_rows, has_audio_repetition_penalty = pool.prepare_active_rows(
            requests
        )
        try:
            frame_graph_max_bs = int(model.frame_graph_max_bs)
        except AttributeError:
            frame_graph_max_bs = 0
        use_forward_native_decode = (
            getattr(model, "_moss_local_native_decode_enabled", False)
            and not has_audio_repetition_penalty
            and batch_size <= frame_graph_max_bs
        )
        model._moss_local_forward_native_decode_active = use_forward_native_decode
        staged_pool_rows = list(pool_rows)
        if batch_size > n_real:
            staged_pool_rows.extend([pool.padding_row] * (batch_size - n_real))
        row_t = torch.tensor(
            staged_pool_rows, dtype=torch.long, device=pool.feedback_embeds.device
        )
        stage_fields = self._DYNAMIC_STAGE_FIELDS
        if use_forward_native_decode:
            stage_fields = stage_fields + self._STATIC_STAGE_FIELDS

        with torch.no_grad():
            model._cg_pool_rows[:batch_size].copy_(
                row_t.to(device=model._cg_pool_rows.device)
            )
            active_rows_t = model._cg_pool_rows[:batch_size].to(
                device=pool.feedback_embeds.device
            )
            for dst_name, src_name in stage_fields:
                dst = getattr(model, dst_name)
                src = getattr(pool, src_name)
                dst[:batch_size].copy_(
                    src[active_rows_t].to(device=dst.device, dtype=dst.dtype)
                )
            stats = getattr(model, "_moss_local_decode_stats", None)
            if isinstance(stats, dict):
                stats["batch_size"] = batch_size
                key = (
                    "forward_native_enabled_count"
                    if use_forward_native_decode
                    else "forward_native_fallback_count"
                )
                stats[key] = stats.get(key, 0) + 1

        row_ids = torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)
        forward_batch.moss_pool_row_t = row_tensor
        forward_batch.moss_pool_rows = pool_rows
        forward_batch.moss_has_audio_repetition_penalty = has_audio_repetition_penalty

        self._forward_sample_pool_rows = pool_rows
        self._forward_sample_pool_row_t = row_tensor
        self._forward_sample_rids = [sched_req.request_id for sched_req in requests]
        self._forward_sample_native_decode = use_forward_native_decode

    def _collect_frame_from_forward_sample(
        self,
        result: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        # Notes (Xinran): Sampling already ran inside
        # MossTTSLocalSGLangModel.forward() via the native decode-forward path.
        # Collection only snapshots the fixed active-slot step buffers and
        # publishes the graph-computed token ids.
        if not requests:
            return
        n_real = len(requests)
        current_rids = [sched_req.request_id for sched_req in requests]
        staged_rids = list(getattr(self, "_forward_sample_rids", []))
        pool_rows = list(getattr(self, "_forward_sample_pool_rows", []))
        if staged_rids[:n_real] != current_rids:
            raise RuntimeError(
                "MOSS-TTS Local decode pool request alignment broken: "
                f"{staged_rids[:n_real]} != {current_rids}"
            )
        if len(pool_rows) < n_real:
            raise RuntimeError(
                "MOSS-TTS Local decode pool row staging is incomplete: "
                f"{len(pool_rows)} < {n_real}"
            )

        model = self.model
        pool = model._state_pool
        rows = model._cg_step_rows[:n_real]
        next_token_ids = model._cg_step_next_token_ids[:n_real]
        result.next_token_ids = next_token_ids
        if schedule_batch is not None:
            schedule_batch.output_ids = next_token_ids

        emit_indices = [
            i
            for i, sched_req in enumerate(requests)
            if not self._is_chunked_request(sched_req)
        ]
        if not emit_indices:
            return

        emit_index_t = torch.tensor(emit_indices, dtype=torch.long, device=rows.device)
        emit_pool_rows = [pool_rows[i] for i in emit_indices]
        cached_row_t = getattr(self, "_forward_sample_pool_row_t", None)
        if (
            isinstance(cached_row_t, torch.Tensor)
            and int(cached_row_t.numel()) >= n_real
        ):
            emit_row_t = cached_row_t.index_select(
                0, emit_index_t.to(device=cached_row_t.device)
            )
        else:
            emit_row_t = torch.tensor(
                emit_pool_rows,
                dtype=torch.long,
                device=pool.feedback_embeds.device,
            )
        emit_row_t = emit_row_t.to(device=pool.feedback_embeds.device, dtype=torch.long)
        pool.feedback_embeds[emit_row_t] = model._cg_active_next_feedback_embeds[
            emit_index_t.to(device=model._cg_active_next_feedback_embeds.device)
        ].to(device=pool.feedback_embeds.device, dtype=pool.feedback_embeds.dtype)
        pool.sampling_steps[emit_row_t] = model._cg_active_next_sampling_steps[
            emit_index_t.to(device=model._cg_active_next_sampling_steps.device)
        ].to(device=pool.sampling_steps.device, dtype=torch.int64)
        result.moss_journal = MossTTSLocalDecodeJournal(
            rids=[requests[i].request_id for i in emit_indices],
            pool_rows=emit_pool_rows,
            rows=rows.index_select(0, emit_index_t),
        )

    def _collect_frame_in_runner(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not requests:
            return
        rows, end_id = self._run_frame_decode(result, forward_batch, requests)
        # Radix key is a capture-safe GPU hash: a device op, no host sync.
        next_text = rows[:, 0]
        next_token_ids = self._row_radix_token_ids(rows, next_text, end_id)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

    def _run_frame_decode(self, result: Any, forward_batch: Any, requests: list):
        """GPU half shared by sync ``_collect_frame_in_runner`` and async
        ``post_decode_launch``. Returns ``(rows, end_id)`` and does NOT publish
        ``next_token_ids``; the caller does, because the async path keeps a
        private device snapshot of the published ids for resolve to restore.
        """
        try:
            hidden_states = result.logits_output.hidden_states
        except AttributeError as exc:
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            ) from exc
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            )
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]

        cfg = self.model.config
        device = hidden_states.device
        pool = self.model._state_pool
        batch_size = len(requests)
        num_channels = int(cfg.n_vq) + 1

        try:
            row_t = forward_batch.moss_pool_row_t
            pool_rows = forward_batch.moss_pool_rows
        except AttributeError:
            row_t, pool_rows, has_audio_repetition_penalty = pool.prepare_active_rows(
                requests
            )
        else:
            try:
                has_audio_repetition_penalty = (
                    forward_batch.moss_has_audio_repetition_penalty
                )
            except AttributeError:
                has_audio_repetition_penalty = pool.rows_have_audio_repetition_penalty(
                    pool_rows
                )
        params = {
            "text_temp": pool.text_temp[row_t],
            "text_top_p": pool.text_top_p[row_t],
            "text_top_k": pool.text_top_k[row_t],
            "audio_temp": pool.audio_temp[row_t],
            "audio_top_p": pool.audio_top_p[row_t],
            "audio_top_k": pool.audio_top_k[row_t],
            "seeds": pool.seeds[row_t],
        }
        text_temp = params["text_temp"]
        text_top_p = params["text_top_p"]
        text_top_k = params["text_top_k"]
        audio_temp = params["audio_temp"]
        audio_top_p = params["audio_top_p"]
        audio_top_k = params["audio_top_k"]
        sampling_seeds = params["seeds"]
        # Advance the launch-side counter only for emitted rows; non-final
        # chunked rows take a read-only position so a mid-prefill chunk's frame
        # cannot shift the final chunk's sampling position off the no-chunk path.
        emit_set = {
            i
            for i, sched_req in enumerate(requests)
            if not self._is_chunked_request(sched_req)
        }
        gen_steps = torch.maximum(
            pool.sampling_steps[row_t].to(device=device),
            pool.generation_steps[row_t].to(device=device),
        )
        rep_penalties = pool.audio_repetition_penalty[row_t].to(
            device=device, dtype=torch.float32
        )

        def sample_text(logits: torch.Tensor) -> torch.Tensor:
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=text_temp,
                top_p=text_top_p,
                top_k=text_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels,
            )

        def sample_audio(logits: torch.Tensor, channel: int) -> torch.Tensor:
            if has_audio_repetition_penalty:
                presence = pool.audio_token_presence[row_t, channel].to(
                    device=logits.device
                )
                if int(presence.shape[-1]) != int(logits.shape[-1]):
                    presence = presence[:, : logits.shape[-1]]
                self._apply_audio_repetition_penalty_mask(
                    logits,
                    presence,
                    rep_penalties.to(device=logits.device, dtype=logits.dtype),
                )
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels + channel + 1,
            )

        try:
            frame_graph_max_bs = int(self.model.frame_graph_max_bs)
        except AttributeError:
            frame_graph_max_bs = 0
        use_graph = (
            not has_audio_repetition_penalty and batch_size <= frame_graph_max_bs
        )
        if use_graph:
            stop_choice, codes, feedback = self.model.decode_frame_graphed(
                hidden_states,
                text_temperature=text_temp,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temp,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                seeds=sampling_seeds,
                base_positions=gen_steps * num_channels,
            )
            # The graph outputs are static buffers that the next replay (any
            # later prefill or decode step) overwrites; snapshot what we keep.
            codes = codes.clone()
            embeds = feedback.clone()
        else:
            stop_choice, codes = self.model.decode_frame(
                hidden_states,
                sample_text=sample_text,
                sample_audio=sample_audio,
            )
            embeds = None

        slot_id = int(cfg.audio_assistant_slot_token_id)
        end_id = int(cfg.audio_end_token_id)
        next_text = torch.where(
            stop_choice == 0,
            torch.full((batch_size,), slot_id, dtype=torch.long, device=device),
            torch.full((batch_size,), end_id, dtype=torch.long, device=device),
        )

        rows = torch.empty((batch_size, num_channels), dtype=torch.long, device=device)
        rows[:, 0] = next_text
        rows[:, 1:] = codes

        if embeds is None:
            embeds = self.model._prepare_multi_modal_inputs(
                rows.to(device=self.model.device)
            )
        emit_indices = sorted(emit_set)
        if emit_indices:
            emit_index_t = torch.tensor(
                emit_indices, dtype=torch.long, device=rows.device
            )
            emit_pool_rows = [pool_rows[i] for i in emit_indices]
            emit_row_t = row_t[emit_index_t.to(device=row_t.device)]
            emit_rows = rows.index_select(0, emit_index_t)
            emit_steps = gen_steps.index_select(
                0, emit_index_t.to(device=gen_steps.device)
            )
            pool.sampling_steps[emit_row_t] = (emit_steps + 1).to(
                device=pool.sampling_steps.device, dtype=torch.int64
            )
            if has_audio_repetition_penalty:
                keep_history = (
                    next_text.index_select(0, emit_index_t.to(device=next_text.device))
                    != end_id
                )
                emit_penalty_active = (
                    pool.audio_repetition_penalty[emit_row_t]
                    .to(device=keep_history.device)
                    .ne(1.0)
                )
                keep_history = keep_history & emit_penalty_active
                pool.update_audio_history(
                    emit_row_t[keep_history.to(device=emit_row_t.device)],
                    emit_rows[keep_history.to(device=emit_rows.device)],
                )
            emit_embeds = embeds.index_select(0, emit_index_t.to(device=embeds.device))
            pool.feedback_embeds[emit_row_t] = emit_embeds.detach().to(
                device=pool.feedback_embeds.device,
                dtype=pool.feedback_embeds.dtype,
            )
            result.moss_journal = MossTTSLocalDecodeJournal(
                rids=[requests[i].request_id for i in emit_indices],
                pool_rows=emit_pool_rows,
                rows=emit_rows,
            )
        # Always return rows so both the sync inline path and the async launch
        # publish next_token_ids; an all-chunked batch just attaches no journal.
        return rows, end_id

    def post_decode_launch(self, result: Any, forward_batch: Any, requests: list):
        """Async-decode GPU half of ``post_decode``: publish the native
        forward-sampled frame ids, no host sync. Returns a private device
        snapshot of those ids for resolve: the base aliases ``next_token_ids``
        onto ``output_ids``, which the next step overwrites in place before this
        step's lagged resolve, clobbering the stop id and silently dropping a
        bs=1 eos finish (4096-frame runaway). The clone preserves it; resolve
        swaps it back.
        """
        if not requests:
            return None
        if not bool(getattr(self, "_forward_sample_native_decode", False)) or (
            forward_batch is not None
            and bool(getattr(forward_batch, "moss_has_audio_repetition_penalty", False))
        ):
            rows, end_id = self._run_frame_decode(result, forward_batch, requests)
            next_token_ids = self._row_radix_token_ids(rows, rows[:, 0], end_id)
            result.next_token_ids = next_token_ids
            return next_token_ids.clone()
        self._collect_frame_from_forward_sample(result, None, requests)
        return result.next_token_ids.clone()

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Async-decode host half: restore the launch-time ``next_token_ids``
        snapshot (a pointer swap) so the shared ``_finalize`` tail reads the real
        stop id, which the next step's in-place write clobbered from the aliased
        tensor before this lagged resolve.
        """
        del forward_batch, schedule_batch, requests
        if launch_buf is not None and result is not None:
            result.next_token_ids = launch_buf

    @staticmethod
    def _row_radix_token_ids(
        rows: torch.Tensor,
        next_text: torch.Tensor,
        end_id: int,
    ) -> torch.Tensor:
        """Radix-cache token ids for generated frames.

        The scheduler appends one token id per frame to the request's KV
        chain, and the radix tree keys on those ids. The text channel alone is
        the same assistant-slot id for every continuing frame of every
        request, so a re-prefill after retraction could falsely prefix-match
        into another identical-prompt request's cached generated region. Hash
        the full multi-channel row — the same keying used for prompt rows —
        so a radix match implies identical audio content (a per-position id
        clash is ~1/151643 and only matters on top of an identical full
        prefix). The hash is folded below the special-token band because the
        scheduler finishes any request whose generated id crosses the vocab
        boundary (``Req._check_vocab_boundary_finish``); the stop decision
        keeps the raw audio_end id so eos detection still fires.

        Unlike the prompt path (``build_row_cache_key_ids``'s host-side
        blake2b), this runs every decode step on a device tensor, so it uses
        the capture-safe tensor-native polynomial hash in :mod:`radix_hash` —
        no GPU->CPU sync. See ``docs/design/gpu_radix_hash.md``.
        """
        return gpu_radix_row_hash(rows, next_text, end_id)

    @staticmethod
    def _advance_sampling_position(data: Any) -> int:
        """RNG position for this collect, advancing the launch-side counter in
        floor mode: ``max(sampling_steps or 0, generation_steps)``. On the sync
        path the two stay equal (generation_steps increments after every collect)
        so the floor is a no-op and the position is bit-identical to before;
        under lookahead generation_steps lags, so the floor lifts launch(N+1) off
        the stale N.
        """
        try:
            sampling_steps = data.sampling_steps
        except AttributeError:
            sampling_steps = None
        s = max(int(sampling_steps or 0), int(data.generation_steps))
        data.sampling_steps = s + 1
        return s

    @staticmethod
    def _apply_audio_repetition_penalty_mask(
        logits: torch.Tensor,
        token_presence: torch.Tensor,
        penalties: torch.Tensor,
    ) -> None:
        """In-place penalty on fp32 logits, matching upstream order (before
        temperature scaling)."""
        penalties = penalties.to(device=logits.device, dtype=logits.dtype)
        active = token_presence.to(device=logits.device, dtype=torch.bool) & (
            penalties != 1.0
        ).unsqueeze(-1)
        penalties = penalties.unsqueeze(-1)
        penalized = torch.where(logits < 0, logits * penalties, logits / penalties)
        logits.copy_(torch.where(active, penalized, logits))

    @staticmethod
    def _is_chunked_request(sched_req: Any) -> bool:
        try:
            req = sched_req.data.req
        except AttributeError:
            return False
        if req is None:
            return False
        try:
            is_chunked = req.is_chunked
        except AttributeError:
            return False
        return int(is_chunked) > 0

    def finalize_skip_rids(self, scheduler_output) -> set[str]:
        """Non-final chunked-prefill rows must not advance ``generation_steps``.

        Their micro-decode still runs (as today), but the spurious step would
        shift the final chunk's sampling position off the no-chunk path; the
        sampling is positional (``position = generation_steps * num_channels +
        channel``), so suppressing the advance keeps the chunked path
        bit-identical to the single-shot prefill path.
        """
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if self._is_chunked_request(sched_req)
        }

    def on_generation_step_advanced(
        self, sched_req: Any, generation_steps: int
    ) -> None:
        try:
            pool = self.model._state_pool
        except AttributeError:
            return
        if pool is not None:
            pool.commit_generation_step(sched_req.request_id, generation_steps)

    def on_generation_steps_advanced(
        self, advanced_steps: list[tuple[Any, int]], forward_batch: Any
    ) -> None:
        try:
            pool = self.model._state_pool
        except AttributeError:
            return
        if pool is None or not advanced_steps:
            return
        steps = [int(generation_steps) for _, generation_steps in advanced_steps]
        try:
            row_t = forward_batch.moss_pool_row_t
        except AttributeError:
            row_t = None
        if row_t is not None and int(row_t.numel()) == len(steps):
            step_t = torch.tensor(steps, dtype=torch.long, device=row_t.device)
            pool.commit_generation_steps(row_t, step_t)
            return
        rows = []
        row_steps = []
        for sched_req, generation_steps in advanced_steps:
            row = pool.row_for(sched_req.request_id)
            if row is None:
                continue
            rows.append(row)
            row_steps.append(int(generation_steps))
        if not rows:
            return
        device = pool.generation_steps.device
        row_t = torch.tensor(rows, dtype=torch.long, device=device)
        step_t = torch.tensor(row_steps, dtype=torch.long, device=device)
        pool.commit_generation_steps(row_t, step_t)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        # The per-step journal is the single source of truth for output
        # collection. A missing journal means no frame was produced this step
        # (e.g. a prefill-only batch), which is the synchronous-baseline early
        # return.
        try:
            journal = result.moss_journal
        except AttributeError:
            return
        if journal is None:
            return

        end_id = int(self.model.config.audio_end_token_id)
        expected_reqs = [
            sched_req
            for sched_req in scheduler_output.requests
            if not self._is_chunked_request(sched_req)
        ]
        expected_rids = [sched_req.request_id for sched_req in expected_reqs]
        rows_len = int(journal.rows.shape[0])
        if len(journal.rids) != rows_len or len(journal.pool_rows) != rows_len:
            raise RuntimeError(
                "MOSS-TTS Local journal length mismatch: "
                f"rids={len(journal.rids)} pool_rows={len(journal.pool_rows)} "
                f"rows={rows_len}"
            )
        if journal.rids != expected_rids:
            raise RuntimeError(
                "MOSS-TTS Local journal/batch alignment broken: "
                f"{journal.rids} != {expected_rids}"
            )
        rows_cpu: torch.Tensor | None = None
        for i, sched_req in enumerate(expected_reqs):
            # Overrun: a request finished or retracted in a PRIOR step is still
            # in this lagged resolve batch; its wasted frame must not reach
            # output_rows / the vocoder. No-op on the sync path.
            req = sched_req.data.req
            if req is not None:
                try:
                    finished_fn = req.finished
                except AttributeError:
                    finished_fn = None
                try:
                    is_retracted = req.is_retracted
                except AttributeError:
                    is_retracted = False
                if (callable(finished_fn) and finished_fn()) or bool(is_retracted):
                    continue
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == end_id:
                continue
            sched_req.data.output_rows.append(journal.rows[i])
            stream_metadata = getattr(sched_req.data, "stream_metadata", None)
            if stream_metadata is None:
                continue
            if self._outbox is None:
                continue
            if rows_cpu is None:
                # One D2H per step regardless of how many requests stream.
                rows_cpu = journal.rows.detach().to("cpu", torch.long)
            self._outbox.put(
                OutgoingMessage(
                    request_id=sched_req.request_id,
                    type="stream",
                    target=self._vocoder_target,
                    data=rows_cpu[i].clone(),
                    metadata=stream_metadata,
                )
            )
