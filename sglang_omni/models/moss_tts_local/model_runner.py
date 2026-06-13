# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts_local.radix_hash import gpu_radix_row_hash
from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeJournal
from sglang_omni.scheduling.types import RequestOutput


class MossTTSLocalModelRunner(ModelRunner):
    """Drives the per-frame local-transformer decode and feedback embeddings.

    Per step: the backbone (radix-cached, CUDA-graphed) produces one hidden
    state per request; :meth:`_collect_frame` then runs the batched local
    micro-decode — a binary continue/stop decision and 12 sequentially
    sampled RVQ codes — and stages the next frame's summed embedding through
    ``model._decode_input_embedding`` so the next decode step stays
    CUDA-graph-replayable (decode input_ids are row indices).
    """

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

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
        self._write_decode_input_embedding(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def lookahead_eligible(self, batch: Any) -> bool:
        """Route to sync when the batch cannot take the graphed frame-decode
        path: any request with ``audio_repetition_penalty != 1`` (its eager
        rep-history gather lags one frame under lookahead and would diverge from
        sync) or ``bs > frame_graph_max_bs``.
        """
        reqs = getattr(batch, "reqs", None) or []
        if len(reqs) > int(getattr(self.model, "frame_graph_max_bs", 0)):
            return False
        for req in reqs:
            data = getattr(req, "_omni_data", None)
            if (
                data is not None
                and float(getattr(data, "audio_repetition_penalty", 1.0)) != 1.0
            ):
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
            data.sampling_steps = int(getattr(data, "generation_steps", 0))
            self.model._state_pool.reset_for_refill(sched_req.request_id)
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

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        pool = self.model._state_pool
        weight = self.model._decode_input_embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-TTS Local decode input_ids must contain one row id per request"
            )
        if batch_size > pool.padding_row:
            raise RuntimeError(
                "MOSS-TTS Local decode batch exceeds the staged decode-embedding "
                f"rows ({batch_size} > {pool.padding_row})"
            )
        pool_rows = [pool.acquire_row(sched_req.request_id) for sched_req in requests]
        row_tensor = torch.tensor(pool_rows, dtype=torch.long, device=weight.device)
        with torch.no_grad():
            weight[:batch_size].copy_(pool.feedback_embeds[row_tensor])

        row_ids = torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def _collect_frame(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return
        rows, end_id = self._run_frame_decode(result, requests)
        # Radix key is a capture-safe GPU hash: a device op, no host sync.
        next_text = rows[:, 0]
        next_token_ids = self._row_radix_token_ids(rows, next_text, end_id)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

    def _run_frame_decode(self, result: Any, requests: list):
        """GPU half shared by sync ``_collect_frame`` and async
        ``post_decode_launch``. Returns ``(rows, end_id)`` and does NOT publish
        ``next_token_ids``; the caller does, because the async path keeps a
        private device snapshot of the published ids for resolve to restore.
        """
        hidden_states = getattr(result.logits_output, "hidden_states", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            )
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]

        cfg = self.model.config
        device = hidden_states.device
        pool = self.model._state_pool
        pool_rows = []
        for sched_req in requests:
            rid = sched_req.request_id
            row = self.model.acquire_row(rid)
            pool_rows.append(row)
            pool.ensure_params(row, rid, sched_req.data)
        datas = [sched_req.data for sched_req in requests]
        batch_size = len(datas)
        num_channels = int(cfg.n_vq) + 1

        row_t = torch.tensor(
            pool_rows, dtype=torch.long, device=pool.feedback_embeds.device
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
        gen_steps = torch.tensor(
            [
                (
                    self._advance_sampling_position(d)
                    if i in emit_set
                    else max(
                        int(getattr(d, "sampling_steps", None) or 0),
                        int(d.generation_steps),
                    )
                )
                for i, d in enumerate(datas)
            ],
            dtype=torch.long,
            device=device,
        )
        rep_penalties = [float(d.audio_repetition_penalty) for d in datas]
        rep_histories = self._gather_rep_histories(datas, rep_penalties, device)

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
            if rep_histories is not None:
                self._apply_audio_repetition_penalty(
                    logits, rep_histories, rep_penalties, channel
                )
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels + channel + 1,
            )

        use_graph = rep_histories is None and batch_size <= getattr(
            self.model, "frame_graph_max_bs", 0
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
            emit_embeds = embeds.index_select(0, emit_index_t.to(device=embeds.device))
            pool.feedback_embeds[emit_row_t] = emit_embeds.detach().to(
                device=pool.feedback_embeds.device,
                dtype=pool.feedback_embeds.dtype,
            )
            result.moss_journal = MossTTSLocalDecodeJournal(
                rids=[requests[i].request_id for i in emit_indices],
                pool_rows=emit_pool_rows,
                rows=rows.index_select(0, emit_index_t),
            )
        # Always return rows so both the sync inline path and the async launch
        # publish next_token_ids; an all-chunked batch just attaches no journal.
        return rows, end_id

    def post_decode_launch(self, result: Any, forward_batch: Any, requests: list):
        """Async-decode GPU half of ``post_decode``: run the frame micro-decode
        (``_run_frame_decode``) and publish the device-computed radix ids, no
        host sync. Returns a private device snapshot of those ids for resolve:
        the base aliases ``next_token_ids`` onto ``output_ids``, which the next
        step overwrites in place before this step's lagged resolve, clobbering
        the stop id and silently dropping a bs=1 eos finish (4096-frame runaway).
        The clone preserves it; resolve swaps it back.
        """
        del forward_batch
        if not requests:
            return None
        rows, end_id = self._run_frame_decode(result, requests)
        next_token_ids = self._row_radix_token_ids(rows, rows[:, 0], end_id)
        result.next_token_ids = next_token_ids
        return next_token_ids.clone()

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
        s = max(
            int(getattr(data, "sampling_steps", None) or 0), int(data.generation_steps)
        )
        data.sampling_steps = s + 1
        return s

    @staticmethod
    def _gather_rep_histories(
        datas: list,
        rep_penalties: list[float],
        device: torch.device,
    ) -> list[torch.Tensor | None] | None:
        """Per-request generated-code history, only when a penalty is active.

        Upstream v1.5 applies the audio repetition penalty over each channel's
        previously *generated* frames only (the prompt's reference codes are
        excluded), so the history snapshot is taken from ``output_rows``.
        """
        if all(penalty == 1.0 for penalty in rep_penalties):
            return None
        histories: list[torch.Tensor | None] = []
        for data, penalty in zip(datas, rep_penalties):
            if penalty == 1.0 or not data.output_rows:
                histories.append(None)
                continue
            stacked = torch.stack(data.output_rows, dim=0)[:, 1:]
            histories.append(stacked.to(device=device, dtype=torch.long))
        return histories

    @staticmethod
    def _apply_audio_repetition_penalty(
        logits: torch.Tensor,
        histories: list[torch.Tensor | None],
        penalties: list[float],
        channel: int,
    ) -> None:
        """In-place penalty on fp32 logits, matching upstream order (before
        temperature scaling)."""
        vocab = logits.shape[-1]
        for row, (history, penalty) in enumerate(zip(histories, penalties)):
            if history is None or penalty == 1.0:
                continue
            tokens = torch.unique(history[:, channel])
            tokens = tokens[(tokens >= 0) & (tokens < vocab)]
            if tokens.numel() == 0:
                continue
            scores = logits[row, tokens]
            logits[row, tokens] = torch.where(
                scores < 0, scores * penalty, scores / penalty
            )

    @staticmethod
    def _is_chunked_request(sched_req: Any) -> bool:
        req = getattr(sched_req.data, "req", None)
        return req is not None and getattr(req, "is_chunked", 0) > 0

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
        journal = getattr(result, "moss_journal", None)
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
        for i, sched_req in enumerate(expected_reqs):
            # Overrun: a request finished or retracted in a PRIOR step is still
            # in this lagged resolve batch; its wasted frame must not reach
            # output_rows / the vocoder. No-op on the sync path.
            req = sched_req.data.req
            if req is not None:
                finished_fn = getattr(req, "finished", None)
                if (callable(finished_fn) and finished_fn()) or bool(
                    getattr(req, "is_retracted", False)
                ):
                    continue
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == end_id:
                continue
            sched_req.data.output_rows.append(journal.rows[i])
