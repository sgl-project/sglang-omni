# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) model runner for OmniScheduler."""

from __future__ import annotations

import bisect
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
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
        if self._forward_sample_enabled():
            self._prepare_forward_sample_inputs(forward_batch, requests)

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
                pool_row = self.model._state_pool.row_for(sched_req.request_id)
                if pool_row is not None:
                    self.model._state_pool.invalidate_params(sched_req.request_id)
                    self.model._state_pool.reset_row(pool_row)
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
        n_real = len(requests)
        raw_batch_size = int(getattr(forward_batch, "batch_size", n_real) or n_real)
        staging_batch_size = self._decode_staging_batch_size(raw_batch_size)
        if n_real == 0 and staging_batch_size == 0:
            return
        pool = self.model._state_pool
        weight = self.model._decode_input_embedding.weight
        if raw_batch_size < n_real:
            raise RuntimeError(
                "MOSS-TTS Local decode graph batch is smaller than the real batch "
                f"({raw_batch_size} < {n_real})"
            )
        if forward_batch.input_ids.numel() < raw_batch_size:
            raise RuntimeError(
                "MOSS-TTS Local decode input_ids must contain one row id per request"
            )
        if staging_batch_size > pool.padding_row:
            raise RuntimeError(
                "MOSS-TTS Local decode batch exceeds the staged decode-embedding "
                f"rows ({staging_batch_size} > {pool.padding_row})"
            )
        pool_rows = [pool.acquire_row(sched_req.request_id) for sched_req in requests]
        if staging_batch_size > n_real:
            pool_rows.extend([pool.padding_row] * (staging_batch_size - n_real))
        row_tensor = torch.tensor(pool_rows, dtype=torch.long, device=weight.device)
        with torch.no_grad():
            weight[:staging_batch_size].copy_(pool.feedback_embeds[row_tensor])

        row_ids = torch.arange(
            raw_batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:raw_batch_size].copy_(row_ids)

    def _decode_staging_batch_size(self, raw_batch_size: int) -> int:
        """Return the model-buffer rows SGLang may read for this decode step.

        SGLang's cuda graph runner receives the raw ForwardBatch, then pads it
        to the next captured ``cuda_graph_bs`` bucket during replay. The model
        staging buffers are outside SGLang's GraphInputBuffers, so they must be
        prefilled up to the same bucket here.
        """
        raw_batch_size = int(raw_batch_size)
        if raw_batch_size <= 0:
            return 0
        if not bool(getattr(self.model, "_moss_local_decode_graph_padding", False)):
            return raw_batch_size
        buckets = sorted(
            {
                int(bs)
                for bs in getattr(self.model, "_moss_local_decode_cuda_graph_bs", [])
                if int(bs) > 0
            }
        )
        if not buckets:
            return raw_batch_size
        idx = bisect.bisect_left(buckets, raw_batch_size)
        return buckets[idx] if idx < len(buckets) else raw_batch_size

    def _forward_sample_enabled(self) -> bool:
        return bool(getattr(self.model, "forward_sample_in_forward", False)) and all(
            hasattr(self.model, name)
            for name in (
                "_cg_text_temp",
                "_cg_text_top_p",
                "_cg_text_top_k",
                "_cg_audio_temp",
                "_cg_audio_top_p",
                "_cg_audio_top_k",
                "_cg_seeds",
                "_cg_base_positions",
                "_cg_rows",
                "_cg_feedback",
            )
        )

    def _prepare_forward_sample_inputs(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        if not requests:
            self._forward_sample_pool_rows = []
            self._forward_sample_rids = []
            return
        n_real = len(requests)
        raw_batch_size = int(getattr(forward_batch, "batch_size", n_real) or n_real)
        staging_batch_size = self._decode_staging_batch_size(raw_batch_size)
        model = self.model
        max_rows = int(model._cg_text_temp.shape[0])
        if raw_batch_size < n_real:
            raise RuntimeError(
                "MOSS-TTS Local forward-sample graph batch is smaller than the "
                f"real batch ({raw_batch_size} < {n_real})"
            )
        if staging_batch_size > max_rows:
            raise RuntimeError(
                "MOSS-TTS Local forward-sample batch exceeds staging buffers "
                f"({staging_batch_size} > {max_rows})"
            )

        pool = model._state_pool
        pool_rows = []
        for sched_req in requests:
            rid = sched_req.request_id
            row = pool.acquire_row(rid)
            pool.ensure_params(row, rid, sched_req.data)
            pool_rows.append(row)

        row_t = torch.tensor(
            pool_rows, dtype=torch.long, device=pool.feedback_embeds.device
        )
        num_channels = int(model.config.n_vq) + 1
        generation_steps = torch.tensor(
            [int(sched_req.data.generation_steps) for sched_req in requests],
            dtype=torch.long,
            device=model._cg_base_positions.device,
        )

        with torch.no_grad():
            if n_real:
                model._cg_text_temp[:n_real].copy_(
                    pool.text_temp[row_t].to(model._cg_text_temp.device)
                )
                model._cg_text_top_p[:n_real].copy_(
                    pool.text_top_p[row_t].to(model._cg_text_top_p.device)
                )
                model._cg_text_top_k[:n_real].copy_(
                    pool.text_top_k[row_t].to(model._cg_text_top_k.device)
                )
                model._cg_audio_temp[:n_real].copy_(
                    pool.audio_temp[row_t].to(model._cg_audio_temp.device)
                )
                model._cg_audio_top_p[:n_real].copy_(
                    pool.audio_top_p[row_t].to(model._cg_audio_top_p.device)
                )
                model._cg_audio_top_k[:n_real].copy_(
                    pool.audio_top_k[row_t].to(model._cg_audio_top_k.device)
                )
                model._cg_seeds[:n_real].copy_(
                    pool.seeds[row_t].to(model._cg_seeds.device)
                )
                model._cg_base_positions[:n_real].copy_(generation_steps * num_channels)
            if staging_batch_size > n_real:
                model._cg_text_temp[n_real:staging_batch_size].fill_(1.0)
                model._cg_text_top_p[n_real:staging_batch_size].fill_(1.0)
                model._cg_text_top_k[n_real:staging_batch_size].fill_(50)
                model._cg_audio_temp[n_real:staging_batch_size].fill_(1.0)
                model._cg_audio_top_p[n_real:staging_batch_size].fill_(1.0)
                model._cg_audio_top_k[n_real:staging_batch_size].fill_(25)
                model._cg_seeds[n_real:staging_batch_size].zero_()
                model._cg_base_positions[n_real:staging_batch_size].zero_()

        self._forward_sample_pool_rows = pool_rows
        self._forward_sample_rids = [sched_req.request_id for sched_req in requests]

    def _use_forward_sample_path(
        self,
        result: Any,
        forward_batch: Any,
        requests: list,
    ) -> bool:
        if not self._forward_sample_enabled() or not requests:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        if not is_decode:
            return False
        if getattr(result, "logits_output", None) is None:
            return False
        return all(
            float(getattr(sched_req.data, "audio_repetition_penalty", 1.0)) == 1.0
            for sched_req in requests
        )

    def _collect_frame(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if self._use_forward_sample_path(result, forward_batch, requests):
            self._collect_frame_from_forward_sample(result, schedule_batch, requests)
        else:
            self._collect_frame_legacy(result, forward_batch, schedule_batch, requests)

    def _collect_frame_from_forward_sample(
        self,
        result: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        # Sampling already ran inside MossTTSLocalSGLangModel.forward() via
        # _decode_frame_graphable(); collection only snapshots the fixed
        # staging buffers and updates scheduler/pool state outside the graph.
        if not requests:
            return
        n_real = len(requests)
        current_rids = [sched_req.request_id for sched_req in requests]
        staged_rids = list(getattr(self, "_forward_sample_rids", []))
        pool_rows = list(getattr(self, "_forward_sample_pool_rows", []))
        if staged_rids[:n_real] != current_rids:
            raise RuntimeError(
                "MOSS-TTS Local forward-sample request alignment broken: "
                f"{staged_rids[:n_real]} != {current_rids}"
            )
        if len(pool_rows) < n_real:
            raise RuntimeError(
                "MOSS-TTS Local forward-sample pool row staging is incomplete: "
                f"{len(pool_rows)} < {n_real}"
            )

        model = self.model
        if (
            int(model._cg_rows.shape[0]) < n_real
            or int(model._cg_feedback.shape[0]) < n_real
        ):
            raise RuntimeError("MOSS-TTS Local forward-sample output buffers too small")
        cfg = model.config
        pool = model._state_pool
        rows = model._cg_rows[:n_real].clone()
        feedback = model._cg_feedback[:n_real].clone()
        next_text = rows[:, 0]
        end_id = int(cfg.audio_end_token_id)
        next_token_ids = self._row_radix_token_ids(rows, next_text, end_id)
        result.next_token_ids = next_token_ids
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
        emit_row_t = torch.tensor(
            emit_pool_rows, dtype=torch.long, device=pool.feedback_embeds.device
        )
        emit_feedback = feedback.index_select(0, emit_index_t.to(feedback.device))
        pool.feedback_embeds[emit_row_t] = emit_feedback.detach().to(
            device=pool.feedback_embeds.device,
            dtype=pool.feedback_embeds.dtype,
        )
        result.moss_journal = MossTTSLocalDecodeJournal(
            rids=[requests[i].request_id for i in emit_indices],
            pool_rows=emit_pool_rows,
            rows=rows.index_select(0, emit_index_t),
        )

    def _collect_frame_legacy(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return
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
        gen_steps = torch.tensor(
            [int(d.generation_steps) for d in datas], dtype=torch.long, device=device
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

        next_token_ids = self._row_radix_token_ids(rows, next_text, end_id)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids
        if embeds is None:
            embeds = self.model._prepare_multi_modal_inputs(
                rows.to(device=self.model.device)
            )
        emit_indices = [
            i
            for i, sched_req in enumerate(requests)
            if not self._is_chunked_request(sched_req)
        ]
        if not emit_indices:
            return

        emit_index_t = torch.tensor(emit_indices, dtype=torch.long, device=rows.device)
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
        """
        from sglang_omni.models.moss_tts.request_builders import build_row_cache_key_ids

        # <|endoftext|> 151643 opens the special/control id band.
        hash_space = 151643
        hashed = torch.tensor(
            build_row_cache_key_ids(rows), dtype=torch.long, device=rows.device
        )
        return torch.where(next_text == end_id, next_text, hashed % hash_space)

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
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == end_id:
                continue
            sched_req.data.output_rows.append(journal.rows[i])
