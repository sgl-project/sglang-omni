# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.types import RequestOutput


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._has_pending_code_step = False
        self._row_ids_cache: torch.Tensor | None = None
        self._mask_last_sampled: torch.Tensor | None = None
        self._mask_prep_rids: list | None = None
        self._mask_rep_active = False
        self._mask_sup_active = False

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        self.model.prepare_decode_buffers(requests)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(
            forward_batch,
            input_embeds,
        )

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
        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(forward_batch, requests)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_codes(result, forward_batch, schedule_batch, requests)

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def _sample_next_token_ids(
        self,
        logits_output: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> Any:
        self._install_semantic_sampling_seeds(forward_batch, requests)
        next_token_ids = super()._sample_next_token_ids(
            logits_output,
            forward_batch,
            schedule_batch,
            requests,
        )
        if isinstance(next_token_ids, torch.Tensor):
            self._mask_last_sampled = next_token_ids
        else:
            self._mask_last_sampled = None
        return next_token_ids

    # ------------------------------------------------------------------
    # Mask-based logit shaping (device-resident, replaces the per-step
    # host index building in the base helpers for this model)
    # ------------------------------------------------------------------

    def _mask_fingerprint(self, requests: list) -> list | None:
        rids = []
        for sched_req in requests:
            rid = getattr(sched_req, "request_id", None)
            epoch = getattr(sched_req.data, "_qwen3_tts_prep_epoch", None)
            if rid is None or epoch is None:
                return None
            rids.append((rid, epoch))
        return rids

    @staticmethod
    def _every_row_grew_by_one(requests: list) -> bool:
        """True when every penalized row's history is exactly one token longer.

        Rows at penalty 1.0 are exempt: their mask bits never reach the logits.
        """
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            if float(req.sampling_params.repetition_penalty) == 1.0:
                continue
            seen_len = getattr(data, "_rep_seen_len", None)
            output_ids = req.output_ids
            if seen_len is None or not output_ids:
                return False
            if len(output_ids) != seen_len + 1:
                return False
        return True

    def _ensure_masks(self, batch_size: int, vocab: int, device: Any) -> None:
        masks = getattr(self, "_shape_masks", None)
        if (
            masks is not None
            and masks[0].shape[0] >= batch_size
            and masks[0].shape[1] == vocab
            and masks[0].device == device
        ):
            return
        rows = max(batch_size, 64)
        self._shape_masks = (
            torch.zeros(rows, vocab, dtype=torch.bool, device=device),
            torch.zeros(rows, vocab, dtype=torch.bool, device=device),
            torch.ones(rows, 1, dtype=torch.float32, device=device),
        )
        self._mask_prep_rids = None

    def _rebuild_masks(self, requests: list, vocab: int, device: Any) -> None:
        rep_mask, sup_mask, pen_col = self._shape_masks
        batch_size = len(requests)
        rep_mask[:batch_size] = False
        sup_mask[:batch_size] = False
        rep_rows: list[int] = []
        rep_toks: list[int] = []
        penalties = [1.0] * batch_size
        sup_rows: list[int] = []
        sup_toks: list[int] = []
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            penalty = float(req.sampling_params.repetition_penalty)
            penalties[row_idx] = penalty
            output_ids = req.output_ids
            if penalty != 1.0 and output_ids:
                seen = ModelRunner._rep_penalty_unique_tokens(data, output_ids, vocab)
                rep_rows.extend([row_idx] * len(seen))
                rep_toks.extend(seen)
            suppress_tokens = data.suppress_tokens
            if not suppress_tokens:
                suppress_tokens = getattr(req, "_codec_suppress_tokens", None)
            if suppress_tokens:
                for token_id in suppress_tokens:
                    tok = int(token_id)
                    if 0 <= tok < vocab:
                        sup_rows.append(row_idx)
                        sup_toks.append(tok)
        if rep_rows:
            pairs = torch.tensor(rep_rows + rep_toks, dtype=torch.long, device=device)
            rep_mask[pairs[: len(rep_rows)], pairs[len(rep_rows) :]] = True
        if sup_rows:
            pairs = torch.tensor(sup_rows + sup_toks, dtype=torch.long, device=device)
            sup_mask[pairs[: len(sup_rows)], pairs[len(sup_rows) :]] = True
        pen_col[:batch_size, 0] = torch.tensor(
            penalties, dtype=torch.float32, device=device
        )
        self._mask_rep_active = bool(rep_rows) or any(p != 1.0 for p in penalties)
        self._mask_sup_active = bool(sup_rows)

    def _apply_repetition_penalty(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        batch_size = len(requests)
        vocab = logits.shape[1]
        self._ensure_masks(batch_size, vocab, logits.device)
        rep_mask, sup_mask, pen_col = self._shape_masks
        fingerprint = self._mask_fingerprint(requests)
        last_sampled = getattr(self, "_mask_last_sampled", None)
        if (
            fingerprint is not None
            and fingerprint == getattr(self, "_mask_prep_rids", None)
            and last_sampled is not None
            and last_sampled.shape[0] >= batch_size
            and self._every_row_grew_by_one(requests)
        ):
            # Note: (Jiaxin Deng) unchanged batch, every history exactly one token
            # longer: the only new information is each row's sampled token, so one
            # scatter replaces the full host-side index rebuild. A history that did
            # not grow by one (retract, restart) can need bits cleared, which the
            # scatter cannot do, so those steps rebuild.
            if self._mask_rep_active:
                rows = torch.arange(batch_size, device=logits.device)
                rep_mask[rows, last_sampled[:batch_size].clamp(0, vocab - 1)] = True
                for sched_req in requests:
                    data = sched_req.data
                    output_ids = sched_req.data.req.output_ids
                    if output_ids:
                        ModelRunner._rep_penalty_unique_tokens(data, output_ids, vocab)
        else:
            self._rebuild_masks(requests, vocab, logits.device)
        self._mask_prep_rids = fingerprint
        if self._mask_rep_active:
            pen = pen_col[:batch_size]
            scores = logits.to(torch.float32)
            penalized = torch.where(scores > 0, scores / pen, scores * pen)
            logits.copy_(
                torch.where(rep_mask[:batch_size], penalized, scores).to(logits.dtype)
            )

    def _apply_codec_suppress_tokens(self, logits_output: Any, requests: list) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2:
            return
        masks = getattr(self, "_shape_masks", None)
        if masks is None or not getattr(self, "_mask_sup_active", False):
            return
        logits.masked_fill_(masks[1][: len(requests)], float("-inf"))

    def _install_semantic_sampling_seeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        forward_batch.sampling_info.sampling_seed = (
            self.model._semantic_sampling_seed_tensor[:batch_size]
        )

    def _collect_codes(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._has_pending_code_step = False
        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        semantic_positions = self._sample_positions(forward_batch, layer0_codes.device)
        self.model.code_predictor_forward(
            layer0_codes,
            hidden,
            semantic_positions=semantic_positions,
        )
        # Note: (Jiaxin Deng) stage the ids into pinned host memory now so the
        # output processor's .tolist() waits on an event instead of issuing a
        # blocking pageable copy inside the decode loop.
        self._stage_token_ids(result, result.next_token_ids)
        self._has_pending_code_step = True

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        if not self._has_pending_code_step:
            return
        self._has_pending_code_step = False
        eos_id = int(self.model.config.codec_eos_token_id)
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        batch_size = len(scheduler_output.requests)
        codes_snap = self.model._output_codes[:batch_size].detach().clone()
        embeds_snap = self.model._output_embeds[:batch_size].detach().clone()
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            code_chunk = codes_snap[row_idx]
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.latest_stream_code_chunk = code_chunk
            sched_req.data.pending_feedback_queue.append(embeds_snap[row_idx])

    def _sample_positions(
        self, forward_batch: Any, device: torch.device
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        if is_decode:
            positions = getattr(forward_batch, "positions", None)
            if positions is not None:
                return positions.to(device=device, dtype=torch.long)

        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            return (seq_lens.to(device=device, dtype=torch.long) - 1).clamp_min(0)

        positions = getattr(forward_batch, "positions", None)
        if positions is not None:
            return positions.to(device=device, dtype=torch.long)

        raise RuntimeError("Qwen3-TTS subtalker sampling requires semantic positions")

    def _write_feedback_buffers(self, forward_batch: Any, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        decode_feedback_embedding = self.model._decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        row_ids = self._decode_row_ids(batch_size, input_ids)
        rows = []

        for row_idx, sched_req in enumerate(requests):
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=decode_feedback_embedding.weight.device,
                dtype=decode_feedback_embedding.weight.dtype,
            )
            if combined is None:
                token_id = input_ids[row_idx : row_idx + 1].to(
                    device=decode_feedback_embedding.weight.device
                )
                combined = self.model.get_input_embeddings()(token_id).reshape(-1)
            QwenTalkerModelRunner._append_decode_input_history(sched_req.data, combined)
            rows.append(combined)
        with torch.no_grad():
            torch.stack(rows, dim=0, out=decode_feedback_embedding.weight[:batch_size])
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)

    def _decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = getattr(self, "_row_ids_cache", None)
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self._row_ids_cache = cached
        return cached[:batch_size]

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            piece = QwenTalkerModelRunner._projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=req_len,
                device=forward_batch.input_ids.device,
            )
            if piece is None or int(piece.shape[0]) != req_len:
                have = 0 if piece is None else int(piece.shape[0])
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {have} rows, need {req_len}"
                )
            pieces.append(piece)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
