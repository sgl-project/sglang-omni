# SPDX-License-Identifier: Apache-2.0
"""Fish Audio S2-Pro model runner built on the phase-aware AR base runner."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.fishaudio_s2_pro.sglang_model import _NO_SEED
from sglang_omni.sampling.seed import resolve_row_seed


def collect_s2pro_step_outputs(
    result: Any,
    requests: list,
    *,
    output_codes: torch.Tensor,
    output_semantic_ids: torch.Tensor,
    im_end_token_id: int,
    rep_history_len: int | None = None,
) -> None:
    batch_size = len(requests)
    if batch_size == 0:
        return

    result.next_token_ids = output_semantic_ids[:batch_size].clone()
    semantic_tokens = output_semantic_ids[:batch_size].tolist()

    for row_idx, sched_req in enumerate(requests):
        data = sched_req.data
        if data.req.is_chunked > 0:
            continue

        semantic_token = semantic_tokens[row_idx]
        if semantic_token == im_end_token_id:
            continue

        codes = output_codes[row_idx].unsqueeze(-1).clone()
        data.last_codebook_values = codes[1:, 0].clone()
        data.previous_semantic_tokens.append(semantic_token)
        if rep_history_len is not None:
            _append_semantic_history(
                data, output_semantic_ids[row_idx], rep_history_len
            )
        data.output_codes.append(codes)
        data.latest_stream_code_chunk = codes


def _append_semantic_history(data: Any, token: torch.Tensor, history_len: int) -> None:
    history = data.semantic_history_tokens
    if (
        history is None
        or history.device != token.device
        or history.shape[0] != history_len
    ):
        history = torch.zeros(history_len, dtype=torch.long, device=token.device)
        data.semantic_history_tokens = history
        data.semantic_history_count = 0

    count = int(data.semantic_history_count)
    if count < history_len:
        history[count].copy_(token)
    else:
        history[:-1].copy_(history[1:].clone())
        history[-1].copy_(token)
    data.semantic_history_count = count + 1


class FishS2ProModelRunner(ModelRunner):
    """Fish TTS runner with unified forward-owned decode and persistent buffers."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._semantic_begin_id = int(self.model._semantic_begin_id)
        self._semantic_end_id = int(self.model._semantic_end_id)
        self._im_end_token_id = int(self.model._im_end_token_id)

    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._sync_decode_state(requests)
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        if input_embeds is not None:
            forward_batch.input_embeds = input_embeds

    def before_decode(
        self,
        forward_batch,
        schedule_batch,
        requests,
        *,
        is_lookahead: bool = False,
    ):
        del is_lookahead
        del schedule_batch
        input_ids = forward_batch.input_ids
        batch_size = input_ids.shape[0]
        is_semantic = (input_ids >= self._semantic_begin_id) & (
            input_ids <= self._semantic_end_id
        )
        self.model._vq_mask[:batch_size].copy_(is_semantic)

        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            self._sync_decode_row_state(row_idx, data)

            last_codes = data.last_codebook_values
            if last_codes is None:
                continue
            self.model._vq_codes[row_idx].copy_(
                last_codes.to(
                    device=self.model._vq_codes.device,
                    dtype=self.model._vq_codes.dtype,
                )
            )

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def _sync_decode_state(self, requests: list) -> None:
        for row_idx, sched_req in enumerate(requests):
            self._sync_decode_row_state(row_idx, sched_req.data)

    def _sync_decode_row_state(self, row_idx: int, data: Any) -> None:
        self.model._sampling_temperature[row_idx] = data.temperature
        self.model._sampling_top_p[row_idx] = data.top_p
        self.model._sampling_top_k[row_idx] = data.top_k
        self.model._sampling_rep_penalty[row_idx] = data.repetition_penalty
        self.model._ras_temperature[row_idx] = data.ras_temperature
        self.model._ras_top_p[row_idx] = data.ras_top_p
        self.model._sampling_seeds[row_idx] = (
            _NO_SEED if data.seed is None else resolve_row_seed(data.seed)
        )
        # semantic_history_count is the uncapped per-request AR step (pre-step).
        self.model._step_count[row_idx] = int(data.semantic_history_count)

        history_len = self.model._rep_history_len
        history = data.semantic_history_tokens
        if history is not None:
            self.model._prev_tokens[row_idx].copy_(
                history.to(
                    device=self.model._prev_tokens.device,
                    dtype=self.model._prev_tokens.dtype,
                )
            )
            self.model._prev_token_count[row_idx] = min(
                int(data.semantic_history_count), history_len
            )
        else:
            self.model._prev_tokens[row_idx].zero_()
            self.model._prev_token_count[row_idx] = 0

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Fish prefill expects tensor input_ids")

        device = input_ids.device
        text_embeds = self.model.get_embed_tokens()(input_ids)
        offset = 0

        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_input_len)

            if (
                data.vq_mask_tokens is None
                or data.vq_parts is None
                or len(data.vq_parts) == 0
            ):
                offset += req_len
                continue

            vq_mask = data.vq_mask_tokens.to(device=device)
            if vq_mask.dim() == 2:
                vq_mask = vq_mask.squeeze(0)

            prefix_len = len(req.prefix_indices)
            mask_slice = vq_mask[prefix_len : prefix_len + req_len]
            if not bool(mask_slice.any()):
                offset += req_len
                continue

            parts = [
                part.to(device=device).T for part in data.vq_parts if part.dim() == 2
            ]
            vq_parts_flat = torch.cat(parts, dim=0) if parts else None
            if vq_parts_flat is None:
                offset += req_len
                continue

            vq_before = int(vq_mask[:prefix_len].sum().item()) if prefix_len > 0 else 0
            num_vq_in_slice = int(mask_slice.sum().item())
            vq_slice = vq_parts_flat[vq_before : vq_before + num_vq_in_slice]

            req_embeds = text_embeds[offset : offset + req_len]
            vq_embeds = self.model._audio_decoder.embed_text_dim(
                req_embeds.unsqueeze(0),
                vq_slice,
                mask_slice.unsqueeze(0),
            )
            mask_indices = mask_slice.nonzero(as_tuple=True)[0] + offset
            text_embeds[mask_indices] = vq_embeds.to(text_embeds.dtype)
            offset += req_len

        return text_embeds

    def _collect_step_outputs(self, result: Any, requests: list) -> None:
        collect_s2pro_step_outputs(
            result,
            requests,
            output_codes=self.model._output_codes,
            output_semantic_ids=self.model._output_semantic_ids,
            im_end_token_id=self._im_end_token_id,
            rep_history_len=self.model._rep_history_len,
        )
