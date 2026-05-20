# SPDX-License-Identifier: Apache-2.0
"""Higgs TTS model runner — phase-aware AR base-runner subclass.

- ``prepare_prefill``: run the model's fused multi-codebook embedding on each
  request's delayed ref codes inline, paste the result at the ``-100``
  placeholder positions, and set ``forward_batch.input_embeds``; also
  propagate ``req_ids`` so :class:`HiggsTTSModel.forward` can route per-row
  slot lookups.
- ``prepare_decode``: just propagate ``req_ids``. The model itself rebuilds
  the per-step embed via ``last_codes`` inside its ``forward``.
- ``post_prefill`` / ``post_decode``: read each request's newly emitted
  multi-codebook row from ``model._slots[req_id].output_codes[-1]``,
  append to ``data.output_codes``, and overwrite ``result.next_token_ids``
  with codebook-0 so the base skips its own (text-vocab) sampler.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.higgs_tts.text_tokenizer import AUDIO_PLACEHOLDER_ID

logger = logging.getLogger(__name__)


class HiggsTTSModelRunner(ModelRunner):
    """ModelRunner for :class:`HiggsTTSModel`."""

    def prepare_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def prepare_decode(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        return None

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        device = input_ids.device
        embed_tokens = self.model.backbone.model.embed_tokens
        fused_embed = self.model.multimodal_embedding.modality_embedding_0

        # embed_tokens would OOB on -100; embed 0 first, overwrite placeholders below.
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

    def _collect_step_outputs(self, result: Any, requests: list) -> None:
        """Pull per-request newly emitted codes from model slots into
        ``data.output_codes`` and overwrite ``result.next_token_ids`` with
        codebook-0 so the base runner skips its text-vocab sampler.
        """
        batch_size = len(requests)
        if batch_size == 0:
            return

        model = self.model
        cb0_per_row: list[int] = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            slot = model._slots.get(sched_req.request_id)
            if req.is_chunked > 0 or slot is None or not slot.output_codes:
                cb0_per_row.append(0)
                continue
            codes_N = slot.output_codes[-1]
            data.output_codes.append(codes_N.detach().cpu().clone())
            data.generation_done = bool(slot.sampler.generation_done)
            cb0_per_row.append(int(codes_N[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )


__all__ = ["HiggsTTSModelRunner"]
