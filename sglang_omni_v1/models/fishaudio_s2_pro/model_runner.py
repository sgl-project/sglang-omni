# SPDX-License-Identifier: Apache-2.0
"""Fish Audio S2-Pro model runner built on the phase-aware AR base runner."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni_v1.model_runner.base import ModelRunner


class FishS2ProModelRunner(ModelRunner):
    """Fish TTS runner with unified forward-owned decode and persistent buffers."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._semantic_begin_id = int(self.model._semantic_begin_id)
        self._semantic_end_id = int(self.model._semantic_end_id)

    def prepare_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        if input_embeds is not None:
            forward_batch.input_embeds = input_embeds
        return None

    def prepare_decode(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        input_ids = forward_batch.input_ids
        batch_size = input_ids.shape[0]
        is_semantic = (input_ids >= self._semantic_begin_id) & (
            input_ids <= self._semantic_end_id
        )
        self.model._vq_mask[:batch_size].copy_(is_semantic)

        for row_idx, sched_req in enumerate(requests):
            if not bool(is_semantic[row_idx].item()):
                continue
            last_codes = sched_req.data.last_codebook_values
            if last_codes is None:
                continue
            self.model._vq_codes[row_idx].copy_(
                last_codes.to(
                    device=self.model._vq_codes.device,
                    dtype=self.model._vq_codes.dtype,
                )
            )
        return None

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._collect_step_outputs(result, requests)

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
        batch_size = len(requests)
        if batch_size == 0:
            return

        result.next_token_ids = self.model._output_semantic_ids[:batch_size].clone()

        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if req.is_chunked > 0:
                continue

            codes = self.model._output_codes[row_idx].unsqueeze(-1).clone()
            data.last_codebook_values = codes[1:, 0].clone()
            data.previous_semantic_tokens.append(int(codes[0, -1].item()))
            data.output_codes.append(codes)
