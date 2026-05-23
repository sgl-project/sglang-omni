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

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.higgs_tts.model import _flat_sampling_attr
from sglang_omni.models.higgs_tts.sampler import K_MAX
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
        self._populate_cg_buffers(forward_batch, requests)
        return None

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._collect_step_outputs_cg(result, forward_batch, requests)

    def _populate_cg_buffers(self, forward_batch, requests) -> None:
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
        """Scatter shadow state back into the pool and append per-request
        codes from the CG output buffers.
        """
        if len(requests) == 0:
            return
        model = self.model
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )

        rows_t = model._cg_row_indices[:n_real]
        pool = model._sampler_pool
        pool.delay_count[rows_t] = model._cg_active_delay_count[:n_real]
        pool.eoc_countdown[rows_t] = model._cg_active_eoc_countdown[:n_real]
        pool.generation_done[rows_t] = model._cg_active_generation_done[:n_real]
        pool.last_codes[rows_t] = model._cg_active_last_codes[:n_real]

        was_done_cpu = model._cg_was_done[:n_real].cpu().tolist()
        codes_BN_cpu = model._cg_codes_BN[:n_real].detach().cpu().clone()
        gen_done_after_cpu = model._cg_active_generation_done[:n_real].cpu().tolist()
        cb0_per_row: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if req.is_chunked > 0:
                cb0_per_row.append(0)
                continue
            if was_done_cpu[b]:
                cb0_per_row.append(0)
                continue
            codes_N = codes_BN_cpu[b]
            data.output_codes.append(codes_N.to(torch.long))
            data.generation_done = bool(gen_done_after_cpu[b])
            cb0_per_row.append(int(codes_N[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
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

    def _collect_step_outputs(self, result: Any, requests: list) -> None:
        """Pull per-request newly emitted codes from the model into
        ``data.output_codes`` and overwrite ``result.next_token_ids``
        with codebook-0 so the base runner skips its text-vocab sampler.
        """
        batch_size = len(requests)
        if batch_size == 0:
            return

        model = self.model
        cb0_per_row: list[int] = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rid = sched_req.request_id
            row = model._rid_to_row.get(rid)
            codes_log = model._output_codes.get(rid)
            if req.is_chunked > 0 or row is None or not codes_log:
                cb0_per_row.append(0)
                continue
            codes_N = codes_log[-1]
            data.output_codes.append(codes_N.detach().cpu().clone())
            data.generation_done = bool(model._sampler_pool.generation_done[row].item())
            cb0_per_row.append(int(codes_N[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )


__all__ = ["HiggsTTSModelRunner"]
