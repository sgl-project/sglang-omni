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
        self._populate_cg_buffers(forward_batch, requests)
        return None

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del schedule_batch
        self._collect_step_outputs_cg(result, forward_batch, requests)

    def _populate_cg_buffers(self, forward_batch, requests) -> None:
        """Write per-row decode state (row indices, sampling params) into
        the model's CUDA-Graph buffers so the captured forward can read
        them without any Python control flow.

        Padding rows (``forward_batch.batch_size > len(requests)``) point
        at the model's reserved padding row; their sampler state is
        reset each step so they never bleed into real rows.
        """
        model = self.model
        bs = int(forward_batch.batch_size)
        n_real = len(requests)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )

        # Always reset the padding row before each decode — keeps its
        # state machine inert so the captured graph can't poison real rows.
        model._sampler_pool.reset_row(model._padding_row)

        rows_py: list[int] = [
            model.acquire_row(req.request_id) for req in requests
        ]
        # Pad with the reserved padding row.
        rows_py.extend([model._padding_row] * (bs - n_real))
        model._cg_row_indices[:bs] = torch.tensor(
            rows_py, dtype=torch.long, device=model._cg_row_indices.device
        )

        # Per-row sampling params: pull off sglang sampling_info if available.
        temps, top_ps, top_k = self._extract_decode_sampling_params(
            forward_batch, n_real
        )
        # Pad to bs with safe defaults (temp=1.0, top_p=1.0).
        temps.extend([1.0] * (bs - n_real))
        top_ps.extend([1.0] * (bs - n_real))
        model._cg_temperature[:bs] = torch.tensor(
            temps, dtype=torch.float32, device=model._cg_temperature.device
        )
        model._cg_top_p[:bs] = torch.tensor(
            top_ps, dtype=torch.float32, device=model._cg_top_p.device
        )
        model._cg_top_k = top_k

    @staticmethod
    def _extract_decode_sampling_params(forward_batch, n_real: int):
        """Pull per-row temperature / top_p (lists) + uniform top_k (scalar
        or None) off sglang's ``sampling_info``. Falls back to safe defaults
        when missing or shapes are unexpected.
        """
        sampling_info = getattr(forward_batch, "sampling_info", None)
        if sampling_info is None or n_real == 0:
            return ([1.0] * n_real, [1.0] * n_real, None)

        def _flat_list(attr: str):
            val = getattr(sampling_info, attr, None)
            if val is None:
                return None
            if hasattr(val, "cpu"):
                # sglang stores some of these as [B, 1] — flatten so we
                # always get a flat per-row list.
                return val.detach().cpu().flatten().tolist()
            return list(val)

        temps_raw = _flat_list("temperatures") or [1.0] * n_real
        top_ps_raw = _flat_list("top_ps") or [1.0] * n_real
        top_ks_raw = _flat_list("top_ks")

        temps = [float(t) for t in temps_raw[:n_real]]
        top_ps = [float(t) for t in top_ps_raw[:n_real]]
        top_k: int | None = None
        if top_ks_raw is not None:
            distinct = {int(t) for t in top_ks_raw[:n_real]}
            if len(distinct) > 1:
                raise ValueError(
                    f"HiggsTTSModelRunner requires uniform top_k across the "
                    f"decode batch; got {distinct}"
                )
            tk = next(iter(distinct))
            top_k = tk if tk > 0 else None
        return temps, top_ps, top_k

    def _collect_step_outputs_cg(
        self, result: Any, forward_batch: Any, requests: list
    ) -> None:
        """Read decode-step outputs out of the model's CG buffers and
        append them to per-request ``output_codes`` logs.

        Mirrors :meth:`_collect_step_outputs` but pulls from the pool /
        CG buffers rather than from the now-removed model._output_codes
        decode path. Padding rows are skipped because we only iterate
        the real ``requests`` slice.
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

        was_done_cpu = model._cg_was_done[:n_real].cpu().tolist()
        codes_BN_cpu = model._cg_codes_BN[:n_real].detach().cpu().clone()
        gen_done_after_cpu = (
            model._sampler_pool.generation_done[
                model._cg_row_indices[:n_real]
            ]
            .cpu()
            .tolist()
        )
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
