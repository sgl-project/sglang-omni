# SPDX-License-Identifier: Apache-2.0
"""Frame-step ModelRunner for CSM TTS.

Prefill embed overlay, decode-buffer collect, and stream emission.
"""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.csm_tts.model import flat_sampling_attr
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
    """ModelRunner for CsmTTSModel."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None
        self._vocoder_target = "vocoder"

    def set_stream_outbox(self, outbox: Any) -> None:
        """Wire the OmniScheduler outbox used by emit_code_chunk."""
        self._outbox = outbox

    @staticmethod
    def is_chunked(req: Any) -> bool:
        """Req.is_chunked if present, else inflight_middle_chunks."""
        raw = getattr(req, "is_chunked", None)
        if raw is not None:
            return int(raw) > 0
        return int(getattr(req, "inflight_middle_chunks", 0) or 0) > 0

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        depth_temps, depth_top_ks = self.extract_depth_sampling_params(requests)
        forward_batch.csm_depth_params = list(zip(depth_temps, depth_top_ks))
        # Prefill CUDA-graph admission rejects ForwardBatch.input_embeds.
        input_embeds = self.build_prefill_input_embeds(forward_batch, requests)
        if input_embeds is not None:
            attach_omni_prefill_inputs(
                forward_batch, OmniPrefillInputs(input_embeds=input_embeds)
            )

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        self.collect_step_outputs(result, requests)

    def build_prefill_input_embeds(
        self, forward_batch: Any, requests: list
    ) -> torch.Tensor | None:
        """Overlay context-frame embeds at 128002 and the zeros frame at 128003."""
        if not any(
            getattr(sched_req.data, "context_codes", None) for sched_req in requests
        ):
            return None

        input_ids = forward_batch.input_ids
        device = input_ids.device
        embed_tokens = self.model.backbone.model.embed_tokens
        frame_embedding = self.model.frame_embedding

        text_embeds = embed_tokens(input_ids)
        eos_frame_embed: torch.Tensor | None = None

        offset = 0
        for sched_req in requests:
            data = sched_req.data
            end = offset + int(data.req.extend_range.length)
            chunk_ids = input_ids[offset:end]

            audio_idx = (chunk_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
            n_placeholders = int(audio_idx.numel())
            if n_placeholders > 0:
                codes = self.concat_context_codes(data, device)
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
    def concat_context_codes(data: Any, device: torch.device) -> torch.Tensor | None:
        """Concatenate per-segment context codes in prompt order."""
        segments = getattr(data, "context_codes", None) or []
        mats = []
        for seg in segments:
            t = seg if isinstance(seg, torch.Tensor) else torch.as_tensor(seg)
            mats.append(t.reshape(-1, NUM_CODEBOOKS).to(torch.long))
        if not mats:
            return None
        return torch.cat(mats, dim=0).to(device)

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
        self.populate_cg_buffers(forward_batch, requests, is_lookahead=is_lookahead)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        self.collect_step_outputs_cg(result, forward_batch, requests)

    def populate_cg_buffers(
        self, forward_batch: Any, requests: list, *, is_lookahead: bool = False
    ) -> None:
        """Fill decode buffers. Extra rows reuse the reset padding slot."""
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
            # Async lookahead can still launch a finished row; park it on padding.
            rows_t_real = model._cg_row_indices[:n_real]
            done = model._sampler_pool.generation_done[rows_t_real]
            model._cg_row_indices[:n_real] = torch.where(
                done, torch.full_like(rows_t_real, model._padding_row), rows_t_real
            )

        temps, top_ps, top_ks = self.extract_decode_sampling_params(
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

        depth_temps, depth_top_ks = self.extract_depth_sampling_params(requests)
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
    def extract_decode_sampling_params(forward_batch: Any, n_real: int) -> Any:
        """cb0 temperature / top_p / top_k from sampling_info."""
        sampling_info = getattr(forward_batch, "sampling_info", None)
        if sampling_info is None or n_real == 0:
            return ([1.0] * n_real, [1.0] * n_real, [None] * n_real)

        temps_raw = flat_sampling_attr(sampling_info, "temperatures") or [1.0] * n_real
        top_ps_raw = flat_sampling_attr(sampling_info, "top_ps") or [1.0] * n_real
        top_ks_raw = flat_sampling_attr(sampling_info, "top_ks")

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
    def extract_depth_sampling_params(requests: list) -> tuple[list[float], list[int]]:
        """Depth temperature / top_k per request."""
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

    def collect_step_outputs_cg(
        self, result: Any, forward_batch: Any, requests: list
    ) -> None:
        """Pack GPU staging, copy to host, then collect codes."""
        if len(requests) == 0:
            return
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )
        staging = self.decode_pack_gpu(n_real)
        combined_cpu = staging[:n_real].cpu()  # one blocking D2H (sync path)
        self.decode_collect_host(
            combined_cpu,
            result,
            requests,
            next_token_device=result.logits_output.next_token_logits.device,
        )

    def decode_pack_gpu(self, n_real: int) -> torch.Tensor:
        """Scatter active buffers into the pool and pack the D2H staging row."""
        model = self.model
        rows_t = model._cg_row_indices[:n_real]
        pool = model._sampler_pool
        pool.generation_done[rows_t] = model._cg_active_generation_done[:n_real]
        pool.last_codes[rows_t] = model._cg_active_last_codes[:n_real]
        pool.frames_emitted[rows_t] += (~model._cg_was_done[:n_real]).to(torch.int32)

        num_codebooks = model._cg_codes_BN.shape[1]
        staging = model._cg_collect_staging
        staging[:n_real, :num_codebooks] = model._cg_codes_BN[:n_real]
        staging[:n_real, num_codebooks] = model._cg_was_done[:n_real]
        staging[:n_real, num_codebooks + 1] = model._cg_active_generation_done[:n_real]
        return staging

    def decode_collect_host(
        self,
        staging_host: torch.Tensor,
        result: Any,
        requests: list,
        *,
        next_token_device: torch.device | None = None,
    ) -> None:
        """Collect one decode step from the staged host snapshot."""
        model = self.model
        num_codebooks = model._cg_codes_BN.shape[1]
        codes_BN_cpu = staging_host[:, :num_codebooks]
        was_done_cpu = staging_host[:, num_codebooks].bool().tolist()
        gen_done_after_cpu = staging_host[:, num_codebooks + 1].bool().tolist()
        cb0_per_row: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req
            if self.is_chunked(req):
                cb0_per_row.append(0)
                continue
            # Skip overrun / length-finished rows (was_done is EOS-only).
            if req.finished():
                cb0_per_row.append(0)
                continue
            if was_done_cpu[b]:
                cb0_per_row.append(0)
                continue
            codes_32 = codes_BN_cpu[b].to(torch.long).clone()
            data.output_codes.append(codes_32)
            data.generation_done = bool(gen_done_after_cpu[b])
            self.mark_sampler_finished(req, data.generation_done)
            if not data.generation_done:
                self.emit_code_chunk(sched_req, codes_32)
            cb0_per_row.append(int(codes_32[0].item()))

        if next_token_device is None:
            result.next_token_ids = torch.tensor(cb0_per_row, dtype=torch.long)
        else:
            result.next_token_ids = torch.tensor(
                cb0_per_row, dtype=torch.long, device=next_token_device
            )

    def collect_step_outputs(self, result: Any, requests: list) -> None:
        """Prefill collect for frame 0. next_token_ids is codebook-0."""
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
            if self.is_chunked(req) or row is None or not codes_log or req.finished():
                cb0_per_row.append(0)
                continue
            codes_32 = codes_log[-1].detach().cpu().clone()
            data.output_codes.append(codes_32)
            data.generation_done = bool(model._sampler_pool.generation_done[row].item())
            self.mark_sampler_finished(req, data.generation_done)
            if not data.generation_done:
                self.emit_code_chunk(sched_req, codes_32)
            cb0_per_row.append(int(codes_32[0].item()))

        result.next_token_ids = torch.tensor(
            cb0_per_row,
            dtype=torch.long,
            device=result.logits_output.next_token_logits.device,
        )

    def post_decode_launch(
        self, result: Any, forward_batch: Any, requests: list
    ) -> Any:
        """Async GPU half: pack staging and copy to a pinned host buffer."""
        if len(requests) == 0:
            return None
        n_real = len(requests)
        bs = int(forward_batch.batch_size)
        if bs < n_real:
            raise ValueError(
                f"forward_batch.batch_size ({bs}) < len(requests) ({n_real})"
            )
        staging = self.decode_pack_gpu(n_real)
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
        """Async host half: collect from the pinned snapshot."""
        del forward_batch, schedule_batch
        if len(requests) == 0:
            return
        n_real = len(requests)
        self.decode_collect_host(
            host_buf[:n_real],
            result,
            requests,
            next_token_device=None,
        )

    def emit_code_chunk(self, sched_req: Any, codes_32: torch.Tensor) -> None:
        """Stream one non-EOS frame to the vocoder."""
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
    def mark_sampler_finished(req: Any, generation_done: bool) -> None:
        """Mark the request finished so upstream can release KV."""
        if generation_done and req.finished_reason is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(CODEBOOK_EOS)
