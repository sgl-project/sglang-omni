# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local model runner: multi-channel embedding feedback + vectorized
per-row sampler + depth-transformer frame decode (``model.decode_frames``)."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.types import RequestOutput

_NEG_INF = float("-inf")


class MossTTSLocalModelRunner(ModelRunner):
    """Depth-transformer frame decode + multi-channel embedding feedback."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_rows: torch.Tensor | None = None
        self._pending_embeds: torch.Tensor | None = None

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )
        return None

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead: bool = False
    ) -> None:
        del is_lookahead, schedule_batch
        self._write_decode_input_embedding(forward_batch, requests)

    def _build_prefill_input_embeds(self, forward_batch, requests) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("MOSS-TTS Local prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            current_rows = rows[prefix_len : prefix_len + req_len]
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
            device=forward_batch.input_ids.device, dtype=self.model.dtype
        )

    def _write_decode_input_embedding(self, forward_batch, requests) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        weight = self.model._decode_input_embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-TTS Local decode input_ids must contain one row id per request"
            )
        if batch_size > int(weight.shape[0]):
            raise RuntimeError(
                "MOSS-TTS Local decode batch exceeds the staged decode-embedding "
                f"rows ({batch_size} > {int(weight.shape[0])})"
            )
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size, device=weight.device))
            elif hasattr(queue, "popleft"):
                rows.append(queue.popleft())
            else:
                rows.append(queue.pop(0))
        stacked = torch.stack(rows, dim=0).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight[:batch_size].copy_(stacked)
        row_ids = torch.arange(
            batch_size, dtype=torch.long, device=forward_batch.input_ids.device
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(result, schedule_batch, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        self._collect_frame(result, schedule_batch, requests)

    def _hidden_from_result(self, result: Any) -> torch.Tensor:
        hidden_states = getattr(result.logits_output, "hidden_states", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            )
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        return hidden_states

    def _make_sampler(self, datas: list):
        device = self.model.device
        num_channels = int(self.model.channels)
        audio_pad = int(self.model.audio_pad_code)

        def t(attr, dtype):
            return torch.tensor(
                [getattr(d, attr) for d in datas], dtype=dtype, device=device
            )

        text_temp = t("text_temperature", torch.float32)
        text_top_p = t("text_top_p", torch.float32)
        text_top_k = t("text_top_k", torch.long)
        audio_temp = t("audio_temperature", torch.float32)
        audio_top_p = t("audio_top_p", torch.float32)
        audio_top_k = t("audio_top_k", torch.long)
        seeds = t("sampling_seed", torch.long)
        gen_steps = t("generation_steps", torch.long)

        # Audio repetition penalty over each codebook's history (audio channels, pre-sampling).
        audio_rep = [float(d.audio_repetition_penalty) for d in datas]
        histories: list | None = None
        if any(p != 1.0 for p in audio_rep):
            histories = []
            for d in datas:
                parts = []
                pr = getattr(d, "prompt_rows", None)
                if pr is not None and pr.numel() > 0:
                    parts.append(pr.to(dtype=torch.long, device=device))
                if d.output_rows:
                    parts.append(
                        torch.stack(d.output_rows, 0).to(
                            dtype=torch.long, device=device
                        )
                    )
                histories.append(torch.cat(parts, 0) if parts else None)

        def apply_rep(logits: torch.Tensor, channel: int) -> torch.Tensor:
            for r, pen in enumerate(audio_rep):
                if pen == 1.0 or histories[r] is None:
                    continue
                toks = torch.unique(histories[r][:, channel])
                toks = toks[(toks >= 0) & (toks < audio_pad)]
                if toks.numel() == 0:
                    continue
                s = logits[r, toks]
                logits[r, toks] = torch.where(s > 0, s / pen, s * pen)
            return logits

        def sampler(channel_idx: int, logits: torch.Tensor) -> torch.Tensor:
            logits = logits.to(torch.float32)
            if channel_idx == 0:
                temp, top_p, top_k = text_temp, text_top_p, text_top_k
            else:
                temp, top_p, top_k = audio_temp, audio_top_p, audio_top_k
                if histories is not None:
                    logits = apply_rep(logits, channel_idx)
            return self._sample_tokens(
                logits,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                seeds=seeds,
                positions=gen_steps * num_channels + channel_idx,
            )

        return sampler

    def _collect_frame(self, result, schedule_batch, requests) -> None:
        if not requests:
            return
        hidden = self._hidden_from_result(result)
        datas = [sched_req.data for sched_req in requests]
        sampler = self._make_sampler(datas)
        rows = self.model.decode_frames(hidden.to(self.model.device), sampler)

        next_token_ids = rows[:, 0].contiguous()
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids
        embeds = self.model._prepare_multi_modal_inputs(rows.to(self.model.device))
        self._pending_rows = rows
        self._pending_embeds = embeds.detach()

    def post_process_outputs(
        self, result: Any, scheduler_output: Any, outputs: dict[str, RequestOutput]
    ) -> None:
        del result
        rows = self._pending_rows
        embeds = self._pending_embeds
        self._pending_rows = None
        self._pending_embeds = None
        if rows is None or embeds is None:
            return
        eos_id = int(self.model.config.audio_end_token_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            sched_req.data.output_rows.append(rows[row_idx].detach().clone())
            sched_req.data.pending_feedback_queue.append(
                embeds[row_idx].detach().clone()
            )

    @staticmethod
    def _as_row_tensor(value, num_rows, dtype, device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=device)
        return torch.full((num_rows,), value, dtype=dtype, device=device)

    @staticmethod
    def _sample_tokens(
        logits: torch.Tensor,
        *,
        temperature,
        top_p,
        top_k,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row temp/top-k/top-p sampling; seeded (reproducible per row) with
        greedy fallback when temperature <= 0. ``logits`` is float32, masked at -inf."""
        num_rows = logits.shape[0]
        if num_rows == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)
        device = logits.device
        cls = MossTTSLocalModelRunner

        temp = cls._as_row_tensor(temperature, num_rows, torch.float32, device)
        top_p_row = cls._as_row_tensor(top_p, num_rows, torch.float32, device)
        top_k_row = cls._as_row_tensor(top_k, num_rows, torch.long, device)
        do_sample = temp > 0
        safe_temp = torch.where(do_sample, temp, torch.ones_like(temp))
        scores = logits / safe_temp.unsqueeze(1)
        scores = cls._apply_top_k(scores, top_k_row)
        scores = cls._apply_top_p(scores, top_p_row)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        seeds_row = cls._as_row_tensor(seeds, num_rows, torch.long, device)
        positions_row = cls._as_row_tensor(positions, num_rows, torch.long, device)
        sampled = multinomial_with_seed(probs, seeds_row, positions_row).view(-1)

        fallback = (~do_sample) | (probs.sum(dim=-1) <= 0)
        if bool(fallback.any()):
            sampled[fallback] = torch.argmax(logits[fallback], dim=-1)
        return sampled.to(torch.long)

    @staticmethod
    def _apply_top_k(scores: torch.Tensor, top_k_row: torch.Tensor) -> torch.Tensor:
        """Per-row top-k mask; rows with k <= 0 or k >= vocab are left untouched."""
        vocab = scores.shape[-1]
        active = (top_k_row > 0) & (top_k_row < vocab)
        if not bool(active.any()):
            return scores
        k_clamped = top_k_row.clamp(min=1, max=vocab)
        max_top_k = int(k_clamped[active].max().item())
        topk_scores, _ = torch.topk(scores, k=max_top_k, dim=-1)
        gather_k = torch.where(active, k_clamped, torch.ones_like(k_clamped))
        gather_k = gather_k.clamp(min=1, max=max_top_k)
        kth = topk_scores.gather(1, (gather_k - 1).unsqueeze(1))
        threshold = torch.where(
            active.unsqueeze(1), kth, torch.full_like(kth, _NEG_INF)
        )
        return scores.masked_fill(scores < threshold, _NEG_INF)

    @staticmethod
    def _apply_top_p(scores: torch.Tensor, top_p_row: torch.Tensor) -> torch.Tensor:
        """Per-row nucleus mask; rows with p <= 0 or p >= 1 are left untouched."""
        active = (top_p_row > 0.0) & (top_p_row < 1.0)
        if not bool(active.any()):
            return scores
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p_row.unsqueeze(1)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove & active.unsqueeze(1)
        remove_scattered = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, sorted_indices, remove
        )
        return scores.masked_fill(remove_scattered, _NEG_INF)
