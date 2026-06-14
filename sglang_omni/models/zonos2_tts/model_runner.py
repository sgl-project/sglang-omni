# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS model runner for OmniScheduler.

Handles multi-codebook sampling with:
- Per-codebook repetition penalty
- Top-k / top-p / min-p sampling
- EOS detection with frame alignment (delayed codebook pattern)
- Speaker embedding injection
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.zonos2_tts.radix_hash import (
    folded_hash_coefficients,
    gpu_radix_row_hash,
)
from sglang_omni.scheduling.types import RequestOutput


class Zonos2TTSModelRunner(ModelRunner):
    """Samples ZONOS2 multi-codebook audio codes and manages EOS detection."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_rows: torch.Tensor | None = None
        self._pending_embeds: torch.Tensor | None = None
        self._radix_hash_coeffs: torch.Tensor | None = None

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Inject 2D prompt embeddings for prefill."""
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
        """Write decode-step input embeddings from the feedback queue."""
        del is_lookahead, schedule_batch
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
        self._collect_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_step(result, forward_batch, schedule_batch, requests)

    def lookahead_eligible(self, batch: Any) -> bool:
        """Allow lookahead only when launch commits the next-step GPU state."""
        if getattr(self.model, "_state_pool", None) is None:
            return False
        reqs = getattr(batch, "reqs", None) or []
        if not reqs:
            return False
        weight_rows = int(self.model._decode_input_embedding.weight.shape[0])
        return len(reqs) <= weight_rows

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        """Build input embeddings for prefill from 2D prompt rows."""
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            rows = data.prompt_rows
            if rows is None:
                raise RuntimeError("ZONOS2 prefill requires prompt_rows")
            if data.output_rows:
                generated = torch.stack(data.output_rows, dim=0)
                rows = torch.cat(
                    [rows.to(device=generated.device), generated], dim=0
                )
                data.pending_feedback_queue.clear()
                pool = getattr(self.model, "_state_pool", None)
                if pool is not None:
                    pool.reset_for_refill(sched_req.request_id, data.output_rows)
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            current_rows = rows[prefix_len : prefix_len + req_len]
            if int(current_rows.shape[0]) != req_len:
                raise RuntimeError(
                    f"ZONOS2 prefill row mismatch for {req.rid}: have "
                    f"{int(current_rows.shape[0])} rows, need {req_len} "
                    f"(prefix={prefix_len}, prompt={int(data.prompt_rows.shape[0])}, "
                    f"generated={len(data.output_rows)})"
                )
            embeds = self.model._prepare_multi_modal_inputs(
                current_rows.to(device=forward_batch.input_ids.device)
            )
            speaker_position = int(getattr(data, "speaker_token_position", -1))
            speaker_embedding = getattr(data, "speaker_embedding", None)
            if (
                speaker_embedding is not None
                and speaker_position >= prefix_len
                and speaker_position < prefix_len + req_len
                and hasattr(self.model, "project_speaker_embedding")
            ):
                local_idx = speaker_position - prefix_len
                embeds[local_idx] = self.model.project_speaker_embedding(
                    speaker_embedding
                ).to(device=embeds.device, dtype=embeds.dtype)
            pieces.append(embeds)
        if not pieces:
            return torch.empty(
                (0, self.model.hidden_size),
                device=forward_batch.input_ids.device,
                dtype=self.model.dtype,
            )
        result = torch.cat(pieces, dim=0)
        return result.to(device=forward_batch.input_ids.device, dtype=self.model.dtype)

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        """Write the next decode step's input embedding from feedback queue."""
        batch_size = len(requests)
        if batch_size == 0:
            return
        embedding = self.model._decode_input_embedding
        weight = embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "ZONOS2 decode input_ids must contain one row id per request"
            )
        if batch_size > int(weight.shape[0]):
            raise RuntimeError(
                "ZONOS2 decode batch exceeds the staged decode-embedding rows "
                f"({batch_size} > {int(weight.shape[0])})"
            )

        rows = []
        pool = getattr(self.model, "_state_pool", None)
        if pool is not None:
            row_t, pool_rows = pool.prepare_active_rows(requests)
            stacked = pool.feedback_embeds[row_t].to(
                device=weight.device,
                dtype=weight.dtype,
            )
            forward_batch.zonos2_pool_row_t = row_t
            forward_batch.zonos2_pool_rows = pool_rows
        else:
            for sched_req in requests:
                queue = sched_req.data.pending_feedback_queue
                if not queue:
                    rows.append(torch.zeros(self.model.hidden_size, device=weight.device))
                    continue
                if hasattr(queue, "popleft"):
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

    def _collect_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Sample multi-codebook tokens and prepare next-step embeddings."""
        self._ensure_pool_rows(forward_batch, requests)
        rows, embeds, next_token_ids = self._run_step_decode(
            result, requests, forward_batch=forward_batch
        )
        if rows is None or embeds is None or next_token_ids is None:
            return
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids
        self._pending_rows = rows
        self._pending_embeds = embeds
        result.zonos2_rows = rows
        result.zonos2_embeds = embeds
        result.zonos2_pool_rows = getattr(forward_batch, "zonos2_pool_rows", None)
        result.zonos2_pool_committed = getattr(
            forward_batch, "zonos2_pool_committed", False
        )

    def _ensure_pool_rows(self, forward_batch: Any, requests: list) -> None:
        if not requests or hasattr(forward_batch, "zonos2_pool_row_t"):
            return
        pool = getattr(self.model, "_state_pool", None)
        if pool is None:
            return
        row_t, pool_rows = pool.prepare_active_rows(requests)
        forward_batch.zonos2_pool_row_t = row_t
        forward_batch.zonos2_pool_rows = pool_rows

    def _run_step_decode(
        self,
        result: Any,
        requests: list,
        forward_batch: Any | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """GPU half shared by sync decode and async launch.

        Returns the sampled frame rows, next-step feedback embeddings, and
        radix-cache token ids. It deliberately does not mutate runner-level
        pending state so async lookahead cannot clobber the lagged step's host
        collection state.
        """
        hidden_states = self._get_hidden_states(result)
        if hidden_states is None or not requests:
            return None, None, None

        # Compute multi-codebook logits
        logits = self.model.compute_multi_codebook_logits(hidden_states)
        # logits: [batch_size, n_codebooks, audio_vocab]

        datas = [sched_req.data for sched_req in requests]
        pool_row_t = getattr(forward_batch, "zonos2_pool_row_t", None)
        pool_rows = getattr(forward_batch, "zonos2_pool_rows", None)
        sampled_codes = self._sample_codebooks(
            logits,
            datas,
            pool_row_t=pool_row_t,
            pool_rows=pool_rows,
        )
        # sampled_codes: [batch_size, n_codebooks]

        # Build full frame: audio codes + text padding
        batch_size = sampled_codes.shape[0]
        text_pad = int(self.model.text_vocab) if self.model.text_vocab is not None else 0
        frame_width = self.model.frame_width
        rows = torch.full(
            (batch_size, frame_width),
            self.model.config.audio_pad_id,
            dtype=torch.long,
            device=sampled_codes.device,
        )
        rows[:, : self.model.n_codebooks] = sampled_codes
        if self.model.text_vocab is not None:
            rows[:, self.model.n_codebooks] = text_pad

        eoa_id = int(self.model.config.eoa_id)
        # ZONOS2 uses delayed EOS (countdown pattern): when any codebook emits
        # eoa_id, we still need n_codebooks + 1 more steps to align the delayed
        # codebook outputs. So we do NOT signal EOS to SGLang here. Instead,
        # all frames get a hash-based radix key, and EOS is handled manually
        # in post_process_outputs when the countdown finishes.
        no_eos_mask = torch.zeros(batch_size, dtype=torch.bool, device=rows.device)

        # Use GPU radix row hash for radix-cache-safe token ids.
        # This hashes the full multi-channel row so that a radix match implies
        # identical audio content, enabling prefix sharing across requests.
        hash_coeffs = self._get_radix_hash_coeffs(frame_width, rows.device)
        next_token_ids = gpu_radix_row_hash(
            rows,
            no_eos_mask,
            eoa_id,
            coeffs=hash_coeffs,
        )

        # Compute embeddings for next decode step
        embeds = self.model._prepare_multi_modal_inputs(rows).detach()
        self._commit_pool_step_state(
            forward_batch,
            rows=rows,
            embeds=embeds,
            requests=requests,
        )
        return rows, embeds, next_token_ids

    def _commit_pool_step_state(
        self,
        forward_batch: Any | None,
        *,
        rows: torch.Tensor,
        embeds: torch.Tensor,
        requests: list,
    ) -> None:
        """Publish feedback/history needed by the following decode launch."""
        if forward_batch is None:
            return
        pool = getattr(self.model, "_state_pool", None)
        row_t = getattr(forward_batch, "zonos2_pool_row_t", None)
        pool_rows = getattr(forward_batch, "zonos2_pool_rows", None)
        if pool is None or row_t is None or pool_rows is None:
            return

        max_rep_window = max(
            int(getattr(sched_req.data, "repetition_window", 0) or 0)
            for sched_req in requests
        )
        if max_rep_window > 0:
            pool.ensure_history_capacity(max_rep_window)
            pool.update_history(
                row_t,
                rows[:, : self.model.n_codebooks],
                row_indices=pool_rows,
            )
        pool.feedback_embeds[row_t] = embeds.to(
            device=pool.device,
            dtype=pool.feedback_embeds.dtype,
        )
        forward_batch.zonos2_pool_committed = True

    def _get_radix_hash_coeffs(
        self,
        frame_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        coeffs = self._radix_hash_coeffs
        if (
            coeffs is None
            or int(coeffs.shape[0]) != int(frame_width)
            or coeffs.device != device
        ):
            coeffs = folded_hash_coefficients(frame_width, device=device)
            self._radix_hash_coeffs = coeffs
        return coeffs

    def post_decode_launch(self, result: Any, forward_batch: Any, requests: list):
        """Async-decode GPU half: sample codebooks and publish radix ids."""
        if not requests:
            return None
        rows, embeds, next_token_ids = self._run_step_decode(
            result, requests, forward_batch=forward_batch
        )
        if rows is None or embeds is None or next_token_ids is None:
            return None
        result.next_token_ids = next_token_ids
        result.zonos2_rows = rows
        result.zonos2_embeds = embeds
        result.zonos2_pool_committed = getattr(
            forward_batch, "zonos2_pool_committed", False
        )
        return (
            next_token_ids.clone(),
            tuple(sched_req.request_id for sched_req in requests),
            rows,
            embeds,
            getattr(forward_batch, "zonos2_pool_rows", None),
            getattr(forward_batch, "zonos2_pool_committed", False),
        )

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Async-decode host half: restore this step's sampled frame payload."""
        del forward_batch, schedule_batch, requests
        if launch_buf is None or result is None:
            return
        pool_committed = False
        if len(launch_buf) == 4:
            next_token_ids, rids, rows, embeds = launch_buf
            pool_rows = None
        elif len(launch_buf) == 5:
            next_token_ids, rids, rows, embeds, pool_rows = launch_buf
        else:
            next_token_ids, rids, rows, embeds, pool_rows, pool_committed = launch_buf
        result.next_token_ids = next_token_ids
        result.zonos2_rids = rids
        result.zonos2_rows = rows
        result.zonos2_embeds = embeds
        result.zonos2_pool_rows = pool_rows
        result.zonos2_pool_committed = pool_committed

    def _get_hidden_states(self, result: Any) -> torch.Tensor | None:
        """Extract hidden states from model output."""
        logits_output = result.logits_output
        hidden_states = getattr(logits_output, "hidden_states", None)
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[:, -1, :]
            return hidden_states
        return None

    def _sample_codebooks(
        self,
        logits: torch.Tensor,
        datas: list,
        pool_row_t: torch.Tensor | None = None,
        pool_rows: list[int] | None = None,
    ) -> torch.Tensor:
        """Sample from multi-codebook logits with per-request parameters.

        Args:
            logits: [batch_size, n_codebooks, audio_vocab]
            datas: Per-request data objects with sampling params

        Returns:
            sampled: [batch_size, n_codebooks] token ids
        """
        batch_size, n_codebooks, vocab_size = logits.shape
        device = logits.device

        # Gather per-request sampling parameters
        temperature_values = [float(d.temperature) for d in datas]
        top_k_values = [int(d.top_k) for d in datas]
        top_p_values = [float(d.top_p) for d in datas]
        min_p_values = [float(d.min_p) for d in datas]
        rep_penalty_values = [float(d.repetition_penalty) for d in datas]

        # Apply repetition penalty
        if any(value > 1.0 for value in rep_penalty_values):
            rep_penalties = torch.tensor(
                rep_penalty_values,
                dtype=torch.float32,
                device=device,
            )
            repetition_token_ids = self._build_repetition_token_ids(
                datas,
                n_codebooks=n_codebooks,
                vocab_size=vocab_size,
                device=device,
                pool_row_t=pool_row_t,
                pool_rows=pool_rows,
            )
            self._apply_repetition_penalty(
                logits,
                repetition_token_ids=repetition_token_ids,
                penalties=rep_penalties,
            )

        # Mask out audio_pad_id and beyond (only valid codes are 0..codebook_size-1 + eoa)
        # eoa_id (1024) is valid, audio_pad_id (1025) is not
        audio_pad_id = int(self.model.config.audio_pad_id)
        if audio_pad_id < vocab_size:
            logits[:, :, audio_pad_id:] = float("-inf")

        if self._can_use_uniform_sampling_fast_path(
            temperature_values,
            top_k_values,
            top_p_values,
            min_p_values,
            datas,
        ):
            return self._sample_uniform_codebooks(
                logits,
                temperature=temperature_values[0],
                top_k=top_k_values[0],
                top_p=top_p_values[0],
                min_p=min_p_values[0],
            )

        top_ks = torch.tensor(top_k_values, dtype=torch.long, device=device)
        top_ps = torch.tensor(top_p_values, dtype=torch.float32, device=device)
        min_ps = torch.tensor(min_p_values, dtype=torch.float32, device=device)
        temperatures = torch.tensor(
            temperature_values, dtype=torch.float32, device=device
        )

        # Per-request temperature, top-k, top-p, and min-p are broadcast over
        # codebooks, then applied as per-row masks. This preserves request
        # semantics under batching while keeping the work vectorized.
        do_sample = temperatures > 0
        safe_temp = torch.where(do_sample, temperatures, torch.ones_like(temperatures))
        scores = logits.to(torch.float32) / safe_temp.view(batch_size, 1, 1)

        flat_scores = scores.reshape(batch_size * n_codebooks, vocab_size)
        flat_top_ks = top_ks.view(batch_size, 1).expand(
            batch_size, n_codebooks
        ).reshape(-1)
        flat_top_ps = top_ps.view(batch_size, 1).expand(
            batch_size, n_codebooks
        ).reshape(-1)
        flat_min_ps = min_ps.view(batch_size, 1).expand(
            batch_size, n_codebooks
        ).reshape(-1)
        flat_greedy = (~do_sample).view(batch_size, 1).expand(
            batch_size, n_codebooks
        ).reshape(-1)

        flat_scores = self._apply_top_k_scores(flat_scores, flat_top_ks)
        if bool(((flat_top_ps > 0.0) & (flat_top_ps < 1.0)).any()):
            probs = F.softmax(flat_scores, dim=-1)
            probs = self._apply_top_p_rows(probs, flat_top_ps)
            probs = self._apply_min_p_rows(probs, flat_min_ps)
        else:
            flat_scores = self._apply_min_p_scores_rows(flat_scores, flat_min_ps)
            probs = F.softmax(flat_scores, dim=-1)

        # Handle all-zero rows (fallback to greedy)
        invalid = probs.sum(dim=-1) <= 0
        if bool(invalid.any()):
            greedy = flat_scores.argmax(dim=-1)
            fallback = torch.zeros_like(probs)
            fallback.scatter_(-1, greedy.unsqueeze(-1), 1.0)
            probs = torch.where(invalid.unsqueeze(-1), fallback, probs)

        # Sample. Match Zyphra's per-request generator semantics when a seed is
        # supplied, while still using the vectorized per-row filtering above.
        if bool(flat_greedy.all()):
            sampled = flat_scores.argmax(dim=-1)
        elif any(getattr(d, "seed", None) is not None for d in datas):
            sampled = torch.empty(
                batch_size * n_codebooks,
                dtype=torch.long,
                device=device,
            )
            for req_idx, data in enumerate(datas):
                start = req_idx * n_codebooks
                end = start + n_codebooks
                if bool(flat_greedy[start]):
                    sampled[start:end] = flat_scores[start:end].argmax(dim=-1)
                    continue
                generator = self._get_sampling_generator(data, device)
                sampled[start:end] = torch.multinomial(
                    probs[start:end],
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
        else:
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Override greedy for zero-temperature requests
            if bool(flat_greedy.any()):
                sampled = torch.where(
                    flat_greedy,
                    flat_scores.argmax(dim=-1),
                    sampled,
                )

        return sampled.view(batch_size, n_codebooks).to(torch.long)

    @staticmethod
    def _can_use_uniform_sampling_fast_path(
        temperatures: list[float],
        top_ks: list[int],
        top_ps: list[float],
        min_ps: list[float],
        datas: list,
    ) -> bool:
        if not temperatures:
            return False
        if any(getattr(data, "seed", None) is not None for data in datas):
            return False
        return (
            len(set(temperatures)) == 1
            and len(set(top_ks)) == 1
            and len(set(top_ps)) == 1
            and len(set(min_ps)) == 1
        )

    def _sample_uniform_codebooks(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> torch.Tensor:
        """Fast path for the common case where a batch shares sampling params."""
        batch_size, n_codebooks, vocab_size = logits.shape
        if temperature <= 0:
            return logits.argmax(dim=-1).to(torch.long)

        scores = logits.to(torch.float32) / max(float(temperature), 1e-8)
        flat_scores = scores.reshape(batch_size * n_codebooks, vocab_size)

        if 0 < top_k < vocab_size:
            topk_scores, _ = torch.topk(flat_scores, k=top_k, dim=-1)
            kth = topk_scores[:, -1].unsqueeze(-1)
            flat_scores = flat_scores.masked_fill(
                flat_scores < kth,
                float("-inf"),
            )

        if 0.0 < top_p < 1.0:
            probs = F.softmax(flat_scores, dim=-1)
            probs = self._apply_top_p(probs, float(top_p))
            probs = self._apply_min_p(probs, float(min_p))
        else:
            flat_scores = self._apply_min_p_scores(flat_scores, float(min_p))
            probs = F.softmax(flat_scores, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled.view(batch_size, n_codebooks).to(torch.long)

    @staticmethod
    def _get_sampling_generator(data: Any, device: torch.device) -> torch.Generator | None:
        seed = getattr(data, "seed", None)
        if seed is None:
            return None
        generator = getattr(data, "_zonos2_sampling_generator", None)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            setattr(data, "_zonos2_sampling_generator", generator)
        return generator

    @staticmethod
    def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering."""
        if p <= 0.0 or p >= 1.0:
            return probs
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = probs.scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs

    @staticmethod
    def _apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
        """Apply min-p filtering."""
        if min_p <= 0.0:
            return probs
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        mask = probs < (min_p * top_probs)
        probs = probs.masked_fill(mask, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs

    @staticmethod
    def _apply_min_p_scores(scores: torch.Tensor, min_p: float) -> torch.Tensor:
        """Apply min-p in score space before softmax.

        ``softmax(x_i) < min_p * max(softmax(x))`` is equivalent to
        ``x_i < max(x) + log(min_p)``. Filtering before softmax preserves the
        sampling distribution after renormalization and avoids an extra
        probability mask + divide on the decode hot path.
        """
        if min_p <= 0.0:
            return scores
        max_scores = scores.max(dim=-1, keepdim=True).values
        threshold = max_scores + math.log(float(min_p))
        return scores.masked_fill(scores < threshold, float("-inf"))

    @staticmethod
    def _apply_top_k_scores(
        scores: torch.Tensor,
        top_k_row: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = scores.shape[-1]
        active = (top_k_row > 0) & (top_k_row < vocab_size)
        if not bool(active.any()):
            return scores
        k_clamped = top_k_row.clamp(min=1, max=vocab_size)
        max_top_k = int(k_clamped[active].max().item())
        topk_scores, _ = torch.topk(scores, k=max_top_k, dim=-1)
        gather_k = torch.where(active, k_clamped, torch.ones_like(k_clamped))
        kth = topk_scores.gather(1, (gather_k - 1).unsqueeze(1))
        threshold = torch.where(
            active.unsqueeze(1), kth, torch.full_like(kth, float("-inf"))
        )
        return scores.masked_fill(scores < threshold, float("-inf"))

    @staticmethod
    def _apply_top_p_rows(
        probs: torch.Tensor,
        top_p_row: torch.Tensor,
    ) -> torch.Tensor:
        active = (top_p_row > 0.0) & (top_p_row < 1.0)
        if not bool(active.any()):
            return probs
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p_row.unsqueeze(1)
        mask = mask & active.unsqueeze(1)
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = probs.scatter(-1, sorted_idx, sorted_probs)
        return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _apply_min_p_rows(
        probs: torch.Tensor,
        min_p_row: torch.Tensor,
    ) -> torch.Tensor:
        active = min_p_row > 0.0
        if not bool(active.any()):
            return probs
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        mask = probs < (min_p_row.unsqueeze(1) * top_probs)
        mask = mask & active.unsqueeze(1)
        probs = probs.masked_fill(mask, 0.0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def _apply_min_p_scores_rows(
        scores: torch.Tensor,
        min_p_row: torch.Tensor,
    ) -> torch.Tensor:
        active = min_p_row > 0.0
        if not bool(active.any()):
            return scores
        max_scores = scores.max(dim=-1, keepdim=True).values
        thresholds = max_scores + torch.log(min_p_row.clamp(min=1e-8)).unsqueeze(1)
        mask = (scores < thresholds) & active.unsqueeze(1)
        return scores.masked_fill(mask, float("-inf"))

    def _build_repetition_token_ids(
        self,
        datas: list,
        n_codebooks: int,
        vocab_size: int,
        device: torch.device,
        pool_row_t: torch.Tensor | None = None,
        pool_rows: list[int] | None = None,
    ) -> torch.Tensor | None:
        """Pack per-request repetition windows into a device tensor."""
        active: list[tuple[int, Any, int, int]] = []
        max_window = 0
        pool = getattr(self.model, "_state_pool", None)
        if pool_rows is not None:
            pool_rows = [int(row) for row in pool_rows]
        elif pool is not None and pool_row_t is not None:
            pool_rows = [int(row) for row in pool_row_t.detach().cpu().tolist()]
        for i, data in enumerate(datas):
            if float(getattr(data, "repetition_penalty", 1.0)) <= 1.0:
                continue
            rep_window = int(getattr(data, "repetition_window", 50))
            rep_codebooks = int(getattr(data, "repetition_codebooks", n_codebooks))
            if rep_window <= 0 or rep_codebooks == 0:
                continue
            if rep_codebooks < 0:
                rep_codebooks = n_codebooks
            rep_codebooks = min(rep_codebooks, n_codebooks)

            history = getattr(data, "_zonos2_repetition_history", None)
            history_len = int(getattr(data, "_zonos2_repetition_history_len", 0) or 0)
            pool_history_len = 0
            if pool is not None and pool_rows is not None:
                pool_history_len = int(pool.history_length(pool_rows[i]))
            if pool_history_len > 0:
                window = min(rep_window, pool_history_len)
            elif history is None or history_len <= 0:
                output_rows = getattr(data, "output_rows", None)
                if not output_rows:
                    continue
                window = min(rep_window, len(output_rows))
            else:
                window = min(rep_window, history_len)
            if window <= 0:
                continue
            active.append((i, data, window, rep_codebooks))
            max_window = max(max_window, window)

        if max_window == 0:
            return None

        token_ids = torch.full(
            (len(datas), n_codebooks, max_window),
            -1,
            dtype=torch.long,
            device=device,
        )
        invalid = token_ids.new_full((), -1)
        for i, data, window, rep_codebooks in active:
            history = self._get_repetition_history(
                data,
                window=window,
                n_codebooks=n_codebooks,
                device=device,
                pool_row=pool_rows[i] if pool_rows is not None else None,
            )
            if history is None:
                continue
            valid = (history >= 0) & (history < vocab_size)
            if rep_codebooks < n_codebooks:
                valid[rep_codebooks:] = False
            token_ids[i, :, -window:] = torch.where(valid, history, invalid)

        return token_ids

    def _get_repetition_history(
        self,
        data: Any,
        *,
        window: int,
        n_codebooks: int,
        device: torch.device,
        pool_row: int | None = None,
    ) -> torch.Tensor | None:
        """Return recent history as [n_codebooks, window] without restacking."""
        if pool_row is not None:
            pool = getattr(self.model, "_state_pool", None)
            if pool is not None:
                recent = pool.recent_history(
                    int(pool_row),
                    window=window,
                    n_codebooks=n_codebooks,
                    device=device,
                )
                if recent is not None:
                    return recent
        history = getattr(data, "_zonos2_repetition_history", None)
        history_len = int(getattr(data, "_zonos2_repetition_history_len", 0) or 0)
        history_pos = int(getattr(data, "_zonos2_repetition_history_pos", 0) or 0)
        if history is not None and history_len > 0:
            history = history.to(device=device, dtype=torch.long, non_blocking=True)
            window = min(window, history_len)
            if window <= 0:
                return None
            capacity = int(history.shape[0])
            if history_len < capacity:
                recent = history[history_len - window : history_len]
            else:
                start = (history_pos - window) % capacity
                if start + window <= capacity:
                    recent = history[start : start + window]
                else:
                    recent = torch.cat(
                        [history[start:], history[: (start + window) % capacity]],
                        dim=0,
                    )
            return recent[:, :n_codebooks].transpose(0, 1).contiguous()

        output_rows = getattr(data, "output_rows", None)
        if not output_rows:
            return None
        history = torch.stack(output_rows[-window:], dim=0).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        return history[:, :n_codebooks].transpose(0, 1).contiguous()

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        *,
        repetition_token_ids: torch.Tensor | None,
        penalties: torch.Tensor,
    ) -> None:
        """Apply per-codebook repetition penalty in-place."""
        if repetition_token_ids is None or repetition_token_ids.numel() == 0:
            return

        batch_size, n_codebooks, vocab_size = logits.shape
        safe_token_ids = repetition_token_ids.clamp(min=0, max=vocab_size - 1)
        valid = (repetition_token_ids >= 0) & (repetition_token_ids < vocab_size)

        repeated = torch.zeros(
            (batch_size, n_codebooks, vocab_size),
            dtype=torch.bool,
            device=logits.device,
        )
        repeated.scatter_(-1, safe_token_ids, valid)

        penalties = penalties.view(batch_size, 1, 1).clamp(min=1.0)
        adjusted = torch.where(logits > 0, logits / penalties, logits * penalties)
        logits.copy_(torch.where(repeated, adjusted, logits))

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Store output rows and feedback embeddings for next decode step.

        Also handles ZONOS2's delayed EOS detection: when the countdown
        finishes, we manually terminate the request via req.to_finish since
        SGLang's automatic stop_token_ids detection is disabled (ZONOS2 needs
        additional frames after the first eoa_id to align the delayed codebook
        pattern).
        """
        rows = getattr(result, "zonos2_rows", None)
        embeds = getattr(result, "zonos2_embeds", None)
        if rows is None or embeds is None:
            rows = self._pending_rows
            embeds = self._pending_embeds
        self._pending_rows = None
        self._pending_embeds = None
        if rows is None or embeds is None:
            return
        expected_rids = tuple(sched_req.request_id for sched_req in scheduler_output.requests)
        actual_rids = getattr(result, "zonos2_rids", expected_rids)
        rows_len = int(rows.shape[0])
        if actual_rids != expected_rids or rows_len != len(expected_rids):
            raise RuntimeError(
                "ZONOS2 decode journal/batch alignment broken: "
                f"rids={actual_rids} expected={expected_rids} rows={rows_len}"
            )

        eoa_id = int(self.model.config.eoa_id)
        n_codebooks = int(self.model.n_codebooks)
        eoa_hits = (
            rows[:, :n_codebooks].eq(eoa_id).any(dim=1).detach().cpu().tolist()
        )
        pool = getattr(self.model, "_state_pool", None)
        pool_rows = getattr(result, "zonos2_pool_rows", None)
        pool_committed = bool(getattr(result, "zonos2_pool_committed", False))
        pool_row_t = None
        if pool is not None:
            if pool_rows is None:
                pool_rows = []
                for rid in expected_rids:
                    pool_row = pool.row_for(rid)
                    if pool_row is None:
                        pool_row = pool.acquire_row(rid)
                    pool_rows.append(pool_row)
            if pool_rows and all(row is not None for row in pool_rows):
                pool_row_t = torch.tensor(
                    [int(row) for row in pool_rows],
                    dtype=torch.long,
                    device=pool.device,
                )
                max_rep_window = max(
                    int(getattr(sched_req.data, "repetition_window", 0) or 0)
                    for sched_req in scheduler_output.requests
                )
                if max_rep_window > 0 and not pool_committed:
                    pool.ensure_history_capacity(max_rep_window)
                    pool.update_history(
                        pool_row_t,
                        rows[:, :n_codebooks],
                        row_indices=[int(row) for row in pool_rows],
                    )

        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs.get(sched_req.request_id)
            if req_output is None:
                continue
            data = sched_req.data
            req = data.req
            if req is not None:
                try:
                    finished_fn = req.finished
                except AttributeError:
                    finished_fn = None
                try:
                    is_retracted = req.is_retracted
                except AttributeError:
                    is_retracted = False
                if (callable(finished_fn) and finished_fn()) or bool(is_retracted):
                    continue

            # Store output row
            row = rows[row_idx].detach().clone()
            data.output_rows.append(row)
            if pool_row_t is None:
                self._update_repetition_history(data, row[:n_codebooks])

            # Check EOS: any codebook emitting eoa_id triggers countdown
            finished = self._check_eos_from_row(
                data,
                row[:n_codebooks],
                has_eoa=bool(eoa_hits[row_idx]),
            )

            if finished:
                # Manually terminate the request. SGLang's automatic EOS is
                # disabled for ZONOS2 because of the delayed codebook pattern.
                if req is not None:
                    from sglang.srt.managers.schedule_batch import (
                        FINISH_MATCHED_TOKEN,
                    )

                    req.to_finish = FINISH_MATCHED_TOKEN(eoa_id)
                if pool is not None:
                    pool.release_row(sched_req.request_id)
            elif pool_row_t is not None:
                if not pool_committed:
                    pool.feedback_embeds[pool_row_t[row_idx]] = (
                        embeds[row_idx]
                        .detach()
                        .to(device=pool.device, dtype=pool.feedback_embeds.dtype)
                    )
            else:
                data.pending_feedback_queue.append(embeds[row_idx].detach())

    @staticmethod
    def _check_eos_from_row(data: Any, audio_row: torch.Tensor, *, has_eoa: bool) -> bool:
        """Run delayed-EOS bookkeeping with the minimum host transfer needed."""
        if bool(getattr(data, "ignore_eos", False)):
            data.total_generated += 1
            return False
        if int(getattr(data, "eos_frame", -1)) >= 0:
            return bool(data.check_eos([]))
        if not has_eoa:
            data.total_generated += 1
            return False
        return bool(data.check_eos(audio_row.detach().cpu().tolist()))

    @staticmethod
    def _update_repetition_history(data: Any, audio_row: torch.Tensor) -> None:
        rep_window = int(getattr(data, "repetition_window", 50))
        rep_codebooks = int(getattr(data, "repetition_codebooks", audio_row.numel()))
        if rep_window <= 0 or rep_codebooks == 0:
            return

        n_codebooks = int(audio_row.numel())
        history = getattr(data, "_zonos2_repetition_history", None)
        if (
            history is None
            or int(history.shape[0]) != rep_window
            or int(history.shape[1]) != n_codebooks
            or history.device != audio_row.device
        ):
            history = torch.full(
                (rep_window, n_codebooks),
                -1,
                dtype=torch.long,
                device=audio_row.device,
            )
            setattr(data, "_zonos2_repetition_history", history)
            setattr(data, "_zonos2_repetition_history_pos", 0)
            setattr(data, "_zonos2_repetition_history_len", 0)

        pos = int(getattr(data, "_zonos2_repetition_history_pos", 0) or 0)
        history[pos].copy_(audio_row.to(dtype=torch.long))
        setattr(data, "_zonos2_repetition_history_pos", (pos + 1) % rep_window)
        history_len = int(getattr(data, "_zonos2_repetition_history_len", 0) or 0)
        setattr(data, "_zonos2_repetition_history_len", min(history_len + 1, rep_window))
