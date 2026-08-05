# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Delay model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.request_builders import _INF_DELAY
from sglang_omni.models.moss_tts.sampler import DelayGraphBatch
from sglang_omni.models.moss_tts.sampling_kernels import (
    multinomial_with_seed_and_token_ids,
)
from sglang_omni.scheduling.types import RequestOutput

_NEG_INF = float("-inf")
_INT64_MAX = torch.iinfo(torch.int64).max
_UINT64_MASK = (1 << 64) - 1
_INT64_SEED_MASK = (1 << 63) - 1


class MossTTSModelRunner(ModelRunner):
    """Samples MOSS-TTS text/audio channels and maintains delay-pattern state."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_rows: torch.Tensor | None = None
        self._pending_embeds: torch.Tensor | None = None

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

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if schedule_batch.is_prefill_only:
            return
        self._collect_moss_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_moss_step(result, forward_batch, schedule_batch, requests)

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
                raise RuntimeError("MOSS-TTS prefill requires prompt_rows")
            req_len = int(req.extend_range.length)
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
            device=forward_batch.input_ids.device,
            dtype=self.model.dtype,
        )

    def _write_decode_input_embedding(
        self,
        forward_batch: Any,
        requests: list,
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        embedding = self.model._decode_input_embedding
        weight = embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "MOSS-TTS decode input_ids must contain one row id per request"
            )
        if batch_size > int(weight.shape[0]):
            raise RuntimeError(
                "MOSS-TTS decode batch exceeds the staged decode-embedding rows "
                f"({batch_size} > {int(weight.shape[0])})"
            )
        rows = []
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
            batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def _collect_moss_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        datas = [sched_req.data for sched_req in requests]
        is_audio = bool(datas) and all(data.is_audio for data in datas)
        channel_logits = self._channel_logits_from_result(
            result,
            forward_batch,
            is_audio=is_audio,
        )
        n_vq = len(channel_logits) - 1
        if n_vq <= 0:
            raise RuntimeError("MOSS-TTS requires at least one audio codebook head")
        if not requests:
            return

        if self._can_use_sampling_cuda_graph(datas, is_audio=is_audio):
            rows = self._sample_rows_graphed(channel_logits, datas)
        else:
            rows = self._sample_rows(
                channel_logits,
                datas,
                n_vq=n_vq,
                is_audio=is_audio,
            )

        next_token_ids = rows[:, 0].contiguous()
        result.next_token_ids = next_token_ids
        embeds = self.model._prepare_multi_modal_inputs(
            rows.to(device=self.model.device)
        )
        self._pending_rows = rows
        self._pending_embeds = embeds.detach()

    def _can_use_sampling_cuda_graph(
        self,
        datas: list,
        *,
        is_audio: bool,
    ) -> bool:
        if not is_audio or not datas:
            return False
        is_compatible = getattr(
            self.model,
            "is_sampling_cuda_graph_compatible",
            None,
        )
        if not callable(is_compatible):
            return False
        return all(
            is_compatible(data) for data in datas
        ) and self._sampling_graph_available(len(datas))

    def _sampling_graph_available(self, batch_size: int) -> bool:
        available = getattr(self.model, "sampling_graph_available", None)
        return bool(callable(available) and available(batch_size))

    def _sample_rows_graphed(
        self,
        channel_logits: list[torch.Tensor],
        datas: list,
    ) -> torch.Tensor:
        device = channel_logits[0].device
        batch_size = len(datas)
        n_vq = len(channel_logits) - 1
        delay_state = torch.stack(
            [self._delay_state_tensor(data, device) for data in datas], dim=0
        )
        seeds = torch.tensor(
            [int(data.sampling_seed) for data in datas],
            dtype=torch.long,
            device=device,
        )
        generation_steps = torch.tensor(
            [int(data.generation_steps) for data in datas],
            dtype=torch.long,
            device=device,
        )
        audio_logits = torch.stack(
            [logits.to(torch.float32) for logits in channel_logits[1:]], dim=1
        )
        if tuple(audio_logits.shape[:2]) != (batch_size, n_vq):
            raise RuntimeError(
                "MOSS-TTS Delay sampling graph audio-logits shape mismatch: "
                f"got {tuple(audio_logits.shape)}"
            )
        sampled = self.model.sample_delay_graphed(
            channel_logits[0].to(torch.float32),
            audio_logits,
            DelayGraphBatch(
                delay_state=delay_state,
                seeds=seeds,
                generation_steps=generation_steps,
            ),
        )
        rows = sampled.rows[:batch_size].clone()
        next_state = sampled.next_delay_state[:batch_size].clone()
        for index, data in enumerate(datas):
            data.delay_state = next_state[index]
        return rows

    def _channel_logits_from_result(
        self,
        result: Any,
        forward_batch: Any,
        *,
        is_audio: bool = False,
    ) -> list[torch.Tensor]:
        logits_output = result.logits_output
        customized = logits_output.customized_info
        if isinstance(customized, dict):
            values = customized.get("moss_tts_channel_logits")
            if isinstance(values, list) and values:
                if is_audio:
                    token_ids = self.model.text_control_token_ids.to(
                        device=values[0].device
                    )
                    return [values[0].index_select(-1, token_ids), *values[1:]]
                return values
        hidden_states = logits_output.hidden_states
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim == 3:
                hidden_states = hidden_states[:, -1, :]
            if is_audio:
                return self.model.compute_channel_logits(
                    hidden_states,
                    forward_batch,
                    is_audio=True,
                )
            return self.model.compute_channel_logits(hidden_states, forward_batch)
        raise RuntimeError("MOSS-TTS model output did not include channel logits")

    @staticmethod
    def _delay_state_tensor(data: Any, device: torch.device) -> torch.Tensor:
        state = getattr(data, "delay_state", None)
        if isinstance(state, torch.Tensor) and tuple(state.shape) == (3,):
            state = state.to(device=device, dtype=torch.long)
        else:
            delayed = (
                _INT64_MAX
                if int(getattr(data, "delayed_length", _INF_DELAY)) == _INF_DELAY
                else int(getattr(data, "delayed_length"))
            )
            state = torch.tensor(
                [
                    int(getattr(data, "audio_length", 0)),
                    delayed,
                    int(data.is_audio),
                ],
                dtype=torch.long,
                device=device,
            )
        data.delay_state = state
        return state

    def _sample_rows(
        self,
        channel_logits: list[torch.Tensor],
        datas: list,
        *,
        n_vq: int,
        is_audio: bool = False,
    ) -> torch.Tensor:
        """One batched MOSS-TTS delay-pattern decode step over the whole batch.

        A vectorized port of one ``MossTTSDelayModel.generate`` step; per-request
        delay state is gathered from the ``data`` objects and scattered back.
        """
        cfg = self.model.config
        device = channel_logits[0].device
        batch_size = len(datas)

        pad_token_id = int(cfg.pad_token_id)
        gen_slot = int(cfg.audio_assistant_gen_slot_token_id)
        delay_slot = int(cfg.audio_assistant_delay_slot_token_id)
        audio_start = int(cfg.audio_start_token_id)
        audio_end = int(cfg.audio_end_token_id)
        im_end = int(cfg.im_end_token_id)
        audio_pad_code = int(cfg.audio_pad_code)

        delay_state = torch.stack(
            [self._delay_state_tensor(d, device) for d in datas], dim=0
        )
        audio_lengths = delay_state[:, 0]
        delayed = delay_state[:, 1]
        audio_mask = delay_state[:, 2].bool()
        gen_steps = torch.tensor(
            [int(d.generation_steps) for d in datas], dtype=torch.long, device=device
        )
        text_temp = torch.tensor(
            [float(d.text_temperature) for d in datas],
            dtype=torch.float32,
            device=device,
        )
        audio_temp = torch.tensor(
            [float(d.audio_temperature) for d in datas],
            dtype=torch.float32,
            device=device,
        )
        text_top_p = torch.tensor(
            [float(d.text_top_p) for d in datas], dtype=torch.float32, device=device
        )
        text_top_k = torch.tensor(
            [int(d.text_top_k) for d in datas], dtype=torch.long, device=device
        )
        audio_top_p = torch.tensor(
            [float(d.audio_top_p) for d in datas], dtype=torch.float32, device=device
        )
        audio_top_k = torch.tensor(
            [int(d.audio_top_k) for d in datas], dtype=torch.long, device=device
        )
        audio_rep = torch.tensor(
            [float(d.audio_repetition_penalty) for d in datas],
            dtype=torch.float32,
            device=device,
        )
        sampling_seeds = torch.tensor(
            [int(d.sampling_seed) for d in datas], dtype=torch.long, device=device
        )
        num_channels = n_vq + 1

        text_logits = channel_logits[0].to(torch.float32)
        next_text = torch.full(
            (batch_size,), pad_token_id, dtype=torch.long, device=device
        )
        next_text[delayed < n_vq] = delay_slot
        is_audio_eos = delayed == n_vq
        next_text[is_audio_eos] = audio_end
        audio_mask = audio_mask & ~is_audio_eos
        text_candidate_token_ids: torch.Tensor | None = None
        if is_audio:
            if int(text_logits.shape[-1]) != 2:
                raise RuntimeError(
                    "MOSS-TTS control-only text logits must have two columns"
                )
            text_candidate_token_ids = self.model.text_control_token_ids.to(
                device=device
            )
            sampling_text_mask = audio_mask & (delayed > n_vq)
            step0 = gen_steps == 0
            if bool(step0.any()):
                text_logits[step0, 1] = _NEG_INF
        else:
            vocab = text_logits.shape[-1]
            sampling_text_mask = delayed > n_vq
            not_audio = ~audio_mask
            if bool(not_audio.any()):
                exclude = torch.tensor(
                    [
                        t
                        for t in (pad_token_id, gen_slot, delay_slot, audio_end)
                        if 0 <= t < vocab
                    ],
                    dtype=torch.long,
                    device=device,
                )
                text_logits[not_audio] = text_logits[not_audio].index_fill(
                    -1, exclude, _NEG_INF
                )
            if bool(audio_mask.any()):
                allow_only = torch.ones(vocab, dtype=torch.bool, device=device)
                for token_id in (gen_slot, delay_slot):
                    if 0 <= token_id < vocab:
                        allow_only[token_id] = False
                text_logits[audio_mask] = text_logits[audio_mask].masked_fill(
                    allow_only, _NEG_INF
                )

            if 0 <= delay_slot < vocab:
                step0 = gen_steps == 0
                if bool(step0.any()):
                    text_logits[step0, delay_slot] = _NEG_INF
            if 0 <= im_end < vocab:
                step_le_nvq = gen_steps <= n_vq
                if bool(step_le_nvq.any()):
                    text_logits[step_le_nvq, im_end] = _NEG_INF

        if bool(sampling_text_mask.any()):
            idx = sampling_text_mask.nonzero(as_tuple=False).squeeze(1)
            next_text[idx] = self._sample_tokens(
                text_logits[idx],
                temperature=text_temp[idx],
                top_p=text_top_p[idx],
                top_k=text_top_k[idx],
                seeds=sampling_seeds[idx],
                positions=gen_steps[idx] * num_channels,
                candidate_token_ids=text_candidate_token_ids,
            )
        audio_mask = audio_mask | (next_text == audio_start)
        audio_mask = audio_mask & (next_text != im_end)

        next_audio = torch.full(
            (batch_size, n_vq), audio_pad_code, dtype=torch.long, device=device
        )
        channel_idx = torch.arange(n_vq, device=device)
        pre_audio = audio_lengths.unsqueeze(1) > channel_idx.unsqueeze(0)
        post_audio = channel_idx.unsqueeze(0) > (delayed.unsqueeze(1) - 1)
        post_audio = post_audio | (delayed == _INT64_MAX).unsqueeze(1)
        sampling_audio_mask = pre_audio & post_audio
        if bool(sampling_audio_mask.any()):
            audio_logits = torch.stack(
                [cl.to(torch.float32) for cl in channel_logits[1:]], dim=1
            )  # [batch, n_vq, vocab_audio]
            if 0 <= audio_pad_code < audio_logits.shape[-1]:
                audio_logits[..., audio_pad_code:] = _NEG_INF
            if bool((audio_rep != 1.0).any()):
                self._apply_audio_repetition_penalty(audio_logits, datas, n_vq=n_vq)
            audio_temp_full = audio_temp.unsqueeze(1).expand(batch_size, n_vq)
            audio_top_p_full = audio_top_p.unsqueeze(1).expand(batch_size, n_vq)
            audio_top_k_full = audio_top_k.unsqueeze(1).expand(batch_size, n_vq)
            mask_idx = sampling_audio_mask.nonzero(as_tuple=False)
            audio_rows = mask_idx[:, 0]
            audio_chans = mask_idx[:, 1]
            next_audio[sampling_audio_mask] = self._sample_tokens(
                audio_logits[sampling_audio_mask],
                temperature=audio_temp_full[sampling_audio_mask],
                top_p=audio_top_p_full[sampling_audio_mask],
                top_k=audio_top_k_full[sampling_audio_mask],
                seeds=sampling_seeds[audio_rows],
                positions=gen_steps[audio_rows] * num_channels + (audio_chans + 1),
            )

        increment = (
            (next_text == audio_start)
            | (next_text == gen_slot)
            | (next_text == delay_slot)
        )
        audio_lengths = audio_lengths + increment.long()
        audio_lengths[next_text == audio_end] = 0
        delayed[(delayed == _INT64_MAX) & (next_text == delay_slot)] = 0
        not_inf = delayed != _INT64_MAX
        delayed[not_inf] = delayed[not_inf] + 1
        delayed[delayed > n_vq] = _INT64_MAX

        next_state = torch.stack((audio_lengths, delayed, audio_mask.long()), dim=1)
        for i, data in enumerate(datas):
            data.delay_state = next_state[i].detach()
            if device.type == "cpu":
                data.audio_length = int(next_state[i, 0])
                delayed_i = int(next_state[i, 1])
                data.delayed_length = (
                    _INF_DELAY if delayed_i == _INT64_MAX else delayed_i
                )
                data.is_audio = bool(int(next_state[i, 2]))

        rows = torch.empty((batch_size, n_vq + 1), dtype=torch.long, device=device)
        rows[:, 0] = next_text
        rows[:, 1:] = next_audio
        return rows

    @staticmethod
    def _as_row_tensor(
        value: torch.Tensor | float | int,
        num_rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a scalar to a per-row tensor, or move an existing one."""
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=device)
        return torch.full((num_rows,), value, dtype=dtype, device=device)

    @staticmethod
    def _sample_tokens(
        logits: torch.Tensor,
        *,
        temperature: torch.Tensor | float,
        top_p: torch.Tensor | float,
        top_k: torch.Tensor | int,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-row temperature / top-k / top-p sampling for ``[N, vocab]`` logits.

        ``temperature``/``top_p``/``top_k`` may be scalars or per-row tensors of
        length N and are applied entirely through vectorized tensor masks (no
        ``.tolist()`` or Python-side grouping, so each row keeps its own params).
        When ``candidate_token_ids`` is provided, ``logits`` must contain exactly
        those two candidates; sampling hashes their original vocabulary ids and
        maps the local sampled column back to the original id. Rows with
        temperature <= 0, or rows left fully masked, fall back to greedy argmax;
        ``logits`` is float32 with disallowed tokens already masked to ``-inf``.
        """
        num_rows = int(logits.shape[0])
        if num_rows == 0:
            return torch.empty(0, dtype=torch.long, device=logits.device)
        device = logits.device
        if candidate_token_ids is not None:
            candidate_token_ids = candidate_token_ids.to(device=device)
            if tuple(logits.shape[1:]) != (2,) or int(candidate_token_ids.numel()) != 2:
                raise ValueError(
                    "MOSS-TTS compact candidate sampling requires exactly two tokens"
                )

        temp = MossTTSModelRunner._as_row_tensor(
            temperature, num_rows, torch.float32, device
        )
        top_p_row = MossTTSModelRunner._as_row_tensor(
            top_p, num_rows, torch.float32, device
        )
        top_k_row = MossTTSModelRunner._as_row_tensor(
            top_k, num_rows, torch.long, device
        )
        do_sample = temp > 0
        safe_temp = torch.where(do_sample, temp, torch.ones_like(temp))
        scores = logits / safe_temp.unsqueeze(1)
        if candidate_token_ids is not None:
            scores = MossTTSModelRunner._apply_two_token_top_k(scores, top_k_row)
            scores = MossTTSModelRunner._apply_top_p(
                scores,
                top_p_row,
                skip_inactive_check=True,
            )
        else:
            scores = MossTTSModelRunner._apply_top_k(scores, top_k_row)
            scores = MossTTSModelRunner._apply_top_p(scores, top_p_row)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        seeds_row = MossTTSModelRunner._as_row_tensor(
            seeds, num_rows, torch.long, device
        )
        positions_row = MossTTSModelRunner._as_row_tensor(
            positions, num_rows, torch.long, device
        )
        fallback = (~do_sample) | (probs.sum(dim=-1) <= 0)
        sampled = torch.argmax(logits, dim=-1)
        if candidate_token_ids is not None:
            if device.type == "cpu":
                candidate_sampled = MossTTSModelRunner._multinomial_with_seed_cpu(
                    probs,
                    seeds_row,
                    positions_row,
                )
            else:
                candidate_sampled = multinomial_with_seed_and_token_ids(
                    scores,
                    seeds_row,
                    positions_row,
                    candidate_token_ids,
                )
            sampled = torch.where(fallback, sampled, candidate_sampled)
            return candidate_token_ids[sampled].to(torch.long)

        sample_mask = ~fallback
        if bool(sample_mask.any()):
            if device.type == "cpu":
                sampled[sample_mask] = MossTTSModelRunner._multinomial_with_seed_cpu(
                    probs[sample_mask],
                    seeds_row[sample_mask],
                    positions_row[sample_mask],
                )
            else:
                # Note:(Chenchen Hong) post1's multinomial_with_seed is Gumbel-max
                # and wants logits, not probs (probs lost the -inf masking and made
                # MOSS emit only the pad code); the CPU fallback still needs probs.
                sampled[sample_mask] = multinomial_with_seed(
                    scores[sample_mask],
                    seeds_row[sample_mask],
                    positions_row[sample_mask],
                ).view(-1)
        return sampled.to(torch.long)

    @staticmethod
    def _apply_two_token_top_k(
        scores: torch.Tensor,
        top_k_row: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the only active compact top-k case without a host sync."""
        top1 = top_k_row == 1
        top1_threshold = scores.max(dim=-1, keepdim=True).values
        return scores.masked_fill(
            top1.unsqueeze(1) & (scores < top1_threshold), _NEG_INF
        )

    @staticmethod
    def _multinomial_with_seed_cpu(
        probs: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """CPU-only seeded fallback; approximates the GPU sampler (not bit-exact)."""
        sampled = torch.empty(probs.shape[0], dtype=torch.long, device=probs.device)
        for row_idx in range(probs.shape[0]):
            row = probs[row_idx]
            if float(row.sum().item()) <= 0.0:
                sampled[row_idx] = 0
                continue
            seed = int(seeds[row_idx].item()) & _UINT64_MASK
            position = int(positions[row_idx].item()) & _UINT64_MASK
            mixed_seed = (
                seed
                ^ ((position + 0x9E3779B97F4A7C15) & _UINT64_MASK)
                ^ ((position << 17) & _UINT64_MASK)
            ) & _INT64_SEED_MASK
            generator = torch.Generator(device="cpu")
            generator.manual_seed(mixed_seed)
            sampled[row_idx] = torch.multinomial(
                row.cpu(), num_samples=1, generator=generator
            ).to(device=probs.device)
        return sampled

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
    def _apply_top_p(
        scores: torch.Tensor,
        top_p_row: torch.Tensor,
        *,
        skip_inactive_check: bool = False,
    ) -> torch.Tensor:
        """Apply per-row nucleus masking.

        ``skip_inactive_check`` keeps the compact hot path tensor-only instead
        of synchronizing with the host just to discover that every row uses
        ``top_p=1``.
        """
        active = (top_p_row > 0.0) & (top_p_row < 1.0)
        if not skip_inactive_check and not bool(active.any()):
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

    @staticmethod
    def _apply_audio_repetition_penalty(
        audio_logits: torch.Tensor,
        datas: list,
        *,
        n_vq: int,
    ) -> None:
        """In-place delay-pattern repetition penalty, per request and codebook.

        Each request's own ``audio_repetition_penalty`` is applied (requests with
        a unit penalty are skipped). Only invoked when at least one request has a
        non-unit penalty (off by default), so this per-request loop is off the
        hot path.
        """
        device = audio_logits.device
        vocab = audio_logits.shape[-1]
        for i, data in enumerate(datas):
            penalty = float(data.audio_repetition_penalty)
            if penalty == 1.0:
                continue
            parts = []
            prompt_rows = getattr(data, "prompt_rows", None)
            if prompt_rows is not None and prompt_rows.numel() > 0:
                parts.append(prompt_rows[:, 1:])
            output_rows = getattr(data, "output_rows", None)
            if output_rows:
                parts.append(torch.stack(output_rows, dim=0)[:, 1:])
            if not parts:
                continue
            history = torch.cat(
                [part.to(device=device, dtype=torch.long) for part in parts], dim=0
            )
            for channel in range(n_vq):
                tokens = torch.unique(history[:, channel])
                tokens = tokens[(tokens >= 0) & (tokens < vocab)]
                if tokens.numel() == 0:
                    continue
                scores = audio_logits[i, channel, tokens]
                audio_logits[i, channel, tokens] = torch.where(
                    scores > 0, scores / penalty, scores * penalty
                )

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        rows = self._pending_rows
        embeds = self._pending_embeds
        self._pending_rows = None
        self._pending_embeds = None
        if rows is None or embeds is None:
            return

        eos_id = int(self.model.config.im_end_token_id)
        audio_start_id = int(self.model.config.audio_start_token_id)
        audio_end_id = int(self.model.config.audio_end_token_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None:
                continue
            text_token_id = int(req_output.data)
            if text_token_id == audio_start_id:
                sched_req.data.is_audio = True
            elif text_token_id == audio_end_id:
                sched_req.data.is_audio = False
            if text_token_id == eos_id:
                continue
            sched_req.data.output_rows.append(rows[row_idx].detach().clone())
            sched_req.data.pending_feedback_queue.append(
                embeds[row_idx].detach().clone()
            )
