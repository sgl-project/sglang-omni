# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.scheduling.types import RequestOutput


class MossTTSLocalModelRunner(ModelRunner):
    """Drives the per-frame local-transformer decode and feedback embeddings.

    Per step: the backbone (radix-cached, CUDA-graphed) produces one hidden
    state per request; :meth:`_collect_frame` then runs the batched local
    micro-decode — a binary continue/stop decision and 12 sequentially
    sampled RVQ codes — and stages the next frame's summed embedding through
    ``model._decode_input_embedding`` so the next decode step stays
    CUDA-graph-replayable (decode input_ids are row indices).
    """

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
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_frame(result, forward_batch, schedule_batch, requests)

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
                raise RuntimeError("MOSS-TTS Local prefill requires prompt_rows")
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            if data.output_rows:
                # KV-pressure retraction re-prefills with an extend region
                # spanning already-generated frames; their rows live in
                # output_rows, not prompt_rows. The resumed prefill samples
                # the next frame itself, superseding any feedback embedding
                # stranded by the retraction.
                generated = torch.stack(data.output_rows, dim=0)
                rows = torch.cat([rows.to(generated.device), generated], dim=0)
                data.pending_feedback_queue.clear()
            current_rows = rows[prefix_len : prefix_len + req_len]
            if int(current_rows.shape[0]) != req_len:
                raise RuntimeError(
                    f"MOSS-TTS Local prefill row mismatch for {req.rid}: have "
                    f"{int(current_rows.shape[0])} rows, need {req_len} "
                    f"(prefix={prefix_len}, prompt={int(data.prompt_rows.shape[0])}, "
                    f"generated={len(data.output_rows)})"
                )
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

    def _collect_frame(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return
        hidden_states = getattr(result.logits_output, "hidden_states", None)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS Local model output did not include hidden states"
            )
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]

        cfg = self.model.config
        device = hidden_states.device
        datas = [sched_req.data for sched_req in requests]
        batch_size = len(datas)
        num_channels = int(cfg.n_vq) + 1

        # The static per-request sampling parameters only change with batch
        # composition, so rebuild them once per composition; gen_steps moves
        # every step and is rebuilt each time.
        cache_key = tuple(sched_req.request_id for sched_req in requests)
        cached = getattr(self, "_param_cache", None)
        if cached is None or cached[0] != cache_key:
            params = {
                "text_temp": torch.tensor(
                    [float(d.text_temperature) for d in datas],
                    dtype=torch.float32,
                    device=device,
                ),
                "text_top_p": torch.tensor(
                    [float(d.text_top_p) for d in datas],
                    dtype=torch.float32,
                    device=device,
                ),
                "text_top_k": torch.tensor(
                    [int(d.text_top_k) for d in datas],
                    dtype=torch.long,
                    device=device,
                ),
                "audio_temp": torch.tensor(
                    [float(d.audio_temperature) for d in datas],
                    dtype=torch.float32,
                    device=device,
                ),
                "audio_top_p": torch.tensor(
                    [float(d.audio_top_p) for d in datas],
                    dtype=torch.float32,
                    device=device,
                ),
                "audio_top_k": torch.tensor(
                    [int(d.audio_top_k) for d in datas],
                    dtype=torch.long,
                    device=device,
                ),
                "seeds": torch.tensor(
                    [int(d.sampling_seed) for d in datas],
                    dtype=torch.long,
                    device=device,
                ),
            }
            self._param_cache = (cache_key, params)
        else:
            params = cached[1]
        text_temp = params["text_temp"]
        text_top_p = params["text_top_p"]
        text_top_k = params["text_top_k"]
        audio_temp = params["audio_temp"]
        audio_top_p = params["audio_top_p"]
        audio_top_k = params["audio_top_k"]
        sampling_seeds = params["seeds"]
        gen_steps = torch.tensor(
            [int(d.generation_steps) for d in datas], dtype=torch.long, device=device
        )
        rep_penalties = [float(d.audio_repetition_penalty) for d in datas]
        rep_histories = self._gather_rep_histories(datas, rep_penalties, device)

        def sample_text(logits: torch.Tensor) -> torch.Tensor:
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=text_temp,
                top_p=text_top_p,
                top_k=text_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels,
            )

        def sample_audio(logits: torch.Tensor, channel: int) -> torch.Tensor:
            if rep_histories is not None:
                self._apply_audio_repetition_penalty(
                    logits, rep_histories, rep_penalties, channel
                )
            return MossTTSModelRunner._sample_tokens(
                logits,
                temperature=audio_temp,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=sampling_seeds,
                positions=gen_steps * num_channels + channel + 1,
            )

        use_graph = rep_histories is None and batch_size <= getattr(
            self.model, "frame_graph_max_bs", 0
        )
        if use_graph:
            stop_choice, codes, feedback = self.model.decode_frame_graphed(
                hidden_states,
                text_temperature=text_temp,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temp,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                seeds=sampling_seeds,
                base_positions=gen_steps * num_channels,
            )
            # The graph outputs are static buffers that the next replay (any
            # later prefill or decode step) overwrites; snapshot what we keep.
            codes = codes.clone()
            embeds = feedback.clone()
        else:
            stop_choice, codes = self.model.decode_frame(
                hidden_states,
                sample_text=sample_text,
                sample_audio=sample_audio,
            )
            embeds = None

        slot_id = int(cfg.audio_assistant_slot_token_id)
        end_id = int(cfg.audio_end_token_id)
        next_text = torch.where(
            stop_choice == 0,
            torch.full((batch_size,), slot_id, dtype=torch.long, device=device),
            torch.full((batch_size,), end_id, dtype=torch.long, device=device),
        )

        rows = torch.empty((batch_size, num_channels), dtype=torch.long, device=device)
        rows[:, 0] = next_text
        rows[:, 1:] = codes

        next_token_ids = self._row_radix_token_ids(rows, next_text, end_id)
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids
        if embeds is None:
            embeds = self.model._prepare_multi_modal_inputs(
                rows.to(device=self.model.device)
            )
        self._pending_rows = rows
        self._pending_embeds = embeds.detach()

    @staticmethod
    def _row_radix_token_ids(
        rows: torch.Tensor,
        next_text: torch.Tensor,
        end_id: int,
    ) -> torch.Tensor:
        """Radix-cache token ids for generated frames.

        The scheduler appends one token id per frame to the request's KV
        chain, and the radix tree keys on those ids. The text channel alone is
        the same assistant-slot id for every continuing frame of every
        request, so a re-prefill after retraction could falsely prefix-match
        into another identical-prompt request's cached generated region. Hash
        the full multi-channel row — the same keying used for prompt rows —
        so a radix match implies identical audio content (a per-position id
        clash is ~1/151643 and only matters on top of an identical full
        prefix). The hash is folded below the special-token band because the
        scheduler finishes any request whose generated id crosses the vocab
        boundary (``Req._check_vocab_boundary_finish``); the stop decision
        keeps the raw audio_end id so eos detection still fires.
        """
        from sglang_omni.models.moss_tts.request_builders import build_row_cache_key_ids

        # <|endoftext|> 151643 opens the special/control id band.
        hash_space = 151643
        hashed = torch.tensor(
            build_row_cache_key_ids(rows), dtype=torch.long, device=rows.device
        )
        return torch.where(next_text == end_id, next_text, hashed % hash_space)

    @staticmethod
    def _gather_rep_histories(
        datas: list,
        rep_penalties: list[float],
        device: torch.device,
    ) -> list[torch.Tensor | None] | None:
        """Per-request generated-code history, only when a penalty is active.

        Upstream v1.5 applies the audio repetition penalty over each channel's
        previously *generated* frames only (the prompt's reference codes are
        excluded), so the history snapshot is taken from ``output_rows``.
        """
        if all(penalty == 1.0 for penalty in rep_penalties):
            return None
        histories: list[torch.Tensor | None] = []
        for data, penalty in zip(datas, rep_penalties):
            if penalty == 1.0 or not data.output_rows:
                histories.append(None)
                continue
            stacked = torch.stack(data.output_rows, dim=0)[:, 1:]
            histories.append(stacked.to(device=device, dtype=torch.long))
        return histories

    @staticmethod
    def _apply_audio_repetition_penalty(
        logits: torch.Tensor,
        histories: list[torch.Tensor | None],
        penalties: list[float],
        channel: int,
    ) -> None:
        """In-place penalty on fp32 logits, matching upstream order (before
        temperature scaling)."""
        vocab = logits.shape[-1]
        for row, (history, penalty) in enumerate(zip(histories, penalties)):
            if history is None or penalty == 1.0:
                continue
            tokens = torch.unique(history[:, channel])
            tokens = tokens[(tokens >= 0) & (tokens < vocab)]
            if tokens.numel() == 0:
                continue
            scores = logits[row, tokens]
            logits[row, tokens] = torch.where(
                scores < 0, scores * penalty, scores / penalty
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

        end_id = int(self.model.config.audio_end_token_id)
        # rows/embeds are step-private tensors (the graph outputs were
        # snapshotted in _collect_frame), so per-request views are stable.
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == end_id:
                # Stop decision: no frame is emitted alongside audio_end.
                continue
            sched_req.data.output_rows.append(rows[row_idx])
            sched_req.data.pending_feedback_queue.append(embeds[row_idx])
