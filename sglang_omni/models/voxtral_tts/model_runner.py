# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.scheduling.types import RequestOutput


class VoxtralTTSModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._pending_audio_codes: torch.Tensor | None = None
        self._pending_audio_embeds: torch.Tensor | None = None

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_input_embeds(
            forward_batch, requests
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del forward_batch, schedule_batch
        self._write_decode_input_embed_buffer(requests)

    def _write_decode_input_embed_buffer(self, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        buffer = self.model._decode_input_embed_buffer
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size, device=buffer.device))
                continue
            rows.append(queue.popleft())
        stacked = torch.stack(rows, dim=0).to(
            device=buffer.device,
            dtype=buffer.dtype,
        )
        buffer[:batch_size].copy_(stacked)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        self._collect_audio_step(result, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        self._collect_audio_step(result, schedule_batch, requests)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        input_embeds = self.model.get_input_embeddings()(input_ids)
        offset = 0
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            full_ids = data.input_ids
            current_ids = full_ids[prefix_len : prefix_len + req_len]
            audio_positions = (current_ids == int(data.audio_token_id)).nonzero(
                as_tuple=True
            )[0]
            if audio_positions.numel() == 0 or data.voice_embedding is None:
                offset += req_len
                continue
            previous_audio = int(
                (full_ids[:prefix_len] == int(data.audio_token_id)).sum()
            )
            voice = data.voice_embedding.to(
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            )
            n_frames = min(
                int(audio_positions.numel()), voice.shape[0] - previous_audio
            )
            if n_frames > 0:
                rows = (
                    audio_positions[:n_frames].to(device=input_embeds.device) + offset
                )
                input_embeds[rows] = voice[previous_audio : previous_audio + n_frames]
            offset += req_len
        return input_embeds

    def _collect_audio_step(
        self,
        result: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del requests
        hidden = result.logits_output.hidden_states
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        codes = self.model.acoustic_transformer(hidden)
        semantic_ids = codes[:, 0].to(dtype=torch.long)
        result.next_token_ids = semantic_ids
        schedule_batch.output_ids = semantic_ids

        self._pending_audio_codes = codes
        self._pending_audio_embeds = self.model.audio_token_embedding(
            codes.unsqueeze(2)
        ).sum(dim=1)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        codes = self._pending_audio_codes
        embeds = self._pending_audio_embeds
        self._pending_audio_codes = None
        self._pending_audio_embeds = None
        if codes is None or embeds is None:
            return

        eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            sched_req.data.output_codes.append(codes[row_idx].detach().clone())
            sched_req.data.pending_feedback_queue.append(
                embeds[row_idx, 0].detach().clone()
            )
