# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens


class VoxtralTTSModelRunner(ModelRunner):
    def prepare_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def prepare_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size))
                continue
            rows.append(queue.popleft())
        input_embeds = torch.stack(rows, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
        return self._forward_with_input_embeds(forward_batch, input_embeds)

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
            full_ids = data.input_ids.to(device=input_ids.device)
            current_ids = full_ids[prefix_len : prefix_len + req_len]
            audio_positions = (current_ids == int(data.audio_token_id)).nonzero(
                as_tuple=True
            )[0]
            if audio_positions.numel() == 0 or data.voice_embedding is None:
                offset += req_len
                continue
            previous_audio = int(
                (full_ids[:prefix_len] == int(data.audio_token_id)).sum().item()
            )
            voice = data.voice_embedding.to(
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            )
            n_frames = min(
                int(audio_positions.numel()), voice.shape[0] - previous_audio
            )
            if n_frames > 0:
                rows = audio_positions[:n_frames] + offset
                input_embeds[rows] = voice[previous_audio : previous_audio + n_frames]
            offset += req_len
        return input_embeds

    def _collect_audio_step(
        self,
        result: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        hidden = result.logits_output.hidden_states
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        codes = self.model.acoustic_transformer(hidden)
        semantic_ids = codes[:, 0].to(dtype=torch.long)
        result.next_token_ids = semantic_ids
        schedule_batch.output_ids = semantic_ids

        eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        embeds = self.model.audio_token_embedding(codes.unsqueeze(2)).sum(dim=1)
        for row_idx, sched_req in enumerate(requests):
            if int(semantic_ids[row_idx].item()) == eos_id:
                continue
            sched_req.data.output_codes.append(codes[row_idx].detach().clone())
            sched_req.data.pending_feedback_queue.append(
                embeds[row_idx, 0].detach().clone()
            )

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
        logits_output = self.model(
            input_ids=forward_batch.input_ids,
            positions=forward_batch.positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=None,
            can_run_cuda_graph=False,
        )
