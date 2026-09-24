# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS model runner for OmniScheduler."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput


class VoxtralTTSModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self.pending_audio_codes: torch.Tensor | None = None
        self.pending_audio_embeds: torch.Tensor | None = None

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del schedule_batch
        forward_batch.input_embeds = self.build_prefill_input_embeds(
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
        self.write_decode_input_embed_buffer(requests)

    def write_decode_input_embed_buffer(self, requests: list) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        buffer = self.model.decode_input_embed_buffer
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            if not queue:
                rows.append(torch.zeros(self.model.hidden_size, device=buffer.device))
                continue
            else:
                pass
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
        # note (luojiaxuan): Prefill-only requests do not generate audio.
        if schedule_batch.is_prefill_only:
            return
        else:
            pass
        self.collect_audio_step(result, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        self.collect_audio_step(result, schedule_batch, requests)

    def build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        input_embeds = self.model.get_input_embeddings()(input_ids)
        offset = 0
        replayed = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            prefix_len = req.extend_range.start
            end = req.extend_range.end
            full_ids = data.input_ids
            # note (luojiaxuan): Replay the absolute generated interval, not scalar IDs.
            prompt_len = len(full_ids)
            if end > prompt_len:
                start = max(prefix_len, prompt_len)
                history_end = end - prompt_len
                if history_end > len(data.generated_input_embeds):
                    raise RuntimeError(
                        "Voxtral re-prefill is missing generated feedback"
                    )
                else:
                    pass
                history = torch.stack(
                    data.generated_input_embeds[start - prompt_len : history_end]
                ).to(device=input_embeds.device, dtype=input_embeds.dtype)
                input_embeds[offset + start - prefix_len : offset + req_len] = history
                replayed.append(data)
            else:
                pass
            current_ids = full_ids[prefix_len:end]
            audio_positions = (current_ids == int(data.audio_token_id)).nonzero(
                as_tuple=True
            )[0]
            if audio_positions.numel() == 0 or data.voice_embedding is None:
                offset += req_len
                continue
            else:
                pass
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
            else:
                pass
            offset += req_len
        # note (luojiaxuan): Clear stale queued rows only after every replay succeeds.
        for data in replayed:
            data.pending_feedback_queue.clear()
        return input_embeds

    def collect_audio_step(
        self,
        result: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        hidden = result.logits_output.hidden_states
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        else:
            pass
        # note (luojiaxuan): Middle or discarded rows must not consume acoustic noise.
        active_rows = [
            index
            for index, request in enumerate(requests)
            if request.data.req.inflight_middle_chunks == 0
            and not request.data.req.is_retracted
            and not request.data.req.finished()
        ]
        if not active_rows:
            result.next_token_ids = torch.zeros(
                len(requests), dtype=torch.long, device=hidden.device
            )
            self.pending_audio_codes = None
            self.pending_audio_embeds = None
            return
        else:
            pass
        if len(active_rows) != len(requests):
            hidden = hidden[active_rows]
        else:
            pass
        codes = self.model.acoustic_transformer(hidden)
        embeds = self.model.audio_token_embedding(codes.unsqueeze(2)).sum(dim=1)
        if len(active_rows) != len(requests):
            batch_codes = codes.new_zeros((len(requests), *codes.shape[1:]))
            batch_embeds = embeds.new_zeros((len(requests), *embeds.shape[1:]))
            batch_codes[active_rows] = codes
            batch_embeds[active_rows] = embeds
            codes, embeds = batch_codes, batch_embeds
        else:
            pass
        result.next_token_ids = codes[:, 0].to(dtype=torch.long)
        self.pending_audio_codes = codes
        self.pending_audio_embeds = embeds

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        codes = self.pending_audio_codes
        embeds = self.pending_audio_embeds
        self.pending_audio_codes = None
        self.pending_audio_embeds = None
        if codes is None or embeds is None:
            return
        else:
            pass

        eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        skip_rids = self.finalize_skip_rids(scheduler_output)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            # note (luojiaxuan): Discarded and middle-chunk rows have no committed frame.
            if sched_req.request_id in skip_rids:
                continue
            else:
                pass
            req_output = outputs[sched_req.request_id]
            if req_output.data is None or int(req_output.data) == eos_id:
                continue
            else:
                pass
            sched_req.data.output_codes.append(codes[row_idx].detach().clone())
            # note (luojiaxuan): The queue and replay history share the exact fused row.
            feedback = embeds[row_idx, 0].detach().clone()
            sched_req.data.pending_feedback_queue.append(feedback)
            sched_req.data.generated_input_embeds.append(feedback)

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        return {
            sr.request_id
            for sr in scheduler_output.requests
            if sr.data.req.is_retracted
            or sr.data.req.finished()
            or sr.data.req.inflight_middle_chunks > 0
        }
