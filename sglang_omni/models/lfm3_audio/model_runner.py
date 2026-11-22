# SPDX-License-Identifier: Apache-2.0
"""LFM3-Audio model runner: audio input embeddings, speech-frame sampling and feedback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.lfm3_audio.depth import sample_frames
from sglang_omni.scheduling.types import SchedulerRequest

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.models.lfm3_audio.request_builders import Lfm3AudioRequestData
    from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
else:
    pass


class Lfm3AudioModelRunner(ModelRunner):
    model: Lfm3AudioSGLangModel

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.tokens = self.model.tokens
        self.stop_id = self.tokens.eos_token_ids[0]

    # Text sampling must happen before the speech step: the state machine reads the sampled token.
    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def write_text_only_rows(self, requests: list[SchedulerRequest]) -> None:
        flags = torch.tensor(
            [bool(request.data.speech.text_only) for request in requests],
            dtype=torch.bool,
        )
        self.model.text_only_buffer[: len(requests)].copy_(flags, non_blocking=True)

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        self.write_text_only_rows(requests)
        forward_batch.input_embeds = self.build_prefill_input_embeds(
            forward_batch, requests
        )

    def build_prefill_input_embeds(
        self, forward_batch: ForwardBatch, requests: list[SchedulerRequest]
    ) -> torch.Tensor:
        input_embeds = self.model.get_input_embeddings()(forward_batch.input_ids)
        offset = 0
        for sched_req in requests:
            data: Lfm3AudioRequestData = sched_req.data
            req = data.req
            extend_len = req.extend_len
            media = self.encode_media(data, input_embeds.device)
            if media:
                if req.prefix_len != 0 or extend_len != len(data.input_ids):
                    raise RuntimeError(
                        "LFM3 places a request's media embeddings in one prefill; "
                        "serve it with chunked prefill and the radix cache off"
                    )
                else:
                    pass
            else:
                pass
            for token_id, embeds in media:
                positions = (data.input_ids == token_id).nonzero(as_tuple=True)[0]
                if positions.numel() != embeds.shape[0]:
                    raise ValueError(
                        f"{positions.numel()} placeholders of token {token_id} for "
                        f"{embeds.shape[0]} media embeddings"
                    )
                else:
                    pass
                input_embeds[positions.to(input_embeds.device) + offset] = embeds.to(
                    input_embeds.dtype
                )
            offset += extend_len
        return input_embeds

    def encode_media(
        self, data: Lfm3AudioRequestData, device: torch.device
    ) -> list[tuple[int, torch.Tensor]]:
        """(placeholder id, embeddings) of the request's audio clips, images and videos; consumes the inputs."""
        media: list[tuple[int, torch.Tensor]] = []
        tokens = self.tokens
        if data.input_features is not None and data.input_features.numel():
            media.append(
                (
                    tokens.audio_token_id,
                    self.model.encode_audio(
                        data.input_features.to(device),
                        data.feature_attention_mask.to(device),
                    ),
                )
            )
        else:
            pass
        for token_id, pixels, grid in (
            (tokens.image_token_id, data.pixel_values, data.image_grid_thw),
            (tokens.video_token_id, data.pixel_values_videos, data.video_grid_thw),
        ):
            if pixels is not None and pixels.numel():
                media.append(
                    (
                        token_id,
                        self.model.encode_vision(pixels.to(device), grid.to(device)),
                    )
                )
            else:
                pass
        data.input_features = None
        data.feature_attention_mask = None
        data.pixel_values = None
        data.image_grid_thw = None
        data.pixel_values_videos = None
        data.video_grid_thw = None
        return media

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del forward_batch, schedule_batch, is_lookahead
        if not requests:
            return
        else:
            pass
        self.write_text_only_rows(requests)
        buffer = self.model.decode_input_embed_buffer
        rows = []
        for sched_req in requests:
            queue = sched_req.data.pending_feedback_queue
            rows.append(
                queue.popleft()
                if queue
                else torch.zeros(self.model.hidden_size, device=buffer.device)
            )
        buffer[: len(rows)].copy_(torch.stack(rows).to(buffer.dtype))

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        self.speech_step(result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        self.speech_step(result, requests)

    @torch.no_grad()
    def speech_step(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        """Advance every row one step: a speech frame or the sampled text token."""
        hidden = result.logits_output.hidden_states
        sampled = result.next_token_ids
        host_ids = sampled.tolist()
        speeches = [request.data.speech for request in requests]
        audio_rows = [
            row for row, speech in enumerate(speeches) if speech.needs_frame()
        ]
        frames = None
        if audio_rows:
            index = torch.tensor(audio_rows, device=hidden.device)
            frames = sample_frames(
                self.model.depth_decoder,
                hidden.index_select(0, index),
                [speeches[row].sampling for row in audio_rows],
                self.tokens.maskgit_schedule,
            )
        else:
            pass
        next_ids = list(host_ids)
        feed_rows: list[int] = []
        feed_codes: list[torch.Tensor] = []
        frame_of_row = {row: slot for slot, row in enumerate(audio_rows)}
        for row, speech in enumerate(speeches):
            was_interleaving = speech.interleaving
            slot = frame_of_row.get(row)
            if slot is not None:
                feed, _ = speech.apply_frame(frames[slot])
                next_ids[row] = self.tokens.audio_out_token_id
                feed_rows.append(row)
                feed_codes.append(feed)
            else:
                speech.apply_text(int(host_ids[row]), self.tokens.eos_token_ids)
                if speech.ended_on_text_end:
                    # The scheduler stops on <|im_end|> only; <|text_end|> ends a text-mode turn.
                    next_ids[row] = self.stop_id
                else:
                    pass
            speech.advance_slot(was_interleaving)
        if feed_rows:
            embeds = self.model.frame_embeddings(torch.stack(feed_codes))
            for row, embed in zip(feed_rows, embeds, strict=True):
                requests[row].data.pending_feedback_queue.append(embed)
        else:
            pass
        if next_ids != host_ids:
            result.next_token_ids = torch.tensor(
                next_ids, dtype=sampled.dtype, device=sampled.device
            )
        else:
            pass
