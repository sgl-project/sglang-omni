# SPDX-License-Identifier: Apache-2.0
"""LFM3 model runner: media input embeddings, speech-frame sampling and feedback.

With the seeded sampler a step's speech state lives on the device (speech_pool.py), so a decode step launches without waiting for the previous one; the host SpeechState follows one resolve later. The exact sampler keeps the host state machine and decodes synchronously.
"""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.lfm3_audio.depth import (
    FrameSampling,
    SpeechSampler,
    sample_frames,
    seeded_frame_codes,
)
from sglang_omni.models.lfm3_audio.request_builders import Lfm3AudioRequestData
from sglang_omni.models.lfm3_audio.sglang_model import Lfm3AudioSGLangModel
from sglang_omni.models.lfm3_audio.speech_pool import (
    MODE_CODES,
    SpeechStatePool,
    advance_speech,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest


class Lfm3AudioModelRunner(ModelRunner):
    """A request keeps its prompt embeddings and fed frame codes, so a retracted request replays its prompt and output in one prefill."""

    model: Lfm3AudioSGLangModel

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        speech_sampler: SpeechSampler,
        repetition_window: int,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.tokens = self.model.tokens
        self.stop_token_id = self.tokens.eos_token_ids[0]
        self.speech_pool: SpeechStatePool | None = None
        if speech_sampler is SpeechSampler.SEEDED:
            self.speech_pool = SpeechStatePool(
                slot_count=tp_worker.model_runner.req_to_token_pool.size,
                hidden_size=self.model.decode_input_embed_buffer.shape[1],
                window=repetition_window,
                num_codebooks=self.tokens.num_codebooks,
                embed_dtype=self.model.decode_input_embed_buffer.dtype,
                device=self.model.decode_input_embed_buffer.device,
            )
        else:
            pass

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """Only the seeded sampler advances speech without the host; the exact sampler draws on the host's generators."""
        return self.speech_pool is not None and super().lookahead_eligible(batch)

    # note (ruilin): the speech step reads the sampled text token, so sampling runs before the post hooks.
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
        embedding_rows = [self.prefill_input_embeds(request) for request in requests]
        forward_batch.input_embeds = torch.cat(embedding_rows, dim=0)

    def prefill_input_embeds(self, request: SchedulerRequest) -> torch.Tensor:
        """The prompt's embeddings, followed on a replay by the inputs of the tokens generated before the retraction."""
        request_data: Lfm3AudioRequestData = request.data
        sglang_request = request_data.req
        if sglang_request.prefix_len != 0:
            raise RuntimeError(
                "LFM3 prefills a request whole; serve it with the radix cache off"
            )
        else:
            pass
        if request_data.prefill_input_embeds is None:
            request_data.prefill_input_embeds = self.prompt_input_embeds(request_data)
        else:
            pass
        prompt_embeds = request_data.prefill_input_embeds
        output_ids = sglang_request.output_ids
        if sglang_request.extend_len != prompt_embeds.shape[0] + len(output_ids):
            raise RuntimeError(
                f"LFM3 prefill of {request.request_id} covers {sglang_request.extend_len} positions, "
                f"not its {prompt_embeds.shape[0]} prompt and {len(output_ids)} output tokens; "
                "serve it with chunked prefill off"
            )
        elif not output_ids:
            return prompt_embeds
        else:
            pass
        output_id_tensor = torch.tensor(output_ids, device=prompt_embeds.device)
        output_embeds = self.model.get_input_embeddings()(output_id_tensor)
        is_frame = output_id_tensor == self.tokens.audio_out_token_id
        if int(is_frame.sum()) != len(request_data.feedback_frame_codes):
            raise RuntimeError(
                f"LFM3 replay of {request.request_id} has {int(is_frame.sum())} frame positions "
                f"for {len(request_data.feedback_frame_codes)} fed frames"
            )
        elif request_data.feedback_frame_codes:
            output_embeds[is_frame] = self.model.frame_embeddings(
                torch.stack(request_data.feedback_frame_codes).to(output_embeds.device)
            ).to(output_embeds.dtype)
        else:
            pass
        # note (ruilin): the queued embedding of the last frame is in output_embeds now.
        request_data.pending_feedback_queue.clear()
        return torch.cat([prompt_embeds, output_embeds], dim=0)

    def prompt_input_embeds(self, request_data: Lfm3AudioRequestData) -> torch.Tensor:
        """Token embeddings of the prompt with its audio, image and video embeddings at their placeholders."""
        embed_tokens = self.model.get_input_embeddings()
        token_ids = request_data.input_ids.to(embed_tokens.weight.device)
        prompt_embeds = embed_tokens(token_ids)
        for placeholder_token_id, media_embeds in self.encode_media(
            request_data, token_ids.device
        ):
            positions = (token_ids == placeholder_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() != media_embeds.shape[0]:
                raise ValueError(
                    f"{positions.numel()} placeholders of token {placeholder_token_id} for "
                    f"{media_embeds.shape[0]} media embeddings"
                )
            else:
                pass
            prompt_embeds[positions] = media_embeds.to(prompt_embeds.dtype)
        return prompt_embeds.detach()

    def encode_media(
        self, request_data: Lfm3AudioRequestData, device: torch.device
    ) -> list[tuple[int, torch.Tensor]]:
        """(placeholder token id, embeddings) of the request's audio clips, images and videos; consumes the inputs."""
        media: list[tuple[int, torch.Tensor]] = []
        tokens = self.tokens
        if request_data.input_features is not None:
            media.append(
                (
                    tokens.audio_token_id,
                    self.model.encode_audio(
                        request_data.input_features.to(device),
                        request_data.feature_attention_mask.to(device),
                    ),
                )
            )
        else:
            pass
        for placeholder_token_id, pixel_values, grid_thw in (
            (
                tokens.image_token_id,
                request_data.pixel_values,
                request_data.image_grid_thw,
            ),
            (
                tokens.video_token_id,
                request_data.pixel_values_videos,
                request_data.video_grid_thw,
            ),
        ):
            if pixel_values is not None:
                media.append(
                    (
                        placeholder_token_id,
                        self.model.encode_vision(
                            pixel_values.to(device), grid_thw.to(device)
                        ),
                    )
                )
            else:
                pass
        request_data.input_features = None
        request_data.feature_attention_mask = None
        request_data.pixel_values = None
        request_data.image_grid_thw = None
        request_data.pixel_values_videos = None
        request_data.video_grid_thw = None
        return media

    def next_decode_input(self, request_data: Lfm3AudioRequestData) -> torch.Tensor:
        """The embedding the request's last output feeds back: its speech frame, else its text token."""
        if request_data.pending_feedback_queue:
            return request_data.pending_feedback_queue.popleft()
        else:
            last_token_id = torch.tensor(
                [request_data.req.output_ids[-1]],
                device=self.model.get_input_embeddings().weight.device,
            )
            return self.model.get_input_embeddings()(last_token_id)[0].detach()

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        if not requests:
            return
        else:
            pass
        self.write_text_only_rows(requests)
        buffer = self.model.decode_input_embed_buffer
        if self.speech_pool is not None:
            slots = forward_batch.req_pool_indices[: len(requests)].long()
            torch.index_select(
                self.speech_pool.next_input_embeds,
                0,
                slots,
                out=buffer[: len(requests)],
            )
            return
        else:
            pass
        decode_inputs = [self.next_decode_input(request.data) for request in requests]
        buffer[: len(decode_inputs)].copy_(torch.stack(decode_inputs).to(buffer.dtype))

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        if self.speech_pool is None:
            self.speech_step(result, requests)
            return
        else:
            pass
        slots = forward_batch.req_pool_indices[: len(requests)].long()
        self.speech_pool.load(slots, [request.data.speech for request in requests])
        host_rows = self.launch_speech_step(
            result, forward_batch, requests, non_blocking=False
        )
        self.commit_speech_step(host_rows, result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        if self.speech_pool is None:
            self.speech_step(result, requests)
            return
        else:
            pass
        host_rows = self.launch_speech_step(
            result, forward_batch, requests, non_blocking=False
        )
        self.commit_speech_step(host_rows, result, requests)

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        if self.speech_pool is None:
            return super().post_decode_launch(result, forward_batch, requests)
        elif not requests:
            return None
        else:
            pass
        return self.launch_speech_step(
            result, forward_batch, requests, non_blocking=True
        )

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        if self.speech_pool is None:
            super().post_decode_resolve(
                launch_buf, result, forward_batch, schedule_batch, requests
            )
            return
        elif launch_buf is None:
            return
        else:
            pass
        # note (ruilin): the pinned staging is reused two launches later; the host state keeps these codes.
        self.commit_speech_step(launch_buf.clone(), result, requests)

    @torch.no_grad()
    def launch_speech_step(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        *,
        non_blocking: bool,
    ) -> torch.Tensor:
        """Advance the batch's device speech state one step and stage, per row, the emitted id, the sampled id, whether it took a frame and the fed codes."""
        pool = self.speech_pool
        assert pool is not None
        batch_size = len(requests)
        slots = forward_batch.req_pool_indices[:batch_size].long()
        rows = pool.gather(slots)
        hidden_states = result.logits_output.hidden_states[:batch_size]
        seeded_inputs = dict(
            seeds=rows.seed,
            frame_positions=rows.frames_sampled,
            temperature=rows.temperature,
            top_k=rows.top_k,
            # note (ruilin): a row without history must leave its scores unchanged.
            repetition_penalty=torch.where(
                rows.history_len > 0, rows.repetition_penalty, 1.0
            ),
            history=rows.history,
        )
        graphs = self.model.depth_graphs
        if graphs is None:
            codes = seeded_frame_codes(
                self.model.depth_decoder,
                hidden_states,
                schedule=self.tokens.maskgit_schedule,
                **seeded_inputs,
            )
        else:
            codes = graphs.seeded_frames(hidden_states, **seeded_inputs)
        sampled_ids = result.next_token_ids[:batch_size]
        step = advance_speech(rows, sampled_ids, codes, self.tokens, self.stop_token_id)
        pool.store(slots, step.rows)
        dtype = pool.next_input_embeds.dtype
        next_inputs = torch.where(
            step.needs_frame[:, None],
            self.model.frame_embeddings(step.fed_codes).to(dtype),
            self.model.get_input_embeddings()(step.token_ids).to(dtype),
        )
        pool.next_input_embeds.index_copy_(0, slots, next_inputs)
        result.next_token_ids = step.token_ids
        advanced_rows = step.rows
        packed = torch.cat(
            [
                step.token_ids[:, None].long(),
                sampled_ids[:, None].long(),
                step.needs_frame[:, None].long(),
                step.fed_codes,
                torch.stack(
                    [
                        advanced_rows.mode,
                        advanced_rows.speaking.long(),
                        advanced_rows.text_done.long(),
                        advanced_rows.finished.long(),
                        advanced_rows.slots_left,
                        advanced_rows.frames_sampled,
                        advanced_rows.history_len,
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        )
        if non_blocking:
            host_rows = self.next_host_staging(packed.shape, packed.dtype)[:batch_size]
            host_rows.copy_(packed, non_blocking=True)
            return host_rows
        else:
            return packed.cpu()

    def commit_speech_step(
        self,
        host_rows: torch.Tensor,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
    ) -> None:
        """Replay a launched step on the host SpeechState, which outputs and replays read, and check the device state it reached."""
        self.stage_token_ids(result, host_rows[:, 0])
        num_codebooks = self.tokens.num_codebooks
        for row, request in enumerate(requests):
            request_data: Lfm3AudioRequestData = request.data
            if request_data.req.finished() or self.req_is_retracted(request_data.req):
                continue
            else:
                pass
            speech = request_data.speech
            took_frame = bool(host_rows[row, 2])
            if speech.needs_frame() != took_frame:
                raise RuntimeError(
                    f"LFM3 speech state of {request.request_id} diverged: the device "
                    f"{'took' if took_frame else 'skipped'} a frame the host state did not"
                )
            else:
                pass
            was_interleaving = speech.interleaving
            if took_frame:
                fed_codes = speech.apply_frame(host_rows[row, 3 : 3 + num_codebooks])
                speech.sampling.frames_sampled += 1
                request_data.feedback_frame_codes.append(fed_codes)
            else:
                speech.apply_text(int(host_rows[row, 1]), self.tokens.eos_token_ids)
            speech.advance_slot(was_interleaving)
            history = speech.sampling.history
            host_state = [
                MODE_CODES[speech.mode],
                int(speech.speaking),
                int(speech.text_done),
                int(speech.finished),
                speech.slots_left,
                speech.sampling.frames_sampled,
                0 if history is None else history.shape[0],
            ]
            if host_rows[row, 3 + num_codebooks :].tolist() != host_state:
                raise RuntimeError(
                    f"LFM3 speech state of {request.request_id} diverged: device "
                    f"{host_rows[row, 3 + num_codebooks:].tolist()}, host {host_state}"
                )
            else:
                pass

    def sample_speaking_frames(
        self, hidden_states: torch.Tensor, samplings: list[FrameSampling]
    ) -> torch.Tensor:
        """One frame per speaking row with the exact sampler, its depth passes from the graphs when they were captured."""
        graphs = self.model.depth_graphs
        depth_logits = (
            self.model.depth_decoder if graphs is None else graphs.depth_logits
        )
        codes = sample_frames(
            depth_logits, hidden_states, samplings, self.tokens.maskgit_schedule
        )
        for sampling in samplings:
            sampling.frames_sampled += 1
        return codes

    @torch.no_grad()
    def speech_step(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        """Advance every row one step on the host, for the exact sampler: a speech frame or the sampled text token."""
        hidden_states = result.logits_output.hidden_states
        sampled_token_ids = result.next_token_ids
        host_token_ids = sampled_token_ids.tolist()
        speeches = [request.data.speech for request in requests]
        speaking_rows = [
            row for row, speech in enumerate(speeches) if speech.needs_frame()
        ]
        frames = None
        if speaking_rows:
            frames = self.sample_speaking_frames(
                hidden_states.index_select(
                    0, torch.tensor(speaking_rows, device=hidden_states.device)
                ),
                [speeches[row].sampling for row in speaking_rows],
            )
        else:
            pass
        next_token_ids = list(host_token_ids)
        feedback_rows: list[int] = []
        feedback_codes: list[torch.Tensor] = []
        frame_index_of_row = {row: index for index, row in enumerate(speaking_rows)}
        for row, speech in enumerate(speeches):
            was_interleaving = speech.interleaving
            frame_index = frame_index_of_row.get(row)
            if frame_index is not None:
                codes = speech.apply_frame(frames[frame_index])
                next_token_ids[row] = self.tokens.audio_out_token_id
                feedback_rows.append(row)
                feedback_codes.append(codes)
                requests[row].data.feedback_frame_codes.append(codes)
            else:
                speech.apply_text(int(host_token_ids[row]), self.tokens.eos_token_ids)
                if speech.finished:
                    # note (ruilin): the scheduler stops on im_end only; text_end and any other end id finish the turn too.
                    next_token_ids[row] = self.stop_token_id
                else:
                    pass
            speech.advance_slot(was_interleaving)
        if feedback_rows:
            frame_embeds = self.model.frame_embeddings(torch.stack(feedback_codes))
            for row, frame_embed in zip(feedback_rows, frame_embeds, strict=True):
                requests[row].data.pending_feedback_queue.append(frame_embed.detach())
        else:
            pass
        if next_token_ids != host_token_ids:
            result.next_token_ids = torch.tensor(
                next_token_ids,
                dtype=sampled_token_ids.dtype,
                device=sampled_token_ids.device,
            )
        else:
            pass
