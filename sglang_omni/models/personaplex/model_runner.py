# SPDX-License-Identifier: Apache-2.0
"""Runs one PersonaPlex request through the timeline, one row per forward.

Prefill embeds every row the request's KV lacks at once: the whole prompt for
a new request, the prompt and its generated positions for a retracted one.
Each decode step then fuses the row for the next position from the last
sampled text token, the agent codes the depformer produced for that position,
and the caller's codes for it. The text token is sampled *before* the post
hook so the depformer can consume it in the same step; the codes it spells
out become the next row's input and, one position later, a finished output
frame streamed to the codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Protocol

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.personaplex.architecture import (
    AGENT_STREAM_OFFSET,
    NUM_STREAMS,
    USER_STREAM_OFFSET,
)
from sglang_omni.models.personaplex.sampling import AudioSampling, sample_token
from sglang_omni.models.personaplex.timeline import Timeline, output_frame
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


class AudioTokenSampler(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...


@dataclass(kw_only=True)
class DecodeFeedback:
    user_rows: torch.Tensor
    agent_row: torch.Tensor
    text_token: torch.Tensor | None = None
    launched_frames: int = 0
    audio_rng_state: torch.Tensor | None = None


@dataclass(kw_only=True)
class PersonaPlexFrameBatch:
    host_rows: torch.Tensor
    agent_rows: torch.Tensor
    audio_rng_states: list[torch.Tensor | None]


@dataclass(frozen=True, kw_only=True)
class SamplingRun:
    """Adjacent batch rows that share one sample_token call."""

    row_slice: slice
    sampling: AudioSampling
    generators: list[torch.Generator | None]


class PersonaPlexModelRunner(ModelRunner):
    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        return True

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        if self.model_device.type != "cuda" or not super().lookahead_eligible(batch):
            return False
        else:
            pass
        for request in batch.reqs:
            data = request.omni_data
            timeline = self.request_timeline(data)
            feedback = self.decode_feedback(data)
            # Note (edwardzh): An over-budget launch would be discarded after its forward.
            if data.return_logprob or feedback.launched_frames >= timeline.num_frames:
                return False
            else:
                pass
        return True

    @property
    def model_device(self) -> torch.device:
        return self.model.fusion_buffer.device

    @staticmethod
    def request_timeline(data: SGLangARRequestData) -> Timeline:
        return data.talker_model_inputs["timeline"]

    def decode_feedback(self, data: SGLangARRequestData) -> DecodeFeedback:
        """Cache device inputs and per-request decode state; user rows a
        session's later units append are copied as they arrive."""
        inputs = data.talker_model_inputs
        timeline = self.request_timeline(data)
        cached = inputs.get("decode_feedback")
        if cached is None:
            cached = DecodeFeedback(
                user_rows=timeline.user_rows.to(self.model_device),
                agent_row=timeline.agent_row_before_start.to(self.model_device),
            )
            inputs["decode_feedback"] = cached
        elif cached.user_rows.shape[0] < timeline.user_rows.shape[0]:
            num_copied = cached.user_rows.shape[0]
            cached.user_rows = torch.cat(
                [
                    cached.user_rows,
                    timeline.user_rows[num_copied:].to(self.model_device),
                ]
            )
        else:
            pass
        return cached

    def agent_row_at(self, data: SGLangARRequestData, position: int) -> torch.Tensor:
        timeline = self.request_timeline(data)
        if position < timeline.num_prompt_positions:
            return timeline.agent_row_before_start.to(self.model_device)
        else:
            return data.talker_model_inputs["agent_rows"][
                position - timeline.num_prompt_positions
            ]

    def prefill_rows(self, data: SGLangARRequestData) -> torch.Tensor:
        timeline = self.request_timeline(data)
        model = self.model
        tokens = timeline.prefill_tokens.to(self.model_device)
        dtype = model.fusion_buffer.dtype
        if not timeline.prefill_embedding_positions:
            return model.embed_rows(tokens).to(dtype)
        else:
            pass
        stored = torch.as_tensor(
            timeline.prefill_embeddings, device=self.model_device
        ).to(dtype)
        known = torch.ones(tokens.shape[0], dtype=torch.bool, device=self.model_device)
        known[timeline.prefill_embedding_positions] = False
        rows = torch.empty(
            tokens.shape[0], stored.shape[1], dtype=dtype, device=self.model_device
        )
        rows[timeline.prefill_embedding_positions] = stored
        rows[known] = model.embed_rows(tokens[known]).to(dtype)
        return rows

    def audio_generator(self, data: SGLangARRequestData) -> torch.Generator | None:
        inputs = data.talker_model_inputs
        audio_seed = inputs["sampling"].audio_seed
        generator = inputs.get("audio_generator")
        if generator is None and audio_seed is not None:
            generator = torch.Generator(device=self.model_device)
            generator.manual_seed(audio_seed)
            inputs["audio_generator"] = generator
        else:
            pass
        return generator

    def audio_sampler(self, requests: list[SchedulerRequest]) -> AudioTokenSampler:
        """Sample row i with requests[i]'s audio settings and generator."""
        runs: list[SamplingRun] = []
        start_row = 0
        # Note (edwardzh): Only adjacent requests share a sampling call, so rows are
        # sliced in place and no index tensor reaches the device.
        for sampling, run_requests in groupby(
            requests,
            key=lambda request: request.data.talker_model_inputs["sampling"].audio,
        ):
            generators = [
                self.audio_generator(request.data) for request in run_requests
            ]
            end_row = start_row + len(generators)
            runs.append(
                SamplingRun(
                    row_slice=slice(start_row, end_row),
                    sampling=sampling,
                    generators=generators,
                )
            )
            start_row = end_row

        def sample(logits: torch.Tensor) -> torch.Tensor:
            picks = [
                sample_token(logits[run.row_slice], run.sampling, run.generators)
                for run in runs
            ]
            return picks[0] if len(picks) == 1 else torch.cat(picks)

        return sample

    def spell_frames(
        self,
        requests: list[SchedulerRequest],
        text_token_B: torch.Tensor,
        forced_BK: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> PersonaPlexFrameBatch:
        """Prepare frame results for the positions just predicted.

        Row i of text_token_B and forced_BK belongs to requests[i].
        """
        batch_size = len(requests)
        codes_BK = self.model.depformer.generate(
            text_token_B[:batch_size],
            self.model.hidden_out[:batch_size],
            forced_BK,
            self.audio_sampler(requests),
        )
        # Note (edwardzh): Sampling buffers can be reused before host results commit.
        text_tokens = text_token_B[:batch_size].clone()
        frames = []
        audio_rng_states = []
        for request, codes, text_token in zip(
            requests, codes_BK, text_tokens, strict=True
        ):
            data = request.data
            inputs = data.talker_model_inputs
            feedback = self.decode_feedback(data)
            frame = output_frame(feedback.agent_row, codes)
            feedback.agent_row = codes
            feedback.text_token = text_token
            feedback.launched_frames += 1
            frames.append(frame)
            generator = inputs.get("audio_generator")
            audio_rng_states.append(
                generator.get_state() if generator is not None else None
            )
        packed_rows = torch.cat((text_tokens[:, None], torch.stack(frames)), dim=1)
        if non_blocking:
            # Note (edwardzh): Reused staging can be larger than a shrinking batch.
            host_rows = self.next_host_staging(packed_rows.shape, packed_rows.dtype)[
                :batch_size
            ]
            host_rows.copy_(packed_rows, non_blocking=True)
        else:
            host_rows = packed_rows.cpu()
        return PersonaPlexFrameBatch(
            host_rows=host_rows,
            agent_rows=codes_BK,
            audio_rng_states=audio_rng_states,
        )

    def commit_frames(
        self,
        frames: PersonaPlexFrameBatch,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
    ) -> None:
        host_rows = frames.host_rows
        self.stage_token_ids(result, host_rows[:, 0])
        for index, request in enumerate(requests):
            data = request.data
            if data.req.finished() or data.req.is_retracted:
                continue
            else:
                pass
            inputs = data.talker_model_inputs
            inputs["agent_rows"].append(frames.agent_rows[index])
            inputs["frames"].append(host_rows[index, 1:])
            inputs["pending_frames"].append(host_rows[index, 1:])
            self.decode_feedback(data).audio_rng_state = frames.audio_rng_states[index]

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> PersonaPlexFrameBatch:
        return self.spell_frames(
            requests,
            result.next_token_ids,
            self.free_codes().expand(len(requests), -1),
            non_blocking=True,
        )

    def post_decode_resolve(
        self,
        launch_buf: PersonaPlexFrameBatch,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        # Note (edwardzh): Committed history outlives the next reuse of pinned staging.
        launch_buf.host_rows = launch_buf.host_rows.clone()
        self.commit_frames(launch_buf, result, requests)

    def free_codes(self) -> torch.Tensor:
        return torch.full(
            (self.model.depformer.spec.steps,),
            -1,
            dtype=torch.long,
            device=self.model_device,
        )

    def missing_rows(self, data: SGLangARRequestData, req: Req) -> torch.Tensor:
        """Embed the positions req's KV lacks, from its cached prefix to its last token.

        A prompt position is the prompt's own row. A later one is the text token
        req holds there, the agent codes the depformer spelled there, and the
        caller's codes.
        """
        model = self.model
        timeline = self.request_timeline(data)
        num_prompt = timeline.num_prompt_positions
        num_origin = len(req.origin_input_ids)
        start = req.prefix_len
        end = num_origin + len(req.output_ids)
        pieces = []
        if start < num_prompt:
            pieces.append(self.prefill_rows(data)[start:])
        else:
            pass
        first = max(start, num_prompt)
        if first < end:
            tokens = [
                *req.origin_input_ids[first:],
                *req.output_ids[max(first - num_origin, 0) :],
            ]
            agent_rows = data.talker_model_inputs["agent_rows"]
            user_rows = self.decode_feedback(data).user_rows
            rows = torch.empty(
                end - first, NUM_STREAMS, dtype=torch.long, device=self.model_device
            )
            rows[:, 0] = torch.tensor(tokens, dtype=torch.long)
            rows[:, AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = torch.stack(
                agent_rows[first - num_prompt : end - num_prompt]
            )
            rows[:, USER_STREAM_OFFSET:] = user_rows[first:end]
            pieces.append(model.embed_rows(rows).to(model.fusion_buffer.dtype))
        else:
            pass
        return torch.cat(pieces, dim=0)

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        rows = []
        for request, req in zip(requests, schedule_batch.reqs, strict=True):
            data = request.data
            inputs = data.talker_model_inputs
            timeline = self.request_timeline(data)
            predicted = len(req.origin_input_ids) + len(req.output_ids)
            feedback = self.decode_feedback(data)
            feedback.launched_frames = predicted - timeline.num_prompt_positions
            generator = inputs.get("audio_generator")
            if generator is not None and feedback.audio_rng_state is not None:
                # Note (edwardzh): Discarded lookahead must not consume a seeded audio draw.
                generator.set_state(feedback.audio_rng_state)
            else:
                pass
            rows.append(self.missing_rows(data, req))
            feedback.agent_row = self.agent_row_at(data, predicted - 1)
            if predicted == timeline.num_prompt_positions:
                inputs["prefill_forced"] = timeline.forced_agent_at_start.to(
                    self.model_device
                )
            else:
                inputs["prefill_forced"] = self.free_codes()
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(rows, dim=0), input_embeds_are_projected=True
            ),
        )

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        forced_BK = torch.stack(
            [
                request.data.talker_model_inputs.pop("prefill_forced")
                for request in requests
            ]
        )
        frames = self.spell_frames(requests, result.next_token_ids, forced_BK)
        self.commit_frames(frames, result, requests)

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        model = self.model
        rows = []
        for request in requests:
            data = request.data
            timeline = self.request_timeline(data)
            feedback = self.decode_feedback(data)
            position = timeline.input_position(feedback.launched_frames)
            row = torch.empty(NUM_STREAMS, dtype=torch.long, device=self.model_device)
            assert feedback.text_token is not None
            row[0] = feedback.text_token
            row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = feedback.agent_row
            row[USER_STREAM_OFFSET:] = feedback.user_rows[position]
            rows.append(row)
        batch = len(rows)
        model.fusion_buffer[:batch] = model.embed_rows(torch.stack(rows)).to(
            model.fusion_buffer.dtype
        )

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        free_BK = self.free_codes().expand(len(requests), -1)
        frames = self.spell_frames(requests, result.next_token_ids, free_BK)
        self.commit_frames(frames, result, requests)
