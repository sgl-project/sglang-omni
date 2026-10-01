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
from sglang_omni.models.personaplex.sampling import sample_token
from sglang_omni.models.personaplex.timeline import Timeline, output_frame
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


class AudioTokenSampler(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...


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
        # Note (wilsonzheng0327): The depformer must see this step's sampled text before
        # the next forward is prepared; a one-step lookahead would run it a step late.
        return False

    @property
    def model_device(self) -> torch.device:
        return self.model.fusion_buffer.device

    @staticmethod
    def request_timeline(data: SGLangARRequestData) -> Timeline:
        return data.talker_model_inputs["timeline"]

    def user_rows_on_device(self, data: SGLangARRequestData) -> torch.Tensor:
        """The timeline's user rows on the device.

        Rows are only ever appended, never rewritten, so a longer timeline
        copies just its new rows.
        """
        inputs = data.talker_model_inputs
        rows = self.request_timeline(data).user_rows
        cached = inputs.get("device_user_rows")
        if cached is None:
            cached = rows.to(self.model_device)
        elif cached.shape[0] < rows.shape[0]:
            cached = torch.cat([cached, rows[cached.shape[0] :].to(self.model_device)])
        else:
            pass
        inputs["device_user_rows"] = cached
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

    def audio_sampler(self, data: SGLangARRequestData) -> AudioTokenSampler:
        inputs = data.talker_model_inputs
        sampling = inputs["sampling"]
        generator = inputs.get("audio_generator")
        if generator is None and sampling.audio_seed is not None:
            generator = torch.Generator(device=self.model_device)
            generator.manual_seed(sampling.audio_seed)
            inputs["audio_generator"] = generator
        else:
            pass
        return lambda logits: sample_token(logits, sampling.audio, generator)

    def spell_frame(
        self,
        index: int,
        request: SchedulerRequest,
        text_token: torch.Tensor,
        forced: torch.Tensor,
    ) -> None:
        """Run the depformer for the position just predicted and record it."""
        data = request.data
        inputs = data.talker_model_inputs
        hidden = self.model.hidden_out[index : index + 1]
        codes = self.model.depformer.generate(
            text_token.view(1), hidden, forced.view(1, -1), self.audio_sampler(data)
        )[0]
        frame = output_frame(inputs["agent_row"], codes)
        inputs["agent_row"] = codes
        inputs["agent_rows"].append(codes)
        inputs["frames"].append(frame)
        inputs["pending_frames"].append(frame)

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
        start = len(req.prefix_indices)
        end = num_origin + len(req.output_ids)
        assert start < end, (start, end)
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
            rows = torch.empty(
                end - first, NUM_STREAMS, dtype=torch.long, device=self.model_device
            )
            rows[:, 0] = torch.tensor(tokens, dtype=torch.long)
            rows[:, AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = torch.stack(
                agent_rows[first - num_prompt : end - num_prompt]
            )
            rows[:, USER_STREAM_OFFSET:] = self.user_rows_on_device(data)[first:end]
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
            rows.append(self.missing_rows(data, req))
            predicted = len(req.origin_input_ids) + len(req.output_ids)
            inputs["agent_row"] = self.agent_row_at(data, predicted - 1)
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
        sampled = result.next_token_ids
        for index, request in enumerate(requests):
            inputs = request.data.talker_model_inputs
            forced = inputs.pop("prefill_forced")
            self.spell_frame(index, request, sampled[index], forced)

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
        for request, req in zip(requests, schedule_batch.reqs, strict=True):
            data = request.data
            position = len(req.origin_input_ids) + len(req.output_ids) - 1
            row = torch.empty(NUM_STREAMS, dtype=torch.long, device=self.model_device)
            row[0] = int(req.output_ids[-1])
            row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = data.talker_model_inputs[
                "agent_row"
            ]
            row[USER_STREAM_OFFSET:] = self.user_rows_on_device(data)[position]
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
        sampled = result.next_token_ids
        free = self.free_codes()
        for index, request in enumerate(requests):
            self.spell_frame(index, request, sampled[index], free)
