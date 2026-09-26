# SPDX-License-Identifier: Apache-2.0
"""Runs one PersonaPlex request through the timeline, one row per forward.

Prefill embeds every prompt row at once. Each decode step then fuses the row
for the next position from the last sampled text token, the agent codes the
depformer produced for that position, and the caller's codes for it. The
text token is sampled *before* the post hook so the depformer can consume it
in the same step; the codes it spells out become the next row's input and,
one position later, a finished output frame streamed to the codec.
"""

from __future__ import annotations

import torch

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


class PersonaPlexModelRunner(ModelRunner):
    def sample_before_post_prefill(
        self, forward_batch, schedule_batch, requests
    ) -> bool:
        return True

    def sample_before_post_decode(
        self, forward_batch, schedule_batch, requests
    ) -> bool:
        return True

    def lookahead_eligible(self, batch) -> bool:
        # Note (wilsonzheng0327): The depformer must see this step's sampled text before
        # the next forward is prepared; a one-step lookahead would run it a step late.
        return False

    @property
    def model_device(self) -> torch.device:
        return self.model.fusion_buffer.device

    @staticmethod
    def request_timeline(data) -> Timeline:
        return data.talker_model_inputs["timeline"]

    def user_rows_on_device(self, data) -> torch.Tensor:
        """The request's user rows on the device, copied once and extended as a
        session appends caller frames."""
        inputs = data.talker_model_inputs
        rows = self.request_timeline(data).user_rows
        cached = inputs.get("device_user_rows")
        if cached is None:
            cached = rows.to(self.model_device)
        elif cached.shape[0] < rows.shape[0]:
            cached = torch.cat([cached, rows[cached.shape[0] :].to(self.model_device)])
        elif cached.shape[0] > rows.shape[0]:
            cached = cached[: rows.shape[0]]
        else:
            pass
        inputs["device_user_rows"] = cached
        return cached

    def agent_row_at(self, data, position: int) -> torch.Tensor:
        """Agent codebooks the timeline holds at position."""
        timeline = self.request_timeline(data)
        if position < timeline.num_prompt_positions:
            return timeline.agent_row_before_start.to(self.model_device)
        else:
            return data.talker_model_inputs["agent_rows"][
                position - timeline.num_prompt_positions
            ]

    def prefill_rows(self, data) -> torch.Tensor:
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

    def audio_sampler(self, data):
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
        self, index: int, request, text_token: torch.Tensor, forced: torch.Tensor
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

    def history_rows(self, data, history: list[int], start: int) -> torch.Tensor:
        """Embed positions start .. len(history)-1.

        Below the prompt length a row is the prompt's own; past it, the text
        token history holds at that position, the agent codes the depformer
        spelled there, and the caller's codes.
        """
        model = self.model
        timeline = self.request_timeline(data)
        num_prompt = timeline.num_prompt_positions
        pieces = []
        if start < num_prompt:
            pieces.append(self.prefill_rows(data)[start:])
        else:
            pass
        first = max(start, num_prompt)
        if first < len(history):
            user_rows = self.user_rows_on_device(data)
            agent_rows = data.talker_model_inputs["agent_rows"]
            rows = torch.empty(
                len(history) - first,
                NUM_STREAMS,
                dtype=torch.long,
                device=self.model_device,
            )
            for index, position in enumerate(range(first, len(history))):
                rows[index, 0] = int(history[position])
                rows[index, AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = agent_rows[
                    position - num_prompt
                ]
                rows[index, USER_STREAM_OFFSET:] = user_rows[position]
            pieces.append(model.embed_rows(rows).to(model.fusion_buffer.dtype))
        else:
            pass
        return torch.cat(pieces, dim=0)

    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        rows = []
        for request, req in zip(requests, schedule_batch.reqs, strict=True):
            data = request.data
            inputs = data.talker_model_inputs
            timeline = self.request_timeline(data)
            # Note (wilsonzheng0327): A retracted request replays its generated
            # positions; a session append extends the positions its KV lacks.
            history = [int(token) for token in req.origin_input_ids]
            history.extend(int(token) for token in req.output_ids)
            rows.append(self.history_rows(data, history, len(req.prefix_indices)))
            predicted = len(history)
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

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        sampled = result.next_token_ids
        for index, request in enumerate(requests):
            inputs = request.data.talker_model_inputs
            forced = inputs.pop("prefill_forced", None)
            if forced is None:
                forced = self.request_timeline(request.data).forced_agent_at_start.to(
                    self.model_device
                )
            else:
                pass
            self.spell_frame(index, request, sampled[index], forced)

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
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

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        sampled = result.next_token_ids
        free = self.free_codes()
        for index, request in enumerate(requests):
            self.spell_frame(index, request, sampled[index], free)
