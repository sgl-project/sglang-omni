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
    def _device(self) -> torch.device:
        return self.model._fusion_buffer.device

    @staticmethod
    def _timeline(data) -> Timeline:
        return data.talker_model_inputs["timeline"]

    def _rows_to_device(self, data) -> dict:
        """Move the request's timeline tensors to the device once."""
        inputs = data.talker_model_inputs
        cached = inputs.get("device_rows")
        if cached is None:
            timeline = self._timeline(data)
            cached = {
                "user_rows": timeline.user_rows.to(self._device),
                "agent_row": timeline.agent_row_before_start.to(self._device),
            }
            inputs["device_rows"] = cached
        return cached

    def _prefill_rows(self, data) -> torch.Tensor:
        timeline = self._timeline(data)
        model = self.model
        tokens = timeline.prefill_tokens.to(self._device)
        dtype = model._fusion_buffer.dtype
        if not timeline.prefill_embedding_positions:
            return model.embed_rows(tokens).to(dtype)
        stored = torch.as_tensor(timeline.prefill_embeddings, device=self._device).to(
            dtype
        )
        known = torch.ones(tokens.shape[0], dtype=torch.bool, device=self._device)
        known[timeline.prefill_embedding_positions] = False
        rows = torch.empty(
            tokens.shape[0], stored.shape[1], dtype=dtype, device=self._device
        )
        rows[timeline.prefill_embedding_positions] = stored
        rows[known] = model.embed_rows(tokens[known]).to(dtype)
        return rows

    def _audio_sampler(self, data):
        inputs = data.talker_model_inputs
        sampling = inputs["sampling"]
        generator = inputs.get("audio_generator")
        if generator is None and sampling.audio_seed is not None:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(sampling.audio_seed)
            inputs["audio_generator"] = generator
        return lambda logits: sample_token(logits, sampling.audio, generator)

    def _spell_frame(
        self, index: int, request, text_token: torch.Tensor, forced: torch.Tensor
    ) -> None:
        """Run the depformer for the position just predicted and record it."""
        data = request.data
        inputs = data.talker_model_inputs
        device_rows = self._rows_to_device(data)
        hidden = self.model._hidden_out[index : index + 1]
        codes = self.model.depformer.generate(
            text_token.view(1), hidden, forced.view(1, -1), self._audio_sampler(data)
        )[0]
        frame = output_frame(device_rows["agent_row"], codes)
        device_rows["agent_row"] = codes
        inputs["agent_rows"].append(codes)
        inputs["frames"].append(frame)
        inputs["pending_frames"].append(frame)

    def _free_codes(self) -> torch.Tensor:
        return torch.full(
            (self.model.depformer.spec.steps,),
            -1,
            dtype=torch.long,
            device=self._device,
        )

    def _generated_rows(self, data, generated: list[int]) -> torch.Tensor:
        """Embed the positions already generated, replayed after a retract."""
        model = self.model
        timeline = self._timeline(data)
        device_rows = self._rows_to_device(data)
        agent_rows = data.talker_model_inputs["agent_rows"]
        start = timeline.num_prompt_positions
        rows = torch.empty(
            len(generated), NUM_STREAMS, dtype=torch.long, device=self._device
        )
        for index, token in enumerate(generated):
            rows[index, 0] = int(token)
            rows[index, AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = agent_rows[index]
            rows[index, USER_STREAM_OFFSET:] = device_rows["user_rows"][start + index]
        return model.embed_rows(rows).to(model._fusion_buffer.dtype)

    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        rows = []
        for request, req in zip(requests, schedule_batch.reqs, strict=True):
            data = request.data
            inputs = data.talker_model_inputs
            generated = [int(token) for token in req.output_ids]
            prompt_rows = self._prefill_rows(data)
            if not generated:
                inputs["prefill_forced"] = self._timeline(
                    data
                ).forced_agent_at_start.to(self._device)
                rows.append(prompt_rows)
                continue
            # Note (wilsonzheng0327): Resuming a retracted request replays the prompt
            # and every generated position, so those rows are embedded again too.
            self._rows_to_device(data)["agent_row"] = inputs["agent_rows"][
                len(generated) - 1
            ]
            inputs["prefill_forced"] = self._free_codes()
            rows.append(
                torch.cat([prompt_rows, self._generated_rows(data, generated)], dim=0)
            )
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(rows, dim=0), input_embeds_are_projected=True
            ),
        )

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        del forward_batch, schedule_batch
        sampled = result.next_token_ids
        for index, request in enumerate(requests):
            inputs = request.data.talker_model_inputs
            forced = inputs.pop("prefill_forced", None)
            if forced is None:
                forced = self._timeline(request.data).forced_agent_at_start.to(
                    self._device
                )
            self._spell_frame(index, request, sampled[index], forced)

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ) -> None:
        del forward_batch, is_lookahead
        model = self.model
        rows = []
        for request, req in zip(requests, schedule_batch.reqs, strict=True):
            data = request.data
            timeline = self._timeline(data)
            device_rows = self._rows_to_device(data)
            position = timeline.input_position(len(req.output_ids))
            row = torch.empty(NUM_STREAMS, dtype=torch.long, device=self._device)
            row[0] = int(req.output_ids[-1])
            row[AGENT_STREAM_OFFSET:USER_STREAM_OFFSET] = device_rows["agent_row"]
            row[USER_STREAM_OFFSET:] = device_rows["user_rows"][position]
            rows.append(row)
        batch = len(rows)
        model._fusion_buffer[:batch] = model.embed_rows(torch.stack(rows)).to(
            model._fusion_buffer.dtype
        )

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        del forward_batch, schedule_batch
        sampled = result.next_token_ids
        free = self._free_codes()
        for index, request in enumerate(requests):
            self._spell_frame(index, request, sampled[index], free)
