from __future__ import annotations

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState


class NemotronVoiceChatModelRunner(ModelRunner):
    def before_prefill(self, forward_batch, schedule_batch, requests) -> None:
        del schedule_batch
        rows = [self._prefill_rows(request.data) for request in requests]
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(rows, dim=0),
                input_embeds_are_projected=True,
            ),
        )

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead: bool = False
    ) -> None:
        del forward_batch, schedule_batch, is_lookahead
        buffer = self.model._fusion_buffer
        batch = len(requests)
        rows = [
            self._decode_row(request.data, buffer.device, buffer.dtype)
            for request in requests
        ]
        buffer[:batch] = torch.stack(rows, dim=0)
        self.model._fusion_mask[:batch] = True

    @staticmethod
    def _acoustic_frames(data) -> torch.Tensor:
        return NemotronVoiceChatState.from_dict(data.stage_payload.data).acoustic_frames

    def _acoustic_row(self, data, index: int) -> torch.Tensor:
        return self._acoustic_frames(data)[index]

    def _prefill_rows(self, data) -> torch.Tensor:
        """The instruction rides the caller's audio channel, text and function
        are padding, and the last position is the first acoustic frame."""
        embeddings = self.model.llm.get_input_embeddings()
        device = embeddings.weight.device
        ids = data.input_ids.to(device)
        row = self._acoustic_row(data, 0)
        spoken = embeddings(ids[:-1])
        heard = torch.cat(
            [spoken, row.to(device=spoken.device, dtype=spoken.dtype).reshape(1, -1)],
            dim=0,
        )
        # The last id the request carries is the padding one, and every
        # position's text and function channels are that same padding.
        pad = embeddings(torch.full_like(ids, int(ids[-1])))
        fusion = self.model.fusion
        return (
            pad * fusion.text_weight
            + heard * fusion.user_weight
            + pad * fusion.function_weight
        )

    def _decode_row(self, data, device, dtype) -> torch.Tensor:
        output_ids = data.req.output_ids
        frame_index = len(output_ids)
        embeddings = self.model.llm.get_input_embeddings()
        tokens = torch.tensor(
            [output_ids[-1], data.extra_model_outputs["function_ids"][-1]],
            dtype=torch.long,
            device=embeddings.weight.device,
        )
        text, function = embeddings(tokens).to(device=device, dtype=dtype)
        acoustic = self._acoustic_row(data, frame_index).to(device=device, dtype=dtype)
        return self.model.fusion(acoustic, text, function)

    def _record_function_ids(self, requests) -> None:
        sampled = self.model._function_ids[: len(requests)].tolist()
        for request, token in zip(requests, sampled):
            request.data.extra_model_outputs.setdefault("function_ids", []).append(
                token
            )

    @staticmethod
    def _record_stream_tokens(result, requests) -> None:
        # Sampling has not happened yet in the post hooks; the thinker decodes
        # greedily, so the argmax IS the token the sampler will record.
        logits = result.logits_output.next_token_logits
        sampled = logits[: len(requests)].argmax(dim=-1).tolist()
        for request, token in zip(requests, sampled):
            request.data.pending_stream_tokens.append(int(token))

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        del forward_batch, schedule_batch
        self._record_function_ids(requests)
        self._record_stream_tokens(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        del forward_batch, schedule_batch
        self._record_function_ids(requests)
        self._record_stream_tokens(result, requests)
