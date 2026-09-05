# SPDX-License-Identifier: Apache-2.0
"""One logical Breeze request per AR batch, two atomic CFG branch rows."""

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)

from .sampling import apply_cfg, sample_logits


class BreezeModelRunner(ModelRunner):
    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        pieces = []
        for request in requests:
            data = request.data
            req = data.req
            if len(req.prefix_indices) or req.output_ids:
                raise RuntimeError(
                    "Breeze initial serving does not support radix reuse or retraction"
                )
            if int(req.extend_range.length) != len(data.prefill_input_embeds):
                raise RuntimeError(
                    "Breeze CFG prompts must be admitted without chunking"
                )
            pieces.append(data.prefill_input_embeds)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(pieces), input_embeds_are_projected=True
            ),
        )

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ):
        del schedule_batch, is_lookahead
        generation = self._generation(requests)
        forward_batch.input_embeds = (
            generation.feedback.reshape(1, -1).expand(2, -1).contiguous()
        )

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._advance(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._advance(result, requests)

    @staticmethod
    def _generation(requests):
        if len(requests) != 2 or requests[0].data.cfg_uncond is not requests[1].data:
            raise RuntimeError("Breeze AR batches must contain one adjacent CFG pair")
        return requests[0].data.generation

    def _advance(self, result, requests):
        generation = self._generation(requests)
        params = generation.sampling
        model = self.model
        eos_id = model.config.audio_vocab_size
        token = sample_logits(
            apply_cfg(result.logits_output.next_token_logits, params.cfg_scale),
            params,
            generation.generator,
            history=generation.history,
            codebook_size=model.config.codec_codebook_size,
            eos_token_id=eos_id,
        )
        token_id = int(token.item())
        # Both branches advance/finish together; the base runner will not
        # resample when next_token_ids has already been supplied.
        result.next_token_ids = token.expand(2).clone()
        if token_id == eos_id:
            return
        codes = model.depth_decoder.decode_frame(
            result.logits_output.hidden_states,
            token,
            params,
            generation.generator,
            codebook_size=model.config.codec_codebook_size,
        ).detach()
        generation.history.append(token_id)
        generation.codes.append(codes)
        generation.pending_chunk = codes
        generation.feedback = model.depth_decoder.embed_frames(codes).detach()
