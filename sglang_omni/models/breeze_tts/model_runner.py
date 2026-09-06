# SPDX-License-Identifier: Apache-2.0
"""Batched Breeze AR execution with adjacent CFG branch rows."""

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
        self._generations(requests)
        for request in requests:
            data = request.data
            req = data.req
            if len(req.prefix_indices) or req.output_ids:
                raise RuntimeError(
                    "Breeze serving does not support radix reuse or retraction"
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
        generations = self._generations(requests)
        if any(generation.feedback is None for generation in generations):
            raise RuntimeError("Breeze decode request has no complete feedback frame")
        feedback = torch.stack([generation.feedback for generation in generations])
        forward_batch.input_embeds = feedback.repeat_interleave(2, dim=0).contiguous()

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._advance(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self._advance(result, requests)

    @staticmethod
    def _generations(requests):
        if not requests or len(requests) % 2:
            raise RuntimeError("Breeze AR batches must contain complete CFG pairs")
        generations = []
        for index in range(0, len(requests), 2):
            cond, uncond = requests[index : index + 2]
            if cond.data.cfg_uncond is not uncond.data:
                raise RuntimeError("Breeze AR batches must contain adjacent CFG pairs")
            generations.append(cond.data.generation)
        return generations

    def _advance(self, result, requests):
        generations = self._generations(requests)
        model = self.model
        logits = result.logits_output.next_token_logits
        if logits is None or logits.shape[0] != len(requests):
            raise RuntimeError("Breeze expected one backbone logit row per CFG branch")

        eos_id = model.config.audio_vocab_size
        tokens = []
        for index, generation in enumerate(generations):
            params = generation.sampling
            tokens.append(
                sample_logits(
                    apply_cfg(logits[2 * index : 2 * index + 2], params.cfg_scale),
                    params,
                    generation.generator,
                    history=generation.history,
                    codebook_size=model.config.codec_codebook_size,
                    eos_token_id=eos_id,
                ).reshape(())
            )
        logical_tokens = torch.stack(tokens)
        # Both branches of every request advance and finish together. The base
        # runner does not resample when next_token_ids is already supplied.
        result.next_token_ids = logical_tokens.repeat_interleave(2)

        active = [
            index
            for index, token in enumerate(logical_tokens.tolist())
            if token != eos_id
        ]
        if not active:
            return

        hidden = result.logits_output.hidden_states
        if hidden is None or hidden.ndim != 2 or hidden.shape[0] != len(requests):
            raise RuntimeError("Breeze expected one backbone hidden row per CFG branch")
        branch_rows = torch.tensor(
            [row for index in active for row in (2 * index, 2 * index + 1)],
            device=hidden.device,
            dtype=torch.long,
        )
        active_generations = [generations[index] for index in active]
        frames = model.depth_decoder.decode_frames(
            hidden.index_select(0, branch_rows),
            logical_tokens[active],
            [generation.sampling for generation in active_generations],
            [generation.generator for generation in active_generations],
            codebook_size=model.config.codec_codebook_size,
        ).detach()
        feedback = model.depth_decoder.embed_frames(frames).detach()

        for row, (request_index, generation) in enumerate(
            zip(active, active_generations, strict=True)
        ):
            frame = frames[row].clone()
            generation.history.append(int(logical_tokens[request_index].item()))
            generation.codes.append(frame)
            generation.pending_chunk = frame
            # Clone views out of the compact batch so one request's retirement
            # cannot retain or alias another request's frame storage.
            generation.feedback = feedback[row].clone()
