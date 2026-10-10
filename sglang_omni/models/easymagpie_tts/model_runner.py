# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie request lifecycle on top of the shared SGLang model runner.

Each AR step conditions on three delayed streams: the text token at the
request offset, the previous phoneme row once the phoneme delay has passed,
and the previous acoustic frame once the speech delay has passed. Prefill
seeds that state on the GPU (``decode_state``); decode steps keep it there,
so the host only collects each step's audio frames, one step behind under
async decode.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.easymagpie_tts.decode_state import EMIT_COLUMN, STOP_COLUMN
from sglang_omni.models.easymagpie_tts.request_builders import (
    EasyMagpieSGLangRequestData,
)


class EasyMagpieTTSModelRunner(ModelRunner):
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def build_prompt_embeds(
        self, data: EasyMagpieSGLangRequestData, device: torch.device
    ) -> torch.Tensor:
        """Speaker rows, context text, then the text lead-in with phoneme BOS.

        The lead-in rows are every step whose conditioning is known before
        any prediction exists, so they are folded into prefill.
        """
        state = data.state
        heads = self.model.heads
        config = self.model.tts_config
        dtype = self.model_dtype()
        rows = []
        if state.speaker_embedding is not None:
            rows.append(state.speaker_embedding.to(device=device, dtype=dtype))
        else:
            pass
        if state.context_token_ids:
            context = torch.tensor(state.context_token_ids, device=device)
            rows.append(heads.text_embedding(context).to(dtype))
        else:
            pass
        lead_in = torch.zeros(
            (state.text_prefill_num, config.embedding_dim), device=device, dtype=dtype
        )
        lead_in_ids = state.text_token_ids[: state.text_prefill_num]
        if lead_in_ids:
            ids = torch.tensor(lead_in_ids, device=device)
            lead_in[: len(lead_in_ids)] = heads.text_embedding(ids).to(dtype)
        else:
            pass
        bos = torch.full(
            (1, config.phoneme_stacking_factor),
            config.phoneme_bos_id,
            device=device,
            dtype=torch.long,
        )
        lead_in[state.phoneme_delay] += heads.embed_phonemes(bos)[0].to(dtype)
        rows.append(lead_in)
        return torch.cat(rows, dim=0)

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        device = forward_batch.input_ids.device
        pieces = []
        for request in requests:
            data = request.data
            prompt = self.build_prompt_embeds(data, device)
            start = len(data.req.prefix_indices)
            length = int(data.req.extend_range.length)
            piece = prompt[start : start + length]
            if piece.shape[0] != length:
                raise RuntimeError(
                    f"EasyMagpie prefill has {piece.shape[0]} rows, "
                    f"the scheduler expects {length}"
                )
            else:
                pass
            pieces.append(piece)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=torch.cat(pieces, dim=0),
                input_embeds_are_projected=True,
            ),
        )
        return None

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch, schedule_batch
        self.model.decode_state.seed(
            slots=[request.data.req.kv.req_pool_idx for request in requests],
            states=[request.data.state for request in requests],
            seeds=[request.data.sampling_seed for request in requests],
            prefill_phonemes=self.model.last_phoneme_tokens[: len(requests)],
        )

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        step = self.model.decode_state.step_output[: len(requests)]
        result.next_token_ids = step[:, STOP_COLUMN].clone()
        self.collect_frames(step.cpu(), requests)

    def post_decode_launch(self, result: Any, forward_batch: Any, requests: list):
        """Copy the step output to pinned host memory without waiting for it."""
        del forward_batch
        if not requests:
            return None
        else:
            pass
        output = self.model.decode_state.step_output
        rows = len(requests)
        host = self.next_host_staging(output.shape, output.dtype)
        host[:rows].copy_(output[:rows], non_blocking=True)
        result.next_token_ids = output[:rows, STOP_COLUMN].clone()
        return host

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        if launch_buf is None or not requests:
            return
        else:
            pass
        step = launch_buf[: len(requests)]
        result.next_token_ids = step[:, STOP_COLUMN]
        self.collect_frames(step, requests)

    def collect_frames(self, step: torch.Tensor, requests: list) -> None:
        """Append each row's audio frame to its request.

        A lookahead step also runs requests that finished in the step before
        it; their rows are dropped.
        """
        frames = step[:, : self.model.tts_config.num_stacked_codebooks].clone()
        emits = step[:, EMIT_COLUMN].tolist()
        for row, request in enumerate(requests):
            req = request.data.req
            if emits[row] and not (req.finished() or self.req_is_retracted(req)):
                request.data.output_codes.append(frames[row])
            else:
                pass


__all__ = ["EasyMagpieTTSModelRunner"]
