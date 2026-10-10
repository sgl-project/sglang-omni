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


class PrefillPlan:
    """Where each prompt row of a prefill batch comes from.

    Row ``p`` of a request's prompt is speaker row ``p`` while ``p`` is
    inside the voice, then the context tokens, then the text lead-in. Rows
    are numbered in the batch's packed order.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.speaker_src: list[int] = []
        self.speaker_dst: list[int] = []
        self.text_ids: list[int] = []
        self.text_dst: list[int] = []
        self.bos_dst: list[int] = []

    def add(
        self, data: EasyMagpieSGLangRequestData, spans: dict[str, tuple[int, int]]
    ) -> None:
        state = data.state
        start = len(data.req.prefix_indices)
        length = int(data.req.extend_range.length)
        speaker_start, speaker_frames = spans[state.voice]
        text_start = speaker_frames + len(state.context_token_ids)
        prompt_len = text_start + state.text_prefill_num
        if start + length > prompt_len:
            raise RuntimeError(
                f"EasyMagpie prefill has {max(prompt_len - start, 0)} rows, "
                f"the scheduler expects {length}"
            )
        else:
            pass
        end = start + length
        packed = self.rows - start
        voice = range(start, min(end, speaker_frames))
        self.speaker_src.extend(speaker_start + position for position in voice)
        self.speaker_dst.extend(packed + position for position in voice)
        text = state.context_token_ids + state.text_token_ids[: state.text_prefill_num]
        for index, token in enumerate(text):
            if start <= speaker_frames + index < end:
                self.text_ids.append(token)
                self.text_dst.append(packed + speaker_frames + index)
            else:
                pass
        if start <= text_start + state.phoneme_delay < end:
            self.bos_dst.append(packed + text_start + state.phoneme_delay)
        else:
            pass
        self.rows += length

    def indices(self) -> list[int]:
        return (
            self.speaker_src
            + self.speaker_dst
            + self.text_ids
            + self.text_dst
            + self.bos_dst
        )

    def sizes(self) -> list[int]:
        return [
            len(self.speaker_src),
            len(self.speaker_dst),
            len(self.text_ids),
            len(self.text_dst),
            len(self.bos_dst),
        ]


class EasyMagpieTTSModelRunner(ModelRunner):
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def build_prefill_embeds(
        self, requests: list, device: torch.device
    ) -> torch.Tensor:
        """Every request's speaker rows, context text, then text lead-in.

        The lead-in rows are every step whose conditioning is known before
        any prediction exists, so they are folded into prefill; the one at
        the phoneme delay also carries the phoneme BOS. Only each request's
        scheduled window of its prompt is built, and the whole batch takes
        one index upload, one speaker gather and one text-embedding lookup.
        """
        plan = PrefillPlan()
        for request in requests:
            plan.add(request.data, self.model.speaker_table.spans)
        heads = self.model.heads
        config = self.model.tts_config
        dtype = self.model_dtype()
        (speaker_src, speaker_dst, text_ids, text_dst, bos_dst) = (
            torch.tensor(plan.indices(), dtype=torch.long)
            .to(device)
            .split(plan.sizes())
        )
        embeds = torch.zeros(
            (plan.rows, config.embedding_dim), device=device, dtype=dtype
        )
        embeds[speaker_dst] = self.model.speaker_table.rows[speaker_src]
        embeds[text_dst] = heads.text_embedding(text_ids).to(dtype)
        bos = torch.full(
            (1, config.phoneme_stacking_factor),
            config.phoneme_bos_id,
            device=device,
            dtype=torch.long,
        )
        bos_row = heads.embed_phonemes(bos).to(dtype)
        embeds.index_add_(0, bos_dst, bos_row.expand(bos_dst.numel(), -1))
        return embeds

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self.build_prefill_embeds(
                    requests, forward_batch.input_ids.device
                ),
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
