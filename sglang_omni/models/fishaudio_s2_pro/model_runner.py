# SPDX-License-Identifier: Apache-2.0
"""Fish Audio S2-Pro model runner built on the phase-aware AR base runner."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.fishaudio_s2_pro.sglang_model import _NO_SEED
from sglang_omni.sampling.seed import resolve_row_seed
from sglang_omni.scheduling.types import SchedulerRequest


def collect_s2pro_step_outputs(
    result: Any,
    requests: list,
    *,
    output_codes: torch.Tensor,
    output_semantic_ids: torch.Tensor,
    im_end_token_id: int,
    rep_history_len: int | None = None,
) -> None:
    batch_size = len(requests)
    if batch_size == 0:
        return
    else:
        pass

    result.next_token_ids = output_semantic_ids[:batch_size].clone()
    semantic_tokens = output_semantic_ids[:batch_size].tolist()

    for row_idx, sched_req in enumerate(requests):
        data = sched_req.data
        if data.req.inflight_middle_chunks > 0:
            continue
        else:
            pass

        semantic_token = semantic_tokens[row_idx]
        if semantic_token == im_end_token_id:
            continue
        else:
            pass

        codes = output_codes[row_idx].unsqueeze(-1).clone()
        data.last_codebook_values = codes[1:, 0].clone()
        data.previous_semantic_tokens.append(semantic_token)
        if rep_history_len is not None:
            append_semantic_history(data, output_semantic_ids[row_idx], rep_history_len)
        else:
            pass
        data.output_codes.append(codes)
        data.latest_stream_code_chunk = codes


def append_semantic_history(data: Any, token: torch.Tensor, history_len: int) -> None:
    history = data.semantic_history_tokens
    if (
        history is None
        or history.device != token.device
        or history.shape[0] != history_len
    ):
        history = torch.zeros(history_len, dtype=torch.long, device=token.device)
        data.semantic_history_tokens = history
        data.semantic_history_count = 0
    else:
        pass

    count = int(data.semantic_history_count)
    if count < history_len:
        history[count].copy_(token)
    else:
        history[:-1].copy_(history[1:].clone())
        history[-1].copy_(token)
    data.semantic_history_count = count + 1


class FishS2ProModelRunner(ModelRunner):
    """Fish TTS runner with unified forward-owned decode and persistent buffers."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self.semantic_begin_id = int(self.model.semantic_begin_id)
        self.semantic_end_id = int(self.model.semantic_end_id)
        self.im_end_token_id = int(self.model.im_end_token_id)

    def lookahead_eligible(self, batch: Any) -> bool:
        # note (Junnan Li): not supported yet; semantic_history_tokens is
        # appended at resolve, one step late under lookahead.
        del batch
        return False

    def before_prefill(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        self.sync_decode_state(requests)
        input_embeds = self.build_prefill_input_embeds(forward_batch, requests)
        if input_embeds is not None:
            forward_batch.input_embeds = input_embeds
        else:
            pass

    def before_decode(
        self,
        forward_batch,
        schedule_batch,
        requests,
        *,
        is_lookahead: bool = False,
    ):
        del is_lookahead
        del schedule_batch
        input_ids = forward_batch.input_ids
        batch_size = input_ids.shape[0]
        is_semantic = (input_ids >= self.semantic_begin_id) & (
            input_ids <= self.semantic_end_id
        )
        self.model.vq_mask[:batch_size].copy_(is_semantic)

        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            self.sync_decode_row_state(row_idx, data)

            last_codes = data.last_codebook_values
            if last_codes is None:
                continue
            else:
                pass
            self.model.vq_codes[row_idx].copy_(
                last_codes.to(
                    device=self.model.vq_codes.device,
                    dtype=self.model.vq_codes.dtype,
                )
            )

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self.collect_step_outputs(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        del forward_batch, schedule_batch
        self.collect_step_outputs(result, requests)

    def sync_decode_state(self, requests: list) -> None:
        for row_idx, sched_req in enumerate(requests):
            self.sync_decode_row_state(row_idx, sched_req.data)

    def sync_decode_row_state(self, row_idx: int, data: Any) -> None:
        self.model.sampling_temperature[row_idx] = data.temperature
        self.model.sampling_top_p[row_idx] = data.top_p
        self.model.sampling_top_k[row_idx] = data.top_k
        self.model.sampling_rep_penalty[row_idx] = data.repetition_penalty
        self.model.ras_temperature[row_idx] = data.ras_temperature
        self.model.ras_top_p[row_idx] = data.ras_top_p
        self.model.sampling_seeds[row_idx] = (
            _NO_SEED if data.seed is None else resolve_row_seed(data.seed)
        )
        # semantic_history_count is the uncapped per-request AR step (pre-step).
        self.model.step_count[row_idx] = int(data.semantic_history_count)

        history_len = self.model.rep_history_len
        history = data.semantic_history_tokens
        if history is not None:
            self.model.prev_tokens[row_idx].copy_(
                history.to(
                    device=self.model.prev_tokens.device,
                    dtype=self.model.prev_tokens.dtype,
                )
            )
            self.model.prev_token_count[row_idx] = min(
                int(data.semantic_history_count), history_len
            )
        else:
            self.model.prev_tokens[row_idx].zero_()
            self.model.prev_token_count[row_idx] = 0

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        input_ids = forward_batch.input_ids
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Fish prefill expects tensor input_ids")
        else:
            pass

        device = input_ids.device
        text_embeds = self.model.get_embed_tokens()(input_ids)
        offset = 0

        for sched_req in requests:
            data = sched_req.data
            req = data.req
            extend_start = req.extend_range.start
            extend_end = req.extend_range.end
            extend_length = req.extend_range.length
            prompt_length = len(data.input_ids)

            if data.vq_mask_tokens is not None and data.vq_parts:
                reference_mask = data.vq_mask_tokens.reshape(-1).to(device=device)
                # note (luojiaxuan): reference masks end at the original prompt.
                reference_length = max(0, min(extend_end, prompt_length) - extend_start)
                mask_slice = reference_mask[
                    extend_start : extend_start + reference_length
                ]
                if bool(mask_slice.any()):
                    reference_codes = torch.cat(
                        [part.to(device=device).T for part in data.vq_parts], dim=0
                    )
                    reference_start = int(reference_mask[:extend_start].sum().item())
                    reference_end = reference_start + int(mask_slice.sum().item())
                    reference_embeds = text_embeds[offset : offset + reference_length]
                    fused = self.model.audio_decoder.embed_text_dim(
                        reference_embeds.unsqueeze(0),
                        reference_codes[reference_start:reference_end],
                        mask_slice.unsqueeze(0),
                    )
                    text_embeds[mask_slice.nonzero(as_tuple=True)[0] + offset] = (
                        fused.to(text_embeds.dtype)
                    )
                else:
                    pass
            else:
                pass

            if extend_end > prompt_length:
                # note (luojiaxuan): a replay chunk can end before the generated tail.
                generated_start = max(extend_start, prompt_length) - prompt_length
                generated_end = extend_end - prompt_length
                assert 0 <= generated_start < generated_end <= len(data.output_codes)
                codes = torch.cat(
                    data.output_codes[generated_start:generated_end], dim=1
                ).T.to(device=device, dtype=torch.long)
                first_row = offset + max(prompt_length - extend_start, 0)
                last_row = offset + extend_length
                assert torch.equal(
                    codes[:, 0], input_ids[first_row:last_row]
                ), "Fish generated codes do not match replay token IDs"
                generated_mask = torch.ones(
                    generated_end - generated_start, dtype=torch.bool, device=device
                )
                fused = self.model.audio_decoder.embed_text_dim(
                    text_embeds[first_row:last_row].unsqueeze(0),
                    codes[:, 1:],
                    generated_mask.unsqueeze(0),
                )
                text_embeds[first_row:last_row] = fused.to(text_embeds.dtype)
            else:
                pass
            offset += extend_length

        return text_embeds

    def collect_step_outputs(self, result: Any, requests: list) -> None:
        collect_s2pro_step_outputs(
            result,
            requests,
            output_codes=self.model.output_codes,
            output_semantic_ids=self.model.output_semantic_ids,
            im_end_token_id=self.im_end_token_id,
            rep_history_len=self.model.rep_history_len,
        )
