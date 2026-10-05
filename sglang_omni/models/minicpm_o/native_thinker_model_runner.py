# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.minicpm_o.duplex_sampler import (
    build_forbidden_token_index,
    duplex_sample,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner as OfflineThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.thinker_state import DuplexUnitRequestData
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import session_prefill_rows
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class MiniCPMOThinkerModelRunner(OfflineThinkerModelRunner):
    """Run duplex units on the shared MiniCPM-o runner with media history."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.special_tokens: MiniCPMOSpecialTokenIds | None = None
        self.forbidden_token_index: torch.Tensor | None = None

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        embed_tokens = self.embed_tokens
        last_token_id = embed_tokens.num_embeddings - 1
        rows = [
            session_prefill_rows(
                request.data,
                lambda token_ids: embed_tokens(token_ids.clamp(0, last_token_id)),
                forward_batch.input_ids.device,
            )
            for request in requests
        ]
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=torch.cat(rows, dim=0)),
        )
        return None

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
        # note (Junnan Li): Lookahead would sample before repetition history advances.
        return False

    def resolve_special_tokens(
        self, data: DuplexUnitRequestData
    ) -> MiniCPMOSpecialTokenIds:
        if self.special_tokens is None:
            self.special_tokens = resolve_special_token_ids(
                data.req.tokenizer,
                bad_token_ids=tuple(data.req.tokenizer.bad_token_ids),
            )
        else:
            pass
        return self.special_tokens

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        logits = logits_output.next_token_logits
        result = torch.empty(len(requests), dtype=torch.long, device=logits.device)
        for index, request in enumerate(requests):
            data = request.data
            special_tokens = self.resolve_special_tokens(data)
            if self.forbidden_token_index is None:
                self.forbidden_token_index = build_forbidden_token_index(
                    special_tokens, logits.shape[-1], logits.device
                )
            else:
                pass
            result[index] = duplex_sample(
                logits[index],
                data.thinker_state,
                special_tokens=special_tokens,
                forbidden_token_index=self.forbidden_token_index,
                generation_step=data.generation_steps,
                is_listen_forced=data.is_listen_forced,
            )
            if data.is_listen_forced and data.generation_steps == 0:
                data.thinker_state.force_listen_counter += 1
            else:
                pass
        return result

    # note (Junnan Li): FULL capture must match decode graphs and retain talker conditioning.
    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        return CaptureHiddenMode.FULL

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Pair generated tokens with their next-step hidden states for the talker."""
        for scheduler_request in scheduler_output.requests:
            request_output = outputs[scheduler_request.request_id]
            data = scheduler_request.data
            sampled_token_id = int(request_output.data)
            special_tokens = self.resolve_special_tokens(data)
            pending_token_id = data.pending_unit_token
            if pending_token_id is not None and data.generation_steps >= 2:
                hidden_state = request_output.extra["hidden_states"]
                hidden_state = (
                    hidden_state.reshape(-1, hidden_state.shape[-1])[-1]
                    .detach()
                    .clone()
                )
                data.talker_conditions.append(
                    (
                        pending_token_id,
                        hidden_state.to("cpu"),
                        pending_token_id == special_tokens.turn_eos,
                    )
                )
            else:
                pass
            if sampled_token_id in special_tokens.chunk_terminators:
                data.pending_unit_token = None
            else:
                if data.generation_steps > 0:
                    data.generated_unit_ids.append(sampled_token_id)
                else:
                    pass
                data.pending_unit_token = sampled_token_id


__all__ = [
    "MiniCPMOThinkerModelRunner",
]
