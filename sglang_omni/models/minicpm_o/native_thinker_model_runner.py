# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker model runner."""

from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
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
    duplex_sample_batch,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_model_runner import (
    MiniCPMOThinkerModelRunner as OfflineThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.thinker_state import DuplexUnitRequestData
from sglang_omni.scheduling.sglang_backend.history_cut import keep_extended_prefix
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
        # note (ruinique): duplex_sample_batch owns the native runner's termination tokens.
        super().__init__(tp_worker, output_processor, eos_token_ids=[])
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
                prefix_len=prefix_len,
                extend_len=extend_len,
            )
            for request, prefix_len, extend_len in zip(
                requests,
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                strict=True,
            )
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
        special_tokens = self.resolve_special_tokens(requests[0].data)
        if self.forbidden_token_index is None:
            self.forbidden_token_index = build_forbidden_token_index(
                special_tokens, logits.shape[-1], logits.device
            )
        else:
            pass
        sampled_rows = [
            row
            for row, request in enumerate(requests)
            if not request.data.draft_token_ids
        ]
        token_ids = dict(
            zip(
                sampled_rows,
                duplex_sample_batch(
                    logits[sampled_rows],
                    [requests[row].data.thinker_state for row in sampled_rows],
                    special_tokens=special_tokens,
                    forbidden_token_index=self.forbidden_token_index,
                    generation_steps=[
                        requests[row].data.generation_steps for row in sampled_rows
                    ],
                    listen_forced_flags=[
                        requests[row].data.is_listen_forced for row in sampled_rows
                    ],
                    unit_indices=[
                        requests[row].data.unit_index for row in sampled_rows
                    ],
                ),
                strict=True,
            )
        )
        drafted_rows = [row for row in range(len(requests)) if row not in sampled_rows]
        if drafted_rows:
            logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
        else:
            logits_metadata = None
        for row in drafted_rows:
            data = requests[row].data
            draft_token_ids = data.draft_token_ids
            segment_end = sum(forward_batch.extend_seq_lens_cpu[: row + 1])
            # note (bianyutong): next_token_logits holds only the last drafted position, so the unit's first position and each drafted one take their logits from the FULL hidden capture.
            position_logits = self.outer_model.language_model.logits_processor._get_logits(  # noqa: leading-underscore  # upstream has no public call for the logits of chosen rows
                logits_output.hidden_states[
                    segment_end - len(draft_token_ids) - 1 : segment_end
                ],
                self.outer_model.lm_head,
                logits_metadata,
                use_logits_buffer=False,
            )
            for position in range(len(draft_token_ids) + 1):
                [token_id] = duplex_sample_batch(
                    position_logits[position : position + 1],
                    [data.thinker_state],
                    special_tokens=special_tokens,
                    forbidden_token_index=self.forbidden_token_index,
                    generation_steps=[data.generation_steps + position],
                    listen_forced_flags=[data.is_listen_forced],
                    unit_indices=[data.unit_index],
                )
                if (
                    position == len(draft_token_ids)
                    or token_id != draft_token_ids[position]
                ):
                    break
                else:
                    pass
            data.accepted_draft_count = position
            token_ids[row] = token_id
        for request in requests:
            data = request.data
            if data.is_listen_forced and data.generation_steps == 0:
                data.thinker_state.force_listen_counter += 1
            else:
                pass
        return torch.tensor(
            [token_ids[row] for row in range(len(requests))],
            dtype=torch.long,
            device=logits.device,
        )

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
        for row, scheduler_request in enumerate(scheduler_output.requests):
            request_output = outputs[scheduler_request.request_id]
            data = scheduler_request.data
            if data.is_draft:
                continue
            else:
                pass
            special_tokens = self.resolve_special_tokens(data)
            draft_count = len(data.draft_token_ids)
            step_token_ids = [
                *data.draft_token_ids[: data.accepted_draft_count],
                int(request_output.data),
            ]
            hidden_states = request_output.extra["hidden_states"]
            hidden_state_rows = hidden_states.reshape(-1, hidden_states.shape[-1])
            # note (bianyutong): A drafted prefill ran one step per decided token, and each step read the prompt row of the token before it, among the last draft_count + 1 rows.
            for position, sampled_token_id in enumerate(step_token_ids):
                pending_token_id = data.pending_unit_token
                if (
                    pending_token_id is not None
                    and data.generation_steps + position >= 2
                ):
                    hidden_state = (
                        hidden_state_rows[
                            hidden_state_rows.shape[0] - draft_count - 1 + position
                        ]
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
                    if data.generation_steps + position > 0:
                        data.generated_unit_ids.append(sampled_token_id)
                    else:
                        pass
                    data.pending_unit_token = sampled_token_id
            if draft_count:
                prompt_token_count = len(data.req.origin_input_ids) - draft_count
                keep_extended_prefix(
                    scheduler_output.batch_data,
                    row,
                    prompt_token_count=prompt_token_count,
                    kept_token_count=prompt_token_count + data.accepted_draft_count,
                )
                # note (bianyutong): finalize advances the step once for the whole forward.
                data.generation_steps += data.accepted_draft_count
                data.draft_token_ids = []
            else:
                pass


__all__ = [
    "MiniCPMOThinkerModelRunner",
]
