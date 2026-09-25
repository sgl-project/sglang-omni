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
    DuplexSamplerState,
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
    """Extend the shared MiniCPM-o runner with duplex sampling and media history."""

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.special_tokens: MiniCPMOSpecialTokenIds | None = None

    @staticmethod
    def is_duplex_request(request: SchedulerRequest) -> bool:
        return isinstance(request.data, DuplexUnitRequestData)

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        duplex = [request for request in requests if self.is_duplex_request(request)]
        if not duplex:
            return super().custom_prefill_forward(
                forward_batch, schedule_batch, requests
            )
        else:
            if len(duplex) != len(requests):
                raise RuntimeError(
                    "offline and duplex thinker requests cannot share prefill"
                )
            else:
                pass

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
                # note (Junnan Li): The text-only wrapper accepts projected embeddings directly.
                OmniPrefillInputs(input_embeds=torch.cat(rows, dim=0)),
            )
            return None

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        if any(self.is_duplex_request(request) for request in requests):
            return True
        else:
            return super().sample_before_post_prefill(
                forward_batch, schedule_batch, requests
            )

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> bool:
        if any(self.is_duplex_request(request) for request in requests):
            return True
        else:
            return super().sample_before_post_decode(
                forward_batch, schedule_batch, requests
            )

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        # note (Junnan Li): Lookahead would sample before repetition history advances.
        return False

    def special_for_data(self, data: DuplexUnitRequestData) -> MiniCPMOSpecialTokenIds:
        if self.special_tokens is None:
            self.special_tokens = resolve_special_token_ids(data.req.tokenizer)
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
        duplex_indices = [
            index
            for index, request in enumerate(requests)
            if self.is_duplex_request(request)
        ]
        if not duplex_indices:
            return super().sample_next_token_ids(
                logits_output, forward_batch, schedule_batch, requests
            )
        else:
            logits = logits_output.next_token_logits
            if logits is None or logits.ndim != 2 or logits.shape[0] != len(requests):
                raise RuntimeError(
                    "duplex thinker requires logits shaped [batch, vocab]"
                )
            else:
                pass
            original_logits = logits.clone()
            if len(duplex_indices) == len(requests):
                result = torch.empty(
                    len(requests), dtype=torch.long, device=logits.device
                )
            else:
                result = (
                    super()
                    .sample_next_token_ids(
                        logits_output, forward_batch, schedule_batch, requests
                    )
                    .clone()
                )

            for index in duplex_indices:
                data = requests[index].data
                session = data.thinker_state
                if session is None:
                    raise RuntimeError("duplex request lost its thinker session state")
                else:
                    pass
                sampling_config = data.sampling_config
                sampler_state = DuplexSamplerState(
                    special_tokens=self.special_for_data(data),
                    generation_step=int(data.generation_steps),
                    force_listen_count=1 if data.forced_listen else 0,
                    force_listen_counter=0,
                    generated_history=session.generated_history,
                    current_turn_ended=session.current_turn_ended,
                    temperature=float(
                        sampling_config.get(
                            "temperature", DuplexSamplerState.temperature
                        )
                    ),
                    top_k=int(sampling_config.get("top_k", DuplexSamplerState.top_k)),
                    top_p=float(sampling_config.get("top_p", DuplexSamplerState.top_p)),
                    repetition_penalty=float(
                        sampling_config.get(
                            "repetition_penalty", DuplexSamplerState.repetition_penalty
                        )
                    ),
                    listen_prob_scale=float(
                        sampling_config.get(
                            "listen_prob_scale", DuplexSamplerState.listen_prob_scale
                        )
                    ),
                    greedy=bool(
                        sampling_config.get("greedy", DuplexSamplerState.greedy)
                    )
                    or str(sampling_config.get("decode_mode", "")) == "greedy",
                )
                token = duplex_sample(original_logits[index], sampler_state)
                session.current_turn_ended = sampler_state.current_turn_ended
                if data.forced_listen and int(data.generation_steps) == 0:
                    session.force_listen_counter += 1
                else:
                    pass
                result[index] = token
            return result

    # note (Junnan Li): FULL capture must match decode graphs and retain talker conditioning.
    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Capture talker conditioning; batch parameters follow the runner interface."""
        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: ScheduleBatch, requests: list[SchedulerRequest]
    ) -> CaptureHiddenMode:
        """Capture talker conditioning; batch parameters follow the runner interface."""
        return CaptureHiddenMode.FULL

    def post_process_outputs(
        self,
        result: GenerationBatchResult,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Pair generated tokens with their next-step hidden states for the talker."""
        offline_outputs = {
            request.request_id: outputs[request.request_id]
            for request in scheduler_output.requests
            if not self.is_duplex_request(request) and request.request_id in outputs
        }
        if offline_outputs:
            super().post_process_outputs(result, scheduler_output, offline_outputs)
        else:
            pass
        for sched_req in scheduler_output.requests:
            req_output = outputs.get(sched_req.request_id)
            data = sched_req.data
            if isinstance(data, DuplexUnitRequestData):
                if req_output is None or req_output.data is None:
                    continue
                else:
                    pass
                sampled = int(req_output.data)
                special = self.special_for_data(data)
                hidden = last_hidden(req_output.extra)
                pending = data.pending_unit_token
                if pending is not None and int(data.generation_steps) >= 2:
                    if hidden is None:
                        raise RuntimeError(
                            "duplex speech conditioning requires thinker hidden state"
                        )
                    else:
                        pass
                    data.unit_pairs.append(
                        (pending, hidden, pending == special.turn_eos)
                    )
                    if len(data.unit_pairs) > 20:
                        del data.unit_pairs[:-20]
                    else:
                        pass
                else:
                    pass
                if sampled in special.chunk_terminators:
                    data.pending_unit_token = None
                    continue
                else:
                    pass
                if int(data.generation_steps) > 0:
                    data.generated_unit_ids.append(sampled)
                else:
                    pass
                data.pending_unit_token = sampled
                state = data.thinker_state
                if state is not None:
                    if sampled == special.turn_eos:
                        state.current_turn_ended = True
                    elif sampled not in special.chunk_terminators:
                        state.current_turn_ended = False
                    else:
                        pass
                else:
                    pass
                continue
            else:
                pass


def last_hidden(
    extra: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None,
) -> torch.Tensor | None:
    if not isinstance(extra, dict):
        return None
    else:
        hidden = extra.get("hidden_states")
        if isinstance(hidden, dict):
            hidden = next(
                (
                    value
                    for value in reversed(list(hidden.values()))
                    if torch.is_tensor(value)
                ),
                None,
            )
        else:
            pass
        if not torch.is_tensor(hidden):
            return None
        else:
            while hidden.ndim > 1 and hidden.shape[0] == 1:
                hidden = hidden[0]
            if hidden.ndim == 2:
                hidden = hidden[-1]
            else:
                pass
            return hidden.detach().clone().to("cpu")


__all__ = [
    "MiniCPMOThinkerModelRunner",
]
