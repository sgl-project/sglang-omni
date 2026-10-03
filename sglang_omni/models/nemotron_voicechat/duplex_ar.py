# SPDX-License-Identifier: Apache-2.0
"""Frame fusion over scheduler-owned streaming KV sessions.

Each unit supplies fusion embeddings; the scheduler owns token history and KV.
Keep prior embeddings so a replayed prefix uses the original model inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from transformers import PreTrainedTokenizerBase

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import capture_cuda_graph
from sglang_omni.models.nemotron_voicechat.model_runner import (
    NemotronVoiceChatModelRunner,
)
from sglang_omni.models.nemotron_voicechat.request_builders import ar_request
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    NUM_ITER,
    NemotronVoiceChatTalkerModelRunner,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FrameHistory:
    fusion_rows: list[torch.Tensor] = field(default_factory=list)
    position_count: int = 0
    previous_text_token_id: int | None = None
    previous_function_token_id: int | None = None
    previous_codes: torch.Tensor | None = None
    text_token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    forwarded_positions: int = 0
    reused_prefix_positions: int = 0
    unit_count: int = 0


def attach_fusion_rows(
    forward_batch: ForwardBatch, requests: list[SchedulerRequest]
) -> None:
    fusion_suffixes: list[torch.Tensor] = []
    for request_index, request in enumerate(requests):
        history: FrameHistory = request.data.talker_model_inputs["duplex_history"]
        cached_positions = int(forward_batch.extend_prefix_lens_cpu[request_index])
        uncached_positions = int(forward_batch.extend_seq_lens_cpu[request_index])
        if cached_positions + uncached_positions != history.position_count:
            raise RuntimeError(
                "VoiceChat fusion history is not aligned with session KV"
            )
        else:
            pass
        # The common path needs just the final row; never concatenate the
        # entire conversation merely to slice off its last position.
        history.forwarded_positions += uncached_positions
        history.reused_prefix_positions += cached_positions
        history.unit_count += 1
        remaining_positions = uncached_positions
        suffix_blocks: list[torch.Tensor] = []
        for block in reversed(history.fusion_rows):
            block_positions = min(remaining_positions, block.shape[0])
            suffix_blocks.append(block[-block_positions:])
            remaining_positions -= block_positions
            if remaining_positions == 0:
                break
            else:
                pass
        if remaining_positions:
            raise RuntimeError("VoiceChat fusion history is incomplete")
        else:
            pass
        fusion_suffixes.extend(reversed(suffix_blocks))
    attach_omni_prefill_inputs(
        forward_batch,
        OmniPrefillInputs(
            input_embeds=torch.cat(fusion_suffixes), input_embeds_are_projected=True
        ),
    )


class DuplexThinkerRunner(NemotronVoiceChatModelRunner):
    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        attach_fusion_rows(forward_batch, requests)

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        raise RuntimeError("VoiceChat session units must perform exactly one forward")


class DuplexTalkerRunner(NemotronVoiceChatTalkerModelRunner):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.sampler_graph: torch.cuda.CUDAGraph | None = None
        self.sampler_hidden: torch.Tensor | None = None
        self.sampler_output: torch.Tensor | None = None

    def generate_codes(self, index: int) -> torch.Tensor:
        # Fixed one-frame sampler only: backbone/session KV remains scheduler-owned.
        # Replaying its small kernels avoids Python dispatch on every 80 ms unit.
        if self.model.hidden_out.device.type != "cuda":
            return super().generate_codes(index)
        else:
            pass
        if self.sampler_graph is None:
            sampler_hidden = self.model.hidden_out[index : index + 1].float().clone()
            iteration_fractions = torch.linspace(
                0, 1, NUM_ITER + 1, device=sampler_hidden.device
            )[:-1]
            remaining_quantizers = torch.ceil(
                (1 - iteration_fractions.pow(self.exponent)).pow(1 / self.exponent)
                * self.model.talker.num_quantizers
            ).long()
            assignment_counts = tuple(
                (
                    remaining_quantizers
                    - torch.cat(
                        [remaining_quantizers[1:], remaining_quantizers.new_zeros(1)]
                    )
                ).tolist()
            )

            def sample() -> torch.Tensor:
                return self.model.talker.generate_codes(
                    sampler_hidden,
                    self.model.mog_head,
                    num_iter=NUM_ITER,
                    exponent=self.exponent,
                    top_p=self.top_p,
                    noise_scale=self.noise_scale,
                    assignment_counts=assignment_counts,
                )

            self.sampler_graph, self.sampler_output = capture_cuda_graph(
                sample, sampler_hidden.device
            )
            self.sampler_hidden = sampler_hidden
        else:
            pass
        assert self.sampler_hidden is not None and self.sampler_output is not None
        self.sampler_hidden.copy_(self.model.hidden_out[index : index + 1])
        self.sampler_graph.replay()
        return self.sampler_output.clone()

    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        attach_fusion_rows(forward_batch, requests)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        for index, request in enumerate(requests):
            request.data.talker_model_inputs["duplex_codes"] = self.generate_codes(
                index
            )

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        raise RuntimeError("VoiceChat session units must perform exactly one forward")


class FrameAdapter(ARSessionAdapter):
    def __init__(
        self, runner: DuplexThinkerRunner | DuplexTalkerRunner, *, context_length: int
    ) -> None:
        self.runner = runner
        self.context_length = context_length
        self.states: dict[SessionIdentity, FrameHistory] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        if request.params.get("instructions"):
            raise ValueError("VoiceChat currently uses the checkpoint system prompt")
        else:
            pass
        self.states[session_identity] = FrameHistory()

    def close(self, session_identity: SessionIdentity) -> None:
        state = self.states.pop(session_identity, None)
        if state is not None:
            logger.info(
                f"VoiceChat {type(self).__name__} session closed: units={state.unit_count} "
                f"forwarded_positions={state.forwarded_positions} reused_prefix_positions={state.reused_prefix_positions}"
            )
        else:
            pass

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload | None:
        if payload.data.get("eos") and payload.data.get("acoustic") is None:
            return payload
        else:
            return None

    def build_unit_request(
        self,
        session_identity: SessionIdentity,
        payload: StagePayload,
        opening_token_ids: list[int],
        fusion_rows: torch.Tensor,
        vocab_size: int,
    ) -> SGLangARRequestData:
        state = self.states[session_identity]
        new_position_count = fusion_rows.shape[0]
        if state.position_count + new_position_count + 1 > self.context_length:
            raise ValueError(
                "VoiceChat session context limit reached; start a new session"
            )
        else:
            pass
        # note (Codex): The next unit forwards the previous sampled token with new fusion input.
        input_token_ids = opening_token_ids if state.position_count == 0 else []
        state.fusion_rows.append(fusion_rows.detach())
        state.position_count += new_position_count
        request_data = ar_request(
            payload, input_ids=input_token_ids, max_new_tokens=1, vocab_size=vocab_size
        )
        request_data.talker_model_inputs["duplex_history"] = state
        request_data.pending_stream_tokens = []
        return request_data


class ThinkerAdapter(FrameAdapter):
    runner: DuplexThinkerRunner

    def __init__(
        self,
        runner: DuplexThinkerRunner,
        *,
        prompt_token_ids: list[int],
        pad_token_id: int,
        tokenizer: PreTrainedTokenizerBase,
        context_length: int,
    ) -> None:
        super().__init__(runner, context_length=context_length)
        self.prompt_token_ids = prompt_token_ids
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.silent_token_ids = set(tokenizer.all_special_ids) | {
            tokenizer.convert_tokens_to_ids(t) for t in ("<s>", "</s>")
        }

    @torch.inference_mode()
    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        state = self.states[session_identity]
        model = self.runner.model
        embeddings = model.llm.get_input_embeddings()
        acoustic = payload.data["acoustic"].to(embeddings.weight).reshape(1, -1)
        if state.position_count == 0:
            opening_token_ids = [*self.prompt_token_ids, self.pad_token_id]
            prompt_ids = torch.tensor(
                opening_token_ids, device=embeddings.weight.device
            )
            padding_embeddings = embeddings(
                torch.full_like(prompt_ids, self.pad_token_id)
            )
            input_embeddings = torch.cat([embeddings(prompt_ids[:-1]), acoustic])
            fusion_rows = model.fusion(
                input_embeddings, padding_embeddings, padding_embeddings
            )
        else:
            opening_token_ids = []
            text, function = embeddings(
                torch.tensor(
                    [state.previous_text_token_id, state.previous_function_token_id],
                    device=embeddings.weight.device,
                )
            )
            fusion_rows = model.fusion(
                acoustic, text.reshape(1, -1), function.reshape(1, -1)
            )
        return self.build_unit_request(
            session_identity,
            payload,
            opening_token_ids,
            fusion_rows,
            model.llm.config.vocab_size,
        )

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        state = self.states[session_identity]
        state.previous_text_token_id = int(request_data.output_ids[-1])
        state.previous_function_token_id = int(
            request_data.extra_model_outputs["function_ids"][-1]
        )
        delta = ""
        if state.previous_text_token_id not in self.silent_token_ids:
            state.text_token_ids.append(state.previous_text_token_id)
            decoded = self.tokenizer.decode(state.text_token_ids)
            # note (Codex): Byte fallback tokens must form complete UTF-8 before publication.
            if decoded.endswith("\ufffd"):
                pass
            elif not decoded.startswith(state.emitted_text):
                raise RuntimeError("VoiceChat detokenization revised committed text")
            else:
                delta = decoded[len(state.emitted_text) :]
                state.emitted_text = decoded
        else:
            pass
        payload = request_data.stage_payload
        payload.data.update(
            text_token=state.previous_text_token_id,
            function_token=state.previous_function_token_id,
            text=delta,
        )
        return payload


class TalkerAdapter(FrameAdapter):
    runner: DuplexTalkerRunner

    @torch.inference_mode()
    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        state = self.states[session_identity]
        runner = self.runner
        previous_codes = (
            runner.pad_codes() if state.previous_codes is None else state.previous_codes
        )
        row = runner.step_row(previous_codes, int(payload.data["text_token"]))
        if state.position_count == 0:
            rows = torch.cat([runner.warmup(), row])
            opening_token_ids = [0] * rows.shape[0]
        else:
            rows, opening_token_ids = row, []
        return self.build_unit_request(
            session_identity,
            payload,
            opening_token_ids,
            rows.to(runner.model.fusion_buffer.dtype),
            runner.model.config.vocab_size,
        )

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        state = self.states[session_identity]
        state.previous_codes = request_data.talker_model_inputs["duplex_codes"]
        payload = request_data.stage_payload
        payload.data.update(codes=state.previous_codes.cpu())
        return payload
