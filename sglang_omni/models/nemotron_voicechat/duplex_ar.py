# SPDX-License-Identifier: Apache-2.0
"""Frame fusion over scheduler-owned KV, retaining inputs for prefix recomputation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from transformers import PreTrainedTokenizerBase

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import capture_cuda_graph
from sglang_omni.models.nemotron_voicechat.model_runner import (
    NemotronVoiceChatModelRunner,
)
from sglang_omni.models.nemotron_voicechat.payload_types import TOOL_CALLS
from sglang_omni.models.nemotron_voicechat.request_builders import (
    NemotronVoiceChatRequestData,
    build_autoregressive_request,
)
from sglang_omni.models.nemotron_voicechat.talker_model_runner import (
    VoiceChatTalkerModelRunner,
)
from sglang_omni.models.nemotron_voicechat.tools import (
    END_OF_TOOL_CALL_TOKEN,
    START_OF_TOOL_CALL_TOKEN,
    SpokenToolCall,
    format_tool_response,
    parse_tool_calls,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import TOOL_RESPONSE
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


@dataclass(kw_only=True)
class FrameHistory:
    fusion_rows: list[torch.Tensor] = field(default_factory=list)
    position_count: int = 0
    previous_text_token_id: int | None = None
    previous_function_token_id: int | None = None
    previous_codes: torch.Tensor | None = None
    text_token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    # note (Junnan Li): None outside a tool call; the function-channel ids of the open call otherwise.
    tool_call_token_ids: list[int] | None = None
    forced_function_token_ids: list[int] = field(default_factory=list)


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
        # Concatenating the full history would make each unit cost grow with the session.
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


class UnitDecodeRunner(ModelRunner):
    def build_forward_batch(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[ForwardBatch | None, ScheduleBatch, bool] | None:
        batch = scheduler_output.batch_data
        # note (Junnan Li): one position appended to retained KV is what a decode forward computes, and sessions keep no Mamba track state.
        if any(length != 1 for length in batch.extend_lens) or 0 in batch.prefix_lens:
            return super().build_forward_batch(scheduler_output)
        else:
            pass
        batch.forward_mode = ForwardMode.DECODE
        forward_batch, schedule_batch, _ = super().build_forward_batch(scheduler_output)
        # note (Junnan Li): the scheduler retires the unit as the extend it admitted.
        batch.forward_mode = ForwardMode.EXTEND
        return forward_batch, schedule_batch, False


class DuplexThinkerRunner(UnitDecodeRunner, NemotronVoiceChatModelRunner):
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
        rows = [
            request.data.talker_model_inputs["duplex_history"].fusion_rows[-1][-1]
            for request in requests
        ]
        self.model.fusion_buffer[: len(rows)] = torch.stack(rows)


class DuplexTalkerRunner(UnitDecodeRunner, VoiceChatTalkerModelRunner):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.sampler_graph: torch.cuda.CUDAGraph | None = None
        self.sampler_hidden: torch.Tensor | None = None
        self.sampler_output: torch.Tensor | None = None

    def generate_codes(self, index: int) -> torch.Tensor:
        if self.model.hidden_out.device.type != "cuda":
            return super().generate_codes(index)
        else:
            pass
        if self.sampler_graph is None:
            sampler_hidden = self.model.hidden_out[index : index + 1].float().clone()

            def sample() -> torch.Tensor:
                return self.model.talker.generate_codes(
                    sampler_hidden,
                    self.model.mog_head,
                    level_schedule=self.level_schedule,
                    top_p=self.top_p,
                    noise_scale=self.noise_scale,
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

    def before_decode(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        rows = [
            request.data.talker_model_inputs["duplex_history"].fusion_rows[-1][-1]
            for request in requests
        ]
        self.model.fusion_buffer[: len(rows)] = torch.stack(rows)
        self.model.fusion_mask[: len(rows)] = True

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

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        self.post_prefill(result, forward_batch, schedule_batch, requests)


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
        self.states.pop(session_identity, None)

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
    ) -> NemotronVoiceChatRequestData:
        state = self.states[session_identity]
        new_position_count = fusion_rows.shape[0]
        positions = state.position_count + new_position_count + 1
        # note (Junnan Li): SGLang holds at most context_length - 1 tokens per request.
        if positions > self.context_length - 1:
            raise ContextExhaustedError(
                f"{ContextExhaustedError.CODE}: VoiceChat session needs {positions} "
                f"positions but the context holds {self.context_length - 1}; "
                "start a new session"
            )
        else:
            pass
        # The next unit forwards the previous sampled token with new fusion input.
        state.fusion_rows.append(fusion_rows.detach())
        state.position_count += new_position_count
        request = build_autoregressive_request(
            payload,
            input_ids=opening_token_ids,
            max_new_tokens=1,
            vocab_size=vocab_size,
        )
        request_data = NemotronVoiceChatRequestData(
            req=request,
            input_ids=torch.tensor(opening_token_ids, dtype=torch.long),
            stage_payload=payload,
            max_new_tokens=1,
            temperature=0.0,
        )
        request_data.talker_model_inputs["duplex_history"] = state
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
        self.start_of_tool_call_id = tokenizer.convert_tokens_to_ids(
            START_OF_TOOL_CALL_TOKEN
        )
        self.end_of_tool_call_id = tokenizer.convert_tokens_to_ids(
            END_OF_TOOL_CALL_TOKEN
        )

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        # note (Junnan Li): The response rides an audio frame; the thinker consumes it in this and later frames.
        tool_response = payload.data.pop(TOOL_RESPONSE, None)
        if tool_response is not None:
            self.states[session_identity].forced_function_token_ids.extend(
                self.tokenizer.encode(
                    format_tool_response(tool_response["responses"]),
                    add_special_tokens=False,
                )
            )
        else:
            pass
        return super().prepare_unit(session_identity, chunk, payload)

    @torch.inference_mode()
    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> NemotronVoiceChatRequestData:
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
            # note (Junnan Li): Response tokens replace the model's own function input one per frame; the user channel keeps live audio because the user is waiting.
            function_token_id = (
                state.forced_function_token_ids.pop(0)
                if state.forced_function_token_ids
                else state.previous_function_token_id
            )
            text, function = embeddings(
                torch.tensor(
                    [state.previous_text_token_id, function_token_id],
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
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
    ) -> StagePayload:
        if not isinstance(request_data, NemotronVoiceChatRequestData):
            raise TypeError("VoiceChat thinker received incompatible request data")
        else:
            pass
        state = self.states[session_identity]
        state.previous_text_token_id = int(request_data.output_ids[-1])
        state.previous_function_token_id = request_data.function_ids[-1]
        tool_calls = self.collect_tool_calls(state)
        delta = ""
        if state.previous_text_token_id not in self.silent_token_ids:
            state.text_token_ids.append(state.previous_text_token_id)
            decoded = self.tokenizer.decode(state.text_token_ids)
            # Byte fallback tokens must form complete UTF-8 before publication.
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
        payload.data[TOOL_CALLS] = tool_calls
        return payload

    def collect_tool_calls(self, state: FrameHistory) -> list[SpokenToolCall]:
        """Track the function channel; return the calls of a span that just closed."""
        function_token_id = state.previous_function_token_id
        assert function_token_id is not None
        if state.forced_function_token_ids:
            # note (Junnan Li): Predictions under a forced response are not the model's own calls.
            return []
        elif function_token_id == self.start_of_tool_call_id:
            state.tool_call_token_ids = [function_token_id]
            return []
        elif (
            state.tool_call_token_ids is None or function_token_id == self.pad_token_id
        ):
            return []
        elif function_token_id != self.end_of_tool_call_id:
            state.tool_call_token_ids.append(function_token_id)
            return []
        else:
            state.tool_call_token_ids.append(function_token_id)
            call_text = self.tokenizer.decode(state.tool_call_token_ids)
            state.tool_call_token_ids = None
            return parse_tool_calls(call_text)


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
