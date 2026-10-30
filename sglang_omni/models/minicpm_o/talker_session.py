# SPDX-License-Identifier: Apache-2.0
"""Incremental speech conditions over the shared SGLang talker and native KV."""

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
)
from sglang_omni.models.minicpm_o.components.talker import build_tts_condition
from sglang_omni.models.minicpm_o.components.tts_runtime import CODEC_CHUNK_SIZE
from sglang_omni.models.minicpm_o.talker_request import CodecTokenizer
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
)
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)

# note (Junnan Li): One 1 s unit of codec tokens plus one, as in the checkpoint's duplex TTS.
TALKER_TOKENS_PER_UNIT = CODEC_CHUNK_SIZE + 1


@dataclass(kw_only=True)
class TalkerSessionState:
    is_reset_pending: bool = False
    is_turn_start: bool = True
    is_rollover_pending: bool = False
    # note (0xtoward): Tokens the turn's talker history holds, and the turn's newest finished units (condition, codes) that a rollover can keep.
    history_length: int = 0
    recent_units: list[tuple[torch.Tensor, list[int]]] = field(default_factory=list)


@dataclass(kw_only=True)
class TalkerUnitRequestData(SGLangARRequestData):
    pass


class TalkerAdapter(ARSessionAdapter):
    def __init__(
        self, model: MiniCPMOTalkerForCausalLM, *, max_request_length: int
    ) -> None:
        self.model = model
        self.max_request_length = max_request_length
        self.boundary_token_ids = (model.audio_bos_token_id,)
        self.tokenizer = CodecTokenizer(eos_token_id=model.codec_eos_id)
        self.states: dict[SessionIdentity, TalkerSessionState] = {}

    def units_to_carry(
        self, units: list[tuple[torch.Tensor, list[int]]], next_unit_length: int
    ) -> list[tuple[torch.Tensor, list[int]]]:
        """The newest units that fit with the next unit in half the request limit."""
        first = len(units)
        kept_length = next_unit_length
        while first > 0:
            condition, codec_tokens = units[first - 1]
            unit_length = int(condition.shape[0]) + len(codec_tokens)
            if kept_length + unit_length > self.max_request_length // 2:
                break
            else:
                kept_length += unit_length
                first -= 1
        return units[first:]

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = TalkerSessionState()

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        state = self.states[session_identity]
        should_reset_history = state.is_reset_pending
        if should_reset_history:
            state.is_turn_start = True
            state.is_reset_pending = False
            state.history_length = 0
            state.recent_units.clear()
        else:
            pass
        thinker_result = payload.data
        thinker_result["codec_tokens"] = []
        thinker_result["speech_turn_start"] = state.is_turn_start
        is_turn_end = thinker_result["end_of_turn"] or chunk.eos
        thinker_result["end_of_turn"] = is_turn_end
        should_bypass_generation = not thinker_result["talker_conditions"] or (
            thinker_result["is_listen"] and not is_turn_end
        )
        if should_bypass_generation and is_turn_end:
            state.is_reset_pending = True
        else:
            pass
        # note (luojiaxuan): A turn that ends mid-reply on a unit the talker skips leaves no unit to reach the codec EOS in.
        thinker_result["is_speech_cut"] = (
            should_bypass_generation and is_turn_end and not state.is_turn_start
        )
        state.is_rollover_pending = not should_bypass_generation and (
            state.history_length
            + len(thinker_result["talker_conditions"])
            + len(self.boundary_token_ids)
            + TALKER_TOKENS_PER_UNIT
            > self.max_request_length
        )
        return ARSessionPreparation(
            reset_history=should_reset_history or state.is_rollover_pending,
            bypass_generation=should_bypass_generation,
        )

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> TalkerUnitRequestData:
        state = self.states[session_identity]
        talker_conditions = payload.data["talker_conditions"]
        tts_condition = build_tts_condition(
            torch.tensor(
                [token_id for token_id, _, _ in talker_conditions], dtype=torch.long
            ),
            torch.stack(
                [
                    torch.as_tensor(hidden_state)
                    for _, hidden_state, _ in talker_conditions
                ]
            ),
            text_embedding=self.model.emb_text,
            semantic_projector=self.model.projector_semantic,
            boundary_tokens=self.boundary_token_ids,
            normalize_projected_hidden=self.model.normalize_projected_hidden,
        )
        sampling_params = SamplingParams(
            max_new_tokens=TALKER_TOKENS_PER_UNIT,
            min_new_tokens=(
                0
                if state.is_turn_start or payload.data["end_of_turn"]
                else TALKER_TOKENS_PER_UNIT
            ),
            temperature=payload.request.params["talker_temperature"],
            # note (luojiaxuan): The official duplex loop samples codec tokens without top-p or top-k and penalizes repeats over the unit's own last 16 tokens, which the runner applies from rep_penalty.
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            stop_token_ids=[self.model.codec_eos_id],
        )
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model.num_audio_tokens)
        condition_rows = int(tts_condition.shape[0])
        carried_token_ids: list[int] = []
        embedding_spans: list[EmbeddingSpan] = []
        if state.is_rollover_pending:
            state.recent_units = self.units_to_carry(
                state.recent_units, condition_rows + TALKER_TOKENS_PER_UNIT
            )
            for condition, codec_tokens in state.recent_units:
                rows = int(condition.shape[0])
                embedding_spans.append(
                    EmbeddingSpan(
                        start=len(carried_token_ids),
                        end=len(carried_token_ids) + rows,
                        input_embeds=condition,
                    )
                )
                carried_token_ids += [self.model.codec_eos_id] * rows + codec_tokens
        else:
            pass
        embedding_spans.append(
            EmbeddingSpan(
                start=len(carried_token_ids),
                end=len(carried_token_ids) + condition_rows,
                input_embeds=tts_condition,
            )
        )
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=[
                *carried_token_ids,
                *[self.model.codec_eos_id] * condition_rows,
            ],
            sampling_params=sampling_params,
            eos_token_ids={self.model.codec_eos_id},
            vocab_size=self.model.num_audio_tokens,
        )
        request.tokenizer = self.tokenizer
        return TalkerUnitRequestData(
            req=request,
            stage_payload=payload,
            unit_embedding_spans=embedding_spans,
            max_new_tokens=TALKER_TOKENS_PER_UNIT,
            talker_model_inputs={
                "rep_penalty": payload.request.params["talker_repetition_penalty"]
            },
        )

    def result(
        self, session_identity: SessionIdentity, request_data: TalkerUnitRequestData
    ) -> StagePayload:
        payload = request_data.stage_payload
        payload.data["codec_tokens"] = list(request_data.output_ids)
        # note (luojiaxuan): A turn's last unit may stop at the codec EOS inside its window; reaching the end of the window instead leaves the rest of the turn's text unspoken, whether its last slot then held the EOS or another codec token.
        payload.data["is_speech_cut"] = (
            payload.data["end_of_turn"]
            and len(request_data.output_ids) >= CODEC_CHUNK_SIZE
        )
        state = self.states[session_identity]
        state.is_turn_start = False
        state.is_reset_pending = payload.data["end_of_turn"]
        state.history_length = len(request_data.req.origin_input_ids) + len(
            request_data.output_ids
        )
        state.recent_units.append(
            (
                request_data.unit_embedding_spans[-1].input_embeds,
                list(request_data.output_ids),
            )
        )
        state.recent_units = self.units_to_carry(state.recent_units, 0)
        return payload
