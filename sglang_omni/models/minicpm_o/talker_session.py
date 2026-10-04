# SPDX-License-Identifier: Apache-2.0
"""Incremental speech conditions over the shared SGLang talker and native KV."""

from dataclasses import dataclass

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


@dataclass(kw_only=True)
class TalkerUnitRequestData(SGLangARRequestData):
    pass


class TalkerAdapter(ARSessionAdapter):
    def __init__(self, model: MiniCPMOTalkerForCausalLM) -> None:
        self.model = model
        self.tokenizer = CodecTokenizer(eos_token_id=model.codec_eos_id)
        self.states: dict[SessionIdentity, TalkerSessionState] = {}

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
        return ARSessionPreparation(
            reset_history=should_reset_history,
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
            boundary_tokens=(self.model.audio_bos_token_id,),
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
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            stop_token_ids=[self.model.codec_eos_id],
        )
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model.num_audio_tokens)
        condition_rows = int(tts_condition.shape[0])
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=[self.model.codec_eos_id] * condition_rows,
            sampling_params=sampling_params,
            eos_token_ids={self.model.codec_eos_id},
            vocab_size=self.model.num_audio_tokens,
        )
        request.tokenizer = self.tokenizer
        return TalkerUnitRequestData(
            req=request,
            stage_payload=payload,
            unit_embedding_spans=[
                EmbeddingSpan(start=0, end=condition_rows, input_embeds=tts_condition)
            ],
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
        state = self.states[session_identity]
        state.is_turn_start = False
        state.is_reset_pending = payload.data["end_of_turn"]
        return payload
