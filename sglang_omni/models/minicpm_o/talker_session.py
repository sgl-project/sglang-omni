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


@dataclass(kw_only=True)
class TalkerSessionState:
    reset_pending: bool = False
    turn_start: bool = True


@dataclass(kw_only=True)
class TalkerUnitRequestData(SGLangARRequestData):
    state: TalkerSessionState


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
        reset = state.reset_pending
        if reset:
            state.turn_start = True
            state.reset_pending = False
        else:
            pass
        result = payload.data
        result["codec_tokens"] = []
        result["speech_turn_start"] = state.turn_start
        end = result["end_of_turn"] or chunk.eos
        result["end_of_turn"] = end
        bypass = not result["pairs"] or (result["is_listen"] and not end)
        if bypass and end:
            state.reset_pending = True
        else:
            pass
        return ARSessionPreparation(reset_history=reset, bypass_generation=bypass)

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> TalkerUnitRequestData:
        state = self.states[session_identity]
        pairs = payload.data["pairs"]
        condition = build_tts_condition(
            torch.tensor([pair[0] for pair in pairs], dtype=torch.long),
            torch.stack([torch.as_tensor(pair[1]) for pair in pairs]),
            text_embedding=self.model.emb_text,
            semantic_projector=self.model.projector_semantic,
            boundary_tokens=(self.model.audio_bos_token_id,),
            normalize_projected_hidden=self.model.normalize_projected_hidden,
        )
        sampling = SamplingParams(
            max_new_tokens=26,
            min_new_tokens=0 if state.turn_start or payload.data["end_of_turn"] else 26,
            temperature=0.8,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            stop_token_ids=[self.model.codec_eos_id],
        )
        sampling.normalize(self.tokenizer)
        sampling.verify(self.model.num_audio_tokens)
        condition_rows = int(condition.shape[0])
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=[self.model.codec_eos_id] * condition_rows,
            sampling_params=sampling,
            eos_token_ids={self.model.codec_eos_id},
            vocab_size=self.model.num_audio_tokens,
        )
        request.tokenizer = self.tokenizer
        return TalkerUnitRequestData(
            req=request,
            stage_payload=payload,
            state=state,
            unit_embedding_spans=[
                EmbeddingSpan(start=0, end=condition_rows, input_embeds=condition)
            ],
            max_new_tokens=26,
            talker_model_inputs={"rep_penalty": 1.05},
        )

    def result(
        self, session_identity: SessionIdentity, request_data: TalkerUnitRequestData
    ) -> StagePayload:
        payload = request_data.stage_payload
        payload.data["codec_tokens"] = list(request_data.output_ids)
        state = self.states[session_identity]
        state.turn_start = False
        state.reset_pending = payload.data["end_of_turn"]
        return payload
