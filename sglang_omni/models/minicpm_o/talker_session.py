# SPDX-License-Identifier: Apache-2.0
"""Incremental speech conditions over the shared SGLang talker and native KV."""

from dataclasses import dataclass
from typing import Literal

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
)
from sglang_omni.models.minicpm_o.components.talker import build_tts_condition
from sglang_omni.models.minicpm_o.components.tts_runtime import CODEC_CHUNK_SIZE
from sglang_omni.models.minicpm_o.native_config import MAX_DRAIN_UNITS
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
# note (wenyao): Drain units speak text a turn the model ended still owes; input_end speaks it in one request, since no unit follows the input end.
TalkerUnitKind = Literal["reply", "turn_end", "drain", "input_end"]


@dataclass(kw_only=True)
class TalkerSessionState:
    is_reset_pending: bool = False
    is_turn_start: bool = True
    drain_units_left: int = 0
    unit_kind: TalkerUnitKind = "reply"
    unit_speech_units: int = 1


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
        thinker_result = payload.data
        has_text = bool(thinker_result["talker_conditions"]) and (
            not thinker_result["is_listen"] or thinker_result["end_of_turn"]
        )
        ends_previous_turn = state.drain_units_left > 0 and has_text
        if ends_previous_turn:
            state.drain_units_left = 0
            state.is_reset_pending = True
        else:
            pass
        should_reset_history = state.is_reset_pending
        if should_reset_history:
            state.is_turn_start = True
            state.is_reset_pending = False
        else:
            pass
        is_turn_open = not state.is_turn_start
        is_drain_stopped = state.drain_units_left > 0 and (
            thinker_result["is_interrupted"] or thinker_result["hears_user"]
        )
        is_turn_end = thinker_result["end_of_turn"] or chunk.eos or is_drain_stopped
        unit_kind: TalkerUnitKind | None
        if is_drain_stopped:
            unit_kind = None
        elif state.drain_units_left > 0 and chunk.eos:
            unit_kind = "input_end"
            state.unit_speech_units = state.drain_units_left
        elif state.drain_units_left > 0:
            unit_kind = "drain"
            state.unit_speech_units = 1
            state.drain_units_left -= 1
        elif (
            chunk.eos
            and not thinker_result["is_interrupted"]
            and (has_text or is_turn_open)
        ):
            unit_kind = "input_end"
            state.unit_speech_units = 1 + MAX_DRAIN_UNITS
        elif (
            thinker_result["end_of_turn"]
            and not thinker_result["is_interrupted"]
            and (has_text or is_turn_open)
        ):
            unit_kind = "turn_end"
            state.unit_speech_units = 1
        elif has_text:
            unit_kind = "reply"
            state.unit_speech_units = 1
        else:
            unit_kind = None
        thinker_result["codec_tokens"] = []
        thinker_result["speech_turn_start"] = state.is_turn_start
        thinker_result["end_of_turn"] = is_turn_end
        thinker_result["ends_previous_turn"] = ends_previous_turn
        if unit_kind is None:
            state.drain_units_left = 0
            state.is_reset_pending = is_turn_end
            # note (luojiaxuan): A turn that ends mid-reply on a unit the talker skips leaves no unit to reach the codec EOS in.
            thinker_result["is_speech_cut"] = is_turn_end and is_turn_open
        else:
            state.unit_kind = unit_kind
            thinker_result["is_speech_cut"] = False
        return ARSessionPreparation(
            reset_history=should_reset_history,
            bypass_generation=unit_kind is None,
        )

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> TalkerUnitRequestData:
        state = self.states[session_identity]
        talker_conditions = payload.data["talker_conditions"]
        max_new_tokens = state.unit_speech_units * CODEC_CHUNK_SIZE + 1
        tts_condition = build_tts_condition(
            torch.tensor(
                [token_id for token_id, _, _ in talker_conditions], dtype=torch.long
            ),
            (
                torch.stack(
                    [
                        torch.as_tensor(hidden_state)
                        for _, hidden_state, _ in talker_conditions
                    ]
                )
                if talker_conditions
                else torch.empty(0)
            ),
            text_embedding=self.model.emb_text,
            semantic_projector=self.model.projector_semantic,
            boundary_tokens=(self.model.audio_bos_token_id,),
            normalize_projected_hidden=self.model.normalize_projected_hidden,
        )
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            min_new_tokens=(
                TALKER_TOKENS_PER_UNIT
                if state.unit_kind == "reply" and not state.is_turn_start
                else 0
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
            max_new_tokens=max_new_tokens,
            talker_model_inputs={
                "rep_penalty": payload.request.params["talker_repetition_penalty"],
                "masks_chunk_boundary_eos": state.unit_speech_units > 1,
            },
        )

    def result(
        self, session_identity: SessionIdentity, request_data: TalkerUnitRequestData
    ) -> StagePayload:
        payload = request_data.stage_payload
        state = self.states[session_identity]
        codec_tokens = list(request_data.output_ids)
        # note (wenyao): Every chunk ends at a codec EOS sample whether or not text is left, so only an EOS before the chunk fills means the talker spoke all its text; the window's extra slot holds that EOS or one more codec token.
        owes_speech = len(codec_tokens) >= state.unit_speech_units * CODEC_CHUNK_SIZE
        if state.unit_kind == "reply":
            is_turn_end = False
        elif state.unit_kind == "input_end" or not owes_speech:
            is_turn_end = True
        elif state.unit_kind == "turn_end":
            state.drain_units_left = MAX_DRAIN_UNITS
            is_turn_end = False
        else:
            is_turn_end = state.drain_units_left == 0
        payload.data["codec_tokens"] = codec_tokens
        payload.data["end_of_turn"] = is_turn_end
        payload.data["is_speech_cut"] = is_turn_end and owes_speech
        state.is_turn_start = False
        state.is_reset_pending = is_turn_end
        if is_turn_end:
            state.drain_units_left = 0
        else:
            pass
        return payload
