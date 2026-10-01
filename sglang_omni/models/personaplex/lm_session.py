# SPDX-License-Identifier: Apache-2.0
"""The LM side of a full-duplex call: one SGLang streaming-session request per unit.

Forward j reads caller frame j - 1 (the timeline's algebra), so once a call
has heard F caller frames it can have run F + 1 forwards. The first unit is
therefore the prompt prefill plus one decode per frame it carries; every later
unit sends no new ids at all. SGLang restores the call's history itself and
extends it by the one position it lacks, the text token the previous unit
sampled last, so the KV cache grows by exactly one row per frame, as in the
offline request.

A failed unit closes the whole call (the coordinator closes a session on any
append failure), so the state here only ever holds what finished units left.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.request_builders import (
    LMInputs,
    RequestSampling,
    lm_sampling_params,
    new_model_inputs,
    prompt_input_ids,
    request_sampling,
    timeline_from_state,
)
from sglang_omni.models.personaplex.session_hooks import no_codes
from sglang_omni.models.personaplex.timeline import Timeline
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class LMCall:
    """What the call's finished units left.

    next_text_id is the text token of the next frame to leave: a frame's audio
    finishes one step after its undelayed text token, so each unit hands on
    the token it sampled last.
    """

    timeline: Timeline
    model_inputs: LMInputs
    next_text_id: int


@dataclass(kw_only=True)
class LMSession:
    sampling: RequestSampling
    call: LMCall | None = None


class PersonaPlexSessionAdapter(ARSessionAdapter):
    def __init__(self, *, vocab_size: int, context_length: int) -> None:
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.sessions: dict[SessionIdentity, LMSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.sessions[session_identity] = LMSession(
            sampling=request_sampling(request)
        )

    def close(self, session_identity: SessionIdentity) -> None:
        # Note (wilsonzheng0327): The bridge registers a session before calling
        # open, so a close can follow an open that raised.
        self.sessions.pop(session_identity, None)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload:
        """Relay an empty end of input to code2wav without stepping the LM."""
        payload.data = PersonaPlexState(codes=no_codes()).to_dict()
        return payload

    def start_call(self, session: LMSession, state: PersonaPlexState) -> LMCall:
        if not state.carries_prompt:
            raise ValueError(
                "PersonaPlex call reached the LM without its prompt, "
                "which only the first unit carries"
            )
        else:
            pass
        timeline = timeline_from_state(state, no_codes())
        return LMCall(
            timeline=timeline,
            model_inputs=new_model_inputs(timeline, session.sampling),
            next_text_id=int(timeline.prefill_tokens[-1, 0]),
        )

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        session = self.sessions[session_identity]
        state = PersonaPlexState.from_dict(payload.data)
        if session.call is None:
            session.call = self.start_call(session, state)
        else:
            pass
        call = session.call
        timeline = call.timeline
        caller_codes = state.user_codes
        assert caller_codes is not None, "Mimi encode sets every unit's codes"
        if caller_codes.shape[0] == 0:
            raise ValueError("PersonaPlex session unit carries no 80 ms frame")
        else:
            pass
        num_forwards = timeline.num_frames + int(caller_codes.shape[0]) + 1
        positions = timeline.num_prompt_positions + num_forwards
        if positions > self.context_length - 1:
            raise ValueError(
                f"PersonaPlex call needs {positions} positions but the LM context "
                f"holds {self.context_length - 1}; raise the lm stage's "
                "context_length for longer calls"
            )
        else:
            pass

        timeline.append_caller_frames(caller_codes)
        forwards_done = len(call.model_inputs["agent_rows"])
        input_ids = prompt_input_ids(timeline) if forwards_done == 0 else []
        max_new_tokens = num_forwards - forwards_done
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=lm_sampling_params(
                session.sampling,
                max_new_tokens=max_new_tokens,
                vocab_size=self.vocab_size,
            ),
            vocab_size=self.vocab_size,
        )
        data = SGLangARRequestData(
            req=req,
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            stage_payload=payload,
            max_new_tokens=max_new_tokens,
            temperature=session.sampling.text_temperature,
        )
        data.talker_model_inputs = call.model_inputs
        return data

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        call = self.sessions[session_identity].call
        assert call is not None, "a unit's result follows its build"
        frames = call.model_inputs["pending_frames"]
        assert len(frames) == len(request_data.output_ids), (
            len(frames),
            len(request_data.output_ids),
        )
        text_ids = [call.next_text_id]
        text_ids.extend(int(token) for token in request_data.output_ids)
        call.next_text_id = text_ids.pop()
        codes = torch.stack(frames).cpu()
        frames.clear()
        call.model_inputs["frames"].clear()
        payload = request_data.stage_payload
        payload.data = PersonaPlexState(codes=codes, text_ids=text_ids).to_dict()
        return payload


__all__ = ["PersonaPlexSessionAdapter"]
