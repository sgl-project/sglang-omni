# SPDX-License-Identifier: Apache-2.0
"""The LM side of a full-duplex session: one SGLang streaming-session request per unit.

Forward j reads caller frame j - 1, so after F caller frames a session can run F + 1
forwards; SGLang extends the retained history by the one position each unit lacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sentencepiece import SentencePieceProcessor
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.models.personaplex.architecture import TEXT_MARKER_IDS
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
from sglang_omni.models.personaplex.timeline import Timeline, no_codes
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

WORD_BOUNDARY = "▁"


@dataclass(kw_only=True)
class LMHistory:
    """What a session's finished units left; a failed unit closes the session.

    next_text_id is the next frame's text token: audio trails text by one step."""

    timeline: Timeline
    model_inputs: LMInputs
    next_text_id: int
    has_spoken: bool = False


@dataclass(kw_only=True)
class LMSessionState:
    sampling: RequestSampling
    history: LMHistory | None = None


class PersonaPlexSessionAdapter(ARSessionAdapter):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        tokenizer: SentencePieceProcessor,
    ) -> None:
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.states: dict[SessionIdentity, LMSessionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        sampling = request_sampling(request)
        # Note (Junnan Li): A session's sampling comes from its client, so it is
        # checked once here and fails the open rather than the first unit.
        lm_sampling_params(sampling, max_new_tokens=1, vocab_size=self.vocab_size)
        self.states[session_identity] = LMSessionState(sampling=sampling)

    def close(self, session_identity: SessionIdentity) -> None:
        # Note (wilsonzheng0327): The bridge registers a session before calling
        # open, so a close can follow an open that raised.
        self.states.pop(session_identity, None)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload:
        """Relay an empty end of input to code2wav without stepping the LM."""
        payload.data = PersonaPlexState(codes=no_codes()).to_dict()
        return payload

    def start_history(
        self, session: LMSessionState, state: PersonaPlexState
    ) -> LMHistory:
        timeline = timeline_from_state(state, no_codes())
        return LMHistory(
            timeline=timeline,
            model_inputs=new_model_inputs(timeline, session.sampling),
            next_text_id=int(timeline.prefill_tokens[-1, 0]),
        )

    def spoken_words(self, history: LMHistory, text_ids: list[int]) -> str:
        """Word pieces as they are spoken, frame markers dropped."""
        # Note (wilsonzheng0327): Piece by piece, since decoding a growing id list
        # could rewrite text already sent.
        pieces = []
        for token in text_ids:
            if token in TEXT_MARKER_IDS:
                continue
            else:
                pass
            piece = self.tokenizer.id_to_piece(token).replace(WORD_BOUNDARY, " ")
            if not history.has_spoken:
                piece = piece.lstrip(" ")
                history.has_spoken = bool(piece)
            else:
                pass
            pieces.append(piece)
        return "".join(pieces)

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        session = self.states[session_identity]
        state = PersonaPlexState.from_dict(payload.data)
        if session.history is None:
            session.history = self.start_history(session, state)
        else:
            pass
        history = session.history
        timeline = history.timeline
        caller_codes = state.user_codes
        num_forwards = timeline.num_frames + int(caller_codes.shape[0]) + 1
        positions = timeline.num_prompt_positions + num_forwards
        if positions > self.context_length - 1:
            raise ContextExhaustedError(
                f"{ContextExhaustedError.CODE}: PersonaPlex session needs {positions} "
                f"positions but the LM context holds {self.context_length - 1}; "
                "raise the lm stage's context_length for longer sessions"
            )
        else:
            pass

        timeline.append_caller_frames(caller_codes)
        forwards_done = len(history.model_inputs["agent_rows"])
        input_ids = prompt_input_ids(timeline) if forwards_done == 0 else []
        max_new_tokens = num_forwards - forwards_done
        sglang_request = Req(
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
            req=sglang_request,
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            stage_payload=payload,
            max_new_tokens=max_new_tokens,
            temperature=session.sampling.text_temperature,
        )
        data.talker_model_inputs = history.model_inputs
        return data

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        history = self.states[session_identity].history
        frames = history.model_inputs["pending_frames"]
        text_ids = [history.next_text_id]
        text_ids.extend(int(token) for token in request_data.output_ids)
        history.next_text_id = text_ids.pop()
        codes = history.model_inputs.pop("pending_codes", None)
        if codes is None:
            codes = torch.stack(frames).cpu()
        else:
            pass
        frames.clear()
        history.model_inputs["frames"].clear()
        payload = request_data.stage_payload
        payload.data = {"codes": codes, "text": self.spoken_words(history, text_ids)}
        return payload


__all__ = ["PersonaPlexSessionAdapter"]
