# SPDX-License-Identifier: Apache-2.0
"""SGLang request/result adapters for Nemotron VoiceChat AR stages."""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import DeferredAdmission, FollowupAdmission

from .payload_types import VoiceChatFrameState


@dataclass
class VoiceChatRequestData(SGLangARRequestData):
    kind: str = "frame"
    started_s: float = 0.0
    custom_inputs: dict[str, Any] = field(default_factory=dict)
    followup: VoiceChatRequestData | None = None


@dataclass
class _ThinkerSessionState:
    next_frame_to_build: int = 0
    next_frame_to_finish: int = 0
    function_token: int | None = None
    pending: dict[int, tuple[VoiceChatRequestData, Future[None]]] = field(
        default_factory=dict
    )


@dataclass
class _TalkerSessionState:
    next_frame_to_build: int = 0
    next_frame_to_finish: int = 0
    previous_audio_codes: list[int] | None = None
    pending: dict[int, tuple[VoiceChatRequestData, Future[None]]] = field(
        default_factory=dict
    )


def _close_pending(
    pending: dict[int, tuple[VoiceChatRequestData, Future[None]]],
    session_id: str,
) -> None:
    error = RuntimeError(f"VoiceChat session {session_id!r} closed")
    for _, ready in pending.values():
        if not ready.done():
            ready.set_exception(error)
    pending.clear()


def _sampling_params(tokenizer: Any, vocab_size: int) -> SamplingParams:
    params = SamplingParams(max_new_tokens=1, temperature=0.0)
    params.normalize(tokenizer)
    params.verify(vocab_size)
    return params


def _request(
    *,
    request_id: str,
    input_ids: list[int],
    tokenizer: Any,
    vocab_size: int,
) -> Req:
    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=_sampling_params(tokenizer, vocab_size),
        vocab_size=vocab_size,
    )
    req.tokenizer = tokenizer
    return req


def _custom_output(data: VoiceChatRequestData, key: str) -> Any:
    if key not in data.extra_model_outputs:
        raise RuntimeError(f"VoiceChat request did not produce {key!r}")
    return data.extra_model_outputs[key]


class ThinkerAdapters:
    def __init__(self, *, config: Any, tokenizer: Any, context_length: int) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self._sessions: dict[str, _ThinkerSessionState] = {}

    def request_builder(
        self, payload: StagePayload
    ) -> VoiceChatRequestData | DeferredAdmission:
        state = VoiceChatFrameState.from_data(payload.data)
        if state.event == "session_close":
            session = self._sessions.pop(state.session_id, None)
            if session is not None:
                _close_pending(session.pending, state.session_id)
            return VoiceChatRequestData(
                stage_payload=payload,
                streaming_session_id=state.session_id,
                close_streaming_session=True,
            )
        if state.acoustic_embedding is None:
            raise ValueError("VoiceChat thinker requires acoustic_embedding")

        session = self._sessions.setdefault(state.session_id, _ThinkerSessionState())
        frame_index = int(state.frame_index)
        if frame_index != session.next_frame_to_build:
            raise ValueError(
                f"VoiceChat thinker expected frame {session.next_frame_to_build}, "
                f"got {state.frame_index}"
            )
        if frame_index == 0:
            prompt = state.instructions or "You are a helpful realtime voice assistant."
            prompt_ids = [
                int(self.config.bos_token_id),
                *self.tokenizer.encode(prompt, add_special_tokens=False),
                int(self.config.eos_token_id),
            ]
            input_ids = [*prompt_ids, int(self.config.pad_token_id)]
            custom_inputs = {
                "is_initial_prefill": True,
                "prompt_length": len(prompt_ids),
                "acoustic_embedding": state.acoustic_embedding,
            }
        else:
            input_ids = []
            custom_inputs = {
                "acoustic_embedding": state.acoustic_embedding,
            }
        state.acoustic_embedding = None
        data = VoiceChatRequestData(
            req=_request(
                request_id=payload.request_id,
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                vocab_size=int(self.config.vocab_size),
            ),
            custom_inputs=custom_inputs,
            stage_payload=StagePayload(
                payload.request_id, payload.request, state.to_dict()
            ),
            streaming_session_id=state.session_id,
            streaming_session_capacity=self.context_length,
            started_s=time.perf_counter(),
        )
        session.next_frame_to_build += 1
        if frame_index == 0:
            return data
        if (
            frame_index == session.next_frame_to_finish
            and session.function_token is not None
        ):
            data.custom_inputs["input_function_ids"] = [session.function_token]
            return data
        ready: Future[None] = Future()
        session.pending[frame_index] = (data, ready)
        return DeferredAdmission(value=data, ready=ready)

    def result_adapter(self, data: VoiceChatRequestData) -> StagePayload:
        payload = data.stage_payload
        state = VoiceChatFrameState.from_data(payload.data)
        if data.close_streaming_session:
            return payload
        if not data.output_ids:
            raise RuntimeError("VoiceChat thinker did not produce a text token")
        session = self._sessions.get(state.session_id)
        if session is None:
            raise RuntimeError(
                f"VoiceChat thinker session {state.session_id!r} is closed"
            )
        frame_index = int(state.frame_index)
        if frame_index != session.next_frame_to_finish:
            raise RuntimeError(
                f"VoiceChat thinker finished frame {frame_index} while expecting "
                f"frame {session.next_frame_to_finish}"
            )
        state.text_token = int(data.output_ids[-1])
        state.function_token = int(_custom_output(data, "function_tokens"))
        state.text_delta = self.tokenizer.decode(
            [state.text_token],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        state.timings_ms["thinker"] = (time.perf_counter() - data.started_s) * 1000
        session.function_token = state.function_token
        session.next_frame_to_finish += 1
        pending = session.pending.pop(session.next_frame_to_finish, None)
        if pending is not None:
            pending_data, ready = pending
            pending_data.custom_inputs["input_function_ids"] = [session.function_token]
            ready.set_result(None)
        return StagePayload(payload.request_id, payload.request, state.to_dict())


class TalkerAdapters:
    def __init__(
        self,
        *,
        config: Any,
        speaker: torch.Tensor,
        context_length: int,
    ) -> None:
        self.config = config
        self.speaker = speaker
        self.context_length = int(context_length)
        self._sessions: dict[str, _TalkerSessionState] = {}

    def _frame_data(
        self,
        payload: StagePayload,
        state: VoiceChatFrameState,
        previous_audio_codes: list[int] | None,
        *,
        started_s: float,
    ) -> VoiceChatRequestData:
        return VoiceChatRequestData(
            req=_request(
                request_id=payload.request_id,
                input_ids=[],
                tokenizer=None,
                vocab_size=int(self.config.vocab_size),
            ),
            custom_inputs={
                "text_token": int(state.text_token),
                "previous_audio_codes": previous_audio_codes,
            },
            stage_payload=payload,
            streaming_session_id=state.session_id,
            streaming_session_capacity=self.context_length,
            kind="frame",
            started_s=started_s,
        )

    def request_builder(
        self, payload: StagePayload
    ) -> VoiceChatRequestData | DeferredAdmission:
        state = VoiceChatFrameState.from_data(payload.data)
        if state.event == "session_close":
            session = self._sessions.pop(state.session_id, None)
            if session is not None:
                _close_pending(session.pending, state.session_id)
            return VoiceChatRequestData(
                stage_payload=payload,
                streaming_session_id=state.session_id,
                close_streaming_session=True,
            )
        if state.text_token is None:
            raise ValueError("VoiceChat talker requires text_token")

        session = self._sessions.setdefault(state.session_id, _TalkerSessionState())
        frame_index = int(state.frame_index)
        if frame_index != session.next_frame_to_build:
            raise ValueError(
                f"VoiceChat talker expected frame {session.next_frame_to_build}, "
                f"got {state.frame_index}"
            )
        started_s = time.perf_counter()
        frame = self._frame_data(
            payload,
            state,
            session.previous_audio_codes,
            started_s=started_s,
        )
        session.next_frame_to_build += 1
        if frame_index > 0:
            if (
                frame_index == session.next_frame_to_finish
                and session.previous_audio_codes is not None
            ):
                return frame
            ready: Future[None] = Future()
            session.pending[frame_index] = (frame, ready)
            return DeferredAdmission(value=frame, ready=ready)

        prefill = VoiceChatRequestData(
            req=_request(
                request_id=payload.request_id,
                input_ids=[0] * int(self.speaker.shape[0]),
                tokenizer=None,
                vocab_size=int(self.config.vocab_size),
            ),
            custom_inputs={
                "is_speaker_prefill": True,
                "speaker_latent": self.speaker,
            },
            stage_payload=payload,
            streaming_session_id=state.session_id,
            streaming_session_capacity=self.context_length,
            kind="speaker_prefill",
            started_s=started_s,
            followup=frame,
        )
        return prefill

    def result_adapter(
        self, data: VoiceChatRequestData
    ) -> StagePayload | FollowupAdmission:
        if data.close_streaming_session:
            return data.stage_payload
        if data.kind == "speaker_prefill":
            if data.followup is None:
                raise RuntimeError("VoiceChat speaker prefill lost its frame follow-up")
            return FollowupAdmission(data.followup)

        payload = data.stage_payload
        state = VoiceChatFrameState.from_data(payload.data)
        session = self._sessions.get(state.session_id)
        if session is None:
            raise RuntimeError(
                f"VoiceChat talker session {state.session_id!r} is closed"
            )
        frame_index = int(state.frame_index)
        if frame_index != session.next_frame_to_finish:
            raise RuntimeError(
                f"VoiceChat talker finished frame {frame_index} while expecting "
                f"frame {session.next_frame_to_finish}"
            )
        codes = _custom_output(data, "audio_codes")
        state.audio_codes = [int(code) for code in codes]
        state.timings_ms["talker"] = (time.perf_counter() - data.started_s) * 1000
        session.previous_audio_codes = state.audio_codes
        session.next_frame_to_finish += 1
        pending = session.pending.pop(session.next_frame_to_finish, None)
        if pending is not None:
            pending_data, ready = pending
            pending_data.custom_inputs["previous_audio_codes"] = list(
                session.previous_audio_codes
            )
            ready.set_result(None)
        return StagePayload(payload.request_id, payload.request, state.to_dict())


__all__ = ["TalkerAdapters", "ThinkerAdapters", "VoiceChatRequestData"]
