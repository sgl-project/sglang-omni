# SPDX-License-Identifier: Apache-2.0
"""Prepare only new user turns while continuation units carry no media."""

import asyncio
import base64
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sglang_omni.audio.wav import pcm16_wav
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import CONFIGURE, control_type
from sglang_omni.scheduling.session import SessionContext, SessionHooks


class QwenSessionSampling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Note (Junnan Li): Thinker defaults follow the ordinary Qwen request path; talker defaults are the checkpoint's generation_config.json.
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_new_tokens: int = Field(default=2048, ge=1)
    talker_max_new_tokens: int = Field(default=4096, ge=1)
    talker_temperature: float = Field(default=0.9, ge=0, le=2)
    talker_top_k: int = Field(default=50, ge=-1)
    talker_top_p: float = Field(default=1.0, gt=0, le=1)
    talker_repetition_penalty: float = Field(default=1.05, ge=1)
    seed: int | None = Field(default=None, ge=0)

    @field_validator("talker_top_k")
    @classmethod
    def top_k_disabled_or_positive(cls, value: int) -> int:
        if value == 0:
            raise ValueError("talker_top_k must be -1 (disabled) or at least 1")
        else:
            return value


DEFAULT_INSTRUCTIONS = (
    "You are a helpful realtime voice assistant. Respond conversationally."
)


class PromptProcessor(Protocol):
    async def __call__(self, payload: StagePayload) -> StagePayload: ...


@dataclass(kw_only=True)
class PreprocessingSession:
    instructions: str
    sampling: QwenSessionSampling
    is_prefix_pending: bool = True


class QwenPreprocessingHooks(SessionHooks):
    def __init__(self, processor: PromptProcessor) -> None:
        self.processor: PromptProcessor = processor
        self.states: dict[SessionIdentity, PreprocessingSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        instructions = request.params.get("instructions", DEFAULT_INSTRUCTIONS)
        if not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        else:
            self.states[session_identity] = PreprocessingSession(
                instructions=instructions,
                sampling=QwenSessionSampling.model_validate(
                    request.params.get("sampling", {})
                ),
            )

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        if control_type(event) != CONFIGURE:
            return
        else:
            state = self.states[session_identity]
        instructions = event.payload.get("instructions", state.instructions)
        if not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        elif instructions != state.instructions and not state.is_prefix_pending:
            raise ValueError(
                "instructions cannot change after Qwen KV has been prefetched; open a new session"
            )
        else:
            sampling = QwenSessionSampling.model_validate(
                {**state.sampling.model_dump(), **event.payload.get("sampling", {})}
            )
            state.instructions = instructions
            state.sampling = sampling

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if context.cancelled.is_set():
            return payload
        else:
            state = self.states[context.session_identity]
        unit = chunk.payload
        session = {"turn": chunk.turn}
        payload.request.params = state.sampling.model_dump(exclude_none=True)
        pcm = unit["pcm"]
        frames = unit["images"]
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + base64.b64encode(frame).decode("ascii")
                },
            }
            for frame in frames
        ]
        content.extend(
            [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "data:audio/wav;base64,"
                        + base64.b64encode(pcm16_wav(pcm, 16000)).decode("ascii")
                    },
                },
                {
                    "type": "text",
                    "text": "Listen to the spoken audio above and respond to it.",
                },
            ]
        )
        payload.request.inputs = {
            "messages": [
                {"role": "system", "content": state.instructions},
                {"role": "user", "content": content},
            ]
        }
        payload = asyncio.run(self.processor(payload))
        if context.cancelled.is_set():
            return payload
        else:
            state.is_prefix_pending = False
        payload.data["session"] = session
        return payload
