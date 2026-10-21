# SPDX-License-Identifier: Apache-2.0
"""Serves PersonaPlex over /v1/realtime as a native full-duplex session."""

from __future__ import annotations

from sglang_omni.client.client import Client
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.config import PersonaPlexSessionPipelineConfig
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import MS_PER_SECOND, Capabilities

CAPABILITIES = Capabilities(
    input_sample_rate_hz=SAMPLE_RATE,
    output_sample_rate_hz=SAMPLE_RATE,
    output_modalities=("audio", "text"),
    native_unit_ms=SAMPLES_PER_FRAME * MS_PER_SECOND // SAMPLE_RATE,
    tail_policy="pad",
)


def build_session_request(config: SessionConfiguration) -> OmniRequest:
    """instructions is the role prompt; the session keeps the default voice."""
    if "instructions" in config:
        params = {"instructions": config["instructions"]}
    else:
        params = {}
    return OmniRequest(inputs=None, params=params)


class PersonaPlexOutputConverter:
    """The agent's side of a session is one response: its audio every frame and the
    words it speaks as it speaks them, finished when the caller's input ends."""

    def __init__(self) -> None:
        self.response_id: str | None = None
        self.item_id: str | None = None
        self.text = ""
        self.audio = False

    def __call__(self, output: OutputChunk) -> list[OutputEvent]:
        value = output.payload
        events: list[OutputEvent] = []
        if value["text"] or value["pcm"]:
            if self.response_id is None:
                session_identity = output.session_identity
                self.response_id = (
                    f"{session_identity.id}-o{session_identity.open_index}"
                )
                self.item_id = self.response_id + "-message"
                events.append(ResponseStarted(self.response_id))
            else:
                pass
            if value["text"]:
                self.text += value["text"]
                events.append(TextDelta(self.response_id, self.item_id, value["text"]))
            else:
                pass
            if value["pcm"]:
                self.audio = True
                events.append(AudioDelta(self.response_id, self.item_id, value["pcm"]))
            else:
                pass
        else:
            pass
        if output.eos and self.response_id is not None:
            events.append(TextFinished(self.response_id, self.item_id, self.text))
            if self.audio:
                events.append(AudioFinished(self.response_id, self.item_id))
            else:
                pass
            events.append(
                ResponseFinished(
                    self.response_id,
                    self.item_id,
                    self.text,
                    self.audio,
                    "completed",
                    "input_audio_ended",
                )
            )
            self.response_id = None
            self.text = ""
            self.audio = False
        else:
            pass
        return events


def create_realtime_deployment(
    client: Client, pipeline_config: PersonaPlexSessionPipelineConfig
) -> RealtimeDeployment:
    def create_adapter() -> CoordinatorAdapter:
        return CoordinatorAdapter(
            client,
            request_builder=build_session_request,
            output_converter=PersonaPlexOutputConverter(),
            input_sample_rate_hz=SAMPLE_RATE,
            atomic_consumption=True,
        )

    return RealtimeDeployment(
        capabilities=CAPABILITIES,
        adapter_factory=create_adapter,
        max_connections=pipeline_config.max_sessions,
    )


__all__ = [
    "CAPABILITIES",
    "PersonaPlexOutputConverter",
    "build_session_request",
    "create_realtime_deployment",
]
