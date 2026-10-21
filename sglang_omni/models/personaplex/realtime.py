# SPDX-License-Identifier: Apache-2.0
"""Serves PersonaPlex over /v1/realtime as a native full-duplex session."""

from __future__ import annotations

from sglang_omni.client.client import Client
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.config import (
    PersonaPlexSessionPipelineConfig,
    PersonaPlexSessionSampling,
)
from sglang_omni.models.personaplex.prompts import packaged_voices
from sglang_omni.models.weight_loader import resolve_model_path
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


def build_session_request(
    config: PersonaPlexSessionPipelineConfig, session: SessionConfiguration
) -> OmniRequest:
    """The deployment's voice, role prompt and sampling under the session's own;
    the request takes offline PersonaPlex params."""
    extension = session.get("sglang", {})
    params = {
        **config.sampling.model_dump(exclude_none=True),
        **extension.get("sampling", {}),
    }
    for name, value in (
        ("voice", extension.get("voice", config.voice)),
        ("instructions", session.get("instructions", config.instructions)),
    ):
        if value is not None:
            params[name] = value
        else:
            pass
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
            request_builder=lambda session: build_session_request(
                pipeline_config, session
            ),
            output_converter=PersonaPlexOutputConverter(),
            input_sample_rate_hz=SAMPLE_RATE,
            atomic_consumption=True,
        )

    return RealtimeDeployment(
        capabilities=Capabilities(
            input_sample_rate_hz=SAMPLE_RATE,
            output_sample_rate_hz=SAMPLE_RATE,
            output_modalities=("audio", "text"),
            native_unit_ms=SAMPLES_PER_FRAME * MS_PER_SECOND // SAMPLE_RATE,
            tail_policy="pad",
            sampling_parameters=tuple(PersonaPlexSessionSampling.model_fields),
            voices=packaged_voices(resolve_model_path(pipeline_config.model_path)),
        ),
        adapter_factory=create_adapter,
        max_connections=pipeline_config.max_sessions,
    )


__all__ = [
    "PersonaPlexOutputConverter",
    "build_session_request",
    "create_realtime_deployment",
]
