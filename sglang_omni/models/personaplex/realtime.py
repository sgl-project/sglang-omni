# SPDX-License-Identifier: Apache-2.0
"""Serves PersonaPlex over /v1/realtime as a native full-duplex conversation."""

from __future__ import annotations

import uuid

from sglang_omni.client import Client
from sglang_omni.models.personaplex.architecture import SAMPLE_RATE, SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.config import REALTIME_STAGES
from sglang_omni.proto import OmniRequest
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
from sglang_omni.serve.realtime.types import Capabilities

NATIVE_UNIT_MS = SAMPLES_PER_FRAME * 1000 // SAMPLE_RATE

CAPABILITIES = Capabilities(
    input_sample_rate_hz=SAMPLE_RATE,
    output_sample_rate_hz=SAMPLE_RATE,
    output_modalities=("audio", "text"),
    native_unit_ms=NATIVE_UNIT_MS,
    tail_policy="pad",
)


def build_session_request(config: SessionConfiguration) -> OmniRequest:
    """instructions becomes the text prompt; the session keeps the default voice."""
    params = {}
    if "instructions" in config:
        params["instructions"] = config["instructions"]
    else:
        pass
    return OmniRequest(inputs=None, params=params)


class PersonaPlexOutputConverter:
    """The agent's side of the call is one response: audio every frame, and
    the words it speaks as they are spoken, until the caller's audio ends."""

    def __init__(self) -> None:
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.item_id = f"item_{uuid.uuid4().hex}"
        self.text = ""
        self.is_started = False
        self.is_finished = False

    def __call__(self, output: OutputChunk) -> list[OutputEvent]:
        if self.is_finished:
            return []
        else:
            pass
        events: list[OutputEvent] = []
        if not self.is_started:
            events.append(ResponseStarted(self.response_id))
            self.is_started = True
        else:
            pass
        payload = output.payload
        if output.modality == "audio" and isinstance(payload, bytes) and payload:
            events.append(AudioDelta(self.response_id, self.item_id, payload))
        elif output.modality == "text" and isinstance(payload, dict):
            text = payload["text"]
            assert isinstance(text, str), type(text)
            self.text += text
            events.append(TextDelta(self.response_id, self.item_id, text))
        else:
            pass
        if output.eos:
            self.is_finished = True
            events.extend(
                (
                    TextFinished(self.response_id, self.item_id, self.text),
                    AudioFinished(self.response_id, self.item_id),
                    ResponseFinished(
                        self.response_id,
                        self.item_id,
                        self.text,
                        True,
                        "completed",
                        "input_audio_ended",
                    ),
                )
            )
        else:
            pass
        return events


def create_realtime_deployment(client: Client) -> RealtimeDeployment:
    def create_adapter() -> CoordinatorAdapter:
        return CoordinatorAdapter(
            client,
            stages=list(REALTIME_STAGES),
            request_builder=build_session_request,
            output_converter=PersonaPlexOutputConverter(),
            input_sample_rate_hz=SAMPLE_RATE,
            atomic_consumption=True,
        )

    return RealtimeDeployment(capabilities=CAPABILITIES, adapter_factory=create_adapter)


__all__ = [
    "CAPABILITIES",
    "PersonaPlexOutputConverter",
    "build_session_request",
    "create_realtime_deployment",
]
