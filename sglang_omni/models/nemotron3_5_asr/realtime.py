# SPDX-License-Identifier: Apache-2.0
"""Native PCM transcription through the shared realtime session protocol."""

from __future__ import annotations

from collections.abc import Iterable

from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import Capabilities


def build_session_request(config: SessionConfiguration) -> OmniRequest:
    if config.get("instructions"):
        raise ValueError("Nemotron ASR does not support instructions")
    else:
        return OmniRequest(inputs=None, params={"language": "auto"})


def convert_session_output(output: OutputChunk) -> Iterable[OutputEvent]:
    payload = output.payload
    if not isinstance(payload, dict):
        raise ValueError("Nemotron text output must be an object")
    else:
        text, full_text = payload["text"], payload["full_text"]
        if not isinstance(text, str) or not isinstance(full_text, str):
            raise ValueError("Nemotron text output must contain strings")
        else:
            identity = output.session_identity
            response_id = f"asr_{identity.id}_{identity.open_index}"
            item_id = f"{response_id}_text"
            events: list[OutputEvent] = []
            if payload["is_first_output"]:
                events.append(ResponseStarted(response_id))
            else:
                pass
            if text:
                events.append(TextDelta(response_id, item_id, text))
            else:
                pass
            if output.eos:
                events.extend([
                    TextFinished(response_id, item_id, full_text),
                    ResponseFinished(response_id, item_id, full_text, False, "completed", "audio_end"),
                ])
            else:
                pass
            return events


def create_realtime_deployment(client: Client, *, native_unit_ms: int = 20) -> RealtimeDeployment:
    def create_adapter() -> CoordinatorAdapter:
        return CoordinatorAdapter(
            client, stages=["asr"], request_builder=build_session_request,
            output_converter=convert_session_output, input_sample_rate_hz=16000,
            atomic_consumption=True,
        )

    return RealtimeDeployment(
        capabilities=Capabilities(native_unit_ms=native_unit_ms),
        adapter_factory=create_adapter, max_connections=64,
    )
