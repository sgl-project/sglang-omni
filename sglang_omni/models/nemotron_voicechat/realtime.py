# SPDX-License-Identifier: Apache-2.0
"""VoiceChat events on the shared realtime protocol, with one continuous response."""

from collections.abc import Iterable

from sglang_omni.client.client import Client
from sglang_omni.models.nemotron_voicechat.duplex_config import STAGES
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk, SessionIdentity, SessionLimits
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
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits


class VoiceChatOutput:
    def __init__(self) -> None:
        self.session_identity: SessionIdentity | None = None
        self.response: str | None = None
        self.text = ""

    def __call__(self, output: OutputChunk) -> Iterable[OutputEvent]:
        payload = output.payload
        if not isinstance(payload, dict):
            raise ValueError("invalid VoiceChat terminal payload")
        else:
            pcm, text, eos = payload.get("pcm"), payload.get("text"), payload.get("eos")
        if (
            not isinstance(pcm, bytes)
            or not isinstance(text, str)
            or not isinstance(eos, bool)
        ):
            raise ValueError("invalid VoiceChat audio, text or EOS")
        else:
            pass
        if self.session_identity != output.session_identity or self.response is None:
            self.session_identity = output.session_identity
            self.response = f"{output.session_identity.id}-o{output.session_identity.open_index}-u{output.input_seq}"
            self.text = ""
            yield ResponseStarted(self.response)
        else:
            pass
        response_id, item_id = self.response, self.response + "-item"
        if text:
            self.text += text
            yield TextDelta(response_id, item_id, text)
        else:
            pass
        if pcm:
            yield AudioDelta(response_id, item_id, pcm)
        else:
            pass
        if eos:
            yield TextFinished(response_id, item_id, self.text)
            yield AudioFinished(response_id, item_id)
            yield ResponseFinished(
                response_id, item_id, self.text, True, "completed", "stop"
            )
            self.response = None
        else:
            pass


def make_adapter(
    client: Client, limits: SessionLimits | None = None
) -> CoordinatorAdapter:
    def request(config: SessionConfiguration) -> OmniRequest:
        if config.get("instructions"):
            raise ValueError("VoiceChat currently uses its checkpoint system prompt")
        else:
            return OmniRequest(inputs=None)

    return CoordinatorAdapter(
        client,
        stages=STAGES,
        request_builder=request,
        output_converter=VoiceChatOutput(),
        atomic_consumption=True,
        limits=limits,
    )


def deployment(
    client: Client, *, session_limits: SessionLimits | None = None
) -> RealtimeDeployment:
    return RealtimeDeployment(
        Capabilities(
            interaction="native",
            input_sample_rate_hz=16000,
            output_sample_rate_hz=22050,
            output_modalities=("audio", "text"),
            native_unit_ms=80,
            tail_policy="pad",
        ),
        lambda: make_adapter(client, session_limits),
        limits=(
            RuntimeLimits(cleanup_timeout_s=session_limits.operation_timeout_s)
            if session_limits is not None
            else RuntimeLimits()
        ),
        max_connections=1,
    )
