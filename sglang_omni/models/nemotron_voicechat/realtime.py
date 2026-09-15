# SPDX-License-Identifier: Apache-2.0
"""VoiceChat events on the shared realtime protocol, with one continuous response."""

from sglang_omni.proto import OmniRequest
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.runtime import Capabilities, RuntimeLimits

from .duplex_config import STAGES


class VoiceChatOutput:
    def __init__(self):
        self.epoch = None
        self.response = None
        self.text = ""

    def __call__(self, output):
        data = output.payload
        if not isinstance(data, dict) or "pcm" not in data:
            raise ValueError("invalid VoiceChat terminal payload")
        if self.epoch != output.ref.epoch or self.response is None:
            self.epoch = output.ref.epoch
            self.response = f"{output.ref.session_id}-e{self.epoch}-u{output.input_seq}"
            self.text = ""
            yield ResponseStarted(self.response)
        rid, item = self.response, self.response + "-item"
        if data["text"]:
            self.text += data["text"]
            yield TextDelta(rid, item, data["text"])
        if data["pcm"]:
            yield AudioDelta(rid, item, data["pcm"])
        if data["eos"]:
            yield TextFinished(rid, item, self.text)
            yield AudioFinished(rid, item)
            yield ResponseFinished(rid, item, self.text, True, "completed", "stop")
            self.response = None


def make_adapter(client):
    def request(config):
        if config.get("instructions"):
            raise ValueError("VoiceChat currently uses its checkpoint system prompt")
        return OmniRequest(inputs=None)

    return CoordinatorAdapter(
        client,
        stages=STAGES,
        request_builder=request,
        output_converter=VoiceChatOutput(),
        atomic_consumption=True,
    )


def deployment(client):
    from sglang_omni.serve.realtime.manager import RealtimeDeployment

    # The bounded 4096-position talker supports <325 s. Leave prompt headroom.
    return RealtimeDeployment(
        Capabilities(
            interaction="native",
            input_rate=16000,
            output_rate=22050,
            output_modalities=("audio", "text"),
            native_unit_ms=80,
            tail_policy="pad",
        ),
        lambda: make_adapter(client),
        RuntimeLimits(session_timeout_s=240),
        max_connections=1,
    )
