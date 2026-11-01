# SPDX-License-Identifier: Apache-2.0
"""VoiceChat events on the shared realtime protocol, with one continuous response."""

import asyncio
from collections.abc import Iterable
from functools import partial

from sglang_omni.client.client import Client
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    NemotronVoiceChatDuplexPipelineConfig,
)
from sglang_omni.models.nemotron_voicechat.payload_types import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk, SessionIdentity, SessionLimits
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.control import Closed, Drained, Failure
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    runtime_limits_for_pipeline,
)
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits


class VoiceChatOutput:
    def __init__(self) -> None:
        self.session_identity: SessionIdentity | None = None
        self.response_id: str | None = None
        self.transcript = ""

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
        if self.session_identity != output.session_identity or self.response_id is None:
            self.session_identity = output.session_identity
            self.response_id = f"{output.session_identity.id}-o{output.session_identity.open_index}-u{output.input_seq}"
            self.transcript = ""
            yield ResponseStarted(self.response_id)
        else:
            pass
        response_id, item_id = self.response_id, self.response_id + "-item"
        if text:
            self.transcript += text
            yield TextDelta(response_id, item_id, text)
        else:
            pass
        if pcm:
            yield AudioDelta(response_id, item_id, pcm)
        else:
            pass
        if eos:
            yield TextFinished(response_id, item_id, self.transcript)
            yield AudioFinished(response_id, item_id)
            yield ResponseFinished(
                response_id=response_id,
                item_id=item_id,
                text=self.transcript,
                has_audio=True,
                status="completed",
                reason="stop",
            )
            self.response_id = None
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
        request_builder=request,
        output_converter=VoiceChatOutput(),
        atomic_consumption=True,
        limits=limits,
    )


def deployment(
    client: Client,
    pipeline_config: NemotronVoiceChatDuplexPipelineConfig,
    *,
    session_limits: SessionLimits | None = None,
) -> RealtimeDeployment:
    """The serving layer passes the pipeline config; the Session shape comes from it through the coordinator."""
    limits = session_limits if session_limits is not None else SessionLimits()
    return RealtimeDeployment(
        Capabilities(
            interaction="native",
            input_sample_rate_hz=INPUT_SAMPLE_RATE,
            output_sample_rate_hz=OUTPUT_SAMPLE_RATE,
            output_modalities=("audio", "text"),
            native_unit_ms=80,
            tail_policy="pad",
        ),
        lambda: make_adapter(client, limits),
        limits=runtime_limits_for_pipeline(
            pipeline_config,
            RuntimeLimits(cleanup_timeout_s=limits.operation_timeout_s),
        ),
        session_limits=limits,
        max_connections=pipeline_config.max_sessions,
        warmup=partial(warm_up, client, pipeline_config),
    )


async def warm_up(
    client: Client,
    pipeline_config: NemotronVoiceChatDuplexPipelineConfig,
    timeout_s: float,
) -> None:
    """Run one silent session so the first conversation compiles no unit kernel."""
    warmup_deployment = deployment(
        client,
        pipeline_config,
        session_limits=SessionLimits(operation_timeout_s=timeout_s),
    )
    runtime = SessionRuntime(
        "nemotron-voicechat",
        warmup_deployment.capabilities,
        warmup_deployment.adapter_factory,
        warmup_deployment.limits,
    )

    async def drain(until: type[Drained] | type[Closed]) -> None:
        async for envelope in runtime.outputs():
            if isinstance(envelope.event, Failure):
                raise RuntimeError(envelope.event.message)
            elif isinstance(envelope.event, until):
                return
            else:
                pass

    drain_task = asyncio.create_task(drain(Drained))
    try:
        await runtime.update({"output_modalities": ["audio"]}, "warmup-configure")
        # note (Junnan Li): consecutive one-position units start their KV slots at an even and an odd pool offset, and SGLang compiles its Triton KV-slot writer once per pointer alignment.
        await runtime.append(
            bytes(warmup_deployment.capabilities.native_unit_bytes * 3),
            0,
            None,
            "warmup-audio",
        )
        await runtime.end("warmup-end")
        await drain_task
        await runtime.close("warmup_finished")
        # note (Junnan Li): close reports a failed session cleanup as a Failure event, not by raising.
        await drain(Closed)
    finally:
        await runtime.close("warmup_finished")
        drain_task.cancel()
        await asyncio.gather(drain_task, return_exceptions=True)
