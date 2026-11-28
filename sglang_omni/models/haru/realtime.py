# SPDX-License-Identifier: Apache-2.0
"""Expose Haru frames through the shared realtime session protocol."""

import asyncio
from dataclasses import replace
from functools import partial

from sglang_omni.client.client import Client
from sglang_omni.models.haru.config import (
    INPUT_FRAME_SAMPLES,
    NATIVE_UNIT_MILLISECONDS,
    SAMPLE_RATE_HZ,
    HaruPipelineConfig,
    HaruSampling,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk, SessionLimits
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
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
    ResponseStatus,
    TextDelta,
    TextFinished,
    TurnFailure,
)
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    MAX_BUFFERED_INPUT_S,
    MS_PER_SECOND,
    PCM16_BYTES_PER_SAMPLE,
    Capabilities,
    OutputSink,
    RuntimeLimits,
    Unit,
)


def build_session_request(
    pipeline_config: HaruPipelineConfig, session_config: SessionConfiguration
) -> OmniRequest:
    extension = session_config.get("sglang", {})
    sampling = HaruSampling.model_validate(
        {
            **pipeline_config.sampling.model_dump(),
            **extension.get("sampling", {}),
        }
    )
    params: dict[str, object] = {
        **sampling.model_dump(),
        "instructions": session_config.get(
            "instructions", pipeline_config.instructions
        ),
    }
    reference_audio = extension.get("reference_audio")
    tts_reference_audio = extension.get("tts_reference_audio")
    if reference_audio is not None and tts_reference_audio is not None:
        raise ValueError("Haru has one voice reference per session")
    else:
        reference_audio = reference_audio or tts_reference_audio
    if reference_audio is not None:
        params["reference_audio"] = reference_audio.data
    else:
        pass
    return OmniRequest(inputs=None, params=params)


class HaruOutput:
    def __init__(self) -> None:
        self.response_id: str | None = None
        self.response_number = 0
        self.transcript = ""
        self.has_audio = False

    def __call__(self, output: OutputChunk) -> list[OutputEvent]:
        payload = output.payload
        if not isinstance(payload, dict):
            raise ValueError("Haru output must contain PCM16 and text")
        else:
            pcm, text = payload.get("pcm"), payload.get("text")
        if not isinstance(pcm, bytes) or not isinstance(text, str):
            raise ValueError("Haru output must contain PCM16 and text")
        elif not pcm and not text:
            return []
        else:
            pass
        events: list[OutputEvent] = []
        if self.response_id is None:
            identity = output.session_identity
            self.response_id = (
                f"{identity.id}-o{identity.open_index}-r{self.response_number}"
            )
            events.append(ResponseStarted(self.response_id))
        else:
            pass
        response_id = self.response_id
        item_id = response_id + "-message"
        if pcm:
            self.has_audio = True
            events.append(AudioDelta(response_id, item_id, pcm))
        else:
            pass
        if text:
            self.transcript += text
            events.append(TextDelta(response_id, item_id, text))
        else:
            pass
        return events

    def finish(self, status: ResponseStatus, reason: str) -> list[OutputEvent]:
        response_id = self.response_id
        if response_id is None:
            return []
        else:
            pass
        item_id = response_id + "-message"
        events: list[OutputEvent] = [
            TextFinished(response_id, item_id, self.transcript)
        ]
        if self.has_audio:
            events.append(AudioFinished(response_id, item_id))
        else:
            pass
        events.append(
            ResponseFinished(
                response_id=response_id,
                item_id=item_id,
                text=self.transcript,
                has_audio=self.has_audio,
                status=status,
                reason=reason,
            )
        )
        self.response_id = None
        self.response_number += 1
        self.transcript = ""
        self.has_audio = False
        return events


class HaruRealtimeAdapter(CoordinatorAdapter):
    def __init__(
        self,
        client: Client,
        pipeline_config: HaruPipelineConfig,
        *,
        session_limits: SessionLimits | None = None,
    ) -> None:
        self.haru_output = HaruOutput()
        self.publish_first_frame = pipeline_config.sampling.speak_first
        self.initial_events: list[OutputEvent] = []
        self.response_lock = asyncio.Lock()
        super().__init__(
            client,
            request_builder=partial(build_session_request, pipeline_config),
            output_converter=self.haru_output,
            input_sample_rate_hz=SAMPLE_RATE_HZ,
            atomic_consumption=True,
            limits=session_limits,
        )

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        async def collect_initial_output(
            event: OutputEvent, unit: Unit | None = None
        ) -> None:
            self.initial_events.append(event)

        self.publish_first_frame = (
            config.get("sglang", {})
            .get("sampling", {})
            .get("speak_first", self.publish_first_frame)
        )
        await super().open(session_id, config, collect_initial_output)
        # Note (wbn): Frame zero comes from prompt prefill, before any caller samples.
        await super().process(Unit(index=0, start_sample=0, pcm=b"", real_samples=0))
        self.output_sink = emit

    def startup_output(self) -> tuple[OutputEvent, ...]:
        if self.publish_first_frame:
            events = tuple(self.initial_events)
            self.initial_events.clear()
            return events
        else:
            return ()

    async def process(self, unit: Unit) -> int:
        assert self.output_sink is not None
        async with self.response_lock:
            for event in self.initial_events:
                await self.output_sink(event, unit)
            self.initial_events.clear()
            consumed_samples = await super().process(
                replace(unit, input_sequence=unit.pipeline_sequence + 1)
            )
            if unit.eos:
                for event in self.haru_output.finish("completed", "stop"):
                    await self.output_sink(event, unit)
            else:
                pass
            return consumed_samples

    async def cancel(self) -> None:
        assert self.output_sink is not None
        async with self.response_lock:
            if self.initial_events:
                return
            else:
                for event in self.haru_output.finish("cancelled", "cancelled"):
                    await self.output_sink(event)


async def warm_up(
    client: Client, pipeline_config: HaruPipelineConfig, timeout_seconds: float
) -> None:
    """Compile the prefill, retained-cache, and codec paths before admission."""
    adapter = HaruRealtimeAdapter(
        client,
        pipeline_config,
        session_limits=SessionLimits(operation_timeout_s=timeout_seconds),
    )

    async def discard_output(event: OutputEvent, unit: Unit | None = None) -> None:
        if isinstance(event, TurnFailure):
            raise RuntimeError(event.message)
        else:
            pass

    session: SessionConfiguration = {
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ}}}
    }
    pcm = bytes(INPUT_FRAME_SAMPLES * PCM16_BYTES_PER_SAMPLE)
    frames_per_unit = (
        NATIVE_UNIT_MILLISECONDS
        * SAMPLE_RATE_HZ
        // (MS_PER_SECOND * INPUT_FRAME_SAMPLES)
    )
    try:
        await adapter.open("haru-warmup", session, discard_output)
        for frame_index in range(frames_per_unit):
            await adapter.process(
                Unit(
                    index=frame_index,
                    start_sample=frame_index * INPUT_FRAME_SAMPLES,
                    pcm=pcm,
                    real_samples=INPUT_FRAME_SAMPLES,
                    eos=frame_index + 1 == frames_per_unit,
                )
            )
    finally:
        await adapter.close()


def create_realtime_deployment(
    client: Client, pipeline_config: HaruPipelineConfig
) -> RealtimeDeployment:
    return RealtimeDeployment(
        capabilities=Capabilities(
            input_sample_rate_hz=SAMPLE_RATE_HZ,
            output_sample_rate_hz=SAMPLE_RATE_HZ,
            native_input_frame_samples=INPUT_FRAME_SAMPLES,
            native_unit_ms=NATIVE_UNIT_MILLISECONDS,
            tail_policy="pad",
            output_modalities=("audio", "text"),
            max_granted_modalities=2,
            sampling_parameters=tuple(HaruSampling.model_fields),
            supports_reference_audio=True,
            supports_cancel=True,
        ),
        adapter_factory=partial(HaruRealtimeAdapter, client, pipeline_config),
        limits=runtime_limits_for_pipeline(
            pipeline_config,
            RuntimeLimits(
                max_input_bytes=MAX_BUFFERED_INPUT_S
                * SAMPLE_RATE_HZ
                * PCM16_BYTES_PER_SAMPLE
            ),
        ),
        max_connections=pipeline_config.max_sessions,
        warmup=partial(warm_up, client, pipeline_config),
    )
