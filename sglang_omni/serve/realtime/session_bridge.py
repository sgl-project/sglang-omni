# SPDX-License-Identifier: Apache-2.0
"""Realtime interaction over a Session described by its published ports."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from functools import partial

import numpy as np

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.client.client import Client
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    CONFIGURED,
    CONTROL,
    IMAGE,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    PCM_F32,
    REPLY_HEARD,
    REPLY_PROGRESS,
    REPLY_TRUNCATED,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT,
    TEXT_DELTA,
    TRANSCRIPT,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    ImagePort,
    SessionPorts,
)
from sglang_omni.serve.realtime.adapters import pipeline_limits_with_idle_backup
from sglang_omni.serve.realtime.constants import CLOSE_REASON_INTERNAL
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    InputTranscription,
    OutputEvent,
    ResponseFinished,
    ResponseHeard,
    ResponseProgress,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    TextDelta,
    TextFinished,
    TurnFailure,
)
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.task_cleanup import cancel_local_tasks
from sglang_omni.serve.realtime.types import (
    DEFAULT_SAMPLE_RATE_HZ,
    PCM16_BYTES_PER_SAMPLE,
    Capabilities,
    InteractionAdapter,
    OutputSink,
    ProtocolError,
    RuntimeLimits,
    Unit,
    failure_code,
    samples_to_ms,
)

logger = logging.getLogger(__name__)

PCM16_SCALE = 32768


def pcm_f32_of(pcm16: bytes) -> bytes:
    return (np.frombuffer(pcm16, dtype="<i2").astype("<f4") / PCM16_SCALE).tobytes()


def pcm16_of(pcm_f32: bytes) -> bytes:
    samples = np.clip(np.frombuffer(pcm_f32, dtype="<f4"), -1.0, 1.0)
    return (samples * (PCM16_SCALE - 1)).astype("<i2").tobytes()


INPUT_ENCODERS: dict[str, Callable[[bytes], bytes]] = {
    PCM16: bytes,
    PCM_F32: pcm_f32_of,
}
OUTPUT_DECODERS: dict[str, Callable[[bytes], bytes]] = {
    PCM16: bytes,
    PCM_F32: pcm16_of,
}


def capabilities_of(ports: SessionPorts) -> Capabilities:
    output_modalities = ("text", "audio") if ports.output is not None else ("text",)
    return Capabilities(
        interaction="turn_based",
        input_sample_rate_hz=ports.audio_input.sample_rate_hz,
        output_sample_rate_hz=(
            ports.output.sample_rate_hz
            if ports.output is not None
            else DEFAULT_SAMPLE_RATE_HZ
        ),
        output_modalities=output_modalities,
        max_granted_modalities=len(output_modalities),
        turn_control=(*ports.turn_detection_types, None),
        native_unit_ms=ports.unit_ms,
        tail_policy="flush",
        input_modalities=ports.input_modalities,
        sampling_parameters=ports.sampling_parameters,
        supports_instruction_updates="instructions" in ports.hot_settings,
        supports_sampling_updates="sampling" in ports.hot_settings,
        supports_truncate=TRUNCATE in ports.controls,
        supports_cancel=INTERRUPT in ports.controls,
        supports_client_commit=COMMIT in ports.controls,
    )


def session_params(config: SessionConfiguration) -> dict[str, object]:
    """Realtime settings under their standard names; unset settings are absent, and an explicit null turn_detection selects manual turns."""
    audio_input = config.get("audio", {}).get("input", {})
    sampling = config.get("sglang", {}).get("sampling")
    settings = {
        "instructions": config.get("instructions"),
        "sampling": dict(sampling) if sampling is not None else None,
    }
    params = {name: value for name, value in settings.items() if value is not None}
    if "turn_detection" in audio_input:
        turn_detection = audio_input["turn_detection"]
        params["turn_detection"] = (
            dict(turn_detection) if turn_detection is not None else None
        )
    else:
        pass
    return params


class SessionBridge(InteractionAdapter):
    def __init__(
        self,
        client: Client,
        ports: SessionPorts,
        *,
        limits: SessionLimits | None = None,
    ) -> None:
        if ports.audio_input.format not in INPUT_ENCODERS or (
            ports.output is not None and ports.output.format not in OUTPUT_DECODERS
        ):
            raise ValueError("bridged audio must be pcm16 or pcm_f32le")
        elif any(
            isinstance(port, ImagePort) and port.format != IMAGE_FRAME
            for port in ports.input
        ):
            raise ValueError(f"bridged images must be {IMAGE_FRAME}")
        else:
            pass
        self.client = client
        self.ports = ports
        self.limits = limits or SessionLimits()
        self.cleanup_timeout_s = self.limits.operation_timeout_s
        self.encode_input = INPUT_ENCODERS[ports.audio_input.format]
        self.projections: dict[
            tuple[str, str | None], Callable[[OutputChunk], list[OutputEvent]]
        ] = {
            (TRANSCRIPT, TRANSCRIPT_FORMAT): self.on_transcript,
            (TEXT, TEXT_DELTA): self.on_text_delta,
            (CONTROL, SESSION_CONTROL): self.on_control,
        }
        if ports.output is not None:
            self.projections[(AUDIO, ports.output.format)] = self.on_audio
            self.decode_output = OUTPUT_DECODERS[ports.output.format]
        else:
            pass
        self.controls: dict[str, Callable[[OutputChunk], list[OutputEvent]]] = {
            SPEECH_STARTED: self.on_speech_started,
            SPEECH_STOPPED: self.on_speech_stopped,
            TURN_QUEUED: self.on_turn_queued,
            TURN_STARTED: self.on_turn_started,
            TURN_DONE: self.on_turn_done,
            TURN_EMPTY: self.on_turn_empty,
            TURN_INTERRUPTED: partial(self.finish_response, "cancelled", "interrupted"),
            CONFIGURED: self.on_configured,
            REPLY_PROGRESS: self.on_progress,
            REPLY_HEARD: self.on_heard,
        }
        self.session_identity: SessionIdentity | None = None
        self.output_sink: OutputSink | None = None
        self.output_reader_task: asyncio.Task[None] | None = None
        self.params: dict[str, object] = {}
        self.append_lock = asyncio.Lock()
        self.next_input_sequence = 0
        self.input_end_ms = 0.0
        self.pending_input_sequence: int | None = None
        self.input_done: asyncio.Future[None] | None = None
        self.reader_error: Exception | None = None
        self.is_closing = False
        self.user_count = 0
        self.user_item_id: str | None = None
        self.is_user_item_open = False
        self.turn_count = 0
        self.response_id: str | None = None
        self.response_item_id = ""
        self.response_turn: int | None = None
        # note (Haoling Pu): a truncate still cuts the last ended response (turn, response id, item id) until the next reply starts.
        self.ended_response: tuple[int | None, str, str] | None = None
        # note (luojiaxuan): The turn of every reply item, so a client truncate reaches the reply it names.
        self.item_turns: dict[str, int | None] = {}
        self.response_text = ""
        self.has_audio = False

    def set_limits(self, runtime_limits: RuntimeLimits) -> None:
        self.cleanup_timeout_s = runtime_limits.cleanup_timeout_s
        self.limits = pipeline_limits_with_idle_backup(self.limits, runtime_limits)

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        negotiated_rate = config["audio"]["input"]["format"]["rate"]
        if negotiated_rate != self.ports.audio_input.sample_rate_hz:
            raise ValueError(
                f"session input sample rate {self.ports.audio_input.sample_rate_hz} "
                f"differs from negotiated rate {negotiated_rate}"
            )
        else:
            pass
        self.output_sink = emit
        self.params = session_params(config)
        self.session_identity = await self.client.open_session(
            OmniRequest(None, params=self.params),
            limits=self.limits,
            session_id=session_id,
        )
        self.output_reader_task = asyncio.create_task(self.read_session_outputs())

    async def append_chunk(self, chunk: TimedChunk) -> None:
        assert self.session_identity is not None
        await self.client.append_session(self.session_identity, chunk)
        self.next_input_sequence += 1

    async def process(self, unit: Unit) -> int:
        if self.reader_error is not None:
            raise self.reader_error
        else:
            pass
        rate = self.ports.audio_input.sample_rate_hz
        t_ms = samples_to_ms(unit.start_sample, rate)
        duration_ms = samples_to_ms(unit.real_samples, rate)
        pcm = unit.pcm[: unit.real_samples * PCM16_BYTES_PER_SAMPLE]
        self.input_done = asyncio.get_running_loop().create_future()
        try:
            async with self.append_lock:
                for image in unit.images:
                    await self.append_chunk(
                        TimedChunk(
                            IMAGE,
                            t_ms,
                            0,
                            self.next_input_sequence,
                            image,
                            format=IMAGE_FRAME,
                        )
                    )
                self.pending_input_sequence = self.next_input_sequence
                await self.append_chunk(
                    TimedChunk(
                        AUDIO,
                        t_ms,
                        duration_ms,
                        self.next_input_sequence,
                        self.encode_input(pcm),
                        format=self.ports.audio_input.format,
                        eos=unit.eos,
                    )
                )
                self.input_end_ms = t_ms + duration_ms
            await self.input_done
            return unit.real_samples
        finally:
            self.pending_input_sequence = None
            self.input_done = None

    async def append_control(
        self, control: dict[str, object], *, turn: int | None = None
    ) -> None:
        async with self.append_lock:
            await self.append_chunk(
                TimedChunk(
                    CONTROL,
                    self.input_end_ms,
                    0,
                    self.next_input_sequence,
                    control,
                    format=SESSION_CONTROL,
                    turn=turn,
                )
            )

    async def cancel(self) -> None:
        assert self.session_identity is not None
        await self.client.control_session(
            self.session_identity,
            TimedChunk(
                CONTROL,
                self.input_end_ms,
                0,
                0,
                {"type": INTERRUPT},
                format=SESSION_CONTROL,
            ),
            should_preempt=True,
        )

    async def truncate(self, item_id: str, audio_end_ms: float) -> None:
        if item_id in self.item_turns:
            await self.append_control(
                {"type": TRUNCATE, "t_ms": audio_end_ms},
                turn=self.item_turns[item_id],
            )
        else:
            raise ProtocolError(
                "item_not_found", f"assistant item {item_id!r} was not found", "item_id"
            )

    async def commit(self) -> None:
        await self.append_control({"type": COMMIT})

    async def update(self, config: SessionConfiguration) -> None:
        params = session_params(config)
        changed = {
            name: params.get(name)
            for name in sorted(self.ports.hot_settings)
            if (name in params, params.get(name))
            != (name in self.params, self.params.get(name))
        }
        if changed:
            await self.append_control({"type": CONFIGURE, **changed})
        else:
            pass
        self.params = params

    async def read_session_outputs(self) -> None:
        assert self.session_identity is not None and self.output_sink is not None
        try:
            async for output in self.client.session_outputs(self.session_identity):
                if output.kind == "input_done":
                    self.finish_input(output.input_seq)
                else:
                    for event in self.project(output):
                        await self.output_sink(event)
            if not self.is_closing:
                raise RuntimeError("session output stream closed")
            else:
                pass
        except Exception as exc:
            if ContextExhaustedError.matches(exc):
                code = ContextExhaustedError.CODE
            else:
                code = failure_code(exc)
            if code == CLOSE_REASON_INTERNAL:
                logger.exception("Session bridge output reader failed")
            else:
                pass
            self.reader_error = exc
            if self.input_done is not None and not self.input_done.done():
                self.input_done.set_exception(exc)
            elif not self.is_closing:
                await self.output_sink(TurnFailure("server_error", code, str(exc)))
            else:
                pass

    def finish_input(self, input_seq: int) -> None:
        if (
            input_seq == self.pending_input_sequence
            and self.input_done is not None
            and not self.input_done.done()
        ):
            self.input_done.set_result(None)
        else:
            pass

    def project(self, output: OutputChunk) -> list[OutputEvent]:
        projection = self.projections.get((output.modality, output.format))
        if projection is None:
            raise ValueError(
                f"unsupported session output {output.modality}/{output.format}"
            )
        else:
            return projection(output)

    def on_control(self, output: OutputChunk) -> list[OutputEvent]:
        projection = self.controls.get(output.payload["type"])
        if projection is None:
            raise ValueError(f"unsupported session control {output.payload['type']}")
        else:
            return projection(output)

    def new_user_item(self) -> str:
        self.user_item_id = f"item_user_{self.user_count}"
        self.user_count += 1
        self.is_user_item_open = True
        return self.user_item_id

    def start_response(self, turn: int | None) -> list[OutputEvent]:
        if self.response_id is not None:
            return []
        else:
            self.response_id = f"resp_{self.turn_count}"
            self.response_turn = turn
            self.response_item_id = f"item_{self.turn_count}"
            self.item_turns[self.response_item_id] = turn
            self.turn_count += 1
            self.response_text = ""
            self.has_audio = False
            return [ResponseStarted(self.response_id)]

    def on_transcript(self, output: OutputChunk) -> list[OutputEvent]:
        payload = output.payload
        item_id = (
            self.user_item_id
            if self.user_item_id is not None and self.is_user_item_open
            else self.new_user_item()
        )
        self.is_user_item_open = not payload["is_final"]
        return [InputTranscription(item_id, payload["text"], payload["is_final"])]

    def on_text_delta(self, output: OutputChunk) -> list[OutputEvent]:
        events = self.start_response(output.turn)
        text = output.payload["text"]
        self.response_text += text
        events.append(TextDelta(self.response_id, self.response_item_id, text))
        return events

    def on_audio(self, output: OutputChunk) -> list[OutputEvent]:
        events = self.start_response(output.turn)
        self.has_audio = True
        events.append(
            AudioDelta(
                self.response_id,
                self.response_item_id,
                self.decode_output(output.payload),
            )
        )
        return events

    def on_progress(self, output: OutputChunk) -> list[OutputEvent]:
        if self.response_id is not None and output.turn == self.response_turn:
            return [
                ResponseProgress(
                    self.response_id,
                    self.response_item_id,
                    output.payload["text_end"],
                    output.payload["audio_ms"],
                )
            ]
        else:
            return []

    def on_heard(self, output: OutputChunk) -> list[OutputEvent]:
        if self.ended_response is not None and output.turn == self.ended_response[0]:
            _, response_id, item_id = self.ended_response
            return [ResponseHeard(response_id, item_id, output.payload["text_end"])]
        else:
            return []

    def on_speech_started(self, output: OutputChunk) -> list[OutputEvent]:
        return [SpeechStarted(self.new_user_item(), output.payload["t_ms"])]

    def on_speech_stopped(self, output: OutputChunk) -> list[OutputEvent]:
        item_id = (
            self.user_item_id if self.user_item_id is not None else self.new_user_item()
        )
        return [SpeechStopped(item_id, output.payload["t_ms"])]

    def on_turn_queued(self, output: OutputChunk) -> list[OutputEvent]:
        return []

    def on_turn_started(self, output: OutputChunk) -> list[OutputEvent]:
        self.ended_response = None
        return self.start_response(output.turn)

    def finish_response(
        self, status: str, reason: str, output: OutputChunk
    ) -> list[OutputEvent]:
        events = self.start_response(output.turn)
        ids = (self.response_id, self.response_item_id)
        events.append(TextFinished(*ids, self.response_text))
        if self.has_audio:
            events.append(AudioFinished(*ids))
        else:
            pass
        events.append(
            ResponseFinished(*ids, self.response_text, self.has_audio, status, reason)
        )
        self.ended_response = (self.response_turn, *ids)
        self.response_id = None
        return events

    def on_turn_done(self, output: OutputChunk) -> list[OutputEvent]:
        """A reply the talker stopped at its token budget is incomplete; any other reply completed."""
        if output.payload.get(REPLY_TRUNCATED) is True:
            return self.finish_response("incomplete", "max_output_tokens", output)
        else:
            return self.finish_response("completed", "stop", output)

    def on_turn_empty(self, output: OutputChunk) -> list[OutputEvent]:
        events = self.start_response(output.turn)
        events.append(
            ResponseFinished(
                self.response_id,
                self.response_item_id,
                "",
                False,
                "completed",
                "empty_turn",
            )
        )
        self.response_id = None
        return events

    def on_configured(self, output: OutputChunk) -> list[OutputEvent]:
        return []

    async def close(self) -> None:
        self.is_closing = True
        try:
            if self.session_identity is not None:
                await self.client.close_session(self.session_identity)
            else:
                pass
        finally:
            await cancel_local_tasks([self.output_reader_task], self.cleanup_timeout_s)
