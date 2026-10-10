from __future__ import annotations

from contextlib import nullcontext

from collections.abc import Callable, Coroutine

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

OUTPUT_SAMPLE_RATE = 22_050
DECODE_WINDOW_FRAMES = 16
TAIL_HOLDBACK_SAMPLES = 256


class StreamingCodec:
    def __init__(self, decoder, device) -> None:
        self.decoder = decoder
        self.device = device
        self.codes_rows: list[torch.Tensor] = []
        self.emitted_samples = 0

    @torch.inference_mode()
    def push(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        for row in codes_TQ.to(self.device):
            self.codes_rows.append(row)
        return self.advance(final=False)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if not self.codes_rows:
            return torch.zeros(0)
        else:
            pass
        return self.advance(final=True)

    def advance(self, *, final: bool) -> torch.Tensor:
        frames = len(self.codes_rows)
        samples_per_frame = self.decoder.samples_per_frame
        first = max(0, frames - DECODE_WINDOW_FRAMES)
        audio = self.decoder(torch.stack(self.codes_rows[first:]))
        available = frames * samples_per_frame - (0 if final else TAIL_HOLDBACK_SAMPLES)
        start = self.emitted_samples - first * samples_per_frame
        fresh = audio[start : available - first * samples_per_frame].float().cpu()
        self.emitted_samples = available
        return fresh


class StreamState:
    def __init__(self, decoder, device) -> None:
        self.codec = StreamingCodec(decoder, device)
        self.audio_parts: list[torch.Tensor] = []
        self.decode_stream: torch.cuda.Stream | None = None


class NemotronCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        
        self,
        decoder,
        device,
        *,
        compute_fn: (
            Callable[
                [StagePayload],
                StagePayload | Coroutine[None, None, StagePayload],
            ]
            | None
        ),
        can_use_local_code_handoff: bool = False
    ) -> None:
        super().__init__(compute_fn)
        self.decoder = decoder
        self.device = torch.device(device)
        self.states: dict[str, StreamState] = {}
        self.decode_stream: torch.cuda.Stream | None = None
        if can_use_local_code_handoff and self.device.type == "cuda":
            self.decode_stream = torch.cuda.Stream(device=self.device)
            self.decode_stream.wait_stream(torch.cuda.current_stream(self.device))
        else:
            pass

    def new_state(self) -> StreamState:
        return StreamState(self.decoder, self.device)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return payload.request_id in self.states

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.states.setdefault(request_id, self.new_state())

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    @torch.inference_mode()
    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        codes: torch.Tensor = item.data
        if codes.is_cuda:
            codes_ready_event = (item.metadata or {}).get("codes_ready_event")
            if not isinstance(codes_ready_event, torch.cuda.Event):
                raise RuntimeError(
                    f"CUDA audio codes for request {request_id!r} require a "
                    "codes_ready_event CUDA event"
                )
            else:
                device_module = torch.get_device_module(codes.device)
                if self.decode_stream is None:
                    decode_stream = device_module.current_stream(codes.device)
                else:
                    decode_stream = self.decode_stream
                decode_stream.wait_event(codes_ready_event)
                codes.record_stream(decode_stream)
        else:
            pass
        state = self.states.setdefault(request_id, self.new_state())
        if codes.is_cuda:
            state.decode_stream = self.decode_stream
        else:
            pass
        with (
            torch.cuda.stream(state.decode_stream)
            if state.decode_stream is not None
            else nullcontext()
        ):
            tail = state.codec.push(codes)
        state.audio_parts.append(tail)
        # Chunks to the coordinator are msgpack'd, so the waveform travels in
        # the shared payload format rather than as a tensor.
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=audio_waveform_payload(
                    tail,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    modality="audio",
                    source_hint="NemotronVoiceChat",
                ),
                metadata={"modality": "audio"},
            )
        ]

    @torch.inference_mode()
    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.states.get(request_id)
        if state is None:
            return []
        else:
            pass
        messages: list[OutgoingMessage] = []
        if state.codec.codes_rows:
            with (
                torch.cuda.stream(state.decode_stream)
                if state.decode_stream is not None
                else nullcontext()
            ):
                tail = state.codec.flush()
            if tail.numel():
                state.audio_parts.append(tail)
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=audio_waveform_payload(
                            tail,
                            sample_rate=OUTPUT_SAMPLE_RATE,
                            modality="audio",
                            source_hint="NemotronVoiceChat",
                        ),
                        metadata={"modality": "audio"},
                    )
                )
            else:
                pass
        else:
            pass
        waveform = torch.cat(state.audio_parts) if state.audio_parts else torch.zeros(0)
        return messages + [
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=request_id,
                    request=self.stream_payloads[request_id].request,
                    data=audio_waveform_payload(
                        waveform,
                        sample_rate=OUTPUT_SAMPLE_RATE,
                        modality="audio",
                        source_hint="NemotronVoiceChat",
                    ),
                ),
            )
        ]
