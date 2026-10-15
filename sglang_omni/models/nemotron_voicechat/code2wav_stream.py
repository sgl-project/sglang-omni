from __future__ import annotations

from collections.abc import Coroutine
from typing import Protocol

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

OUTPUT_SAMPLE_RATE = 22_050
DECODE_WINDOW_FRAMES = 16
TAIL_HOLDBACK_SAMPLES = 256


class CodecDecoder(Protocol):
    samples_per_frame: int
    num_quantizers: int

    def __call__(self, codes_TQ: torch.Tensor) -> torch.Tensor: ...

    def required_context_frames(self, holdback_samples: int) -> int: ...


class DecodePayload(Protocol):
    def __call__(
        self, payload: StagePayload
    ) -> StagePayload | Coroutine[None, None, StagePayload]: ...


def validate_chunk_frames(decoder: CodecDecoder, chunk_frames: tuple[int, ...]) -> None:
    if not chunk_frames or any(
        type(frames) is not int or frames < 1 for frames in chunk_frames
    ):
        raise ValueError("chunk_frames must contain positive integers")
    elif (
        max(chunk_frames) + decoder.required_context_frames(TAIL_HOLDBACK_SAMPLES)
        > DECODE_WINDOW_FRAMES
    ):
        raise ValueError("Chunk plus required context exceeds the decoder window")
    else:
        pass


class StreamingCodec:
    def __init__(
        self,
        decoder: CodecDecoder,
        device: str | torch.device,
        *,
        chunk_frames: tuple[int, ...],
    ) -> None:
        validate_chunk_frames(decoder, chunk_frames)
        self.decoder: CodecDecoder = decoder
        self.device: str | torch.device = device
        self.chunk_frames: tuple[int, ...] = chunk_frames
        self.chunk_index: int = 0
        self.decoded_frames: int = 0
        self.window_codes_TQ: torch.Tensor = torch.empty(
            0, decoder.num_quantizers, dtype=torch.long, device=device
        )
        self.pushed_frames: int = 0
        self.emitted_samples: int = 0

    @torch.inference_mode()
    def push(self, codes_TQ: torch.Tensor) -> list[torch.Tensor]:
        audio_parts: list[torch.Tensor] = []
        for row in codes_TQ.to(self.device):
            self.window_codes_TQ = torch.cat(
                (self.window_codes_TQ, row[None])
            )[-DECODE_WINDOW_FRAMES:]
            self.pushed_frames += 1
            target_frames = self.chunk_frames[
                min(self.chunk_index, len(self.chunk_frames) - 1)
            ]
            if self.pushed_frames - self.decoded_frames >= target_frames:
                audio_parts.append(self.advance(final=False))
                self.chunk_index += 1
            else:
                pass
        return audio_parts

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if self.emitted_samples == self.pushed_frames * self.decoder.samples_per_frame:
            return torch.zeros(0)
        else:
            pass
        return self.advance(final=True)

    def advance(self, *, final: bool) -> torch.Tensor:
        frames = self.pushed_frames
        samples_per_frame = self.decoder.samples_per_frame
        first = frames - self.window_codes_TQ.shape[0]
        audio = self.decoder(self.window_codes_TQ)
        available = frames * samples_per_frame - (0 if final else TAIL_HOLDBACK_SAMPLES)
        start = self.emitted_samples - first * samples_per_frame
        fresh = audio[start : available - first * samples_per_frame].float().cpu()
        self.decoded_frames = frames
        self.emitted_samples = available
        return fresh


class StreamState:
    def __init__(
        self,
        decoder: CodecDecoder,
        device: str | torch.device,
        chunk_frames: tuple[int, ...],
    ) -> None:
        self.codec: StreamingCodec = StreamingCodec(
            decoder, device, chunk_frames=chunk_frames
        )
        self.audio_parts: list[torch.Tensor] = []


class NemotronCode2WavScheduler(StreamingSimpleScheduler):
    def __init__(
        self,
        decoder: CodecDecoder,
        device: str | torch.device,
        *,
        chunk_frames: tuple[int, ...],
        compute_fn: DecodePayload | None,
    ) -> None:
        validate_chunk_frames(decoder, chunk_frames)
        super().__init__(compute_fn)
        self.decoder: CodecDecoder = decoder
        self.device: str | torch.device = device
        self.chunk_frames: tuple[int, ...] = chunk_frames
        self.states: dict[str, StreamState] = {}

    def new_state(self) -> StreamState:
        return StreamState(self.decoder, self.device, self.chunk_frames)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return payload.request_id in self.states

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        if request_id not in self.states:
            self.states[request_id] = self.new_state()
        else:
            pass

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    @torch.inference_mode()
    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        assert isinstance(item.data, torch.Tensor)
        if request_id not in self.states:
            self.states[request_id] = self.new_state()
        else:
            pass
        state = self.states[request_id]
        audio_parts = state.codec.push(item.data)
        state.audio_parts.extend(audio_parts)
        # Chunks to the coordinator are msgpack'd, so the waveform travels in
        # the shared payload format rather than as a tensor.
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=audio_waveform_payload(
                    audio,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    modality="audio",
                    source_hint="NemotronVoiceChat",
                ),
                metadata={"modality": "audio"},
            )
            for audio in audio_parts
        ]

    @torch.inference_mode()
    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self.states.get(request_id)
        if state is None:
            return []
        else:
            pass
        messages: list[OutgoingMessage] = []
        if state.codec.pushed_frames:
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
