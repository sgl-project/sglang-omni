from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 22_050
DECODE_WINDOW_FRAMES = 16
TAIL_HOLDBACK_SAMPLES = 256


@dataclass(frozen=True)
class DecodeWindow:
    # note (Xinhao Tan): codes is [T, Q] with at most DECODE_WINDOW_FRAMES frames.
    codes: torch.Tensor
    first_sample: int
    end_sample: int

    def fresh(self, audio_S: torch.Tensor) -> torch.Tensor:
        return audio_S[self.first_sample : self.end_sample]


class StreamingCodec:
    def __init__(self, decoder, device) -> None:
        self.decoder = decoder
        self.device = device
        self.window_codes_TQ: torch.Tensor = torch.empty(
            0, decoder.num_quantizers, dtype=torch.long, device=device
        )
        self.pushed_frames: int = 0
        self.emitted_samples = 0

    @torch.inference_mode()
    def push(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        return self.decode(self.plan(codes_TQ))

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if not self.pushed_frames:
            return torch.zeros(0)
        else:
            pass
        return self.decode([self.next_window(final=True)])

    def plan(self, codes_TQ: torch.Tensor) -> list[DecodeWindow]:
        window_codes_TQ = torch.cat((self.window_codes_TQ, codes_TQ.to(self.device)))
        self.window_codes_TQ = window_codes_TQ[-DECODE_WINDOW_FRAMES:]
        self.pushed_frames += codes_TQ.shape[0]
        return [self.next_window(final=False)]

    def next_window(self, *, final: bool) -> DecodeWindow:
        samples_per_frame = self.decoder.samples_per_frame
        first = (self.pushed_frames - self.window_codes_TQ.shape[0]) * samples_per_frame
        available = self.pushed_frames * samples_per_frame - (
            0 if final else TAIL_HOLDBACK_SAMPLES
        )
        window = DecodeWindow(
            self.window_codes_TQ, self.emitted_samples - first, available - first
        )
        self.emitted_samples = available
        return window

    def decode(self, windows: list[DecodeWindow]) -> torch.Tensor:
        return (
            torch.cat([window.fresh(self.decoder(window.codes)) for window in windows])
            .float()
            .cpu()
        )


class StreamState:
    def __init__(self, decoder, device) -> None:
        self.codec = StreamingCodec(decoder, device)
        self.audio_parts: list[torch.Tensor] = []


class NemotronCode2WavScheduler(StreamingSimpleScheduler):
    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

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
        max_batch_size: int,
    ) -> None:
        super().__init__(compute_fn, max_batch_size=max_batch_size)
        self.decoder = decoder
        self.device = device
        self.states: dict[str, StreamState] = {}

    def new_state(self) -> StreamState:
        return StreamState(self.decoder, self.device)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return payload.request_id in self.states

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.states.setdefault(request_id, self.new_state())

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)

    def audio_message(self, request_id: str, audio_S: torch.Tensor) -> OutgoingMessage:
        # Chunks to the coordinator are msgpack'd, so the waveform travels in
        # the shared payload format rather than as a tensor.
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=audio_waveform_payload(
                audio_S,
                sample_rate=OUTPUT_SAMPLE_RATE,
                modality="audio",
                source_hint="NemotronVoiceChat",
            ),
            metadata={"modality": "audio"},
        )

    def decode_windows(self, windows: list[DecodeWindow]) -> list[torch.Tensor]:
        codes_BTQ = torch.stack([window.codes for window in windows])
        padded_size = 1 << (len(windows) - 1).bit_length()
        # note (Xinhao Tan): few batch sizes let the allocator reuse its cached blocks.
        codes_BTQ = torch.cat(
            (codes_BTQ, codes_BTQ[:1].expand(padded_size - len(windows), -1, -1))
        )
        audio_BS = self.decoder.decode_batch(codes_BTQ)[: len(windows)]
        return [
            window.fresh(audio_S)
            for window, audio_S in zip(windows, audio_BS, strict=True)
        ]

    @torch.inference_mode()
    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        errors: dict[str, Exception] = {}
        jobs: list[tuple[str, DecodeWindow]] = []
        with self.state_lock:
            for request_id, item in items:
                try:
                    codec = self.states.setdefault(request_id, self.new_state()).codec
                    jobs += [(request_id, window) for window in codec.plan(item.data)]
                except Exception as exc:
                    errors[request_id] = exc
        # note (Xinhao Tan): no lock while the GPU decodes, so an abort does not wait for it.
        jobs = [
            (request_id, window)
            for request_id, window in jobs
            if request_id not in errors
        ]
        indices_by_window_frames: dict[int, list[int]] = {}
        for index, (_, window) in enumerate(jobs):
            indices_by_window_frames.setdefault(window.codes.shape[0], []).append(index)
        fresh_by_index: dict[int, torch.Tensor] = {}
        for indices in indices_by_window_frames.values():
            try:
                fresh_audio = self.decode_windows([jobs[index][1] for index in indices])
            except Exception:
                logger.exception(
                    f"code2wav decode failed for {len(indices)} windows, "
                    "decoding them one at a time"
                )
                for index in indices:
                    try:
                        (fresh_by_index[index],) = self.decode_windows([jobs[index][1]])
                    except Exception as exc:
                        errors.setdefault(jobs[index][0], exc)
            else:
                fresh_by_index.update(zip(indices, fresh_audio, strict=True))
        emitted = [
            (request_id, fresh_by_index[index])
            for index, (request_id, _) in enumerate(jobs)
            if request_id not in errors
        ]
        host_audio: list[torch.Tensor] = []
        if emitted:
            try:
                # note (Xinhao Tan): every host copy waits for the GPU, so a step copies once.
                host_audio = list(
                    torch.cat([audio_S for _, audio_S in emitted])
                    .float()
                    .cpu()
                    .split([audio_S.numel() for _, audio_S in emitted])
                )
            except Exception as exc:
                for request_id, _ in emitted:
                    errors.setdefault(request_id, exc)
                emitted = []
        else:
            pass
        with self.state_lock:
            for request_id, exc in errors.items():
                logger.error(f"code2wav failed for {request_id}: {exc}")
                self.emit_error(request_id, exc)
                self.abort_state(request_id)
            for (request_id, _), audio_S in zip(emitted, host_audio, strict=True):
                state = self.states.get(request_id)
                if state is None or self.is_aborted(request_id):
                    continue
                else:
                    pass
                # note (Xinhao Tan): a split view would pin the whole step's audio.
                state.audio_parts.append(audio_S.clone())
                self.outbox.put(self.audio_message(request_id, audio_S))
        for request_id in errors:
            self.cleanup_aborted_request(request_id)

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
                messages.append(self.audio_message(request_id, tail))
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
