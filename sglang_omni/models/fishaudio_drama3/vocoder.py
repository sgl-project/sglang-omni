# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder: Fish codec decoding of sampled frames, streamed in fixed windows."""

from __future__ import annotations

import asyncio
import bisect
import collections
import logging
from dataclasses import dataclass, field

import torch

from sglang_omni.models.fishaudio_drama3.incremental_decoder import (
    IncrementalWindowDecoder,
)
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationResult
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.profiler.event_recorder import emit, get_recorder
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.audio_stream import (
    FinalAudioData,
    StreamVocoderState,
    WindowDecoder,
    build_audio_chunk_payload,
    build_stream_vocoder_chunk,
)
from sglang_omni.scheduling.audio_vocoder import StreamingVocoderScheduler
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


def generation_result_of(payload: StagePayload) -> Drama3GenerationResult:
    """The engine's result in a vocoder payload, which must hold an audio frame."""
    result = payload.data
    if not isinstance(result, Drama3GenerationResult):
        raise TypeError(
            f"Drama 3 vocoder payloads carry Drama3GenerationResult; got "
            f"{type(result).__name__}"
        )
    elif result.frames.shape[0] == 0:
        raise ValueError(
            f"Request {payload.request_id}: Drama 3 generated no audio frames"
        )
    else:
        return result


def non_streaming_decode_groups(
    frame_counts: list[int], max_padded_frames_per_decode: int | None
) -> list[list[int]]:
    if max_padded_frames_per_decode is None or (
        len(frame_counts) * max(frame_counts) <= max_padded_frames_per_decode
    ):
        return [list(range(len(frame_counts)))]
    else:
        groups: list[list[int]] = []
        for index in sorted(
            range(len(frame_counts)), key=frame_counts.__getitem__, reverse=True
        ):
            if (
                groups
                and (len(groups[-1]) + 1) * frame_counts[groups[-1][0]]
                <= max_padded_frames_per_decode
            ):
                groups[-1].append(index)
            else:
                groups.append([index])
        return groups


class WindowGraphDecoder:
    def __init__(
        self,
        decode: WindowDecoder,
        *,
        num_codebooks: int,
        frame_length: int,
        device: torch.device,
        window_lengths: tuple[int, ...],
    ) -> None:
        graph_backend = current_platform.get_device_graph_backend(device)
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {device}; set "
                f"stages.vocoder.factory.window_cuda_graphs to false to decode eagerly"
            )
        else:
            pass
        self.decode = decode
        self.frame_length = frame_length
        self.window_lengths = tuple(sorted(window_lengths))
        self.static_indices: dict[int, torch.Tensor] = {}
        self.static_audio: dict[int, torch.Tensor] = {}
        self.graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        current_stream = device_module.current_stream(device)
        capture_stream = device_module.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.no_grad(), device_module.device(device):
            with device_module.stream(capture_stream):
                for window_length in reversed(self.window_lengths):
                    decode(
                        torch.zeros(
                            (1, num_codebooks, window_length),
                            dtype=torch.long,
                            device=device,
                        )
                    )
            capture_stream.synchronize()
            graph_pool = device_module.graph_pool_handle()
            for window_length in reversed(self.window_lengths):
                static_indices = torch.zeros(
                    (1, num_codebooks, window_length), dtype=torch.long, device=device
                )
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream
                ) as graph:
                    self.static_audio[window_length] = decode(static_indices)
                self.static_indices[window_length] = static_indices
                self.graphs[window_length] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(device)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        frame_count = int(indices.shape[-1])
        if indices.shape[0] != 1 or frame_count > self.window_lengths[-1]:
            return self.decode(indices)
        else:
            pass
        window_length = self.window_lengths[
            bisect.bisect_left(self.window_lengths, frame_count)
        ]
        static_indices = self.static_indices[window_length]
        static_indices[..., :frame_count].copy_(indices)
        static_indices[..., frame_count:].zero_()
        self.graphs[window_length].replay()
        return self.static_audio[window_length][
            ..., : frame_count * self.frame_length
        ].clone()


class Drama3VocoderScheduler(StreamingVocoderScheduler):
    """Decode complete windows regardless of incoming frame chunk sizes."""

    def __init__(
        self,
        codec: DAC,
        *,
        device: str,
        stream_stride: int,
        stream_followup_stride_ramp: tuple[int, ...],
        stream_followup_stride: int,
        stream_overlap_tokens: int,
        stream_crossfade_samples: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        decode_window: WindowDecoder | None = None,
        max_padded_frames_per_decode: int | None = None,
    ) -> None:
        if not all(stride > 0 for stride in stream_followup_stride_ramp):
            raise ValueError(
                f"stream_followup_stride_ramp entries must be > 0; got "
                f"{stream_followup_stride_ramp}"
            )
        else:
            pass
        super().__init__(
            codec,
            device=device,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=stream_overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            decode_window=decode_window,
        )
        self.stream_followup_stride_ramp = tuple(stream_followup_stride_ramp)
        self.stream_window_counts: dict[str, int] = {}
        self.max_padded_frames_per_decode = max_padded_frames_per_decode

    def followup_stride(self, request_id: str) -> int:
        window_count = self.stream_window_counts.get(request_id, 0)
        if window_count < len(self.stream_followup_stride_ramp):
            return self.stream_followup_stride_ramp[window_count]
        else:
            return self.stream_followup_stride

    def decode_event_metadata(
        self,
        request_id: str,
        state: StreamVocoderState,
        *,
        new_frame_count: int,
        chunk_id: int,
        is_final: bool,
    ) -> dict[str, int | float]:
        """The window a decode is about to run, once new_frame_count more frames
        join the stream; chunk_id is -1 when no stream chunk completed the window
        (frames only the final result holds, or the tail flush)."""
        total_frame_count = state.total_tokens + new_frame_count
        held_sample_count = (
            state.pending_tail.shape[-1] if state.pending_tail is not None else 0
        )
        window_start_frame = max(
            state.code_start_token,
            state.last_vocode_tokens - self.stream_overlap_tokens,
        )
        return {
            "window_index": self.stream_window_counts.get(request_id, 0),
            "chunk_id": chunk_id,
            "is_final": int(is_final),
            "emitted_frames": total_frame_count - state.last_vocode_tokens,
            "decoded_frames": total_frame_count - window_start_frame,
            "emitted_audio_s": (
                state.last_vocode_tokens * self.codec.frame_length - held_sample_count
            )
            / self.codec.sample_rate,
        }

    def stream_frames(
        self, request_id: str, frame_codes: torch.Tensor, *, chunk_id: int
    ) -> list[OutgoingMessage]:
        """Audio chunks for frames [1 + num_codebooks, frames] given in order."""
        state = self.stream_states.setdefault(request_id, StreamVocoderState())
        messages: list[OutgoingMessage] = []
        incoming_frame_count = frame_codes.shape[1]
        first_frame_index = 0
        while first_frame_index < incoming_frame_count:
            window_end_frame = state.next_vocode_tokens or self.stream_stride
            remaining_window_frame_count = window_end_frame - state.total_tokens
            if incoming_frame_count - first_frame_index < remaining_window_frame_count:
                end_frame_index = incoming_frame_count
                is_window_end = False
            else:
                end_frame_index = first_frame_index + remaining_window_frame_count
                is_window_end = True
            if is_window_end and get_recorder().is_active():
                decode_metadata = self.decode_event_metadata(
                    request_id,
                    state,
                    new_frame_count=end_frame_index - first_frame_index,
                    chunk_id=chunk_id,
                    is_final=False,
                )
                emit(
                    request_id=request_id,
                    stage=None,
                    event_name="drama3_vocoder_window_decode_started",
                    metadata=decode_metadata,
                )
            else:
                decode_metadata = None
            audio_chunk = build_stream_vocoder_chunk(
                state,
                frame_codes[:, first_frame_index:end_frame_index],
                codec=self.codec,
                device=self.device,
                stream_stride=self.stream_stride,
                stream_followup_stride=self.followup_stride(request_id),
                stream_overlap_tokens=self.stream_overlap_tokens,
                stream_crossfade_samples=self.stream_crossfade_samples,
                decode_window=self.decode_window,
            )
            if decode_metadata is not None:
                emit(
                    request_id=request_id,
                    stage=None,
                    event_name="drama3_vocoder_window_decode_finished",
                    metadata=decode_metadata,
                )
            else:
                pass
            if is_window_end:
                self.stream_window_counts[request_id] = (
                    self.stream_window_counts.get(request_id, 0) + 1
                )
            else:
                pass
            if audio_chunk is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=audio_chunk,
                        metadata={"modality": "audio"},
                    )
                )
            else:
                pass
            first_frame_index = end_frame_index
        return messages

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        frame_codes = chunk.data
        if not isinstance(frame_codes, torch.Tensor):
            raise TypeError(
                f"Drama 3 stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(frame_codes).__name__}"
            )
        else:
            return self.stream_frames(request_id, frame_codes, chunk_id=chunk.chunk_id)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        """Stream the frames no chunk carried, skipping those already streamed, then
        flush the tail and send the final result."""
        frames = generation_result_of(self.payloads[request_id]).frames
        streamed_frame_count = self.stream_states.setdefault(
            request_id, StreamVocoderState()
        ).total_tokens
        if streamed_frame_count > frames.shape[0]:
            raise RuntimeError(
                f"Request {request_id}: {streamed_frame_count} frames streamed but "
                f"the result holds {frames.shape[0]}"
            )
        else:
            pass
        messages = self.stream_frames(
            request_id, frames[streamed_frame_count:].T, chunk_id=-1
        )
        state = self.stream_states[request_id]
        # note (Haoling Pu): the tail flush decodes only frames no window took.
        if get_recorder().is_active() and state.total_tokens > state.last_vocode_tokens:
            tail_metadata = self.decode_event_metadata(
                request_id, state, new_frame_count=0, chunk_id=-1, is_final=True
            )
            emit(
                request_id=request_id,
                stage=None,
                event_name="drama3_vocoder_window_decode_started",
                metadata=tail_metadata,
            )
        else:
            tail_metadata = None
        done_messages = super().on_stream_done(request_id)
        if tail_metadata is not None:
            emit(
                request_id=request_id,
                stage=None,
                event_name="drama3_vocoder_window_decode_finished",
                metadata=tail_metadata,
            )
        else:
            pass
        return messages + done_messages

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        generation_result_of(payload)

    def clear_stream_state(self, request_id: str) -> None:
        super().clear_stream_state(request_id)
        self.stream_window_counts.pop(request_id, None)

    def final_result_data(self, payload: StagePayload) -> FinalAudioData:
        result = generation_result_of(payload)
        completion_tokens = int(result.frames.shape[0])
        final_data: FinalAudioData = {
            "modality": "audio",
            "sample_rate": int(self.codec.sample_rate),
            "usage": {
                "prompt_tokens": result.prompt_row_count,
                "completion_tokens": completion_tokens,
                "total_tokens": result.prompt_row_count + completion_tokens,
            },
        }
        if result.finish_reason is not None:
            final_data["finish_reason"] = result.finish_reason
        else:
            pass
        return final_data

    def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        codebook_codes = [
            generation_result_of(payload).frames[:, 1:].T for payload in payloads
        ]
        audio_by_index: dict[int, torch.Tensor] = {}
        for group in non_streaming_decode_groups(
            [int(codes.shape[-1]) for codes in codebook_codes],
            self.max_padded_frames_per_decode,
        ):
            audio_by_index.update(
                zip(
                    group,
                    self.decode_codebook_codes([codebook_codes[i] for i in group]),
                    strict=True,
                )
            )
        audios = [audio_by_index[index] for index in range(len(payloads))]
        return [
            StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={**self.final_result_data(payload), "audio_data": audio.tolist()},
            )
            for payload, audio in zip(payloads, audios, strict=True)
        ]


@dataclass(kw_only=True)
class QueuedIncrementalWindow:
    codes: torch.Tensor
    chunk_id: int
    is_final: int


@dataclass(kw_only=True)
class IncrementalStream:
    arrival: int
    held: list[torch.Tensor] = field(default_factory=list)
    held_frame_count: int = 0
    received_frame_count: int = 0
    window_count: int = 0
    windows: collections.deque[QueuedIncrementalWindow] = field(
        default_factory=collections.deque
    )
    decoded_frame_count: int = 0
    decoded_window_count: int = 0
    slot: int | None = None
    is_done: bool = False


class Drama3IncrementalVocoderScheduler(Drama3VocoderScheduler):
    DECODE_STEP_MESSAGE_BUDGET = 8

    def __init__(
        self,
        codec: DAC,
        *,
        window_decoder: IncrementalWindowDecoder,
        max_stream_batch_size: int,
        stream_slot_count: int,
        max_stream_frames: int,
        decode_step_message_budget: int = 8,
        device: str,
        stream_stride: int,
        stream_followup_stride_ramp: tuple[int, ...],
        stream_followup_stride: int,
        stream_overlap_tokens: int,
        stream_crossfade_samples: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        decode_window: WindowDecoder | None = None,
        max_padded_frames_per_decode: int | None = None,
    ) -> None:
        super().__init__(
            codec,
            device=device,
            stream_stride=stream_stride,
            stream_followup_stride_ramp=stream_followup_stride_ramp,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=stream_overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            decode_window=decode_window,
            max_padded_frames_per_decode=max_padded_frames_per_decode,
        )
        if self.stream_crossfade_samples != 0:
            raise ValueError(
                "incremental stream decoding has no window overlap to crossfade; "
                "set stream_crossfade_samples to 0"
            )
        else:
            pass
        self.window_decoder = window_decoder
        self.max_stream_batch_size = max_stream_batch_size
        self.max_stream_frames = max_stream_frames
        self.free_slots = list(range(stream_slot_count - 1, -1, -1))
        self.streams: dict[str, IncrementalStream] = {}
        self.stream_arrivals = 0
        self.messages_since_decode_step = 0
        self.DECODE_STEP_MESSAGE_BUDGET = decode_step_message_budget

    def handle_message(
        self, msg: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        super().handle_message(msg, loop)
        self.messages_since_decode_step += 1
        if (
            self.messages_since_decode_step >= self.DECODE_STEP_MESSAGE_BUDGET
            and self.has_ready_work()
        ):
            self.run_ready_step()
        else:
            pass

    def stream_of(self, request_id: str) -> IncrementalStream:
        stream = self.streams.get(request_id)
        if stream is None:
            self.stream_arrivals += 1
            stream = IncrementalStream(arrival=self.stream_arrivals)
            self.streams[request_id] = stream
        else:
            pass
        return stream

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.stream_of(request_id)

    def stride_of_window(self, window_index: int) -> int:
        if window_index == 0:
            return self.stream_stride
        elif window_index <= len(self.stream_followup_stride_ramp):
            return self.stream_followup_stride_ramp[window_index - 1]
        else:
            return self.stream_followup_stride

    def stream_frames(
        self,
        request_id: str,
        frame_codes: torch.Tensor,
        *,
        chunk_id: int = -1,
    ) -> list[OutgoingMessage]:
        stream = self.stream_of(request_id)
        frame_count = int(frame_codes.shape[1])
        if frame_count == 0:
            return []
        else:
            pass
        stream.received_frame_count += frame_count
        if stream.received_frame_count > self.max_stream_frames:
            raise ValueError(
                f"Request {request_id}: stream exceeds the codec's "
                f"{self.max_stream_frames} frames"
            )
        else:
            pass
        stream.held.append(frame_codes[1:])
        stream.held_frame_count += frame_count
        stride = self.stride_of_window(stream.window_count)
        if stream.held_frame_count < stride:
            return []
        else:
            pass
        held = torch.cat(stream.held, dim=1)
        first_frame = 0
        while stream.held_frame_count - first_frame >= stride:
            stream.windows.append(
                QueuedIncrementalWindow(
                    codes=held[:, first_frame : first_frame + stride],
                    chunk_id=chunk_id,
                    is_final=0,
                )
            )
            first_frame += stride
            stream.window_count += 1
            stride = self.stride_of_window(stream.window_count)
        stream.held = [held[:, first_frame:]] if first_frame < held.shape[1] else []
        stream.held_frame_count -= first_frame
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        frames = generation_result_of(self.payloads[request_id]).frames
        stream = self.stream_of(request_id)
        if stream.received_frame_count > frames.shape[0]:
            raise RuntimeError(
                f"Request {request_id}: {stream.received_frame_count} frames streamed "
                f"but the result holds {frames.shape[0]}"
            )
        else:
            pass
        self.stream_frames(request_id, frames[stream.received_frame_count :].T)
        if stream.held_frame_count > 0:
            stream.windows.append(
                QueuedIncrementalWindow(
                    codes=torch.cat(stream.held, dim=1),
                    chunk_id=-1,
                    is_final=1,
                )
            )
            stream.held = []
            stream.held_frame_count = 0
        else:
            pass
        stream.is_done = True
        if stream.windows:
            return None
        else:
            return [self.stream_result_message(request_id)]

    def stream_result_message(self, request_id: str) -> OutgoingMessage:
        payload = self.payloads[request_id]
        return OutgoingMessage(
            request_id=request_id,
            type="result",
            data=StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=self.final_result_data(payload),
            ),
        )

    def clear_stream_state(self, request_id: str) -> None:
        super().clear_stream_state(request_id)
        stream = self.streams.pop(request_id, None)
        if stream is not None and stream.slot is not None:
            self.free_slots.append(stream.slot)
        else:
            pass

    def ready_streams(self) -> list[tuple[str, IncrementalStream]]:
        return [
            (request_id, stream)
            for request_id, stream in self.streams.items()
            if stream.windows and (stream.slot is not None or self.free_slots)
        ]

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return bool(self.ready_streams())

    def next_group(self) -> list[tuple[str, IncrementalStream]]:
        ready = sorted(
            self.ready_streams(),
            key=lambda item: (item[1].decoded_frame_count, item[1].arrival),
        )
        if not ready:
            return []
        else:
            pass
        length_of = self.window_decoder.window_length_for
        group_length = length_of(int(ready[0][1].windows[0].codes.shape[1]))
        slotless_capacity = len(self.free_slots)
        group: list[tuple[str, IncrementalStream]] = []
        for request_id, stream in ready:
            if length_of(int(stream.windows[0].codes.shape[1])) != group_length:
                continue
            elif stream.slot is None and slotless_capacity == 0:
                continue
            elif stream.slot is None:
                slotless_capacity -= 1
            else:
                pass
            group.append((request_id, stream))
            if len(group) == self.max_stream_batch_size:
                break
            else:
                pass
        return group

    def run_ready_step(self) -> None:
        self.messages_since_decode_step = 0
        failed: list[str] = []
        with self.state_lock:
            group = self.next_group()
            if group:
                try:
                    self.decode_group(group)
                except Exception as exc:
                    logger.exception(
                        "Drama 3 decode of %d stream windows failed", len(group)
                    )
                    for request_id, _ in group:
                        self.emit_error(request_id, exc)
                        self.abort_state(request_id)
                        failed.append(request_id)
            else:
                pass
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def incremental_decode_event_metadata(
        self, stream: IncrementalStream, window: QueuedIncrementalWindow
    ) -> dict[str, int | float]:
        window_frames = int(window.codes.shape[1])
        emitted_audio_s = (
            stream.decoded_frame_count
            * self.codec.frame_length
            / self.codec.sample_rate
        )
        return {
            "window_index": stream.decoded_window_count,
            "chunk_id": window.chunk_id,
            "is_final": window.is_final,
            "emitted_frames": window_frames,
            "decoded_frames": window_frames,
            "emitted_audio_s": emitted_audio_s,
        }

    def decode_group(self, group: list[tuple[str, IncrementalStream]]) -> None:
        new_slots: list[int] = []
        for _, stream in group:
            if stream.slot is None:
                stream.slot = self.free_slots.pop()
                new_slots.append(stream.slot)
            else:
                pass
        queued = [stream.windows[0] for _, stream in group]
        windows = [window.codes for window in queued]
        decode_metadata: list[dict[str, int | float]] | None = None
        if get_recorder().is_active():
            decode_metadata = [
                self.incremental_decode_event_metadata(stream, window)
                for (_, stream), window in zip(group, queued, strict=True)
            ]
            for (request_id, _), metadata in zip(group, decode_metadata, strict=True):
                emit(
                    request_id=request_id,
                    stage=None,
                    event_name="drama3_vocoder_window_decode_started",
                    metadata=metadata,
                )
        else:
            pass
        audio = self.window_decoder.decode(
            windows,
            [stream.slot for _, stream in group],
            [stream.decoded_frame_count for _, stream in group],
            new_slots,
        )
        frame_length = int(self.codec.frame_length)
        sample_rate = int(self.codec.sample_rate)
        finished: list[str] = []
        for row, (request_id, stream) in enumerate(group):
            window = stream.windows.popleft()
            window_frames = int(window.codes.shape[1])
            stream.decoded_frame_count += window_frames
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=build_audio_chunk_payload(
                        audio[row, : window_frames * frame_length],
                        sample_rate=sample_rate,
                    ),
                    metadata={"modality": "audio"},
                )
            )
            if decode_metadata is not None:
                emit(
                    request_id=request_id,
                    stage=None,
                    event_name="drama3_vocoder_window_decode_finished",
                    metadata=decode_metadata[row],
                )
            else:
                pass
            stream.decoded_window_count += 1
            if stream.is_done and not stream.windows:
                finished.append(request_id)
            else:
                pass
        for request_id in finished:
            self.complete_stream_request(
                request_id, [self.stream_result_message(request_id)]
            )
