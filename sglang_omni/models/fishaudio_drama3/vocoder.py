# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder: Fish codec decoding of sampled frames, streamed in fixed windows."""

from __future__ import annotations

import asyncio
import collections
import logging
from dataclasses import dataclass, field

import torch

from sglang_omni.models.fishaudio_drama3.incremental_decoder import (
    IncrementalWindowDecoder,
)
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationResult
from sglang_omni.models.fishaudio_drama3.stream_windows import (
    crossfade_frames,
    discarded_frames,
)
from sglang_omni.models.fishaudio_drama3.window_graphs import WindowBatchDecoder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.pipeline.stage.stream_queue import StreamItem
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
from sglang_omni.scheduling.streaming_vocoder import vocoder_decode_stream_priority
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

STREAM_CHUNK_DRAIN_LIMIT = 64


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


@dataclass(frozen=True)
class QueuedWindow:
    """A stream window whose last frames arrived, waiting for its decode."""

    request_id: str
    frame_codes: torch.Tensor
    codebook_codes: torch.Tensor
    discard_frames: int
    # note (Wenyao Gao): the window's decode event metadata while a profile records.
    decode_metadata: dict[str, int | float] | None


def decoded_window(audio: torch.Tensor, frame_count: int) -> WindowDecoder:
    """A window decoder that returns audio already decoded for the window."""

    def decode(indices: torch.Tensor) -> torch.Tensor:
        assert int(indices.shape[-1]) == frame_count
        return audio

    return decode


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
        window_batch_decoder: WindowBatchDecoder | None = None,
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
        self.window_batch_decoder = window_batch_decoder
        self.crossfade_frames = crossfade_frames(
            stream_crossfade_samples, int(codec.frame_length)
        )
        self.can_batch_stream_chunks = window_batch_decoder is not None
        self.stream_chunk_batch_max = STREAM_CHUNK_DRAIN_LIMIT
        # note (Wenyao Gao): in the engine's process, decode ahead of the engine's
        # default-priority stream instead of queueing behind each decode step.
        torch_device = torch.device(device)
        if torch_device.type == "cuda":
            self.decode_stream: torch.cuda.Stream | None = torch.cuda.Stream(
                device=torch_device,
                priority=vocoder_decode_stream_priority(torch.cuda),
            )
        else:
            self.decode_stream = None

    def start(self) -> None:
        if self.decode_stream is None:
            super().start()
        else:
            # note (Wenyao Gao): construction wrote the codec weights and the window
            # graphs' static buffers on this thread's default stream; a new stream
            # does not wait for it, so serving waits once for that work.
            self.decode_stream.wait_stream(
                torch.cuda.current_stream(self.decode_stream.device)
            )
            with torch.cuda.stream(self.decode_stream):
                super().start()

    def flush_decode_window(
        self, stream_state: StreamVocoderState
    ) -> WindowDecoder | None:
        window_batch_decoder = self.window_batch_decoder
        if window_batch_decoder is None:
            return self.decode_window
        else:
            _, discard_frames = self.window_start_and_discard(stream_state)

            def decode(indices: torch.Tensor) -> torch.Tensor:
                return window_batch_decoder.decode_windows(
                    [indices[0]], discard_frames=discard_frames
                )[0]

            return decode

    def window_start_and_discard(
        self, stream_state: StreamVocoderState
    ) -> tuple[int, int]:
        """The token decode_stream_window starts the request's next window at, and the
        window's leading frames whose audio the stream never reads."""
        window_start_token = max(
            stream_state.code_start_token,
            stream_state.last_vocode_tokens - self.stream_overlap_tokens,
        )
        return window_start_token, discarded_frames(
            stream_state.last_vocode_tokens - window_start_token, self.crossfade_frames
        )

    def followup_stride(self, request_id: str) -> int:
        window_count = self.stream_window_counts.get(request_id, 0)
        if window_count < len(self.stream_followup_stride_ramp):
            return self.stream_followup_stride_ramp[window_count]
        else:
            return self.stream_followup_stride

    def stream_frame_slice(
        self,
        request_id: str,
        frame_codes: torch.Tensor,
        *,
        is_window_end: bool,
        decode_window: WindowDecoder | None,
    ) -> OutgoingMessage | None:
        """The audio chunk, if any, for frames that stay within the current window."""
        audio_chunk = build_stream_vocoder_chunk(
            self.stream_states.setdefault(request_id, StreamVocoderState()),
            frame_codes,
            codec=self.codec,
            device=self.device,
            stream_stride=self.stream_stride,
            stream_followup_stride=self.followup_stride(request_id),
            stream_overlap_tokens=self.stream_overlap_tokens,
            stream_crossfade_samples=self.stream_crossfade_samples,
            decode_window=decode_window,
        )
        if is_window_end:
            self.stream_window_counts[request_id] = (
                self.stream_window_counts.get(request_id, 0) + 1
            )
        else:
            pass
        if audio_chunk is None:
            return None
        else:
            return OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=audio_chunk,
                metadata={"modality": "audio"},
            )

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
            message = self.stream_frame_slice(
                request_id,
                frame_codes[:, first_frame_index:end_frame_index],
                is_window_end=is_window_end,
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
            if message is not None:
                messages.append(message)
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

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Stream queued chunks in rounds: each round decodes every request's next
        complete window, windows of one shape together and shortest first, so a first
        window never waits behind longer ones; nothing waits for more chunks."""
        failed: list[str] = []
        with self.state_lock:
            frame_chunks: dict[str, list[StreamItem]] = {}
            for request_id, stream_item in items:
                if self.is_aborted(request_id):
                    continue
                elif not isinstance(stream_item.data, torch.Tensor):
                    self.fail_stream(
                        request_id,
                        TypeError(
                            f"Drama 3 stream chunk for {request_id!r} must carry a "
                            f"torch.Tensor, got {type(stream_item.data).__name__}"
                        ),
                        failed,
                    )
                else:
                    frame_chunks.setdefault(request_id, []).append(stream_item)
            unread_frames: dict[str, torch.Tensor] = {}
            # note (Wenyao Gao): the chunk that carried each unread frame; a window's
            # decode events name the chunk that carried its last frame.
            unread_chunk_ids: dict[str, list[int]] = {}
            for request_id, chunks in frame_chunks.items():
                try:
                    unread_frames[request_id] = torch.cat(
                        [chunk.data for chunk in chunks], dim=1
                    )
                    unread_chunk_ids[request_id] = [
                        chunk.chunk_id
                        for chunk in chunks
                        for _ in range(chunk.data.shape[1])
                    ]
                except Exception as exc:
                    self.fail_stream(request_id, exc, failed)
            while unread_frames:
                windows_by_length: dict[tuple[int, int], list[QueuedWindow]] = {}
                next_unread_frames: dict[str, torch.Tensor] = {}
                for request_id, frame_codes in unread_frames.items():
                    if self.is_aborted(request_id):
                        continue
                    else:
                        pass
                    try:
                        window = self.queue_next_window(
                            request_id, frame_codes, unread_chunk_ids[request_id]
                        )
                    except Exception as exc:
                        self.fail_stream(request_id, exc, failed)
                        continue
                    if window is None:
                        continue
                    else:
                        pass
                    windows_by_length.setdefault(
                        (int(window.codebook_codes.shape[-1]), window.discard_frames),
                        [],
                    ).append(window)
                    read_frame_count = int(window.frame_codes.shape[1])
                    if frame_codes.shape[1] > read_frame_count:
                        next_unread_frames[request_id] = frame_codes[
                            :, read_frame_count:
                        ]
                        unread_chunk_ids[request_id] = unread_chunk_ids[request_id][
                            read_frame_count:
                        ]
                    else:
                        pass
                for window_shape in sorted(windows_by_length):
                    self.decode_queued_windows(windows_by_length[window_shape], failed)
                unread_frames = next_unread_frames
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def queue_next_window(
        self, request_id: str, frame_codes: torch.Tensor, chunk_ids: list[int]
    ) -> QueuedWindow | None:
        """Ingest frames up to the end of the request's current window; the window
        to decode once they complete it, else None. chunk_ids holds the stream chunk
        that carried each frame."""
        stream_state = self.stream_states.setdefault(request_id, StreamVocoderState())
        window_frame_count = (
            stream_state.next_vocode_tokens or self.stream_stride
        ) - stream_state.total_tokens
        if frame_codes.shape[1] == 0:
            return None
        elif frame_codes.shape[1] < window_frame_count:
            self.stream_frame_slice(
                request_id,
                frame_codes,
                is_window_end=False,
                decode_window=self.decode_window,
            )
            return None
        else:
            if get_recorder().is_active():
                decode_metadata = self.decode_event_metadata(
                    request_id,
                    stream_state,
                    new_frame_count=window_frame_count,
                    chunk_id=chunk_ids[window_frame_count - 1],
                    is_final=False,
                )
            else:
                decode_metadata = None
            window_frames = frame_codes[:, :window_frame_count].to(
                device=self.device, dtype=torch.long, non_blocking=True
            )
            window_start_token, discard_frames = self.window_start_and_discard(
                stream_state
            )
            return QueuedWindow(
                request_id=request_id,
                frame_codes=window_frames,
                codebook_codes=torch.cat([*stream_state.codes, window_frames], dim=1)[
                    1:, window_start_token - stream_state.code_start_token :
                ],
                discard_frames=discard_frames,
                decode_metadata=decode_metadata,
            )

    def decode_queued_windows(
        self, windows: list[QueuedWindow], failed: list[str]
    ) -> None:
        """Decode windows of one length and discard from distinct requests and emit
        their chunks."""
        assert self.window_batch_decoder is not None
        # note (Wenyao Gao): the windows share one decode, so each window's span from
        # started to finished holds that whole decode.
        for window in windows:
            if window.decode_metadata is not None:
                emit(
                    request_id=window.request_id,
                    stage=None,
                    event_name="drama3_vocoder_window_decode_started",
                    metadata=window.decode_metadata,
                )
            else:
                pass
        try:
            with torch.no_grad():
                audios = self.window_batch_decoder.decode_windows(
                    [window.codebook_codes for window in windows],
                    discard_frames=windows[0].discard_frames,
                )
        except Exception as exc:
            for window in windows:
                if self.is_aborted(window.request_id):
                    pass
                else:
                    self.fail_stream(window.request_id, exc, failed)
            return
        for window, audio in zip(windows, audios, strict=True):
            if self.is_aborted(window.request_id):
                continue
            else:
                pass
            try:
                message = self.stream_frame_slice(
                    window.request_id,
                    window.frame_codes,
                    is_window_end=True,
                    decode_window=decoded_window(
                        audio, int(window.codebook_codes.shape[-1])
                    ),
                )
            except Exception as exc:
                self.fail_stream(window.request_id, exc, failed)
                continue
            if window.decode_metadata is not None:
                emit(
                    request_id=window.request_id,
                    stage=None,
                    event_name="drama3_vocoder_window_decode_finished",
                    metadata=window.decode_metadata,
                )
            else:
                pass
            if message is not None and not self.is_aborted(window.request_id):
                self.outbox.put(message)
            else:
                pass

    def fail_stream(
        self, request_id: str, exc: BaseException, failed: list[str]
    ) -> None:
        """Report the error and drop the request's state; abort cleanup runs once
        the state lock is released."""
        logger.error(f"Drama 3 vocoder failed for {request_id}", exc_info=exc)
        self.emit_error(request_id, exc)
        self.abort_state(request_id)
        failed.append(request_id)

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
                data={
                    **self.final_result_data(payload),
                    **audio_waveform_payload(audio, source_hint="Drama 3 codec"),
                },
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
