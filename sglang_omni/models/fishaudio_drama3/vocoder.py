# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder: Fish codec decoding of sampled frames, streamed in fixed windows."""

from __future__ import annotations

import bisect
import functools
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import torch

from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationResult
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.audio_stream import (
    FinalAudioData,
    StreamVocoderState,
    WindowDecoder,
    build_stream_vocoder_chunk,
)
from sglang_omni.scheduling.audio_vocoder import StreamingVocoderScheduler
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import vocoder_decode_stream_priority
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

BATCHED_WINDOW_FRAME_LIMIT_IN_LONGEST_WINDOWS = 2
WINDOW_BATCH_MIN_SAVING = 0.05
CALIBRATION_REPLAYS = 8
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


class WindowBatchDecoder(Protocol):
    def decode_windows(
        self, codebook_codes: list[torch.Tensor], discard_frames: int = 0
    ) -> list[torch.Tensor]:
        """[1, 1, samples] audio per [num_codebooks, frames] window of one length;
        the first discard_frames frames' audio may be left unset."""


class SuffixDecoder(Protocol):
    def __call__(
        self, indices: torch.Tensor, *, discard_frames: int = 0
    ) -> torch.Tensor:
        """[1, 1, samples] audio of a [1, num_codebooks, frames] window; the first
        discard_frames frames' audio may be zero."""


# note (Wenyao Gao): (batch size, frames, discarded frames) of a captured graph.
GraphKey = tuple[int, int, int]


def fastest_replay_milliseconds(
    graph: ReplayableGraph, device_module: ModuleType
) -> float:
    start_event = device_module.Event(enable_timing=True)
    end_event = device_module.Event(enable_timing=True)
    fastest_milliseconds = math.inf
    for _ in range(CALIBRATION_REPLAYS):
        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        fastest_milliseconds = min(
            fastest_milliseconds, start_event.elapsed_time(end_event)
        )
    return fastest_milliseconds


def crossfade_frames(stream_crossfade_samples: int, frame_length: int) -> int:
    """Frames of the overlap a crossfade reads back into."""
    return -(-stream_crossfade_samples // frame_length)


def discarded_frames(overlap_frames: int, crossfade_frames: int) -> int:
    """Leading frames of a window whose audio the stream never reads."""
    return max(0, overlap_frames - crossfade_frames)


def window_graph_keys(
    window_lengths: tuple[int, ...],
    utterance_lengths: tuple[int, ...],
    max_batch_size: int,
    window_discards: Mapping[int, int] | None = None,
    flush_keys: Iterable[tuple[int, int]] = (),
) -> list[GraphKey]:
    """(batch size, frames, discarded frames) of each graph to capture, largest
    first: batch 1 whole for every length and at each stream window's or flush
    shape's discard, and batched at the discard of stream windows that fit the limit."""
    batched_frame_limit = BATCHED_WINDOW_FRAME_LIMIT_IN_LONGEST_WINDOWS * max(
        window_lengths
    )
    discards = window_discards or {}
    graph_keys = {
        *((1, length, 0) for length in (*window_lengths, *utterance_lengths)),
        *(
            (1, length, discards[length])
            for length in window_lengths
            if discards.get(length)
        ),
        *(
            (1, length, discard_frames)
            for length, discard_frames in flush_keys
            if discard_frames
        ),
        *(
            (batch_size, window_length, discards.get(window_length, 0))
            for window_length in window_lengths
            for batch_size in range(2, max_batch_size + 1)
            if batch_size * window_length <= batched_frame_limit
        ),
    }
    # note (Wenyao Gao): largest first, so the first capture sizes the shared pool.
    return sorted(
        graph_keys,
        key=lambda graph_key: (graph_key[0] * graph_key[1], graph_key[1], graph_key[2]),
        reverse=True,
    )


class WindowGraphDecoder:
    def __init__(
        self,
        decode: WindowDecoder,
        *,
        num_codebooks: int,
        frame_length: int,
        device: torch.device,
        window_lengths: tuple[int, ...],
        utterance_lengths: tuple[int, ...] = (),
        max_batch_size: int = 1,
        window_discards: Mapping[int, int] | None = None,
        flush_shapes: Iterable[tuple[int, int]] = (),
    ) -> None:
        """flush_shapes: (frames, discarded frames) of the windows a stream flushes
        at its end; each gets a graph at the length its frames pad to."""
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
        self.window_lengths = tuple(sorted(set(window_lengths)))
        self.single_lengths = tuple(sorted({*window_lengths, *utterance_lengths}))
        self.window_discards = dict(window_discards or {})
        graph_keys = window_graph_keys(
            self.window_lengths,
            utterance_lengths,
            max_batch_size,
            self.window_discards,
            {
                (self.padded_length(frame_count), discard_frames)
                for frame_count, discard_frames in flush_shapes
                if frame_count <= self.single_lengths[-1]
            },
        )
        self.static_indices: dict[GraphKey, torch.Tensor] = {}
        self.static_audio: dict[GraphKey, torch.Tensor] = {}
        self.graphs: dict[GraphKey, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        current_stream = device_module.current_stream(device)
        capture_stream = device_module.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.no_grad(), device_module.device(device):
            with device_module.stream(capture_stream):
                for batch_size, window_length, discard_frames in graph_keys:
                    audio_dtype = self.decode_from(
                        torch.zeros(
                            (batch_size, num_codebooks, window_length),
                            dtype=torch.long,
                            device=device,
                        ),
                        discard_frames,
                    ).dtype
            capture_stream.synchronize()
            graph_pool = device_module.graph_pool_handle()
            for graph_key in graph_keys:
                batch_size, window_length, discard_frames = graph_key
                static_indices = torch.zeros(
                    (batch_size, num_codebooks, window_length),
                    dtype=torch.long,
                    device=device,
                )
                if discard_frames == 0:
                    with graph_backend.capture(
                        pool=graph_pool, stream=capture_stream
                    ) as graph:
                        self.static_audio[graph_key] = decode(static_indices)
                else:
                    # note (Wenyao Gao): the discarded frames' audio stays zero and
                    # the replay fills the rest, so the result reads like a
                    # whole-window decode.
                    static_audio = torch.zeros(
                        (batch_size, 1, window_length * frame_length),
                        dtype=audio_dtype,
                        device=device,
                    )
                    with graph_backend.capture(
                        pool=graph_pool, stream=capture_stream
                    ) as graph:
                        static_audio[..., discard_frames * frame_length :].copy_(
                            self.decode_from(static_indices, discard_frames)
                        )
                    self.static_audio[graph_key] = static_audio
                self.static_indices[graph_key] = static_indices
                self.graphs[graph_key] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(device)
        self.batch_sizes = (
            self.calibrated_batch_sizes(device_module) if max_batch_size > 1 else {}
        )

    def padded_length(self, frame_count: int) -> int:
        """The shortest graph length holding frame_count frames."""
        return self.single_lengths[bisect.bisect_left(self.single_lengths, frame_count)]

    def decode_from(self, indices: torch.Tensor, discard_frames: int) -> torch.Tensor:
        """The window's audio from discard_frames on; a decoder without
        keep_from_frame only gets discard 0."""
        if discard_frames == 0:
            return self.decode(indices)
        else:
            return self.decode(indices, keep_from_frame=discard_frames)

    def graph_key(
        self, batch_size: int, frame_count: int, discard_frames: int
    ) -> GraphKey | None:
        """The captured graph for the shape that skips the discarded frames' conv
        stack, else the one decoding every frame, else None."""
        for candidate_discard_frames in (discard_frames, 0):
            candidate_graph_key = (batch_size, frame_count, candidate_discard_frames)
            if candidate_graph_key in self.graphs:
                return candidate_graph_key
            else:
                pass
        return None

    def calibrated_batch_sizes(
        self, device_module: ModuleType
    ) -> dict[int, tuple[int, ...]]:
        """Per window length, the batch sizes, largest first, whose replay measured
        faster than decoding its windows one at a time."""
        batch_sizes: dict[int, tuple[int, ...]] = {}
        ratios: list[str] = []
        for window_length in self.window_lengths:
            discard_frames = self.window_discards.get(window_length, 0)
            single_key = self.graph_key(1, window_length, discard_frames)
            assert single_key is not None
            single_milliseconds = fastest_replay_milliseconds(
                self.graphs[single_key], device_module
            )
            enabled_batch_sizes: list[int] = []
            for (
                batch_size,
                length,
                graph_discard_frames,
            ), graph in self.graphs.items():
                if (
                    length != window_length
                    or batch_size == 1
                    or graph_discard_frames != discard_frames
                ):
                    continue
                else:
                    pass
                ratio = fastest_replay_milliseconds(graph, device_module) / (
                    batch_size * single_milliseconds
                )
                if ratio <= 1 - WINDOW_BATCH_MIN_SAVING:
                    enabled_batch_sizes.append(batch_size)
                else:
                    pass
                ratios.append(f"{batch_size}x{window_length}={ratio:.2f}")
            batch_sizes[window_length] = tuple(
                sorted(enabled_batch_sizes, reverse=True)
            )
        batched_lengths = {
            length: sizes for length, sizes in batch_sizes.items() if sizes
        }
        logger.info(
            f"Drama 3 vocoder: batched window replay over one-at-a-time GPU time "
            f"{' '.join(ratios)}; batching {batched_lengths or 'no window length'}"
        )
        return batch_sizes

    def __call__(
        self, indices: torch.Tensor, *, discard_frames: int = 0
    ) -> torch.Tensor:
        """The window's audio; the first discard_frames frames' audio may be zero."""
        frame_count = int(indices.shape[-1])
        if indices.shape[0] != 1 or frame_count > self.single_lengths[-1]:
            return self.decode(indices)
        else:
            pass
        graph_key = self.graph_key(1, self.padded_length(frame_count), discard_frames)
        assert graph_key is not None
        static_indices = self.static_indices[graph_key]
        static_indices[..., :frame_count].copy_(indices)
        static_indices[..., frame_count:].zero_()
        self.graphs[graph_key].replay()
        return self.static_audio[graph_key][
            ..., : frame_count * self.frame_length
        ].clone()

    def decode_windows(
        self, codebook_codes: list[torch.Tensor], discard_frames: int = 0
    ) -> list[torch.Tensor]:
        """[1, 1, samples] audio per [num_codebooks, frames] window of one length;
        windows share a replay at the batch sizes calibration enabled."""
        frame_count = int(codebook_codes[0].shape[-1])
        batch_sizes = self.batch_sizes.get(frame_count, ())
        audios: list[torch.Tensor] = []
        while len(audios) < len(codebook_codes):
            remaining_codes = codebook_codes[len(audios) :]
            graph_key = next(
                (
                    graph_key
                    for candidate_batch_size in batch_sizes
                    if candidate_batch_size <= len(remaining_codes)
                    for graph_key in [
                        self.graph_key(
                            candidate_batch_size, frame_count, discard_frames
                        )
                    ]
                    if graph_key is not None
                ),
                None,
            )
            if graph_key is None:
                audios.append(
                    self(remaining_codes[0][None], discard_frames=discard_frames)
                )
            else:
                batch_size = graph_key[0]
                torch.stack(
                    remaining_codes[:batch_size], out=self.static_indices[graph_key]
                )
                self.graphs[graph_key].replay()
                audios.extend(
                    self.static_audio[graph_key][..., : frame_count * self.frame_length]
                    .clone()
                    .split(1)
                )
        return audios


@dataclass(frozen=True)
class QueuedWindow:
    """A stream window whose last frames arrived, waiting for its decode."""

    request_id: str
    frame_codes: torch.Tensor
    codebook_codes: torch.Tensor
    discard_frames: int


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
        suffix_decoder: SuffixDecoder | None = None,
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
        self.suffix_decoder = suffix_decoder
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
        if self.suffix_decoder is None:
            return self.decode_window
        else:
            _, discard_frames = self.window_start_and_discard(stream_state)
            return functools.partial(self.suffix_decoder, discard_frames=discard_frames)

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

    def stream_frames(
        self, request_id: str, frame_codes: torch.Tensor
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
            message = self.stream_frame_slice(
                request_id,
                frame_codes[:, first_frame_index:end_frame_index],
                is_window_end=is_window_end,
                decode_window=self.decode_window,
            )
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
            return self.stream_frames(request_id, frame_codes)

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Stream queued chunks in rounds: each round decodes every request's next
        complete window, windows of one shape together and shortest first, so a first
        window never waits behind longer ones; nothing waits for more chunks."""
        failed: list[str] = []
        with self.state_lock:
            frame_chunks: dict[str, list[torch.Tensor]] = {}
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
                    frame_chunks.setdefault(request_id, []).append(stream_item.data)
            unread_frames: dict[str, torch.Tensor] = {}
            for request_id, chunks in frame_chunks.items():
                try:
                    unread_frames[request_id] = torch.cat(chunks, dim=1)
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
                        window = self.queue_next_window(request_id, frame_codes)
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
                    else:
                        pass
                for window_shape in sorted(windows_by_length):
                    self.decode_queued_windows(windows_by_length[window_shape], failed)
                unread_frames = next_unread_frames
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def queue_next_window(
        self, request_id: str, frame_codes: torch.Tensor
    ) -> QueuedWindow | None:
        """Ingest frames up to the end of the request's current window; the window
        to decode once they complete it, else None."""
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
            )

    def decode_queued_windows(
        self, windows: list[QueuedWindow], failed: list[str]
    ) -> None:
        """Decode windows of one length and discard from distinct requests and emit
        their chunks."""
        assert self.window_batch_decoder is not None
        try:
            with torch.no_grad():
                audios = self.window_batch_decoder.decode_windows(
                    [window.codebook_codes for window in windows],
                    discard_frames=windows[0].discard_frames,
                )
        except Exception as exc:
            for window in windows:
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
        return self.stream_frames(
            request_id, frames[streamed_frame_count:].T
        ) + super().on_stream_done(request_id)

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
