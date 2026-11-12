# SPDX-License-Identifier: Apache-2.0
"""Drama 3 vocoder window decoders that replay captured CUDA graphs."""

from __future__ import annotations

import bisect
import logging
import math
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Protocol

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.scheduling.audio_stream import WindowDecoder

logger = logging.getLogger(__name__)

BATCHED_WINDOW_FRAME_LIMIT_IN_LONGEST_WINDOWS = 2
WINDOW_BATCH_MIN_SAVING = 0.05
CALIBRATION_REPLAYS = 8


class WindowBatchDecoder(Protocol):
    def decode_windows(
        self, codebook_codes: list[torch.Tensor], discard_frames: int = 0
    ) -> list[torch.Tensor]:
        """[1, 1, samples] audio per [num_codebooks, frames] window of one length;
        the first discard_frames frames' audio may be left unset."""


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


def window_graph_keys(
    window_lengths: tuple[int, ...],
    utterance_lengths: tuple[int, ...],
    max_batch_size: int,
    window_discards: Mapping[int, int],
    flush_keys: Iterable[tuple[int, int]],
) -> list[GraphKey]:
    """(batch size, frames, discarded frames) of each graph to capture, largest
    first: batch 1 whole for every length and at each stream window's or flush
    shape's discard, and batched at the discard of stream windows that fit the limit."""
    batched_frame_limit = BATCHED_WINDOW_FRAME_LIMIT_IN_LONGEST_WINDOWS * max(
        window_lengths
    )
    graph_keys = {
        *((1, length, 0) for length in (*window_lengths, *utterance_lengths)),
        *(
            (1, length, window_discards[length])
            for length in window_lengths
            if window_discards.get(length)
        ),
        *(
            (1, length, discard_frames)
            for length, discard_frames in flush_keys
            if discard_frames
        ),
        *(
            (batch_size, window_length, window_discards.get(window_length, 0))
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
