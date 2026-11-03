# SPDX-License-Identifier: Apache-2.0
"""Streaming window schedules and per-window-length CUDA graph replay."""

from __future__ import annotations

import bisect

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.scheduling.audio_stream import WindowDecoder


def stream_window_lengths(
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
) -> tuple[int, ...]:
    strides = (stream_stride, *stream_followup_stride_ramp)
    lengths: set[int] = set()
    emitted_frames = 0
    window_index = 0
    while True:
        stride = (
            strides[window_index]
            if window_index < len(strides)
            else stream_followup_stride
        )
        window_end = emitted_frames + stride
        lengths.add(window_end - max(0, emitted_frames - stream_overlap_tokens))
        if window_index >= len(strides) and emitted_frames >= stream_overlap_tokens:
            return tuple(sorted(lengths))
        else:
            pass
        emitted_frames = window_end
        window_index += 1


def frames_in_complete_windows(
    frame_count: int,
    *,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
) -> int:
    window_end = 0
    for stride in (stream_stride, *stream_followup_stride_ramp):
        if window_end + stride > frame_count:
            return window_end
        else:
            window_end += stride
    return (
        window_end
        + (frame_count - window_end) // stream_followup_stride * stream_followup_stride
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
