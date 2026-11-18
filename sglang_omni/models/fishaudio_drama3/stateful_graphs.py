# SPDX-License-Identifier: Apache-2.0
"""Exact-shape shared graphs over the fixed incremental codec arena."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

import torch

from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_drama3.stateful_state import (
    StreamState,
    decode_attention,
)
from sglang_omni.platforms.device_graph import (
    CudaDeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

CAPTURE_FRAME_COUNTS = tuple(range(1, 65))


class DeltaGraph(Protocol):
    def replay(self) -> torch.Tensor: ...


class GraphBoundary(Protocol):
    def capture(
        self,
        backend: StatefulDACDecoder,
        codes: torch.Tensor,
        device_slot_index: torch.Tensor,
        frame_count: int,
    ) -> DeltaGraph: ...

    def wait_before_submit(self) -> None: ...

    def record_submission(self) -> None: ...

    def synchronize(self) -> None: ...


class CudaDeltaGraph:
    def __init__(self, graph: ReplayableGraph, audio: torch.Tensor) -> None:
        self.graph: ReplayableGraph = graph
        self.audio: torch.Tensor = audio

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.audio


class CudaGraphBoundary:
    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("Stateful graph capture requires a CUDA device")
        else:
            pass
        self.device: torch.device = device
        self.capture_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)
        self.submitting_stream: torch.cuda.Stream | None = None
        self.completion: torch.cuda.Event = torch.cuda.Event()
        self.has_submission: bool = False
        self.graph_backend: CudaDeviceGraphBackend = CudaDeviceGraphBackend()
        self.graph_pools: dict[int, DeviceGraphPool] = {}

    def capture(
        self,
        backend: StatefulDACDecoder,
        codes: torch.Tensor,
        device_slot_index: torch.Tensor,
        frame_count: int,
    ) -> DeltaGraph:
        self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.submitting_stream = self.capture_stream
        with torch.cuda.device(self.device), torch.cuda.stream(self.capture_stream):
            backend.decode_slot(codes, device_slot_index, frame_count)
            self.capture_stream.synchronize()
            backend.reset(backend.scratch_slot_index)
            self.capture_stream.synchronize()
            self.graph_pools[frame_count] = (
                next(iter(self.graph_pools.values()))
                if self.graph_pools
                else self.graph_backend.graph_pool_handle()
            )
            with self.graph_backend.capture(
                pool=self.graph_pools[frame_count], stream=self.capture_stream
            ) as graph:
                audio = backend.decode_slot(codes, device_slot_index, frame_count)
            self.completion.record(self.capture_stream)
            self.has_submission = True
        return CudaDeltaGraph(graph, audio)

    def wait_before_submit(self) -> None:
        self.submitting_stream = torch.cuda.current_stream(self.device)
        if self.has_submission:
            self.submitting_stream.wait_event(self.completion)
        else:
            pass

    def record_submission(self) -> None:
        self.submitting_stream = torch.cuda.current_stream(self.device)
        self.completion.record(self.submitting_stream)
        self.has_submission = True

    def synchronize(self) -> None:
        if self.submitting_stream is not None:
            self.submitting_stream.synchronize()
        else:
            pass
        if self.has_submission:
            self.completion.synchronize()
        else:
            pass


class StatefulGraphDecoder:
    """Serialize shared graph buffers and return independently owned audio."""

    def __init__(
        self,
        backend: StatefulDACDecoder,
        *,
        graph_boundary: GraphBoundary | None = None,
        capture_frame_counts: tuple[int, ...] = CAPTURE_FRAME_COUNTS,
        compiled_frame_counts: tuple[int, ...] | None = None,
    ) -> None:
        if any(count not in CAPTURE_FRAME_COUNTS for count in capture_frame_counts):
            raise ValueError("Captured delta lengths must be within 1–64 frames")
        else:
            pass
        self.backend: StatefulDACDecoder = backend
        self.graph_boundary: GraphBoundary = (
            CudaGraphBoundary(backend.device)
            if graph_boundary is None
            else graph_boundary
        )
        self.state_lock: RLock = RLock()
        self.is_stopped: bool = False
        self.device_slot_index: torch.Tensor = torch.full(
            (1,), backend.scratch_slot_index, dtype=torch.long, device=backend.device
        )
        self.static_codes: dict[int, torch.Tensor] = {
            frame_count: torch.zeros(
                (1, backend.num_codebooks, frame_count),
                dtype=torch.long,
                device=backend.device,
            )
            for frame_count in capture_frame_counts
        }
        self.graphs: dict[int, DeltaGraph] = {}
        with torch.no_grad():
            for frame_count, codes in self.static_codes.items():
                attention_decode = backend.attention_decode
                if (
                    compiled_frame_counts is not None
                    and frame_count not in compiled_frame_counts
                ):
                    backend.attention_decode = decode_attention
                else:
                    pass
                try:
                    self.graphs[frame_count] = self.graph_boundary.capture(
                        backend, codes, self.device_slot_index, frame_count
                    )
                finally:
                    backend.attention_decode = attention_decode
                    self.graph_boundary.synchronize()
                    backend.reset(backend.scratch_slot_index)
                    self.graph_boundary.record_submission()
                    self.graph_boundary.synchronize()
            self.graph_boundary.synchronize()

    def create_state(self, slot_index: int | None = None) -> StreamState:
        with self.state_lock:
            if self.is_stopped:
                raise RuntimeError(
                    "Stateful graph decoder stopped after execution failure"
                )
            else:
                return self.backend.create_state(slot_index)

    def try_create_state(self) -> StreamState | None:
        with self.state_lock:
            if self.is_stopped:
                raise RuntimeError(
                    "Stateful graph decoder stopped after execution failure"
                )
            else:
                return self.backend.try_create_state()

    def reset(self, slot_index: int) -> None:
        with self.state_lock:
            self.graph_boundary.synchronize()
            self.backend.reset(slot_index)
            self.graph_boundary.record_submission()
            self.graph_boundary.synchronize()

    def release_state(self, state: StreamState) -> None:
        with self.state_lock:
            self.graph_boundary.synchronize()
            self.backend.release_state(state)
            self.graph_boundary.record_submission()
            self.graph_boundary.synchronize()

    @torch.no_grad()
    def decode_delta(
        self, codes: torch.Tensor, state: StreamState, is_final: bool = False
    ) -> torch.Tensor:
        with self.state_lock:
            if self.is_stopped:
                raise RuntimeError(
                    "Stateful graph decoder stopped after execution failure"
                )
            else:
                pass
            frame_count = self.backend.validate_delta(codes, state)
            if frame_count == 0:
                self.backend.commit_delta(state, frame_count, is_final)
                return torch.empty(
                    (1, 1, 0), device=self.backend.device, dtype=self.backend.dtype
                )
            else:
                pass
            try:
                self.graph_boundary.wait_before_submit()
                self.device_slot_index.fill_(state.slot_index)
                if frame_count in self.graphs:
                    self.static_codes[frame_count].copy_(codes)
                    audio = self.graphs[frame_count].replay()
                else:
                    attention_decode = self.backend.attention_decode
                    self.backend.attention_decode = decode_attention
                    try:
                        audio = self.backend.decode_slot(
                            codes, self.device_slot_index, frame_count
                        )
                    finally:
                        self.backend.attention_decode = attention_decode
                owned_audio = audio.clone()
                self.graph_boundary.record_submission()
                self.backend.commit_delta(state, frame_count, is_final)
                return owned_audio
            except Exception:
                self.is_stopped = True
                try:
                    self.graph_boundary.synchronize()
                finally:
                    self.backend.discard_state(state)
                    self.graph_boundary.record_submission()
                    self.graph_boundary.synchronize()
                raise
