# SPDX-License-Identifier: Apache-2.0
"""Resident Mimi encode state rows and CUDA graphs for realtime audio units."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiEncodeState
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

logger = logging.getLogger(__name__)
GRAPH_WARMUP_STEPS = 2


@dataclass(kw_only=True)
class CapturedEncode:
    graph: ReplayableGraph
    slot_indices: torch.Tensor
    input_waveform: torch.Tensor
    output_codes: torch.Tensor


class MimiEncodeSlots:
    """One resident encode state row per session, advanced in ready batches."""

    def __init__(
        self,
        codec: MimiCodec,
        *,
        num_slots: int,
        max_batch_size: int,
        enable_graphs: bool,
    ) -> None:
        assert num_slots > 0 and max_batch_size > 0
        self.codec: MimiCodec = codec
        self.num_slots: int = num_slots
        self.max_batch_size: int = min(max_batch_size, num_slots)
        self.encode_state: MimiEncodeState = codec.init_encode_state(num_slots)
        self.free_slots: list[int] = list(reversed(range(num_slots)))
        self.graphs: dict[int, CapturedEncode] = {}
        graph_backend = (
            current_platform.get_device_graph_backend(codec.device)
            if enable_graphs and codec.device.type == "cuda"
            else None
        )
        if graph_backend is None:
            logger.info(
                f"PersonaPlex Mimi streaming encode runs eager on {codec.device}"
            )
        else:
            self.capture_graphs(graph_backend)

    def capture_graphs(self, graph_backend: DeviceGraphBackend) -> None:
        device = self.codec.device
        device_module = torch.get_device_module(device)
        capture_stream = device_module.Stream(device=device)
        graph_pool = graph_backend.graph_pool_handle()
        capture_started = time.perf_counter()
        allocated_bytes_before = device_module.memory_allocated(device)
        with (
            torch.inference_mode(),
            device_module.device(device),
            current_platform.graph_capture_attention(),
        ):
            capture_stream.wait_stream(device_module.current_stream(device))
            for batch_size in range(self.max_batch_size, 0, -1):
                with device_module.stream(capture_stream):
                    slot_indices = torch.arange(batch_size, device=device)
                    input_waveform = torch.zeros(
                        (batch_size, 1, self.codec.samples_per_frame),
                        dtype=torch.float32,
                        device=device,
                    )
                    for _ in range(GRAPH_WARMUP_STEPS):
                        self.encode_rows(input_waveform, slot_indices)
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream, thread_local_errors=True
                ) as graph:
                    output_codes = self.encode_rows(input_waveform, slot_indices)
                self.graphs[batch_size] = CapturedEncode(
                    graph=graph,
                    slot_indices=slot_indices,
                    input_waveform=input_waveform,
                    output_codes=output_codes,
                )
            device_module.current_stream(device).wait_stream(capture_stream)
            self.encode_state.reset()
            device_module.synchronize(device)
        graph_bytes = device_module.memory_allocated(device) - allocated_bytes_before
        logger.info(
            f"PersonaPlex Mimi encode graphs: batches of 1-{self.max_batch_size} "
            f"over {self.num_slots} slots in {time.perf_counter() - capture_started:.1f}s, "
            f"{graph_bytes / 2**20:.0f}MiB"
        )

    @torch.inference_mode()
    def acquire_slot(self) -> int | None:
        if not self.free_slots:
            return None
        else:
            slot_index = self.free_slots.pop()
            self.encode_state.reset_row(slot_index)
            return slot_index

    def release_slot(self, slot_index: int) -> None:
        self.free_slots.append(slot_index)

    @torch.inference_mode()
    def encode_step(
        self, waveform: torch.Tensor, slot_indices: list[int]
    ) -> torch.Tensor:
        """Encode [B, 1, samples] for distinct slots; codes last until the next step."""
        assert waveform.shape[0] == len(slot_indices) == len(set(slot_indices))
        assert waveform.shape[1] == 1
        assert waveform.shape[-1] > 0
        assert waveform.shape[-1] % self.codec.samples_per_frame == 0
        captured_encode = (
            self.graphs.get(len(slot_indices))
            if waveform.shape[-1] == self.codec.samples_per_frame
            else None
        )
        if captured_encode is None:
            rows = torch.tensor(
                slot_indices, dtype=torch.long, device=self.codec.device
            )
            return self.encode_rows(waveform, rows)
        else:
            captured_encode.slot_indices.copy_(
                torch.tensor(slot_indices, dtype=torch.long)
            )
            captured_encode.input_waveform.copy_(waveform)
            captured_encode.graph.replay()
            return captured_encode.output_codes

    def encode_rows(self, waveform: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        selected = self.encode_state.select_rows(rows)
        if waveform.shape[-1] == self.codec.samples_per_frame:
            codes = self.codec.encode_step(waveform, selected)
        else:
            codes = torch.cat(
                [
                    self.codec.encode_step(waveform_chunk, selected)
                    for waveform_chunk in waveform.split(
                        self.codec.samples_per_frame, dim=-1
                    )
                ],
                dim=-1,
            )
        self.encode_state.store_rows(rows, selected)
        return codes


__all__ = ["MimiEncodeSlots"]
