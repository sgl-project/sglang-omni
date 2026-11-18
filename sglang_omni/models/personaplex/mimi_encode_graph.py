# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from sglang.srt.utils.common import is_pin_memory_available

from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.models.personaplex.mimi_decode_graph import GRAPH_WARMUP_STEPS
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CapturedEncode:
    graph: ReplayableGraph
    slot_indices: torch.Tensor
    input_waveform: torch.Tensor
    output_codes: torch.Tensor


class MimiEncodeSlots:
    def __init__(self, codec: MimiCodec, *, num_slots: int) -> None:
        self.codec = codec
        self.num_slots = num_slots
        self.pin_memory = is_pin_memory_available(codec.device)
        self.host_slot_indices = self.slot_index_rows(1)
        self.encode_state = codec.init_encode_state(batch_size=num_slots)
        self.free_slots = list(reversed(range(num_slots)))
        self.graphs: dict[int, CapturedEncode] = {}
        device = codec.device
        graph_backend = (
            current_platform.get_device_graph_backend(device)
            if current_platform.is_cuda()
            else None
        )
        if graph_backend is None:
            logger.info(f"PersonaPlex Mimi streaming encode runs eager on {device}")
        else:
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
                # Note (Junnan Li): Largest first, so smaller graphs reuse its blocks from
                # the shared pool.
                for batch_size in range(self.num_slots, 0, -1):
                    with device_module.stream(capture_stream):
                        slot_indices = torch.arange(batch_size, device=device)
                        input_waveform = torch.zeros(
                            (batch_size, 1, SAMPLES_PER_FRAME),
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
            graph_bytes = (
                device_module.memory_allocated(device) - allocated_bytes_before
            )
            logger.info(
                f"PersonaPlex Mimi encode graphs: batches of 1-{self.num_slots} over "
                f"{self.num_slots} slots in {time.perf_counter() - capture_started:.1f}s, "
                f"{graph_bytes / 2**20:.0f}MiB"
            )

    @torch.inference_mode()
    def acquire(self) -> int:
        slot_index = self.free_slots.pop()
        self.encode_state.reset_row(slot_index)
        return slot_index

    def release(self, slot_index: int) -> None:
        self.free_slots.append(slot_index)

    @torch.inference_mode()
    def slot_index_rows(self, frames: int) -> torch.Tensor:
        return torch.empty(
            frames, self.num_slots, dtype=torch.long, pin_memory=self.pin_memory
        )

    @torch.inference_mode()
    def encode_step(
        self, waveform_B1T: torch.Tensor, slot_indices: list[int], *, frame: int
    ) -> torch.Tensor:
        """Codes [B, 8, F] are valid only until the next step."""
        captured = (
            self.graphs.get(len(slot_indices))
            if waveform_B1T.shape[-1] == SAMPLES_PER_FRAME
            else None
        )
        # Note (Junnan Li): One index row per frame step, as the steps run back to back
        # without a host sync.
        if frame >= self.host_slot_indices.shape[0]:
            self.host_slot_indices = self.slot_index_rows(frame + 1)
        else:
            pass
        host_rows = self.host_slot_indices[frame, : len(slot_indices)]
        host_rows.numpy()[:] = slot_indices
        if captured is None:
            rows = host_rows.to(self.codec.device, non_blocking=True)
            return self.encode_rows(waveform_B1T, rows)
        else:
            captured.slot_indices.copy_(host_rows, non_blocking=True)
            captured.input_waveform.copy_(waveform_B1T)
            captured.graph.replay()
            return captured.output_codes

    def encode_rows(
        self, waveform_B1T: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        selected = self.encode_state.select_rows(rows)
        codes = self.codec.encode_step(waveform_B1T, selected)
        self.encode_state.store_rows(rows, selected)
        return codes


__all__ = ["MimiEncodeSlots"]
