# SPDX-License-Identifier: Apache-2.0
"""Mimi's streaming decode for many streams at once, replayed through device graphs.

Each stream owns a slot: one row of a decode state stacked over every slot,
kept at fixed addresses for the life of the stage and zeroed when the next
stream takes it. A step decodes any set of slots whose chunks share a width:
it gathers their rows, decodes them as one batch with each row at its own
point in its stream, and writes the rows back. A step of up to max_batch_size
slots and max_graph_frames frames replays the device graph captured for that
batch size and width; other widths, and platforms without codec decode graphs,
run the same step eagerly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CODEBOOKS_PER_STREAM
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

logger = logging.getLogger(__name__)

# Note (wilsonzheng0327): Eager steps before the first capture of a width, so lazy
# library handles and workspaces are created outside the capture.
GRAPH_WARMUP_STEPS = 2


@dataclass(kw_only=True)
class CapturedDecode:
    graph: ReplayableGraph
    slot_indices: torch.Tensor
    input_codes: torch.Tensor
    output_waveform: torch.Tensor


class MimiDecodeSlots:
    """Streaming decode states for num_slots streams, decoded in batches."""

    def __init__(
        self,
        codec: MimiCodec,
        *,
        num_slots: int,
        max_batch_size: int,
        max_graph_frames: int,
        max_chunk_frames: int,
    ) -> None:
        self.codec = codec
        self.num_slots = num_slots
        self.max_batch_size = max_batch_size
        self.max_chunk_frames = max(max_chunk_frames, max_graph_frames)
        self.decode_state = codec.init_decode_state(
            batch_size=num_slots, max_chunk_frames=self.max_chunk_frames
        )
        self.free_slots = list(reversed(range(num_slots)))
        self.graphs: dict[tuple[int, int], CapturedDecode] = {}
        device = codec.device
        graph_backend = (
            current_platform.get_device_graph_backend(device)
            if current_platform.enable_codec_decode_graph()
            else None
        )
        if graph_backend is None or max_graph_frames == 0:
            logger.info(f"PersonaPlex Mimi streaming decode runs eager on {device}")
        else:
            self.capture_graphs(
                graph_backend,
                max_batch_size=min(max_batch_size, num_slots),
                max_graph_frames=max_graph_frames,
            )

    def capture_graphs(
        self,
        graph_backend: DeviceGraphBackend,
        *,
        max_batch_size: int,
        max_graph_frames: int,
    ) -> None:
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
            # Note (nightdecode): Largest batch and widest chunk first, so smaller
            # graphs reuse their blocks from the shared pool. The static inputs are
            # filled on the capture stream, which reads them first.
            for batch_size in range(max_batch_size, 0, -1):
                for chunk_frames in range(max_graph_frames, 0, -1):
                    with device_module.stream(capture_stream):
                        slot_indices = torch.arange(batch_size, device=device)
                        input_codes = torch.zeros(
                            (batch_size, AUDIO_CODEBOOKS_PER_STREAM, chunk_frames),
                            dtype=torch.long,
                            device=device,
                        )
                        for _ in range(GRAPH_WARMUP_STEPS):
                            self.decode_rows(input_codes, slot_indices)
                    with graph_backend.capture(
                        pool=graph_pool, stream=capture_stream, thread_local_errors=True
                    ) as graph:
                        output_waveform = self.decode_rows(input_codes, slot_indices)
                    self.graphs[(batch_size, chunk_frames)] = CapturedDecode(
                        graph=graph,
                        slot_indices=slot_indices,
                        input_codes=input_codes,
                        output_waveform=output_waveform,
                    )
            device_module.current_stream(device).wait_stream(capture_stream)
            self.decode_state.reset()
            device_module.synchronize(device)
        graph_bytes = device_module.memory_allocated(device) - allocated_bytes_before
        logger.info(
            f"PersonaPlex Mimi decode graphs: batches of 1-{max_batch_size} x chunks "
            f"of 1-{max_graph_frames} frames over {self.num_slots} slots in "
            f"{time.perf_counter() - capture_started:.1f}s, {graph_bytes / 2**20:.0f}MiB"
        )

    @property
    def active_slot_count(self) -> int:
        return self.num_slots - len(self.free_slots)

    @torch.inference_mode()
    def acquire(self) -> int | None:
        """A zeroed slot for a new stream, or None when every slot is taken."""
        if not self.free_slots:
            return None
        else:
            pass
        slot_index = self.free_slots.pop()
        self.decode_state.reset_row(slot_index)
        return slot_index

    def release(self, slot_index: int) -> None:
        self.free_slots.append(slot_index)

    @torch.inference_mode()
    def decode_step(
        self, codes_BKF: torch.Tensor, slot_indices: list[int]
    ) -> torch.Tensor:
        """Codes [B, 8, F] for B distinct slots → waveform [B, 1, F * 1920].

        Row i continues slot_indices[i]'s stream. The waveform is valid until
        the next step.
        """
        assert codes_BKF.shape[0] == len(slot_indices) == len(set(slot_indices))
        assert 0 < codes_BKF.shape[-1] <= self.max_chunk_frames
        captured_decode = self.graphs.get((len(slot_indices), codes_BKF.shape[-1]))
        if captured_decode is None:
            rows = torch.tensor(
                slot_indices, dtype=torch.long, device=self.codec.device
            )
            return self.decode_rows(codes_BKF, rows)
        else:
            captured_decode.slot_indices.copy_(
                torch.tensor(slot_indices, dtype=torch.long)
            )
            captured_decode.input_codes.copy_(codes_BKF)
            captured_decode.graph.replay()
            return captured_decode.output_waveform

    def decode_rows(self, codes_BKF: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """The step a graph captures: gather the rows, decode, write them back."""
        selected = self.decode_state.select_rows(rows)
        waveform = self.codec.decode_step(codes_BKF, selected)
        self.decode_state.store_rows(rows, selected)
        return waveform


__all__ = ["MimiDecodeSlots"]
