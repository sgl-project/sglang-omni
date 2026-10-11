# SPDX-License-Identifier: Apache-2.0
"""Mimi's streaming decode, replayed through device graphs per slot and chunk width.

A graph reads and writes its decode state at fixed addresses, so each slot keeps
one stream's state for the life of the stage, with graphs of its own, and is
zeroed when the next stream takes it. Other chunk widths, and streams that find
every slot taken, decode eagerly; without graphs there are no slots, and every
stream decodes eagerly on a state of its own.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CODEBOOKS_PER_STREAM
from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiDecodeState
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)

# Note (wilsonzheng0327): Eager steps before the first capture of a width, so lazy
# library handles and workspaces are created outside the capture.
GRAPH_WARMUP_STEPS = 2


@dataclass(kw_only=True)
class DecodeSlot:
    """One stream's decode state; index is None for a state outside the pool."""

    index: int | None
    decode_state: MimiDecodeState


@dataclass(kw_only=True)
class CapturedDecode:
    graph: ReplayableGraph
    input_codes: torch.Tensor
    output_waveform: torch.Tensor


class MimiDecodeSlots:
    """Streaming decode states at fixed addresses, with graphs for short chunks."""

    def __init__(
        self, codec: MimiCodec, *, num_slots: int, max_graph_frames: int
    ) -> None:
        self.codec = codec
        self.decode_states: list[MimiDecodeState] = []
        self.free_slots: list[int] = []
        self.graphs: dict[tuple[int, int], CapturedDecode] = {}
        self.has_warned_full = False
        device = codec.device
        graph_backend = (
            current_platform.get_device_graph_backend(device)
            if current_platform.enable_codec_decode_graph()
            else None
        )
        if graph_backend is None or max_graph_frames == 0 or num_slots == 0:
            logger.info(f"PersonaPlex Mimi streaming decode runs eager on {device}")
            return
        else:
            pass

        self.decode_states = [
            codec.init_decode_state(batch_size=1) for _ in range(num_slots)
        ]
        self.free_slots = list(reversed(range(num_slots)))
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
            # Note (wilsonzheng0327): Widest first, so narrower graphs reuse its blocks.
            for chunk_frames in range(max_graph_frames, 0, -1):
                input_codes = torch.zeros(
                    (1, AUDIO_CODEBOOKS_PER_STREAM, chunk_frames),
                    dtype=torch.long,
                    device=device,
                )
                with device_module.stream(capture_stream):
                    for _ in range(GRAPH_WARMUP_STEPS):
                        codec.decode_step(input_codes, self.decode_states[0])
                for slot_index, decode_state in enumerate(self.decode_states):
                    with graph_backend.capture(
                        pool=graph_pool, stream=capture_stream, thread_local_errors=True
                    ) as graph:
                        output_waveform = codec.decode_step(input_codes, decode_state)
                    self.graphs[(slot_index, chunk_frames)] = CapturedDecode(
                        graph=graph,
                        input_codes=input_codes,
                        output_waveform=output_waveform,
                    )
            device_module.current_stream(device).wait_stream(capture_stream)
            for decode_state in self.decode_states:
                decode_state.reset()
            device_module.synchronize(device)
        graph_bytes = device_module.memory_allocated(device) - allocated_bytes_before
        logger.info(
            f"PersonaPlex Mimi decode graphs: {num_slots} slots x chunks of "
            f"1-{max_graph_frames} frames in "
            f"{time.perf_counter() - capture_started:.1f}s, {graph_bytes / 2**20:.0f}MiB"
        )

    @torch.inference_mode()
    def acquire(self) -> DecodeSlot:
        """A fresh state for a new stream; outside the pool when no slot is free."""
        if self.free_slots:
            slot_index = self.free_slots.pop()
            decode_state = self.decode_states[slot_index]
            decode_state.reset()
            return DecodeSlot(index=slot_index, decode_state=decode_state)
        elif self.decode_states and not self.has_warned_full:
            logger.warning(
                f"All {len(self.decode_states)} PersonaPlex Mimi decode slots are "
                "taken, so further streams decode eagerly; raise "
                "--code2wav.factory.num_decode_slots to at least the LM's "
                "max_running_requests"
            )
            self.has_warned_full = True
        else:
            pass
        return DecodeSlot(
            index=None, decode_state=self.codec.init_decode_state(batch_size=1)
        )

    def release(self, slot: DecodeSlot) -> None:
        if slot.index is not None:
            self.free_slots.append(slot.index)
        else:
            pass

    @torch.inference_mode()
    def decode_step(self, codes_BKF: torch.Tensor, slot: DecodeSlot) -> torch.Tensor:
        """Codes [1, 8, F] → waveform [1, 1, F * 1920], valid until the next step."""
        captured_decode = self.graphs.get((slot.index, codes_BKF.shape[-1]))
        if captured_decode is None:
            return self.codec.decode_step(codes_BKF, slot.decode_state)
        else:
            captured_decode.input_codes.copy_(codes_BKF)
            captured_decode.graph.replay()
            return captured_decode.output_waveform


__all__ = ["DecodeSlot", "MimiDecodeSlots"]
