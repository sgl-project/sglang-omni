# SPDX-License-Identifier: Apache-2.0
"""A fixed-frame Mimi encode graph with independently owned stream history."""

from __future__ import annotations

import torch

from sglang_omni.models.personaplex.components.mimi import MimiCodec, MimiEncodeState
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph

GRAPH_WARMUP_STEPS = 2


class MimiEncodeGraph:
    @torch.inference_mode()
    def __init__(self, codec: MimiCodec) -> None:
        self.state: MimiEncodeState = codec.init_encode_state(batch_size=1)
        self.input_waveform: torch.Tensor = torch.zeros(
            1, 1, codec.samples_per_frame, device=codec.device, dtype=torch.float32
        )
        backend = current_platform.get_device_graph_backend(codec.device)
        if backend is None:
            raise ValueError(f"Mimi encode graphs are unavailable on {codec.device}")
        else:
            pass
        device_module = torch.get_device_module(codec.device)
        capture_stream = device_module.Stream(device=codec.device)
        with (
            device_module.device(codec.device),
            current_platform.graph_capture_attention(),
        ):
            capture_stream.wait_stream(device_module.current_stream(codec.device))
            with device_module.stream(capture_stream):
                for _ in range(GRAPH_WARMUP_STEPS):
                    codec.encode_step(self.input_waveform, self.state)
            with backend.capture(
                pool=backend.graph_pool_handle(),
                stream=capture_stream,
                thread_local_errors=True,
            ) as graph:
                self.output_codes: torch.Tensor = codec.encode_step(
                    self.input_waveform, self.state
                )
            self.graph: ReplayableGraph = graph
            device_module.current_stream(codec.device).wait_stream(capture_stream)
            self.state.reset()
            device_module.synchronize(codec.device)

    @torch.inference_mode()
    def encode_step(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode one frame; returned codes remain valid until the next replay."""
        assert waveform.shape == self.input_waveform.shape
        self.input_waveform.copy_(waveform)
        self.graph.replay()
        return self.output_codes
