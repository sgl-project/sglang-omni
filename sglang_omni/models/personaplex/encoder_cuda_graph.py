# SPDX-License-Identifier: Apache-2.0
"""Exact-length Mimi encoder graphs for the serial audio input stage."""

from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph
from sglang_omni.utils.device import device_guard


@dataclass
class CapturedMimiEncoder:
    graph: ReplayableGraph
    waveform: torch.Tensor
    codes: torch.Tensor


class MimiEncoderCudaGraphRunner:
    """Capture configured lengths at startup and return independent CPU codes."""

    def __init__(
        self,
        codec: MimiCodec,
        *,
        frames: list[int],
        graph_backend: DeviceGraphBackend,
        compile_quantizer: bool,
    ) -> None:
        self.codec = codec
        self.graphs: dict[int, CapturedMimiEncoder] = {}
        device = codec.device
        device_module = torch.get_device_module(device)
        if compile_quantizer:
            quantize = torch.compile(
                codec.quantizer.encode,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

            def encode_for_capture(waveform: torch.Tensor) -> torch.Tensor:
                latent = codec.encoder_transformer(codec.encoder(waveform))
                return quantize(codec.downsample(latent))

        else:
            encode_for_capture = codec.encode
        with device_guard(device), torch.inference_mode():
            stream = device_module.Stream(device=device)
            for frame_count in sorted(set(frames)):
                waveform = torch.zeros(
                    (1, 1, frame_count * codec.samples_per_frame),
                    device=device,
                    dtype=torch.float32,
                )
                stream.wait_stream(device_module.current_stream(device))
                with device_module.stream(stream):
                    for _ in range(2):
                        encode_for_capture(waveform)
                with graph_backend.capture(stream=stream) as graph:
                    codes = encode_for_capture(waveform)
                stream.synchronize()
                self.graphs[waveform.numel()] = CapturedMimiEncoder(
                    graph=graph, waveform=waveform, codes=codes
                )

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode a CPU waveform [samples] into independent CPU codes [frames, 8]."""
        captured = self.graphs.get(waveform.numel())
        with device_guard(self.codec.device):
            if captured is None:
                codes = self.codec.encode(
                    waveform.to(device=self.codec.device, dtype=torch.float32).view(
                        1, 1, -1
                    )
                )
            else:
                captured.waveform.copy_(waveform.view(1, 1, -1))
                captured.graph.replay()
                codes = captured.codes
            return codes[0].T.cpu()
