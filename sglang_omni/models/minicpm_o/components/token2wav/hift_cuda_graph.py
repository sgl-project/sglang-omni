# SPDX-License-Identifier: Apache-2.0
"""Exact-shape CUDA graphs for the real-valued HiFT decode body."""

from dataclasses import dataclass
from typing import Protocol

import torch


class HiftDecodeBody(Protocol):
    def __call__(
        self, x: torch.Tensor, s_stft: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(kw_only=True)
class CapturedHiftBody:
    graph: torch.cuda.CUDAGraph
    mel: torch.Tensor
    source_spectrum: torch.Tensor
    magnitude: torch.Tensor
    phase: torch.Tensor


class HiftBodyCudaGraphRunner:
    """Use on one serialized CUDA stream; returned spectra own their storage."""

    def __init__(self, decode_body: HiftDecodeBody) -> None:
        self.decode_body = decode_body
        self.graphs: dict[tuple[int, int], CapturedHiftBody] = {}

    def capture(
        self,
        shapes: tuple[tuple[int, int], ...],
        *,
        device: torch.device,
        mel_channels: int,
        samples_per_frame: int,
        fft_samples: int,
        hop_samples: int,
    ) -> None:
        if len(set(shapes)) != len(shapes) or any(
            batch_size <= 0 or mel_frames <= 0 for batch_size, mel_frames in shapes
        ):
            raise ValueError(
                "HiFT graph shapes must be unique positive batch/frame pairs"
            )
        else:
            pass
        caller_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(caller_stream)
        graphs: dict[tuple[int, int], CapturedHiftBody] = {}
        pool = torch.cuda.graph_pool_handle()
        try:
            with torch.cuda.device(device), torch.cuda.stream(capture_stream):
                for batch_size, mel_frames in sorted(
                    shapes, key=lambda shape: shape[0] * shape[1], reverse=True
                ):
                    # note (MayDomine): match serving's normal mel and inference-mode spectrum guards.
                    with torch.inference_mode(False):
                        mel = torch.zeros(
                            batch_size, mel_channels, mel_frames, device=device
                        )
                    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                        spectrum_frames = (
                            mel_frames * samples_per_frame // hop_samples + 1
                        )
                        source_spectrum = torch.zeros(
                            batch_size, fft_samples + 2, spectrum_frames, device=device
                        )
                        for _ in range(3):
                            self.decode_body(mel, source_spectrum)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                            magnitude, phase = self.decode_body(mel, source_spectrum)
                        graphs[(batch_size, mel_frames)] = CapturedHiftBody(
                            graph=graph,
                            mel=mel,
                            source_spectrum=source_spectrum,
                            magnitude=magnitude,
                            phase=phase,
                        )
        finally:
            caller_stream.wait_stream(capture_stream)
        self.graphs = graphs

    def run(
        self, mel: torch.Tensor, source_spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured = self.graphs.get((mel.shape[0], mel.shape[2]))
        if (
            captured is None
            or mel.device != captured.mel.device
            or mel.dtype != captured.mel.dtype
            or mel.shape != captured.mel.shape
            or source_spectrum.device != captured.source_spectrum.device
            or source_spectrum.dtype != captured.source_spectrum.dtype
            or source_spectrum.shape != captured.source_spectrum.shape
            or not torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cuda")
        ):
            return self.decode_body(mel, source_spectrum)
        else:
            # note (MayDomine): time padding changes the noncausal convolution boundary.
            captured.mel.copy_(mel)
            captured.source_spectrum.copy_(source_spectrum)
            captured.graph.replay()
            # note (MayDomine): other shapes share the pool and can overwrite both spectra.
            return captured.magnitude.clone(), captured.phase.clone()
