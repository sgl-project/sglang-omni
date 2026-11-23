# SPDX-License-Identifier: Apache-2.0
"""Padded CUDA graphs with valid-frame masks for the HiFT decode body."""

from dataclasses import dataclass
from typing import Protocol

import torch


class HiftDecodeBody(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        s_stft: torch.Tensor,
        frame_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(kw_only=True)
class CapturedHiftBody:
    graph: torch.cuda.CUDAGraph
    mel: torch.Tensor
    source_spectrum: torch.Tensor
    frame_lengths: torch.Tensor
    magnitude: torch.Tensor
    phase: torch.Tensor


class HiftBodyCudaGraphRunner:
    """Use on one serialized CUDA stream; returned spectra own their storage."""

    def __init__(self, decode_body: HiftDecodeBody) -> None:
        self.decode_body = decode_body
        self.graphs: dict[tuple[int, int], CapturedHiftBody] = {}
        self.spectrum_frames_per_mel_frame: int = 0

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
        self.spectrum_frames_per_mel_frame = samples_per_frame // hop_samples
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
                        frame_lengths = torch.full(
                            (batch_size,), mel_frames, device=device, dtype=torch.int64
                        )
                        for _ in range(3):
                            self.decode_body(mel, source_spectrum, frame_lengths)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                            magnitude, phase = self.decode_body(
                                mel, source_spectrum, frame_lengths
                            )
                        graphs[(batch_size, mel_frames)] = CapturedHiftBody(
                            graph=graph,
                            mel=mel,
                            source_spectrum=source_spectrum,
                            frame_lengths=frame_lengths,
                            magnitude=magnitude,
                            phase=phase,
                        )
        finally:
            caller_stream.wait_stream(capture_stream)
        self.graphs = graphs

    def run(
        self, mel: torch.Tensor, source_spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, mel_channels, mel_frames = mel.shape
        capture_shape = min(
            (
                shape
                for shape in self.graphs
                if shape[0] >= batch_size and shape[1] >= mel_frames
            ),
            key=lambda shape: (shape[0] * shape[1], shape[1], shape[0]),
            default=None,
        )
        captured = self.graphs.get(capture_shape)
        spectrum_frames = mel_frames * self.spectrum_frames_per_mel_frame + 1
        if (
            captured is None
            or mel.device != captured.mel.device
            or mel.dtype != captured.mel.dtype
            or mel_channels != captured.mel.shape[1]
            or source_spectrum.device != captured.source_spectrum.device
            or source_spectrum.dtype != captured.source_spectrum.dtype
            or source_spectrum.shape
            != (batch_size, captured.source_spectrum.shape[1], spectrum_frames)
            or not torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cuda")
        ):
            return self.decode_body(mel, source_spectrum)
        else:
            captured.mel.zero_()
            captured.source_spectrum.zero_()
            captured.mel[:batch_size, :, :mel_frames].copy_(mel)
            captured.source_spectrum[:batch_size, :, :spectrum_frames].copy_(
                source_spectrum
            )
            captured.frame_lengths.fill_(mel_frames)
            captured.graph.replay()
            # note (MayDomine): crop before iSTFT so its window envelope ends at the real tail.
            return (
                captured.magnitude[:batch_size, :, :spectrum_frames].clone(),
                captured.phase[:batch_size, :, :spectrum_frames].clone(),
            )
