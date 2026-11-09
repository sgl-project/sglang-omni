# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture on a dedicated stream for the YuE2 acoustic solver."""

import ctypes
import weakref

import torch


def release_capture(
    graph: torch.cuda.CUDAGraph, stream: torch.cuda.ExternalStream
) -> None:
    with torch.cuda.device(stream.device):
        graph.reset()
        torch.cuda.check_error(
            torch.cuda.cudart().cudaStreamDestroy(stream.cuda_stream)
        )


class CapturedGraph:
    def __init__(self, device: torch.device) -> None:
        # note (ruiling): PyTorch #193402 lets pooled-stream reuse invalidate another live graph's cuBLAS workspace.
        with torch.cuda.device(device):
            handle = ctypes.c_void_p()
            torch.cuda.check_error(
                torch.cuda.cudart().cudaStreamCreate(ctypes.addressof(handle))
            )
            assert handle.value is not None
            try:
                self.stream = torch.cuda.ExternalStream(handle.value, device=device)
                self.graph = torch.cuda.CUDAGraph()
            except Exception:
                torch.cuda.check_error(
                    torch.cuda.cudart().cudaStreamDestroy(handle.value)
                )
                raise
        weakref.finalize(self, release_capture, self.graph, self.stream)

    def replay(self) -> None:
        self.graph.replay()
