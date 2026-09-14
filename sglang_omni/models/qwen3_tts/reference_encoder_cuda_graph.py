# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for the Qwen3-TTS reference encoder at bucketed lengths."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.mimi.modeling_mimi import MimiConv1d

logger = logging.getLogger(__name__)

# note(ratish): 2.6 to 20.5 s of reference audio at 12.5 Hz; a key costs 32 MiB.
DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES = (32, 48, 64, 96, 128, 192, 256)


def move_conv_padding_to_host(encoder: torch.nn.Module) -> int:
    """Put every conv's padding integers on the CPU; returns the conv count.

    note(ratish): on the device the padding arithmetic before each conv is six
    launches and a host read, and a host read cannot be captured.
    """
    count = 0
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.stride = module.stride.cpu()
            module.kernel_size = module.kernel_size.cpu()
            module.padding_total = module.padding_total.cpu()
            module.padding_right = module.padding_total // 2
            module.padding_left = module.padding_total - module.padding_right
            count += 1
    return count


def smallest_bucket(frames: int, buckets: Iterable[int]) -> int | None:
    """The smallest bucket at or above frames; None if none is."""
    fitting = [bucket for bucket in buckets if bucket >= frames]
    return min(fitting) if fitting else None


@dataclass
class _CapturedEncoderGraph:
    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    static_codes: torch.Tensor


class Qwen3TTSReferenceEncoderCudaGraphRunner:
    """One captured encode per bucket length, batch 1, replayed on the current stream."""

    def __init__(
        self,
        encoder: Any,
        *,
        hop: int,
        num_quantizers: int,
        bucket_frames: Iterable[int],
        stream: torch.cuda.Stream,
    ) -> None:
        self._encoder = encoder
        self._hop = int(hop)
        self._num_quantizers = int(num_quantizers)
        self._bucket_frames = tuple(sorted({int(f) for f in bucket_frames}))
        self._stream = stream
        param = next(encoder.parameters())
        self._device = param.device
        self._dtype = param.dtype
        self._graphs: dict[int, _CapturedEncoderGraph] = {}
        self._pool: Any | None = None
        self._disable_reason: str | None = None
        self._replays = 0
        self._misses = 0

    def capture(self) -> None:
        graphs: dict[int, _CapturedEncoderGraph] = {}
        try:
            with torch.cuda.device(self._device):
                pool = torch.cuda.graph_pool_handle()
                # note(ratish): largest first so the shared pool is sized once.
                for frames in reversed(self._bucket_frames):
                    graphs[frames] = self._capture_bucket(frames, pool)
        except Exception as exc:
            self._disable_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3-TTS reference encoder graph capture disabled the runner: %s",
                self._disable_reason,
                exc_info=True,
            )
            return
        self._graphs = {frames: graphs[frames] for frames in self._bucket_frames}
        self._pool = pool
        logger.info(
            "Qwen3-TTS reference encoder graphs captured for %s frames",
            list(self._bucket_frames),
        )

    def _capture_bucket(self, frames: int, pool: Any) -> _CapturedEncoderGraph:
        static_input = torch.zeros(
            (1, 1, frames * self._hop), device=self._device, dtype=self._dtype
        )
        self._stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.inference_mode(), torch.cuda.stream(self._stream):
            for _ in range(2):
                self._encode(static_input)
        graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(
                graph,
                pool=pool,
                stream=self._stream,
                capture_error_mode="thread_local",
            ),
        ):
            static_codes = self._encode(static_input)
        self._stream.synchronize()
        return _CapturedEncoderGraph(
            graph=graph, static_input=static_input, static_codes=static_codes
        )

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        return self._encoder.encode(
            values, num_quantizers=self._num_quantizers, return_dict=True
        ).audio_codes

    def bucket_for(self, frames: int) -> int | None:
        return smallest_bucket(frames, self._graphs)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """Codes (frames, quantizers) of a waveform (samples,); None above the largest bucket."""
        samples = waveform.numel()
        frames = -(-samples // self._hop)
        bucket = self.bucket_for(frames)
        if bucket is None:
            self._misses += 1
            return None
        captured = self._graphs[bucket]
        captured.static_input[0, 0, :samples].copy_(waveform)
        captured.static_input[0, 0, samples:].zero_()
        captured.graph.replay()
        self._replays += 1
        # note(ratish): the graph rewrites its output on the next replay.
        return captured.static_codes[0, :, :frames].transpose(0, 1).clone()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": bool(self._graphs),
            "disable_reason": self._disable_reason,
            "bucket_frames": list(self._bucket_frames),
            "captured": list(self._graphs),
            "replays": self._replays,
            "misses": self._misses,
        }


__all__ = [
    "DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES",
    "Qwen3TTSReferenceEncoderCudaGraphRunner",
    "move_conv_padding_to_host",
    "smallest_bucket",
]
