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

# note(ratish): a replay costs a floor plus a small per frame term and a key costs
# 32 MiB, so keys below 32 frames save nothing measurable; the step keeps the padding
# under half a clip, and clips past 256 frames run eager.
DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES = (32, 48, 64, 96, 128, 192, 256)


def move_conv_padding_to_host(encoder: torch.nn.Module) -> int:
    """Put every conv's padding integers on the CPU; returns the conv count."""
    count = 0
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.stride = module.stride.cpu()
            module.kernel_size = module.kernel_size.cpu()
            module.padding_total = module.padding_total.cpu()
            module.padding_right = module.padding_total // 2
            module.padding_left = module.padding_total - module.padding_right
            count += 1
        else:
            pass
    return count


def smallest_bucket(frames: int, buckets: Iterable[int]) -> int | None:
    """The smallest bucket at or above frames; None if none is."""
    fitting = [bucket for bucket in buckets if bucket >= frames]
    return min(fitting) if fitting else None


@dataclass
class CapturedEncoderGraph:
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
        self.encoder = encoder
        self.hop = int(hop)
        self.num_quantizers = int(num_quantizers)
        self.bucket_frames = tuple(sorted({int(f) for f in bucket_frames}))
        self.stream = stream
        param = next(encoder.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self.graphs: dict[int, CapturedEncoderGraph] = {}
        self.pool: Any | None = None
        self.disable_reason: str | None = None
        self.replays = 0
        self.misses = 0

    def capture(self) -> None:
        graphs: dict[int, CapturedEncoderGraph] = {}
        try:
            with torch.cuda.device(self.device):
                pool = torch.cuda.graph_pool_handle()
                # note(ratish): largest first so the shared pool is sized once.
                for frames in reversed(self.bucket_frames):
                    graphs[frames] = self.capture_bucket(frames, pool)
        except Exception as exc:
            self.disable_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3-TTS reference encoder graph capture disabled the runner: %s",
                self.disable_reason,
                exc_info=True,
            )
            return
        self.graphs = {frames: graphs[frames] for frames in self.bucket_frames}
        self.pool = pool
        logger.info(
            "Qwen3-TTS reference encoder graphs captured for %s frames",
            list(self.bucket_frames),
        )

    def capture_bucket(self, frames: int, pool: Any) -> CapturedEncoderGraph:
        static_input = torch.zeros(
            (1, 1, frames * self.hop), device=self.device, dtype=self.dtype
        )
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for _ in range(2):
                self._encode(static_input)
        graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(
                graph,
                pool=pool,
                stream=self.stream,
                capture_error_mode="thread_local",
            ),
        ):
            static_codes = self._encode(static_input)
        self.stream.synchronize()
        return CapturedEncoderGraph(
            graph=graph, static_input=static_input, static_codes=static_codes
        )

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(
            values, num_quantizers=self.num_quantizers, return_dict=True
        ).audio_codes

    def bucket_for(self, frames: int) -> int | None:
        return smallest_bucket(frames, self.graphs)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """Codes (frames, quantizers) of a waveform (samples,); None above the largest bucket."""
        samples = waveform.numel()
        frames = -(-samples // self.hop)
        bucket = self.bucket_for(frames)
        if bucket is None:
            self.misses += 1
            return None
        else:
            pass
        captured = self.graphs[bucket]
        captured.static_input[0, 0, :samples].copy_(waveform)
        captured.static_input[0, 0, samples:].zero_()
        captured.graph.replay()
        self.replays += 1
        # note(ratish): the graph rewrites its output on the next replay.
        return captured.static_codes[0, :, :frames].transpose(0, 1).clone()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.graphs),
            "disable_reason": self.disable_reason,
            "bucket_frames": list(self.bucket_frames),
            "captured": list(self.graphs),
            "replays": self.replays,
            "misses": self.misses,
        }


__all__ = [
    "DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES",
    "Qwen3TTSReferenceEncoderCudaGraphRunner",
    "move_conv_padding_to_host",
    "smallest_bucket",
]
