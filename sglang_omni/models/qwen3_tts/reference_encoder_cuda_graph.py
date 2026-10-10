# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the Qwen3-TTS reference encoder at bucketed lengths."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

import torch
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1d

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph
from sglang_omni.utils.device import device_guard

logger = logging.getLogger(__name__)

# note(ratish): a replay costs a floor plus a small per frame term and a key costs
# 32 MiB, so keys below 32 frames save nothing measurable; the step keeps the padding
# under half a clip, and clips past 256 frames run eager.
DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES = (32, 48, 64, 96, 128, 192, 256)


class ReferenceEncoderGraphStats(TypedDict):
    enabled: bool
    disable_reason: str | None
    bucket_frames: list[int]
    captured: list[int]
    replays: int
    misses: int


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
    graph: ReplayableGraph
    static_input: torch.Tensor
    static_codes: torch.Tensor


class Qwen3TTSReferenceEncoderCudaGraphRunner:
    """One captured encode per bucket length, batch 1, replayed on the current stream."""

    def __init__(
        self,
        encoder: MimiModel,
        *,
        hop: int,
        num_quantizers: int,
        bucket_frames: Iterable[int],
        stream: torch.Stream,
    ) -> None:
        self.encoder = encoder
        self.hop = int(hop)
        self.num_quantizers = int(num_quantizers)
        self.bucket_frames = tuple(sorted({int(f) for f in bucket_frames}))
        self.stream = stream
        param = next(encoder.parameters())
        self.device = param.device
        self.device_module = torch.get_device_module(self.device)
        self.graph_backend: DeviceGraphBackend | None = (
            current_platform.get_device_graph_backend(self.device)
        )
        self.dtype = param.dtype
        self.graphs: dict[int, CapturedEncoderGraph] = {}
        self.pool: tuple[int, int] | None = None
        self.disable_reason: str | None = None
        self.replays = 0
        self.misses = 0

    def capture(self) -> None:
        if self.graph_backend is None:
            self.disable_reason = f"no graph backend on {self.device.type}"
            return
        else:
            pass
        graphs: dict[int, CapturedEncoderGraph] = {}
        try:
            with device_guard(self.device):
                pool = self.graph_backend.graph_pool_handle()
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

    def capture_bucket(
        self, frames: int, pool: tuple[int, int]
    ) -> CapturedEncoderGraph:
        static_input = torch.zeros(
            (1, 1, frames * self.hop), device=self.device, dtype=self.dtype
        )
        self.stream.wait_stream(self.device_module.current_stream(self.device))
        with torch.inference_mode(), self.device_module.stream(self.stream):
            for _ in range(2):
                self._encode(static_input)
        with (
            torch.inference_mode(),
            self.graph_backend.capture(
                pool=pool,
                stream=self.stream,
                thread_local_errors=True,
            ) as graph,
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

    def stats(self) -> ReferenceEncoderGraphStats:
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
