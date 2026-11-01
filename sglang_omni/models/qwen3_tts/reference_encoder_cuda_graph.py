# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the Qwen3-TTS reference encoder at bucketed lengths."""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
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
# note (ratish): a kernel with several configs benchmarks them on its first call and keeps the fastest, so its pick can change between compiles, and a split LayerNorm's combine sums in the picked order, which moves codes; pointwise kernels get one heuristic config, and an unsplit LayerNorm gets one persistent config, so no kernel is picked by timing.
REFERENCE_ENCODER_INDUCTOR_OPTIONS: dict[str, bool] = {
    "triton.autotune_pointwise": False,
    "split_reductions": False,
}


class ReferenceEncoderGraphStats(TypedDict):
    enabled: bool
    disable_reason: str | None
    bucket_frames: list[int]
    captured: list[int]
    replays: int
    misses: int


class ConvForward(Protocol):
    def __call__(
        self, hidden_states: torch.Tensor, padding_cache: None = None
    ) -> torch.Tensor: ...


def integer_padding_forward(conv: MimiConv1d) -> ConvForward:
    """The causal conv's forward with Mimi's padding arithmetic on Python integers."""
    stride = int(conv.conv.stride[0])
    kernel_size = (int(conv.conv.kernel_size[0]) - 1) * int(conv.conv.dilation[0]) + 1
    padding_left = kernel_size - stride

    def forward(
        hidden_states: torch.Tensor, padding_cache: None = None
    ) -> torch.Tensor:
        # note (ratish): Mimi's ceil division, not -length % stride, which on a symbolic length took Inductor 82 s instead of 8 s to build the encode; torch copies the input even for zero widths.
        length = hidden_states.shape[-1]
        frames = -(-(length - kernel_size + padding_left) // stride)
        padding_right = frames * stride + kernel_size - padding_left - length
        if padding_left or padding_right:
            hidden_states = F.pad(
                hidden_states, (padding_left, padding_right), mode=conv.pad_mode
            )
        else:
            pass
        return conv.conv(hidden_states)

    return forward


def pad_convs_by_integers(encoder: torch.nn.Module) -> int:
    """Give every conv a forward that pads by Python integers; returns the conv count."""
    # note (ratish): Mimi computes each conv's padding from tensor buffers, which waits on the device per conv and breaks a compiled graph.
    count = 0
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.forward = integer_padding_forward(module)
            count += 1
        else:
            pass
    return count


def pad_tail(
    clip_samples: torch.Tensor,
    samples_per_step: int,
    conv: MimiConv1d,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Fill a strided conv's input past the clip's steps as the conv pads the clip."""
    hidden_states = inputs[0]
    clip_steps = (clip_samples + samples_per_step - 1) // samples_per_step
    is_inside_clip = (
        torch.arange(hidden_states.shape[-1], device=hidden_states.device) < clip_steps
    )
    if conv.pad_mode == "replicate":
        tail_values = hidden_states.index_select(-1, (clip_steps - 1).view(1))
    else:
        tail_values = 0.0
    return (torch.where(is_inside_clip, hidden_states, tail_values), *inputs[1:])


@contextlib.contextmanager
def exact_length_encode(
    encoder: torch.nn.Module, clip_samples: torch.Tensor
) -> Iterator[None]:
    """Inside the block, the encoder gives the codes of its input's first clip_samples samples at their exact length.

    The hooks act on every forward of the encoder inside the block, so the block must hold the encoder's only caller.
    """
    # note (ratish): at the exact length each strided conv pads past the clip in its own mode, and a stride-s conv yields ceil(L / s) steps, so the clip covers its samples over the strides before the conv, rounded up.
    hook_handles: list[RemovableHandle] = []
    samples_per_step = 1
    try:
        for conv in encoder.modules():
            if isinstance(conv, MimiConv1d) and conv.conv.stride[0] > 1:
                hook_handles.append(
                    conv.register_forward_pre_hook(
                        functools.partial(pad_tail, clip_samples, samples_per_step)
                    )
                )
                samples_per_step *= conv.conv.stride[0]
            else:
                pass
        yield
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()


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
        # note (ratish): one program serves every bucket, so a boot compiles once; it keeps the exact-length hooks it was traced with, since module hooks are not guarded, so it runs only inside exact_length_encode, and the eager encode past the ladder stays uncompiled.
        self.compiled_encode = (
            torch.compile(
                self._encode,
                fullgraph=True,
                options=dict(REFERENCE_ENCODER_INDUCTOR_OPTIONS),
            )
            if current_platform.is_cuda()
            else self._encode
        )
        # note (ratish): every bucket's graph reads the clip's samples from here, set before each replay.
        self.clip_samples = torch.zeros((), dtype=torch.long, device=self.device)
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
        # note (ratish): a hint, so the length starts symbolic and every bucket reuses the first bucket's program.
        dynamo.maybe_mark_dynamic(static_input, 2)
        self.clip_samples.fill_(frames * self.hop)
        self.stream.wait_stream(self.device_module.current_stream(self.device))
        with exact_length_encode(self.encoder, self.clip_samples):
            with torch.inference_mode(), self.device_module.stream(self.stream):
                for _ in range(2):
                    self.compiled_encode(static_input)
            with (
                torch.inference_mode(),
                self.graph_backend.capture(
                    pool=pool,
                    stream=self.stream,
                    thread_local_errors=True,
                ) as graph,
            ):
                static_codes = self.compiled_encode(static_input)
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
        # note (ratish): the captured tail padding keeps the last clip's samples past this one out of every code.
        self.clip_samples.fill_(samples)
        captured.static_input[0, 0, :samples].copy_(waveform)
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
    "exact_length_encode",
    "pad_convs_by_integers",
    "smallest_bucket",
]
