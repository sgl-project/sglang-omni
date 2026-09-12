# SPDX-License-Identifier: Apache-2.0
"""Graph replay for one AuK Euler step, over bucketed padded shapes.

A DiT step is launch-bound: after block fusion it still issues ~2k kernels of a
few microseconds each, so at batch size 1 the launch gap, not the math, sets the
step time. Everything a step reads except the noised latent and the timestep --
text conditioning, reference latent, padding masks, rope positions -- is fixed
for a whole trajectory, so one captured graph serves every NFE step, and every
later request whose padded shape rounds to the same bucket.

Padding is what keeps that shape space small enough to capture. Target frames,
reference frames, and text tokens each round up to a bucket multiple, with the
real lengths carried in the masks the backbone already builds: the -inf
attention bias makes padded keys contribute exact zeros and the conv position
embedding masks its own input, so a padded step computes what the unpadded one
does plus waste. Requests already pad to the in-batch maximum whenever the
sampling batch holds more than one item.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend

logger = logging.getLogger(__name__)

# Rounding granularity in rows for the three padded axes. Fine buckets keep the
# padded waste below the launch gap they buy: measured on H200, 64/64/32 inflates
# the joint sequence ~29% and loses to eager at batch size 8, while 16/16/8 wins
# at every batch size.
FRAME_BUCKET = 16
REF_BUCKET = 16
TEXT_BUCKET = 8

# Step signature the runner captures: (constant inputs, timestep, latent).
StepFn = Callable[[Mapping[str, Any], torch.Tensor, torch.Tensor], torch.Tensor]


def round_to_bucket(length: int, bucket: int) -> int:
    """Round ``length`` up to a multiple of ``bucket``; an empty axis stays empty."""
    if length <= 0:
        return 0
    return -(-length // bucket) * bucket


def build_step_graph_runner(
    device: torch.device, **kwargs: Any
) -> AuKStepCudaGraphRunner | None:
    """The runner for ``device``, or None where the platform records no graphs."""
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        return None
    return AuKStepCudaGraphRunner(backend=backend, device=device, **kwargs)


class AuKStepCudaGraphRunner:
    """Capture one Euler step per padded shape, replay it for every NFE step."""

    def __init__(
        self,
        *,
        backend: DeviceGraphBackend,
        device: torch.device,
        frame_bucket: int = FRAME_BUCKET,
        ref_bucket: int = REF_BUCKET,
        text_bucket: int = TEXT_BUCKET,
        max_graphs: int = 16,
        min_free_gb: float = 4.0,
        warmup_iters: int = 3,
    ) -> None:
        self._backend = backend
        self._device = device
        self._module = torch.get_device_module(device)
        self._buckets = (frame_bucket, ref_bucket, text_bucket)
        if min(self._buckets) < 1:
            raise ValueError("AuK DiT graph buckets must be positive")
        self._max_graphs = max(int(max_graphs), 1)
        self._min_free_bytes = int(float(min_free_gb) * (1024**3))
        self._warmup_iters = int(warmup_iters)
        self._graphs: dict[tuple, tuple] = {}
        self._rejected: set[tuple] = set()
        self._pool: Any | None = None
        self.graph_bytes = 0

    def pad_lengths(self, *, frames: int, ref: int, text: int) -> tuple[int, int, int]:
        """Bucketed row counts for the three padded axes of a sampling batch."""
        frame_bucket, ref_bucket, text_bucket = self._buckets
        return (
            round_to_bucket(frames, frame_bucket),
            round_to_bucket(ref, ref_bucket),
            round_to_bucket(text, text_bucket),
        )

    def bind(
        self,
        step: StepFn,
        inputs: Mapping[str, Any],
        *,
        x: torch.Tensor,
        time: torch.Tensor,
        baked: Sequence[Any] = (),
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None:
        """Load this trajectory's constants into a captured step, or return None.

        ``baked`` names the values ``step`` closes over rather than reads from
        ``inputs`` -- they select a different computation, so they belong in the
        key. None means the caller runs the step eagerly instead.
        """
        key = self._key(inputs, x, baked)
        if key in self._rejected:
            return None
        entry = self._graphs.get(key) or self._prepare(key, step, inputs, x, time)
        if entry is None:
            return None
        graph, statics, static_x, static_time, static_out = entry
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                statics[name].copy_(value)

        def replay(time: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            static_time.copy_(time)
            static_x.copy_(x)
            graph.replay()
            return static_out.clone()

        return replay

    def _key(
        self, inputs: Mapping[str, Any], x: torch.Tensor, baked: Sequence[Any]
    ) -> tuple:
        shapes: list[Any] = [(tuple(x.shape), x.dtype)]
        for name, value in sorted(inputs.items()):
            if isinstance(value, torch.Tensor):
                shapes.append((name, tuple(value.shape), value.dtype))
            else:
                shapes.append((name, value))
        return (tuple(shapes), tuple(baked))

    def _prepare(
        self,
        key: tuple,
        step: StepFn,
        inputs: Mapping[str, Any],
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple | None:
        if len(self._graphs) >= self._max_graphs:
            logger.warning(
                "AuK DiT step graph: %d shapes captured (max_graphs); "
                "running x=%s eager",
                self._max_graphs,
                tuple(x.shape),
            )
            self._rejected.add(key)
            return None
        free, _ = self._module.mem_get_info(self._device)
        if free < self._min_free_bytes:
            logger.warning(
                "AuK DiT step graph: free memory %.1fGB below the %.1fGB "
                "headroom; running x=%s eager",
                free / 1024**3,
                self._min_free_bytes / 1024**3,
                tuple(x.shape),
            )
            self._rejected.add(key)
            return None
        try:
            with self._module.device(self._device):
                entry = self._capture(step, inputs, x, time)
        except Exception as exc:
            logger.warning(
                "AuK DiT step graph capture failed for x=%s: %s; "
                "running this shape eager",
                tuple(x.shape),
                exc,
            )
            self._rejected.add(key)
            return None
        self._graphs[key] = entry
        # A graph's tensors live in its own pool, which the allocator totals do
        # not follow, so the device's own free count is what measures a capture.
        self.graph_bytes += max(0, free - self._module.mem_get_info(self._device)[0])
        logger.info(
            "Captured AuK DiT step graph x=%s (%d cached, %.0fMiB total)",
            tuple(x.shape),
            len(self._graphs),
            self.graph_bytes / 2**20,
        )
        return entry

    def _capture(
        self,
        step: StepFn,
        inputs: Mapping[str, Any],
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple:
        statics = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        static_x, static_time = x.clone(), time.clone()
        # Warm up on a side stream: the block compile and the allocator blocks a
        # first call needs must both be settled before the capture records.
        stream = self._module.Stream(device=self._device)
        stream.wait_stream(self._module.current_stream(self._device))
        with self._module.stream(stream):
            for _ in range(self._warmup_iters):
                step(statics, static_time, static_x)
        self._module.current_stream(self._device).wait_stream(stream)
        self._module.synchronize(self._device)

        if self._pool is None:
            self._pool = self._module.graph_pool_handle()
        # thread_local: the conditioning and decode stages launch on their own
        # streams in this process and must not poison this thread's capture.
        with self._backend.capture(pool=self._pool, thread_local_errors=True) as graph:
            static_out = step(statics, static_time, static_x)
        self._module.synchronize(self._device)
        return graph, statics, static_x, static_time, static_out


__all__ = [
    "AuKStepCudaGraphRunner",
    "build_step_graph_runner",
    "round_to_bucket",
]
