# SPDX-License-Identifier: Apache-2.0
"""Graph replay for one AuK Euler step, over bucketed padded shapes.

A DiT step is launch-bound at small batch. Block fusion cuts it from ~2400
kernel launches to ~600, yet at batch size 1 the GPU still idles for about half
the step, so the launch gap and not the math sets the step time. Everything a
step reads except the noised latent and the timestep -- text conditioning,
reference latent, padding masks, rope positions -- is fixed for a whole
trajectory, so one captured graph serves every NFE step, and every later
request whose padded shape rounds to the same bucket.

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
from typing import Any, NamedTuple

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend

logger = logging.getLogger(__name__)

# Rounding granularity in rows for the three padded axes. Coarse enough that a
# varied corpus reuses graphs: over a 288-request SeedTTS-en sweep, 16/16/8 asked
# for 61 shapes and got 16, while 32/64/64 asks for 25 and gets all of them. The
# waste that buys is nearly free where graphs are used, see MAX_GRAPH_BATCH.
FRAME_BUCKET = 32
REF_BUCKET = 64
TEXT_BUCKET = 64

# A step is launch-bound only at small batch: measured on H200, a graph takes
# batch 1 from 15.9 to 7.8 ms/step but batch 8 only from 38.4 to 36.8, less than
# the padding it needs costs there. Above this the caller runs the step eager and
# leaves the cache for the shapes that gain from it.
MAX_GRAPH_BATCH = 2

# Step signature the runner captures: (constant inputs, timestep, latent).
StepFn = Callable[[Mapping[str, Any], torch.Tensor, torch.Tensor], torch.Tensor]


class _CapturedStep(NamedTuple):
    """A recorded step and the buffers a replay reads from and writes into.

    ``graph`` is the platform's graph object, so it is typed by the backend that
    recorded it rather than by torch.cuda.
    """

    graph: Any
    static_inputs: dict[str, Any]
    static_x: torch.Tensor
    static_time: torch.Tensor
    static_out: torch.Tensor


def round_to_bucket(length: int, bucket: int) -> int:
    """Round ``length`` up to a multiple of ``bucket``; an empty axis stays empty."""
    if length <= 0:
        return 0
    return (length + bucket - 1) // bucket * bucket


def build_step_graph_runner(device: torch.device) -> AuKStepCudaGraphRunner | None:
    """The runner for ``device``, or None where the platform records no graphs."""
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        return None
    return AuKStepCudaGraphRunner(backend=backend, device=device)


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
        max_graph_batch: int = MAX_GRAPH_BATCH,
        max_graphs: int = 32,
        min_free_gb: float = 4.0,
        warmup_iters: int = 3,
    ) -> None:
        self._backend = backend
        self._device = device
        self._module = torch.get_device_module(device)
        self._buckets = (frame_bucket, ref_bucket, text_bucket)
        if min(self._buckets) < 1:
            raise ValueError("AuK DiT graph buckets must be positive")
        if max_graph_batch < 1 or max_graphs < 1:
            raise ValueError("AuK DiT graph batch and cache limits must be positive")
        if warmup_iters < 1:
            raise ValueError("AuK DiT graph capture needs a warmup iteration")
        self._max_graph_batch = max_graph_batch
        self._max_graphs = max_graphs
        self._min_free_bytes = int(min_free_gb * 1024**3)
        self._warmup_iters = warmup_iters
        self._graphs: dict[tuple, _CapturedStep] = {}
        self._rejected: set[tuple] = set()
        self._pool: Any | None = None
        self._graph_bytes = 0

    def pad_lengths(
        self, *, frames: int, ref: int, text: int, batch: int
    ) -> tuple[int, int, int] | None:
        """Bucketed row counts for the three padded axes, or None for no padding.

        None means this batch is too wide to be worth a graph, so the caller
        should neither pad it nor try to bind it.
        """
        if batch > self._max_graph_batch:
            return None
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
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._prepare(key, step, inputs, x, time)
        if entry is None:
            return None
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                entry.static_inputs[name].copy_(value)

        def replay(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            entry.static_time.copy_(t)
            entry.static_x.copy_(x)
            entry.graph.replay()
            # The next replay overwrites this buffer, so the caller gets a copy.
            return entry.static_out.clone()

        return replay

    def _key(
        self, inputs: Mapping[str, Any], x: torch.Tensor, baked: Sequence[Any]
    ) -> tuple:
        parts: list[Any] = [(tuple(x.shape), x.dtype)]
        for name, value in sorted(inputs.items()):
            if isinstance(value, torch.Tensor):
                parts.append((name, tuple(value.shape), value.dtype))
            else:
                parts.append((name, value))
        return (tuple(parts), tuple(baked))

    def _prepare(
        self,
        key: tuple,
        step: StepFn,
        inputs: Mapping[str, Any],
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> _CapturedStep | None:
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
        # A capture's tensors live in the graph's own pool, which the allocator
        # totals do not follow, and a warm server usually has cached blocks to
        # lend it. The device free delta is the closest measure left, and it
        # reads high if another tenant allocates during the capture.
        self._graph_bytes += max(0, free - self._module.mem_get_info(self._device)[0])
        logger.info(
            "Captured AuK DiT step graph x=%s (%d cached, %.0fMiB new device memory)",
            tuple(x.shape),
            len(self._graphs),
            self._graph_bytes / 2**20,
        )
        return entry

    def _capture(
        self,
        step: StepFn,
        inputs: Mapping[str, Any],
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> _CapturedStep:
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
        return _CapturedStep(graph, statics, static_x, static_time, static_out)


__all__ = [
    "AuKStepCudaGraphRunner",
    "build_step_graph_runner",
    "round_to_bucket",
]
