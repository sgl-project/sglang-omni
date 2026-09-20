# SPDX-License-Identifier: Apache-2.0
"""Graph replay for one AuK Euler step, over a declared list of padded shapes.

A DiT step is launch-bound at small batch. Block fusion cuts it from ~2400
kernel launches to ~600, yet at batch size 1 the GPU still idles for about half
the step, so the launch gap and not the math sets the step time. Everything a
step reads except the noised latent and the timestep -- text conditioning,
reference latent, padding masks, rope positions -- is fixed for a whole
trajectory, so one captured graph serves every NFE step, and every later
request whose shape fits the same declared shape.

Padding is what lets one capture serve many requests. Target frames, reference
frames, and text tokens each round up to a declared shape that covers them,
with the real lengths carried in the masks the backbone already builds: the
-inf attention bias makes padded keys contribute exact zeros and the conv
position embedding masks its own input, so a padded step computes what the
unpadded one does plus waste. Requests already pad to the in-batch maximum
whenever the sampling batch holds more than one item.

Captures happen once, at startup, over the declared shapes. Entering a capture
synchronizes the device and empties the allocator cache, which is not something
to do underneath a request while the conditioning and decode stages share the
process; a request whose shape no captured graph covers runs the step eagerly,
and unpadded, since padding only pays for itself when it buys a replay.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend

logger = logging.getLogger(__name__)


class AuKGraphShape(NamedTuple):
    """One declared capture: a batch size and the three padded row counts."""

    batch: int
    frames: int
    ref: int
    text: int


# Target frames a capture is declared for. Dense where a step is launch-bound
# and a graph roughly halves it, and stopping at 15s because a longer step
# spends long enough inside its kernels that padding up to the next rung costs
# more than the launches the replay saves.
_FRAME_LADDER = (192, 320, 448, 576, 768)

# (reference frames, text tokens) a capture is declared for: a request with no
# reference audio carries the instruction alone, one cloning a voice carries
# that reference's own tokens alongside it. A request that fits neither -- a
# long reference, say -- runs eager rather than padding to a shape this wide.
_CONDITIONING = ((0, 192), (320, 384))

DEFAULT_CAPTURE_SHAPES: tuple[AuKGraphShape, ...] = tuple(
    AuKGraphShape(batch, frames, ref, text)
    # Batch sizes a capture is declared for. A wide batch is not launch-bound:
    # measured on H200, a graph takes batch 1 from 15.9 to 7.8 ms/step but
    # batch 8 only from 38.4 to 36.8, less than the padding it needs costs.
    for batch in (1, 2)
    for frames in _FRAME_LADDER
    for ref, text in _CONDITIONING
)

# Step signature the runner captures: (constant inputs, timestep, latent).
StepFn = Callable[[Mapping[str, Any], torch.Tensor, torch.Tensor], torch.Tensor]


class CapturedStep(NamedTuple):
    """A recorded step and the buffers a replay reads from and writes into.

    The graph is the platform's own object, so it is typed by the backend that
    recorded it rather than by torch.cuda.
    """

    graph: Any
    static_inputs: dict[str, Any]
    static_x: torch.Tensor
    static_time: torch.Tensor
    static_out: torch.Tensor


def verify_capture_shapes(
    shapes: Iterable[Sequence[int]],
) -> tuple[AuKGraphShape, ...]:
    """Normalize declared shapes, cheapest first, so a lookup takes the closest.

    Ordering is by padded rows rather than by any one axis: the rows are what a
    step pays for, and the first declared shape that covers a request is then
    the least wasteful one that does.
    """
    verified = set()
    for shape in shapes:
        shape = AuKGraphShape(*shape)
        if shape.batch < 1 or shape.frames < 1 or shape.text < 1 or shape.ref < 0:
            raise ValueError(
                "AuK DiT graph capture shapes need a positive batch, frame and "
                f"text count and a non-negative reference count; got {shape!r}"
            )
        verified.add(shape)
    if not verified:
        raise ValueError("AuK DiT graph capture shapes must not be empty")
    return tuple(
        sorted(verified, key=lambda s: (s.batch, s.frames + s.ref + s.text, s))
    )


def build_step_graph_runner(
    device: torch.device,
    capture_shapes: Iterable[Sequence[int]] | None = None,
) -> AuKStepCudaGraphRunner | None:
    """The runner for a device, or None where the platform records no graphs."""
    backend = current_platform.get_device_graph_backend(device)
    if backend is None:
        return None
    return AuKStepCudaGraphRunner(
        backend=backend,
        device=device,
        capture_shapes=capture_shapes,
    )


class AuKStepCudaGraphRunner:
    """Capture one Euler step per declared shape, replay it for every NFE step."""

    def __init__(
        self,
        *,
        backend: DeviceGraphBackend,
        device: torch.device,
        capture_shapes: Iterable[Sequence[int]] | None = None,
        min_free_gb: float = 4.0,
        warmup_iters: int = 3,
    ) -> None:
        self._backend = backend
        self._device = device
        self._module = torch.get_device_module(device)
        self._declared = verify_capture_shapes(
            DEFAULT_CAPTURE_SHAPES if capture_shapes is None else capture_shapes
        )
        if warmup_iters < 1:
            raise ValueError("AuK DiT graph capture needs a warmup iteration")
        self._min_free_bytes = int(min_free_gb * 1024**3)
        self._warmup_iters = warmup_iters
        self._graphs: dict[tuple, CapturedStep] = {}
        # The declared shapes that hold a graph, and the one being captured now.
        self._ready: set[AuKGraphShape] = set()
        self._capturing: AuKGraphShape | None = None
        self._pool: Any | None = None
        self._graph_bytes = 0

    def capture_declared(self, run_trajectory: Callable[[AuKGraphShape], Any]) -> None:
        """Capture every declared shape by running one trajectory through each.

        The trajectories come from the caller so that a capture records exactly
        what a request runs, down to the padded input shapes and the sampling
        settings baked into the step. A shape that fails to capture is simply
        left out, and the requests that would have used it run eager.
        """
        started = time.perf_counter()
        for shape in self._declared:
            self._capturing = shape
            try:
                run_trajectory(shape)
            except Exception as exc:
                # The trajectory, not just the capture inside it, can fail: a
                # declared shape is a list the caller may set, and one too wide
                # for the device should cost that shape rather than startup.
                logger.warning(
                    "AuK DiT step graph: %r did not capture (%s); it will run eager",
                    shape,
                    exc,
                )
            finally:
                self._capturing = None
        logger.info(
            "AuK DiT step graphs: captured %d of %d declared shapes in %.1fs "
            "(%.0fMiB new device memory)",
            len(self._ready),
            len(self._declared),
            time.perf_counter() - started,
            self._graph_bytes / 2**20,
        )

    def pad_lengths(
        self, *, frames: int, ref: int, text: int, batch: int
    ) -> tuple[int, int, int] | None:
        """Padded row counts for the three padded axes, or None for no padding.

        None means no captured graph covers this batch, so the caller should
        neither pad it nor try to bind it: an eager step is cheaper unpadded.
        """
        shape = self._capturing or self.fit(
            frames=frames, ref=ref, text=text, batch=batch
        )
        if shape is None:
            return None
        return (shape.frames, shape.ref, shape.text)

    def fit(
        self, *, frames: int, ref: int, text: int, batch: int
    ) -> AuKGraphShape | None:
        """The cheapest captured shape that covers this batch on every axis."""
        for shape in self._declared:
            if (
                shape in self._ready
                and shape.batch == batch
                and shape.frames >= frames
                and shape.ref >= ref
                and shape.text >= text
            ):
                return shape
        return None

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

        baked names the values the step closes over rather than reads from
        its inputs -- they select a different computation, so they belong in the
        key. None means the caller runs the step eagerly instead.
        """
        key = self.graph_key(inputs, x, baked)
        entry = self._graphs.get(key)
        if entry is None:
            # Outside capture_declared a miss stays a miss: a capture costs
            # the whole device a synchronization and its allocator cache.
            if self._capturing is None:
                return None
            entry = self.prepare(key, step, inputs, x, time)
            if entry is None:
                return None
            self._ready.add(self._capturing)
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

    def graph_key(
        self, inputs: Mapping[str, Any], x: torch.Tensor, baked: Sequence[Any]
    ) -> tuple:
        parts: list[Any] = [(tuple(x.shape), x.dtype)]
        for name, value in sorted(inputs.items()):
            if isinstance(value, torch.Tensor):
                parts.append((name, tuple(value.shape), value.dtype))
            else:
                parts.append((name, value))
        return (tuple(parts), tuple(baked))

    def prepare(
        self,
        key: tuple,
        step: StepFn,
        inputs: Mapping[str, Any],
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> CapturedStep | None:
        free, _ = self._module.mem_get_info(self._device)
        if free < self._min_free_bytes:
            logger.warning(
                "AuK DiT step graph: free memory %.1fGB below the %.1fGB "
                "headroom; x=%s will run eager",
                free / 1024**3,
                self._min_free_bytes / 1024**3,
                tuple(x.shape),
            )
            return None
        try:
            with self._module.device(self._device):
                entry = self.capture(step, inputs, x, time)
        except Exception as exc:
            logger.warning(
                "AuK DiT step graph capture failed for x=%s: %s; "
                "this shape will run eager",
                tuple(x.shape),
                exc,
            )
            return None
        self._graphs[key] = entry
        # A capture's tensors live in the graph's own pool, which the allocator
        # totals do not follow, and a warm server usually has cached blocks to
        # lend it. The device free delta is the closest measure left, and it
        # reads high if another tenant allocates during the capture.
        self._graph_bytes += max(0, free - self._module.mem_get_info(self._device)[0])
        logger.debug(
            "Captured AuK DiT step graph x=%s (%d cached, %.0fMiB new device memory)",
            tuple(x.shape),
            len(self._graphs),
            self._graph_bytes / 2**20,
        )
        return entry

    def capture(
        self,
        step: StepFn,
        inputs: Mapping[str, Any],
        x: torch.Tensor,
        time: torch.Tensor,
    ) -> CapturedStep:
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
        return CapturedStep(graph, statics, static_x, static_time, static_out)


__all__ = [
    "DEFAULT_CAPTURE_SHAPES",
    "AuKGraphShape",
    "AuKStepCudaGraphRunner",
    "build_step_graph_runner",
    "verify_capture_shapes",
]
