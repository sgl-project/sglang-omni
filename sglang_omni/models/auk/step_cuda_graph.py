# SPDX-License-Identifier: Apache-2.0
"""Graph replay for one AuK Euler step, over a declared list of padded shapes.

Everything a step reads except the noised latent and the timestep is constant
for a whole trajectory, so one capture per declared shape serves every NFE step
and every later request that shape covers.
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


# note(Dayuxiaoshui): the ladder stops at 15s because a longer step spends long
# enough inside its kernels that padding up to the next rung costs more than
# the launches a replay saves.
FRAME_LADDER = (192, 320, 448, 576, 768)

CONDITIONING = ((0, 192), (320, 384))

# note(Dayuxiaoshui): only narrow batches are declared. A wide batch already
# spends long enough inside each kernel to hide the issue time a replay saves,
# so the graph buys it less than the padding it needs costs.
DEFAULT_CAPTURE_SHAPES: tuple[AuKGraphShape, ...] = tuple(
    AuKGraphShape(batch, frames, ref, text)
    for batch in (1, 2)
    for frames in FRAME_LADDER
    for ref, text in CONDITIONING
)

StepFn = Callable[[Mapping[str, Any], torch.Tensor, torch.Tensor], torch.Tensor]


class CapturedStep(NamedTuple):
    """A recorded step and the buffers a replay reads from and writes into.

    The graph is typed by the backend that recorded it rather than torch.cuda.
    """

    graph: Any
    static_inputs: dict[str, Any]
    static_x: torch.Tensor
    static_time: torch.Tensor
    static_out: torch.Tensor


def verify_capture_shapes(
    shapes: Iterable[Sequence[int]],
) -> tuple[AuKGraphShape, ...]:
    """Normalize declared shapes, ordered by padded rows so a lookup takes the cheapest."""
    verified = set()
    for shape in shapes:
        shape = AuKGraphShape(*shape)
        if shape.batch < 1 or shape.frames < 1 or shape.text < 1 or shape.ref < 0:
            raise ValueError(
                "AuK DiT graph capture shapes need a positive batch, frame and "
                f"text count and a non-negative reference count; got {shape!r}"
            )
        else:
            pass
        verified.add(shape)
    if not verified:
        raise ValueError("AuK DiT graph capture shapes must not be empty")
    else:
        pass
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
    else:
        pass
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
        self.backend = backend
        self.device = device
        self.module = torch.get_device_module(device)
        self.declared = verify_capture_shapes(
            DEFAULT_CAPTURE_SHAPES if capture_shapes is None else capture_shapes
        )
        if warmup_iters < 1:
            raise ValueError("AuK DiT graph capture needs a warmup iteration")
        else:
            pass
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self.warmup_iters = warmup_iters
        self.graphs: dict[tuple, CapturedStep] = {}
        self.ready: set[AuKGraphShape] = set()
        self.capturing: AuKGraphShape | None = None
        self.pool: Any | None = None
        self.graph_bytes = 0

    def capture_declared(self, run_trajectory: Callable[[AuKGraphShape], Any]) -> None:
        """Capture every declared shape by running one trajectory through each.

        The trajectories come from the caller so that a capture records exactly
        what a request runs. A shape that fails is left out and runs eager.
        """
        started = time.perf_counter()
        for shape in self.declared:
            self.capturing = shape
            try:
                run_trajectory(shape)
            except Exception as exc:
                # note(Dayuxiaoshui): the trajectory, not just the capture
                # inside it, can fail, and a declared shape too wide for the
                # device should cost that shape rather than startup.
                logger.warning(
                    f"AuK DiT step graph: {shape!r} did not capture ({exc}); "
                    "it will run eager"
                )
            finally:
                self.capturing = None
        logger.info(
            f"AuK DiT step graphs: captured {len(self.ready)} of "
            f"{len(self.declared)} declared shapes in "
            f"{time.perf_counter() - started:.1f}s "
            f"({self.graph_bytes / 2**20:.0f}MiB new device memory)"
        )

    def pad_lengths(
        self, *, frames: int, ref: int, text: int, batch: int
    ) -> tuple[int, int, int] | None:
        """Padded row counts for the three padded axes, or None for no padding.

        None means no captured graph covers this batch, so the caller should
        neither pad it nor try to bind it: an eager step is cheaper unpadded.
        """
        shape = self.capturing or self.fit(
            frames=frames, ref=ref, text=text, batch=batch
        )
        if shape is None:
            return None
        else:
            pass
        return (shape.frames, shape.ref, shape.text)

    def fit(
        self, *, frames: int, ref: int, text: int, batch: int
    ) -> AuKGraphShape | None:
        """The cheapest captured shape that covers this batch on every axis."""
        for shape in self.declared:
            if (
                shape in self.ready
                and shape.batch == batch
                and shape.frames >= frames
                and shape.ref >= ref
                and shape.text >= text
            ):
                return shape
            else:
                pass
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

        baked names the values the step closes over rather than reads from its
        inputs, so they belong in the key. None means the caller runs eagerly.
        """
        key = self.graph_key(inputs, x, baked)
        entry = self.graphs.get(key)
        if entry is None:
            # note(Dayuxiaoshui): outside capture_declared a miss stays a miss,
            # because a capture costs the whole device a synchronization and
            # its allocator cache.
            if self.capturing is None:
                return None
            else:
                pass
            entry = self.prepare(key, step, inputs, x, time)
            if entry is None:
                return None
            else:
                pass
            self.ready.add(self.capturing)
        else:
            pass
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                entry.static_inputs[name].copy_(value)
            else:
                pass

        def replay(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            entry.static_time.copy_(t)
            entry.static_x.copy_(x)
            entry.graph.replay()
            # note(Dayuxiaoshui): the next replay overwrites this buffer, so
            # the caller gets a copy.
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
        free, _ = self.module.mem_get_info(self.device)
        if free < self.min_free_bytes:
            logger.warning(
                f"AuK DiT step graph: free memory {free / 1024**3:.1f}GB below "
                f"the {self.min_free_bytes / 1024**3:.1f}GB headroom; "
                f"x={tuple(x.shape)} will run eager"
            )
            return None
        else:
            pass
        try:
            with self.module.device(self.device):
                entry = self.capture(step, inputs, x, time)
        except Exception as exc:
            logger.warning(
                f"AuK DiT step graph capture failed for x={tuple(x.shape)}: "
                f"{exc}; this shape will run eager"
            )
            return None
        self.graphs[key] = entry
        # note(Dayuxiaoshui): a capture's tensors live in the graph's own pool,
        # which the allocator totals do not follow, so the device free delta is
        # the closest measure left and it reads high under another tenant.
        self.graph_bytes += max(0, free - self.module.mem_get_info(self.device)[0])
        logger.debug(
            f"Captured AuK DiT step graph x={tuple(x.shape)} "
            f"({len(self.graphs)} cached, {self.graph_bytes / 2**20:.0f}MiB "
            "new device memory)"
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
        # note(Dayuxiaoshui): warming on a side stream settles the block compile
        # and the allocator blocks a first call needs before the capture records.
        stream = self.module.Stream(device=self.device)
        stream.wait_stream(self.module.current_stream(self.device))
        with self.module.stream(stream):
            for _ in range(self.warmup_iters):
                step(statics, static_time, static_x)
        self.module.current_stream(self.device).wait_stream(stream)
        self.module.synchronize(self.device)

        if self.pool is None:
            self.pool = self.module.graph_pool_handle()
        else:
            pass
        # note(Dayuxiaoshui): thread_local because the conditioning and decode
        # stages launch on their own streams in this process and must not
        # poison this thread's capture.
        with self.backend.capture(pool=self.pool, thread_local_errors=True) as graph:
            static_out = step(statics, static_time, static_x)
        self.module.synchronize(self.device)
        return CapturedStep(graph, statics, static_x, static_time, static_out)


__all__ = [
    "CONDITIONING",
    "DEFAULT_CAPTURE_SHAPES",
    "FRAME_LADDER",
    "AuKGraphShape",
    "AuKStepCudaGraphRunner",
    "build_step_graph_runner",
    "verify_capture_shapes",
]
