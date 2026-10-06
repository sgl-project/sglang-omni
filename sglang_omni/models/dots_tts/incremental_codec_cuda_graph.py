# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for fixed-shape incremental dots.tts AudioVAE decodes."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.models.dots_tts.codec_state_arena import DotsCodecStateArena
from sglang_omni.models.dots_tts.incremental_codec import DotsIncrementalDecoder
from sglang_omni.utils.cuda_staging import indices_to_device

logger = logging.getLogger(__name__)

WARMUP_ITERATIONS = 2
CAPTURE_INPUT_SEED = 20261003
# note (0xtoward): a replay that drifts further than this from eager means the
# graph captured a stale buffer, so startup fails instead of serving it.
MAX_REPLAY_RELATIVE_ERROR = 1e-3

WarmInputs = tuple[torch.Tensor, torch.Tensor]
ColdInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class IncrementalCodecGraphKey:
    """One captured decode shape: cold or warm, batch size and latent frames."""

    mode: Literal["cold", "warm"]
    batch_size: int
    frames: int


@dataclass(kw_only=True)
class CapturedIncrementalCodecGraph:
    graph: torch.cuda.CUDAGraph
    inputs: WarmInputs | ColdInputs
    output: torch.Tensor


class DotsIncrementalCodecCudaGraphRunner:
    """Run cold and warm decode steps, replaying a captured graph when one fits.

    Graphs cover every batch size up to max_batch_size, for each warm fresh
    frame count and each cold window bucket. Other shapes run eagerly.
    """

    def __init__(
        self,
        decoder: DotsIncrementalDecoder,
        arena: DotsCodecStateArena,
        *,
        max_batch_size: int,
        warm_fresh_frames: list[int],
        cold_window_frames: list[int],
        capture_graphs: bool = True,
    ) -> None:
        self.decoder = decoder
        self.arena = arena
        self.cold_window_frames = sorted(cold_window_frames)
        self.graphs: dict[IncrementalCodecGraphKey, CapturedIncrementalCodecGraph] = {}
        self.replay_calls = 0
        self.eager_calls = 0
        if capture_graphs and decoder.device.type == "cuda":
            # note (0xtoward): with cuDNN benchmarking on, capturing the 5 to 8 row
            # graphs left a later replay faulting on an illegal address; the
            # heuristic algorithm choice decodes every captured shape correctly.
            previous_cudnn_benchmark = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = False
            try:
                # note (0xtoward): a flush decodes lookahead zero frames, so that
                # count is captured next to the regular step sizes.
                self.capture(
                    max_batch_size, sorted(set(warm_fresh_frames) | {decoder.lookahead})
                )
            finally:
                torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        else:
            pass
        logger.info(
            f"dots.tts incremental codec ready: stage_contexts={decoder.stage_contexts} "
            f"upsample_factors={decoder.upsample_factors} "
            f"warm_history_frames={decoder.warm_history_frames} graphs={len(self.graphs)}"
        )

    def forward(
        self, key: IncrementalCodecGraphKey, inputs: WarmInputs | ColdInputs
    ) -> torch.Tensor:
        if key.mode == "warm":
            frames, slot_index = inputs
            return self.decoder.warm_forward(self.arena, frames, slot_index)
        else:
            window, slot_index, stable, valid = inputs
            return self.decoder.cold_forward(
                self.arena, window, slot_index, stable, valid
            )

    def cold_window_bucket(self, frames: int) -> int | None:
        """Smallest captured cold window that holds frames, or None when none does."""
        for bucket in self.cold_window_frames:
            if bucket >= frames:
                return bucket
            else:
                pass
        return None

    @torch.no_grad()
    def capture(self, max_batch_size: int, warm_fresh_frames: list[int]) -> None:
        started_seconds = time.perf_counter()
        decoder = self.decoder
        generator = torch.Generator(device=decoder.device).manual_seed(
            CAPTURE_INPUT_SEED
        )
        capture_stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        plans: list[tuple[IncrementalCodecGraphKey, WarmInputs | ColdInputs]] = []
        for batch_size in range(1, max_batch_size + 1):
            slot_index = torch.arange(batch_size, device=decoder.device)
            for frames in warm_fresh_frames:
                inputs = (
                    torch.randn(
                        batch_size,
                        decoder.latent_channels,
                        frames,
                        device=decoder.device,
                        generator=generator,
                    )
                    * 0.01,
                    slot_index.clone(),
                )
                plans.append(
                    (IncrementalCodecGraphKey("warm", batch_size, frames), inputs)
                )
            for frames in self.cold_window_frames:
                inputs = (
                    torch.randn(
                        batch_size,
                        decoder.latent_channels,
                        frames,
                        device=decoder.device,
                        generator=generator,
                    )
                    * 0.01,
                    slot_index.clone(),
                    torch.full(
                        (batch_size,),
                        frames - decoder.lookahead,
                        device=decoder.device,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (batch_size,), frames, device=decoder.device, dtype=torch.long
                    ),
                )
                plans.append(
                    (IncrementalCodecGraphKey("cold", batch_size, frames), inputs)
                )
        history = self.arena.tensors()
        for key, inputs in plans:
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(WARMUP_ITERATIONS):
                    self.forward(key, inputs)
                snapshot = [tensor.clone() for tensor in history]
                expected = self.forward(key, inputs)
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                output = self.forward(key, inputs)
            # note (0xtoward): capturing records the kernels without running them,
            # so restore the history the eager reference started from and replay.
            for tensor, saved in zip(history, snapshot):
                tensor.copy_(saved)
            graph.replay()
            torch.cuda.synchronize()
            relative_error = (
                (output - expected).norm() / expected.norm().clamp_min(1e-12)
            ).item()
            if (
                not math.isfinite(relative_error)
                or relative_error > MAX_REPLAY_RELATIVE_ERROR
            ):
                raise RuntimeError(
                    f"Incremental codec graph {key} replay differs from eager: "
                    f"{relative_error}"
                )
            else:
                pass
            self.graphs[key] = CapturedIncrementalCodecGraph(
                graph=graph, inputs=inputs, output=output
            )
        for tensor in history:
            tensor.zero_()
        logger.info(
            f"dots.tts incremental codec captured {len(self.graphs)} graphs in "
            f"{time.perf_counter() - started_seconds:.1f}s"
        )

    def replay(
        self, key: IncrementalCodecGraphKey, inputs: WarmInputs | ColdInputs
    ) -> torch.Tensor:
        captured = self.graphs.get(key)
        if captured is None:
            self.eager_calls += 1
            if self.eager_calls == 1 and self.graphs:
                logger.warning(f"dots.tts incremental codec eager fallback for {key}")
            else:
                pass
            return self.forward(key, inputs)
        else:
            for static, value in zip(captured.inputs, inputs):
                static.copy_(value, non_blocking=True)
            captured.graph.replay()
            self.replay_calls += 1
            # note (0xtoward): the next replay overwrites the static output.
            return captured.output.clone()

    @torch.no_grad()
    def decode_warm(self, frames: torch.Tensor, slots: list[int]) -> torch.Tensor:
        """Decode new latent frames [B, C, n] of warm slots."""
        slot_index = indices_to_device(slots, self.decoder.device)
        key = IncrementalCodecGraphKey(
            "warm", int(frames.shape[0]), int(frames.shape[-1])
        )
        return self.replay(key, (frames.to(self.decoder.dtype), slot_index))

    @torch.no_grad()
    def decode_cold(
        self,
        window: torch.Tensor,
        slots: list[int],
        stable: list[int],
        valid: list[int],
    ) -> torch.Tensor:
        """Decode the latent windows [B, C, L] of slots that are not warm yet."""
        frames = self.cold_window_bucket(max(valid))
        if frames is None:
            frames = int(window.shape[-1])
        else:
            pass
        inputs = (
            window[..., :frames].to(self.decoder.dtype),
            indices_to_device(slots, self.decoder.device),
            indices_to_device(stable, self.decoder.device),
            indices_to_device(valid, self.decoder.device),
        )
        return self.replay(
            IncrementalCodecGraphKey("cold", int(window.shape[0]), frames), inputs
        )

    @torch.no_grad()
    def flush(self, slot: int) -> torch.Tensor:
        """Finish a warm stream: lookahead zero frames release its last audio."""
        frames = torch.zeros(
            1,
            self.decoder.latent_channels,
            self.decoder.lookahead,
            device=self.decoder.device,
            dtype=self.decoder.dtype,
        )
        return self.decode_warm(frames, [slot])


__all__ = ["DotsIncrementalCodecCudaGraphRunner", "IncrementalCodecGraphKey"]
