# SPDX-License-Identifier: Apache-2.0
"""Noise comes from the default generator inside the graph."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from sglang_omni.models.personaplex.architecture import NUM_STREAMS
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.components.depformer_cuda_graph import (
    CUDA_GRAPH_WARMUP_STEPS,
)
from sglang_omni.models.personaplex.sampling import AudioSampling, sample_token
from sglang_omni.models.personaplex.sglang_model import PersonaPlexForCausalLM
from sglang_omni.models.personaplex.timeline import output_frame
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

logger = logging.getLogger(__name__)


def graph_buckets(batch_sizes: tuple[int, ...], max_batch_size: int) -> tuple[int, ...]:
    sizes = list(batch_sizes)
    while sizes[-1] < max_batch_size:
        sizes.append(min(2 * sizes[-1], max_batch_size))
    return tuple(sizes)


def captured_buckets(
    batch_sizes: tuple[int, ...], max_batch_size: int, rows: int
) -> tuple[int, ...]:
    """No bucket exceeds the rows of the buffers the graphs read."""
    buckets = graph_buckets(batch_sizes, max_batch_size)
    limit = min(next(size for size in buckets if size >= max_batch_size), rows)
    return tuple(sorted({min(size, limit) for size in buckets}))


@dataclass(frozen=True)
class LiveStepSampling:
    """Greedy settings are stored as AudioSampling(0.0, 0), whatever their top-k."""

    text: AudioSampling
    audio: AudioSampling

    @classmethod
    def of(cls, text: AudioSampling, audio: AudioSampling) -> LiveStepSampling:
        return cls(
            text=AudioSampling(0.0, 0) if text.greedy else text,
            audio=AudioSampling(0.0, 0) if audio.greedy else audio,
        )


GREEDY_LIVE_STEP = LiveStepSampling(
    text=AudioSampling(0.0, 0), audio=AudioSampling(0.0, 0)
)


@dataclass(kw_only=True)
class CapturedLiveStep:
    graph: ReplayableGraph
    logits: torch.Tensor
    previous_agent_rows: torch.Tensor
    packed: torch.Tensor
    # Note (Junnan Li): Written by every replay, so they live as long as the graph.
    key_value_caches: list[torch.Tensor]
    output_codes: torch.Tensor


def live_step(
    depformer: Depformer,
    logits_BV: torch.Tensor,
    hidden_BD: torch.Tensor,
    previous_agent_rows_BK: torch.Tensor,
    sampling: LiveStepSampling,
    *,
    key_value_caches: list[torch.Tensor],
    output_codes: torch.Tensor,
    packed: torch.Tensor,
) -> torch.Tensor:
    """[B, 1 + 2K]: each row's text token, output frame and new agent codes."""
    batch_size = logits_BV.shape[0]
    steps = depformer.spec.steps
    unseeded = [None] * batch_size
    text_token_B = sample_token(logits_BV.float(), sampling.text, unseeded)
    codes_BK = depformer.generate(
        text_token_B,
        hidden_BD,
        torch.full((batch_size, steps), -1, dtype=torch.long, device=logits_BV.device),
        lambda logits: sample_token(logits, sampling.audio, unseeded),
        key_value_caches=key_value_caches,
        output_codes=output_codes,
    )
    packed[:, 0] = text_token_B
    packed[:, 1 : 1 + steps] = output_frame(previous_agent_rows_BK, codes_BK)
    packed[:, 1 + steps :] = codes_BK
    return packed


class LiveStepGraphRunner:
    def __init__(
        self,
        depformer: Depformer,
        hidden_out: torch.Tensor,
        vocab_size: int,
        batch_sizes: tuple[int, ...],
    ) -> None:
        self.depformer: Depformer = depformer
        self.hidden_out: torch.Tensor = hidden_out
        self.vocab_size: int = vocab_size
        self.batch_sizes: tuple[int, ...] = tuple(sorted(set(batch_sizes)))
        self.graphs: dict[tuple[int, LiveStepSampling], CapturedLiveStep] = {}

    @torch.inference_mode()
    def capture(
        self,
        batch_size: int,
        sampling: LiveStepSampling,
        *,
        graph_backend: DeviceGraphBackend,
        graph_pool: DeviceGraphPool,
        capture_stream: torch.Stream,
    ) -> CapturedLiveStep:
        device = self.hidden_out.device
        steps = self.depformer.spec.steps
        logits = torch.zeros(
            batch_size, self.vocab_size, dtype=torch.float32, device=device
        )
        previous_agent_rows = torch.zeros(
            batch_size, steps, dtype=torch.long, device=device
        )
        packed = torch.empty(batch_size, 1 + 2 * steps, dtype=torch.long, device=device)
        hidden = self.hidden_out[:batch_size]
        key_value_caches = self.depformer.create_key_value_caches(hidden)
        output_codes = torch.empty(batch_size, steps, dtype=torch.long, device=device)

        def run() -> None:
            live_step(
                self.depformer,
                logits,
                hidden,
                previous_agent_rows,
                sampling,
                key_value_caches=key_value_caches,
                output_codes=output_codes,
                packed=packed,
            )

        device_module = torch.get_device_module(device)
        capture_stream.wait_stream(device_module.current_stream(device))
        with device_module.stream(capture_stream):
            for _ in range(CUDA_GRAPH_WARMUP_STEPS):
                run()
        with graph_backend.capture(
            pool=graph_pool, stream=capture_stream, thread_local_errors=True
        ) as graph:
            run()
        return CapturedLiveStep(
            graph=graph,
            logits=logits,
            previous_agent_rows=previous_agent_rows,
            packed=packed,
            key_value_caches=key_value_caches,
            output_codes=output_codes,
        )

    @torch.inference_mode()
    def capture_graphs(
        self, samplings: tuple[LiveStepSampling, ...], *, max_batch_size: int
    ) -> None:
        self.batch_sizes = captured_buckets(
            self.batch_sizes, max_batch_size, self.hidden_out.shape[0]
        )
        device = self.hidden_out.device
        graph_backend = current_platform.get_device_graph_backend(device)
        assert graph_backend is not None
        device_module = torch.get_device_module(device)
        capture_stream = device_module.Stream(device=device)
        graph_pool = graph_backend.graph_pool_handle()
        capture_started = time.perf_counter()
        allocated_bytes_before = device_module.memory_allocated(device)
        with device_module.device(device), current_platform.graph_capture_attention():
            current_stream = device_module.current_stream(device)
            capture_stream.wait_stream(current_stream)
            # Note (Junnan Li): Largest first, so smaller graphs reuse its blocks from
            # the shared pool.
            for batch_size in reversed(self.batch_sizes):
                for sampling in samplings:
                    self.graphs[(batch_size, sampling)] = self.capture(
                        batch_size,
                        sampling,
                        graph_backend=graph_backend,
                        graph_pool=graph_pool,
                        capture_stream=capture_stream,
                    )
            current_stream.wait_stream(capture_stream)
            device_module.synchronize(device)
        graph_bytes = device_module.memory_allocated(device) - allocated_bytes_before
        logger.info(
            f"PersonaPlex live-step graphs: batches {self.batch_sizes} x "
            f"{len(samplings)} samplings in "
            f"{time.perf_counter() - capture_started:.1f}s, "
            f"{graph_bytes / 2**20:.0f}MiB"
        )

    def graph_for(
        self, batch_size: int, sampling: LiveStepSampling
    ) -> CapturedLiveStep | None:
        bucket_size = next(
            (size for size in self.batch_sizes if size >= batch_size), None
        )
        if bucket_size is None:
            return None
        else:
            return self.graphs.get((bucket_size, sampling))

    @torch.inference_mode()
    def run(
        self,
        captured: CapturedLiveStep,
        logits_BV: torch.Tensor,
        previous_agent_rows: list[torch.Tensor],
    ) -> torch.Tensor:
        """The returned rows are the caller's to keep."""
        batch_size = logits_BV.shape[0]
        captured.logits[:batch_size].copy_(logits_BV)
        torch.stack(previous_agent_rows, out=captured.previous_agent_rows[:batch_size])
        captured.graph.replay()
        return captured.packed[:batch_size].clone()


class LiveRowGraphRunner:
    def __init__(
        self, model: PersonaPlexForCausalLM, batch_sizes: tuple[int, ...]
    ) -> None:
        self.model: PersonaPlexForCausalLM = model
        self.batch_sizes: tuple[int, ...] = tuple(sorted(set(batch_sizes)))
        self.graphs: dict[int, ReplayableGraph] = {}
        self.rows: torch.Tensor | None = None
        self.host_rows: torch.Tensor | None = None

    @torch.inference_mode()
    def capture_graphs(self, *, max_batch_size: int) -> None:
        self.batch_sizes = captured_buckets(
            self.batch_sizes, max_batch_size, self.model.fusion_buffer.shape[0]
        )
        fusion_buffer = self.model.fusion_buffer
        device = fusion_buffer.device
        self.rows = torch.zeros(
            self.batch_sizes[-1], NUM_STREAMS, dtype=torch.long, device=device
        )
        self.host_rows = torch.zeros_like(self.rows, device="cpu").pin_memory()
        graph_backend = current_platform.get_device_graph_backend(device)
        assert graph_backend is not None
        device_module = torch.get_device_module(device)
        capture_stream = device_module.Stream(device=device)
        graph_pool = graph_backend.graph_pool_handle()

        def embed(batch_size: int) -> None:
            fusion_buffer[:batch_size] = self.model.embed_rows(
                self.rows[:batch_size]
            ).to(fusion_buffer.dtype)

        with device_module.device(device):
            current_stream = device_module.current_stream(device)
            for batch_size in reversed(self.batch_sizes):
                capture_stream.wait_stream(current_stream)
                with device_module.stream(capture_stream):
                    embed(batch_size)
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream, thread_local_errors=True
                ) as graph:
                    embed(batch_size)
                self.graphs[batch_size] = graph
            current_stream.wait_stream(capture_stream)
            device_module.synchronize(device)

    def staging(self, batch_size: int) -> np.ndarray:
        # Note (Junnan Li): One buffer suffices: the previous step's read-back blocks
        # until its copy has run.
        return self.host_rows[:batch_size].numpy()

    @torch.inference_mode()
    def embed(self, batch_size: int) -> None:
        bucket_size = next(size for size in self.batch_sizes if size >= batch_size)
        self.rows[:batch_size].copy_(self.host_rows[:batch_size], non_blocking=True)
        self.graphs[bucket_size].replay()


__all__ = [
    "GREEDY_LIVE_STEP",
    "LiveRowGraphRunner",
    "LiveStepGraphRunner",
    "LiveStepSampling",
    "live_step",
]
