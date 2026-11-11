# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for a complete frame of Depformer audio codes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CARD
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.sampling import (
    AudioSampling,
    SamplingRun,
    fill_sampling_noise,
    sample_runs,
    sample_token,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

logger = logging.getLogger(__name__)

CUDA_GRAPH_WARMUP_STEPS = 2


@dataclass(kw_only=True)
class CapturedDepformerGraph:
    graph: ReplayableGraph
    text_tokens: torch.Tensor
    hidden_states: torch.Tensor
    forced_codes: torch.Tensor
    output_codes: torch.Tensor
    key_value_caches: list[torch.Tensor]
    sampling_noise: list[torch.Tensor]


class DepformerCudaGraphRunner:
    def __init__(self, model: Depformer, batch_sizes: tuple[int, ...]) -> None:
        if any(batch_size <= 0 for batch_size in batch_sizes):
            raise ValueError("Depformer CUDA graph batch sizes must be positive")
        else:
            pass
        self.model: Depformer = model
        self.batch_sizes: tuple[int, ...] = tuple(sorted(set(batch_sizes)))
        self.graphs: dict[
            tuple[tuple[int, AudioSampling], ...], CapturedDepformerGraph
        ] = {}

    @torch.inference_mode()
    def capture(
        self,
        layout: tuple[tuple[int, AudioSampling], ...],
        *,
        graph_backend: DeviceGraphBackend,
        graph_pool: DeviceGraphPool,
        capture_stream: torch.Stream,
    ) -> CapturedDepformerGraph:
        projection_weight = self.model.depformer_in_weight
        batch_size = sum(rows for rows, _ in layout)
        device = projection_weight.device
        hidden_states = projection_weight.new_zeros(
            batch_size, self.model.spec.input_dim
        )
        text_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        forced_codes = torch.full(
            (batch_size, self.model.spec.steps), -1, dtype=torch.long, device=device
        )
        output_codes = torch.empty_like(forced_codes)
        key_value_caches = self.model.create_key_value_caches(hidden_states)
        sampling_noise = [
            torch.ones(
                self.model.spec.steps,
                rows,
                (
                    0
                    if sampling.greedy
                    else (
                        min(sampling.top_k, AUDIO_CARD)
                        if sampling.top_k > 0
                        else AUDIO_CARD
                    )
                ),
                dtype=torch.float32,
                device=device,
            )
            for rows, sampling in layout
        ]

        def generate_frame() -> None:
            noise_steps = iter(zip(*(noise.unbind(0) for noise in sampling_noise)))

            def sample(logits: torch.Tensor) -> torch.Tensor:
                picks: list[torch.Tensor] = []
                start_row = 0
                for (rows, sampling), noise in zip(
                    layout, next(noise_steps), strict=True
                ):
                    end_row = start_row + rows
                    picks.append(
                        sample_token(logits[start_row:end_row], sampling, noise=noise)
                    )
                    start_row = end_row
                return picks[0] if len(picks) == 1 else torch.cat(picks)

            self.model.generate(
                text_tokens,
                hidden_states,
                forced_codes,
                sample,
                key_value_caches=key_value_caches,
                output_codes=output_codes,
            )

        device_module = torch.get_device_module(device)
        capture_stream.wait_stream(device_module.current_stream(device))
        with device_module.stream(capture_stream):
            for _ in range(CUDA_GRAPH_WARMUP_STEPS):
                generate_frame()
        with graph_backend.capture(
            pool=graph_pool,
            stream=capture_stream,
            thread_local_errors=True,
        ) as graph:
            generate_frame()
        return CapturedDepformerGraph(
            graph=graph,
            text_tokens=text_tokens,
            hidden_states=hidden_states,
            forced_codes=forced_codes,
            output_codes=output_codes,
            key_value_caches=key_value_caches,
            sampling_noise=sampling_noise,
        )

    @torch.inference_mode()
    def capture_graphs(self, sampling: AudioSampling, *, max_batch_size: int) -> None:
        """Capture default-sampling buckets used by the configured concurrency."""
        capture_limit = next(
            (
                batch_size
                for batch_size in self.batch_sizes
                if batch_size >= max_batch_size
            ),
            max_batch_size,
        )
        batch_sizes = tuple(
            batch_size for batch_size in self.batch_sizes if batch_size <= capture_limit
        )
        if not batch_sizes:
            return
        else:
            pass
        device = self.model.depformer_in_weight.device
        graph_backend = current_platform.get_device_graph_backend(device)
        assert graph_backend is not None
        device_module = torch.get_device_module(device)
        capture_stream = device_module.Stream(device=device)
        graph_pool = graph_backend.graph_pool_handle()
        capture_started = time.perf_counter()
        allocated_bytes_before = device_module.memory_allocated(device)
        with (
            device_module.device(device),
            current_platform.graph_capture_attention(),
        ):
            current_stream = device_module.current_stream(device)
            capture_stream.wait_stream(current_stream)
            # Note (Kokoro2336): Larger graphs claim the shared pool first so
            # smaller buckets can reuse their blocks.
            for batch_size in reversed(batch_sizes):
                layout = ((batch_size, sampling),)
                self.graphs[layout] = self.capture(
                    layout,
                    graph_backend=graph_backend,
                    graph_pool=graph_pool,
                    capture_stream=capture_stream,
                )
            current_stream.wait_stream(capture_stream)
            device_module.synchronize(device)
        graph_bytes = device_module.memory_allocated(device) - allocated_bytes_before
        logger.info(
            f"PersonaPlex Depformer graphs: batches {batch_sizes} in "
            f"{time.perf_counter() - capture_started:.1f}s, "
            f"{graph_bytes / 2**20:.0f}MiB"
        )

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        forced_codes: torch.Tensor,
        runs: list[SamplingRun],
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        bucket_size = next(
            (size for size in self.batch_sizes if size >= batch_size), None
        )
        projection_weight = self.model.depformer_in_weight
        captured: CapturedDepformerGraph | None = None
        if (
            batch_size > 0
            and bucket_size is not None
            and hidden_states.is_cuda
            and hidden_states.device == projection_weight.device
            and hidden_states.dtype == projection_weight.dtype
        ):
            graph_key = tuple(
                (
                    (bucket_size if index == len(runs) - 1 else run.row_slice.stop)
                    - run.row_slice.start,
                    run.sampling,
                )
                for index, run in enumerate(runs)
            )
            captured = self.graphs.get(graph_key)
        else:
            pass

        if captured is None:
            return self.model.generate(
                text_tokens,
                hidden_states,
                forced_codes,
                lambda logits: sample_runs(logits, runs),
            )
        else:
            captured.text_tokens[:batch_size].copy_(text_tokens)
            captured.hidden_states[:batch_size].copy_(hidden_states)
            captured.forced_codes[:batch_size].copy_(forced_codes)
            # note (Kokoro2336): Preserve eager draw order and never draw for padded rows.
            for step in range(self.model.spec.steps):
                for run, noise in zip(runs, captured.sampling_noise, strict=True):
                    if run.sampling.greedy:
                        pass
                    else:
                        rows = len(run.generators)
                        fill_sampling_noise(noise[step, :rows], run.generators)
            captured.graph.replay()
            # note (Kokoro2336): Request timelines retain codes across later graph replays.
            return captured.output_codes[:batch_size].clone()
