# SPDX-License-Identifier: Apache-2.0
"""Bucketed CUDA graphs for a complete frame of Depformer audio codes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from sglang_omni.models.personaplex.architecture import AUDIO_CARD
from sglang_omni.models.personaplex.components.depformer import Depformer
from sglang_omni.models.personaplex.sampling import AudioSampling, sample_token

logger = logging.getLogger(__name__)

CUDA_GRAPH_WARMUP_STEPS = 2
MAX_CACHED_GRAPHS = 16


@dataclass(kw_only=True)
class CapturedDepformerGraph:
    graph: torch.cuda.CUDAGraph
    text_tokens: torch.Tensor
    hidden_states: torch.Tensor
    forced_codes: torch.Tensor
    output_codes: torch.Tensor
    key_value_caches: list[torch.Tensor]
    sampling_noise: torch.Tensor


class DepformerCudaGraphRunner:
    def __init__(self, model: Depformer, batch_sizes: tuple[int, ...]) -> None:
        if any(batch_size <= 0 for batch_size in batch_sizes):
            raise ValueError("Depformer CUDA graph batch sizes must be positive")
        else:
            pass
        self.model: Depformer = model
        self.batch_sizes: tuple[int, ...] = tuple(sorted(set(batch_sizes)))
        self.graphs: dict[tuple[int, AudioSampling], CapturedDepformerGraph | None] = {}

    @torch.inference_mode()
    def capture(
        self, batch_size: int, sampling: AudioSampling
    ) -> CapturedDepformerGraph:
        projection_weight = self.model.depformer_in[0].weight
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
        noise_width = (
            min(sampling.top_k, AUDIO_CARD) if sampling.top_k > 0 else AUDIO_CARD
        )
        sampling_noise = torch.ones(
            self.model.spec.steps,
            batch_size,
            0 if sampling.greedy else noise_width,
            dtype=torch.float32,
            device=device,
        )

        def generate_frame() -> None:
            noise_steps = iter(sampling_noise.unbind(0))

            def sample(logits: torch.Tensor) -> torch.Tensor:
                return sample_token(logits, sampling, noise=next(noise_steps))

            self.model.generate(
                text_tokens,
                hidden_states,
                forced_codes,
                sample,
                key_value_caches=key_value_caches,
                output_codes=output_codes,
            )

        with torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                for _ in range(CUDA_GRAPH_WARMUP_STEPS):
                    generate_frame()
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                generate_frame()
            current_stream.wait_stream(capture_stream)
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
    def generate(
        self,
        text_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        forced_codes: torch.Tensor,
        sampling: AudioSampling,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        bucket_size = next(
            (size for size in self.batch_sizes if size >= batch_size), None
        )
        projection_weight = self.model.depformer_in[0].weight
        captured: CapturedDepformerGraph | None = None
        if (
            batch_size > 0
            and bucket_size is not None
            and hidden_states.is_cuda
            and hidden_states.device == projection_weight.device
            and hidden_states.dtype == projection_weight.dtype
        ):
            graph_key = (bucket_size, sampling)
            if graph_key not in self.graphs and len(self.graphs) < MAX_CACHED_GRAPHS:
                try:
                    self.graphs[graph_key] = self.capture(bucket_size, sampling)
                except RuntimeError as error:
                    logger.warning(
                        f"Depformer CUDA graph capture failed for batch={bucket_size} "
                        f"sampling={sampling}; using eager execution: {error}"
                    )
                    self.graphs[graph_key] = None
            else:
                pass
            captured = self.graphs.get(graph_key)
        else:
            pass

        if captured is None:
            return self.model.generate(
                text_tokens,
                hidden_states,
                forced_codes,
                lambda logits: sample_token(logits, sampling, generator),
            )
        else:
            captured.text_tokens[:batch_size].copy_(text_tokens)
            captured.hidden_states[:batch_size].copy_(hidden_states)
            captured.forced_codes[:batch_size].copy_(forced_codes)
            if batch_size < captured.text_tokens.shape[0]:
                captured.text_tokens[batch_size:].zero_()
                captured.hidden_states[batch_size:].zero_()
                captured.forced_codes[batch_size:].fill_(-1)
                captured.sampling_noise[:, batch_size:].fill_(1.0)
            else:
                pass
            if not sampling.greedy:
                # note (Kokoro2336): Match eager draw shapes without consuming RNG for padding.
                for noise in captured.sampling_noise.unbind(0):
                    noise[:batch_size].exponential_(1.0, generator=generator)
            else:
                pass
            captured.graph.replay()
            # note (Kokoro2336): Request timelines retain codes across later graph replays.
            return captured.output_codes[:batch_size].clone()
