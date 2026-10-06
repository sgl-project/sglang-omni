# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs for the streaming vocoder's latent front end: post_proj, the SLSTM and its output projection."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol

import torch

logger = logging.getLogger(__name__)

# note (0xtoward): a replay further than this from the eager front end, in its output
# or in either carried LSTM state, means the graph captured the wrong work.
PARITY_RELATIVE_L2 = 1e-4
CAPTURE_INPUT_SEED = 20261003
# note (0xtoward): capture-time LSTM states at the scale of a running stream's states.
CAPTURE_STATE_SCALE = 0.1
WARMUP_ITERATIONS = 2


class StreamLatentDecode(Protocol):
    def __call__(
        self, latents: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


def relative_errors(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> list[float]:
    """Relative L2 of each tensor pair; a non-finite value on either side is an infinite error."""
    errors: list[float] = []
    for value, reference in zip(actual, expected, strict=True):
        value = value.float()
        reference = reference.float()
        if bool(torch.isfinite(value).all()) and bool(torch.isfinite(reference).all()):
            errors.append(
                ((value - reference).norm() / reference.norm().clamp_min(1e-12)).item()
            )
        else:
            errors.append(math.inf)
    return errors


@dataclass(kw_only=True)
class StreamLatentGraph:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    hidden_state: torch.Tensor
    cell_state: torch.Tensor
    decoder_input: torch.Tensor
    next_hidden_state: torch.Tensor
    next_cell_state: torch.Tensor


class StreamLatentGraphs:
    """Replays one captured graph per (batch, frames) instead of launching the front end op by op.

    Shapes outside the captured set run eagerly. A replay returns the graph's static
    output buffers, which hold until the next call with the same shape.
    """

    def __init__(
        self,
        inference: torch.nn.Module,
        *,
        max_batch_size: int,
        frame_counts: list[int],
    ) -> None:
        self.eager_decode: StreamLatentDecode = (
            inference._decode_stream_latents
        )  # noqa: leading-underscore  # upstream spelling
        self.graphs: dict[tuple[int, int], StreamLatentGraph] = {}
        self.replay_calls: int = 0
        self.fallback_calls: int = 0
        self.exact: bool = True
        if (
            int(inference._lstm_num_layers) == 0
        ):  # noqa: leading-underscore  # upstream spelling
            inference._prepare_lstm_stream_params()  # noqa: leading-underscore  # upstream spelling
        else:
            pass
        self.latent_dim: int = int(inference.vocoder.h.latent_dim)
        self.num_layers: int = int(
            inference._lstm_num_layers
        )  # noqa: leading-underscore  # upstream spelling
        self.hidden_size: int = int(
            inference._lstm_hidden_size
        )  # noqa: leading-underscore  # upstream spelling
        self.device: torch.device = next(inference.vocoder.parameters()).device
        started_seconds = time.perf_counter()
        generator = torch.Generator(device=self.device).manual_seed(CAPTURE_INPUT_SEED)
        capture_stream = torch.cuda.Stream(device=self.device)
        with torch.no_grad():
            for batch_size in range(1, max_batch_size + 1):
                for frames in frame_counts:
                    self.graphs[(batch_size, frames)] = self.capture_shape(
                        batch_size, frames, generator, capture_stream
                    )
        logger.info(
            f"Stream latent graphs ready shapes={sorted(self.graphs)} exact={self.exact} "
            f"startup_seconds={time.perf_counter() - started_seconds:.3f}"
        )

    def random_latents(
        self, batch_size: int, frames: int, generator: torch.Generator
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.latent_dim,
            frames,
            device=self.device,
            generator=generator,
        )

    def random_state(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        return (
            torch.randn(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=self.device,
                generator=generator,
            )
            * CAPTURE_STATE_SCALE
        )

    def capture_shape(
        self,
        batch_size: int,
        frames: int,
        generator: torch.Generator,
        capture_stream: torch.cuda.Stream,
    ) -> StreamLatentGraph:
        """Capture one shape and check two consecutive replays against the eager front end."""
        latents = self.random_latents(batch_size, frames, generator)
        hidden_state = self.random_state(batch_size, generator)
        cell_state = self.random_state(batch_size, generator)
        expected_input, (expected_hidden_state, expected_cell_state) = (
            self.eager_decode(latents, (hidden_state, cell_state))
        )
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(WARMUP_ITERATIONS):
                self.eager_decode(latents, (hidden_state, cell_state))
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            decoder_input, (next_hidden_state, next_cell_state) = self.eager_decode(
                latents, (hidden_state, cell_state)
            )
        captured = StreamLatentGraph(
            graph=graph,
            latents=latents,
            hidden_state=hidden_state,
            cell_state=cell_state,
            decoder_input=decoder_input,
            next_hidden_state=next_hidden_state,
            next_cell_state=next_cell_state,
        )
        graph.replay()
        torch.cuda.synchronize(self.device)
        observed = [replay_outputs(captured)]
        expected = [(expected_input, expected_hidden_state, expected_cell_state)]
        # note (0xtoward): replay again from the graph's own states, so a state that is
        # wrong while the first output still matches fails the gate too.
        next_latents = self.random_latents(batch_size, frames, generator)
        latents.copy_(next_latents)
        hidden_state.copy_(observed[0][1])
        cell_state.copy_(observed[0][2])
        graph.replay()
        torch.cuda.synchronize(self.device)
        observed.append(replay_outputs(captured))
        next_expected_input, (next_expected_hidden_state, next_expected_cell_state) = (
            self.eager_decode(
                next_latents, (expected_hidden_state, expected_cell_state)
            )
        )
        expected.append(
            (next_expected_input, next_expected_hidden_state, next_expected_cell_state)
        )
        self.check_parity(batch_size, frames, observed, expected)
        return captured

    def check_parity(
        self,
        batch_size: int,
        frames: int,
        observed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        expected: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        exact = all(
            torch.equal(value, reference)
            for replay_observed, replay_expected in zip(observed, expected, strict=True)
            for value, reference in zip(replay_observed, replay_expected, strict=True)
        )
        if not exact:
            errors = [
                relative_errors(replay_observed, replay_expected)
                for replay_observed, replay_expected in zip(
                    observed, expected, strict=True
                )
            ]
            logger.warning(
                f"Stream latent graph B{batch_size} T{frames} relative_l2 "
                f"(input, hidden state, cell state) per replay={errors}"
            )
            if max(max(replay) for replay in errors) > PARITY_RELATIVE_L2:
                raise RuntimeError(
                    f"Stream latent graph B{batch_size} T{frames} failed parity gate: {errors}"
                )
            else:
                pass
        else:
            pass
        self.exact &= exact

    def __call__(
        self, latents: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        captured = self.graphs.get((int(latents.shape[0]), int(latents.shape[-1])))
        if captured is None:
            self.fallback_calls += 1
            if self.fallback_calls == 1:
                logger.warning(
                    f"Stream latent graph fallback shape={tuple(latents.shape)} dtype={latents.dtype}"
                )
            else:
                pass
            return self.eager_decode(latents, hidden)
        else:
            captured.latents.copy_(latents)
            captured.hidden_state.copy_(hidden[0])
            captured.cell_state.copy_(hidden[1])
            captured.graph.replay()
            self.replay_calls += 1
            return captured.decoder_input, (
                captured.next_hidden_state,
                captured.next_cell_state,
            )


def replay_outputs(
    captured: StreamLatentGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copies of a replay's outputs, which the next replay overwrites."""
    return (
        captured.decoder_input.clone(),
        captured.next_hidden_state.clone(),
        captured.next_cell_state.clone(),
    )
