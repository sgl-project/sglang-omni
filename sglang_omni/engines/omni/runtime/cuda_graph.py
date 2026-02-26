# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph support for DualAR models in sglang-omni.

DualARTransformer has two distinct forward paths with different tensor
shapes, both of which benefit from CUDA Graph capture:

1. **Slow graph** (``forward_generate``):
   - Input:  ``[1, num_codebooks+1, 1]``  (decode) or
             ``[1, num_codebooks+1, seq_len]`` (prefill — not graphed)
   - Output: ``BaseTransformerForwardResult(logits, hidden_states)``

2. **Fast graph** (``forward_generate_fast``):
   - Input:  ``[1, 1, fast_dim]`` hidden state + scalar ``input_pos``
   - Output: ``[1, 1, codebook_size]`` logits
   - Called ``num_codebooks`` times per decode step

Prefill is NOT captured (variable length). Only the decode step is graphed.

This module is a **patch** — it wraps an existing ``DualARTransformer``
model and replaces its decode-time forward calls with graph replays.

Usage:
    model = DualARTransformer.from_pretrained(...)
    runner = DualARCudaGraphRunner(model, num_codebooks=4)
    runner.capture()  # one-time warmup

    # Inside the decode loop:
    result = runner.replay_slow(x, input_pos, audio_masks, audio_parts)
    for cb_idx in range(num_codebooks):
        logits = runner.replay_fast(hidden, torch.tensor([cb_idx]))
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DualARCudaGraphRunner:
    """Manages CUDA Graphs for the DualARTransformer decode step.

    Captures two graphs:
    - ``_slow_graph``: one slow-transformer forward (decode mode, seq_len=1)
    - ``_fast_graph``: one fast-transformer forward (single codebook step)

    The fast graph is replayed ``num_codebooks`` times per decode step
    with updated input tensors (copied into the static buffers).
    """

    def __init__(
        self,
        model: Any,
        *,
        num_codebooks: int = 4,
        codebook_size: int = 1024,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)

        self._model = model
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._device = device
        self._dtype = dtype

        # Static input/output buffers (allocated during capture)
        self._slow_graph: torch.cuda.CUDAGraph | None = None
        self._slow_x: torch.Tensor | None = None
        self._slow_input_pos: torch.Tensor | None = None
        self._slow_out_logits: torch.Tensor | None = None
        self._slow_out_hidden: torch.Tensor | None = None

        self._fast_graph: torch.cuda.CUDAGraph | None = None
        self._fast_x: torch.Tensor | None = None
        self._fast_input_pos: torch.Tensor | None = None
        self._fast_out_logits: torch.Tensor | None = None

        self._captured = False

    @property
    def is_captured(self) -> bool:
        return self._captured

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, warmup_iters: int = 3) -> None:
        """Capture CUDA Graphs for slow and fast decode.

        Must be called after ``model.setup_caches()`` and before any
        decode calls. Runs a few warmup iterations to stabilise CUDA
        state before recording.
        """
        if self._captured:
            logger.warning("CUDA Graphs already captured, skipping")
            return

        logger.info("Capturing DualAR CUDA Graphs (%d warmup iters)", warmup_iters)

        self._capture_slow_graph(warmup_iters)
        self._capture_fast_graph(warmup_iters)
        self._captured = True

        logger.info("DualAR CUDA Graphs captured successfully")

    def _capture_slow_graph(self, warmup_iters: int) -> None:
        dim = self._model.config.dim
        codebook_dim = self._num_codebooks + 1

        # Static buffers
        self._slow_x = torch.zeros(
            (1, codebook_dim, 1), dtype=torch.int, device=self._device
        )
        self._slow_input_pos = torch.zeros(
            (1,), dtype=torch.long, device=self._device
        )

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                out = self._model.forward_generate(
                    self._slow_x, self._slow_input_pos
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._slow_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._slow_graph):
            out = self._model.forward_generate(
                self._slow_x, self._slow_input_pos
            )
        self._slow_out_logits = out.logits
        self._slow_out_hidden = out.hidden_states

    def _capture_fast_graph(self, warmup_iters: int) -> None:
        fast_dim = self._model.config.fast_dim or self._model.config.dim

        self._fast_x = torch.zeros(
            (1, fast_dim), dtype=self._dtype, device=self._device
        )
        self._fast_input_pos = torch.zeros(
            (1,), dtype=torch.long, device=self._device
        )

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self._model.forward_generate_fast(
                    self._fast_x, self._fast_input_pos
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._fast_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._fast_graph):
            out = self._model.forward_generate_fast(
                self._fast_x, self._fast_input_pos
            )
        self._fast_out_logits = out

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay_slow(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        audio_masks: torch.Tensor | None = None,
        audio_parts: torch.Tensor | None = None,
    ) -> Any:
        """Replay the slow-transformer decode graph.

        If ``x`` is a prefill (seq_len > 1) or CUDA Graphs are not
        captured, falls back to eager execution.
        """
        is_prefill = x.shape[-1] > 1
        if is_prefill or not self._captured:
            return self._model.forward_generate(
                x, input_pos,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            )

        # Copy live data into static buffers
        self._slow_x.copy_(x)
        self._slow_input_pos.copy_(input_pos)

        self._slow_graph.replay()

        # Return references to static output buffers.
        # Caller must consume or clone before next replay.
        from dataclasses import dataclass
        from torch import Tensor

        @dataclass
        class _Result:
            logits: Tensor
            hidden_states: Tensor

        return _Result(
            logits=self._slow_out_logits,
            hidden_states=self._slow_out_hidden,
        )

    def replay_fast(
        self, x: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        """Replay the fast-transformer codebook decode graph."""
        if not self._captured:
            return self._model.forward_generate_fast(x, input_pos)

        self._fast_x.copy_(x)
        self._fast_input_pos.copy_(input_pos)

        self._fast_graph.replay()

        return self._fast_out_logits

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Release captured graphs (e.g., before re-capture)."""
        self._slow_graph = None
        self._fast_graph = None
        self._captured = False
