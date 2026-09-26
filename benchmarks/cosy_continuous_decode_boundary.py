# SPDX-License-Identifier: Apache-2.0
"""Measure CUDA readback boundaries using Cosy stop-position traces."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

TRACE_PATH = Path(__file__).with_name("cosy_continuous_decode_stop_steps.json")
WINDOWS = (1, 4, 8)
ORDER = (1, 4, 8, 8, 4, 1)


def replay(
    stop_steps: list[int],
    concurrency: int,
    window: int,
    state: torch.Tensor,
    checkpoint: torch.Tensor,
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(stop_steps), concurrency):
        wave = stop_steps[offset : offset + concurrency]
        physical_steps = max(((length + window - 1) // window) * window for length in wave)
        state.zero_()
        for step in range(physical_steps):
            state.add_(1)
            checkpoint[step % window].copy_(state)
            if (step + 1) % window == 0:
                observed = checkpoint.tolist()
                assert observed[-1][0] == step + 1
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    traces: dict[str, dict[str, int | list[int]]] = json.loads(TRACE_PATH.read_text())
    for name, trace in traces.items():
        concurrency = trace["concurrency"]
        stop_steps = trace["stop_steps"]
        assert isinstance(concurrency, int)
        assert isinstance(stop_steps, list)
        state = torch.empty(concurrency, dtype=torch.int64, device="cuda")
        checkpoints = {
            window: torch.empty(window, concurrency, dtype=torch.int64, device="cuda")
            for window in WINDOWS
        }
        for window in WINDOWS:
            replay(stop_steps[:concurrency], concurrency, window, state, checkpoints[window])
        seconds = [
            replay(stop_steps, concurrency, window, state, checkpoints[window])
            for window in ORDER
        ]
        print(
            json.dumps(
                {
                    "trace": name,
                    "requests": len(stop_steps),
                    "concurrency": concurrency,
                    "windows": list(ORDER),
                    "seconds": seconds,
                    "extra_row_steps": {
                        window: sum(-length % window for length in stop_steps)
                        for window in WINDOWS
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
