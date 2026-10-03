# SPDX-License-Identifier: Apache-2.0
"""Capture tables for MiniCPM-o dense Flow."""

from __future__ import annotations

# note (wirybeaver): Batch-1 SeedTTS EN uses 260-714 frames; sparse tails retain coverage.
FLOW_CUDA_GRAPH_FRAME_BUCKETS = (
    *range(128, 257, 32),
    *range(272, 721, 16),
    *range(752, 1009, 32),
    1024,
)
# note (wirybeaver): Sparse tails cover zero-wait SeedTTS batches without broad padding.
FLOW_CUDA_GRAPH_BATCH_FRAME_BUCKETS: dict[int, tuple[int, ...]] = {
    2: (
        336,
        384,
        400,
        416,
        432,
        448,
        464,
        480,
        496,
        512,
        528,
        544,
        560,
        576,
        592,
        608,
        624,
        640,
        672,
        736,
    ),
    3: (
        400,
        416,
        448,
        464,
        480,
        496,
        512,
        528,
        544,
        560,
        576,
        608,
        640,
        656,
        672,
        688,
        736,
    ),
    4: (432, 448, 464, 480, 496, 512, 528, 576, 592, 608, 624, 640),
    5: (416, 432, 448, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624),
    6: (464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656),
    7: (480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 656, 672, 720, 736),
    8: (480, 496, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 784),
}


def build_default_flow_cuda_graph_shapes() -> tuple[tuple[int, int], ...]:
    """Build the default batch/mel-frame graph grid."""
    return (
        *((1, frames) for frames in FLOW_CUDA_GRAPH_FRAME_BUCKETS),
        *(
            (batch, frames)
            for batch, frame_buckets in FLOW_CUDA_GRAPH_BATCH_FRAME_BUCKETS.items()
            for frames in frame_buckets
        ),
    )
