# SPDX-License-Identifier: Apache-2.0
"""Capture tables for MiniCPM-o Flow and packed DiT."""

from __future__ import annotations

# note (wirybeaver): SeedTTS EN uses 260-714 frames; sparse tails retain coverage.
FLOW_CUDA_GRAPH_FRAME_BUCKETS = (
    *range(128, 257, 32),
    *range(272, 721, 16),
    *range(752, 1009, 32),
    1024,
)
# note (wirybeaver): B2-B8 buckets cover 294 timed SeedTTS EN packed fits.
FLOW_CUDA_GRAPH_PACKED_FRAME_BUCKETS: dict[int, tuple[int, ...]] = {
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
    ),
    3: (400, 416, 448, 464, 480, 496, 512, 528, 544, 560, 576, 608, 640, 656, 672),
    4: (432, 448, 464, 480, 496, 512, 528, 576, 592, 608, 624),
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
            for batch, frame_buckets in FLOW_CUDA_GRAPH_PACKED_FRAME_BUCKETS.items()
            for frames in frame_buckets
        ),
    )


# note (wirybeaver): Capacities projected from the September 2026 SeedTTS EN census.
SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES: tuple[tuple[int, int], ...] = (
    (2, 2144),
    (3, 3088),
    (4, 3936),
    (6, 5680),
    (8, 7616),
    (5, 4944),
    (7, 6384),
    (5, 4192),
    (8, 6832),
    (7, 7136),
    (2, 2448),
    (6, 6464),
    (2, 1744),
    (2, 1936),
    (2, 1616),
    (2, 2080),
    (3, 2688),
    (3, 3392),
    (3, 3072),
    (3, 2496),
    (4, 3520),
    (4, 4048),
    (4, 3216),
    (4, 4432),
    (5, 4128),
    (5, 5376),
    (5, 4592),
    (5, 3760),
    (6, 5376),
    (6, 5888),
    (6, 5248),
    (6, 5424),
    (7, 6032),
    (7, 6448),
    (7, 4944),
    (7, 6736),
    (8, 7760),
    (8, 8416),
    (8, 7440),
    (8, 6352),
    (3, 2000),
    (5, 3216),
    (2, 1280),
    (7, 5520),
    (7, 5552),
    (8, 6896),
)
