# SPDX-License-Identifier: Apache-2.0
"""Drama 3 expert trace files and the cache simulation that reads them, on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from sglang.srt.layers.moe.topk import StandardTopKOutput
from torch import nn

from benchmarks.drama3_expert_trace import (
    EXPERT_IDS_FILE_NAME,
    FORWARDS_FILE_NAME,
    METADATA_FILE_NAME,
    ExpertTrace,
    TopKIdsRecorder,
)
from benchmarks.expert_cache_simulation import load_trace, simulate

ROUTED_EXPERT_COUNT = 4
TOP_K = 2
BYTES_PER_ROUTED_EXPERT = 1_000_000
# One expert miss costs 1 ms at this bandwidth.
HOST_TO_DEVICE_GIGABYTES_PER_SECOND = 1.0
# Per forward: is_prefill, request ids, token counts, then each layer's top-k ids.
FORWARDS = [
    (True, ["a"], [2], [[[0, 1], [1, 2]], [[3, 2], [3, 0]]]),
    (False, ["a", "b"], [1, 1], [[[0, 3], [3, 1]], [[1, 2], [2, 1]]]),
    (False, ["a"], [1], [[[2, 0]], [[0, 3]]]),
]


def write_trace(process_dir: Path) -> None:
    """Route FORWARDS through the hooks and the trace, as the model runner does."""
    process_dir.mkdir()
    (process_dir / METADATA_FILE_NAME).write_text(
        json.dumps(
            {
                "checkpoint_path": "synthetic",
                "quantization": None,
                "device_name": "cpu",
                "moe_layer_ids": [0, 1],
                "routed_expert_count": ROUTED_EXPERT_COUNT,
                "top_k": TOP_K,
                "bytes_per_routed_expert": BYTES_PER_ROUTED_EXPERT,
            }
        )
    )
    layer_expert_ids = [torch.full((8, TOP_K), -1, dtype=torch.int16) for _ in range(2)]
    recorders = [TopKIdsRecorder(expert_ids) for expert_ids in layer_expert_ids]
    trace = ExpertTrace(
        moe_layer_ids=[0, 1],
        layer_expert_ids=layer_expert_ids,
        expert_ids_file=(process_dir / EXPERT_IDS_FILE_NAME).open("ab"),
        forwards_file=(process_dir / FORWARDS_FILE_NAME).open("a"),
    )
    for is_prefill, request_ids, token_counts, layer_topk_ids in FORWARDS:
        for recorder, topk_ids in zip(recorders, layer_topk_ids, strict=True):
            topk_ids_tensor = torch.tensor(topk_ids, dtype=torch.int32)
            recorder(
                nn.Identity(),
                (),
                StandardTopKOutput(
                    topk_weights=torch.ones(topk_ids_tensor.shape),
                    topk_ids=topk_ids_tensor,
                    router_logits=torch.zeros(
                        topk_ids_tensor.shape[0], ROUTED_EXPERT_COUNT
                    ),
                ),
            )
        trace.record(
            is_prefill=is_prefill, request_ids=request_ids, token_counts=token_counts
        )
    trace.expert_ids_file.close()
    trace.forwards_file.close()


def test_trace_round_trips_each_forwards_experts(tmp_path: Path) -> None:
    write_trace(tmp_path / "pid1")
    metadata, forwards = load_trace(tmp_path / "pid1")
    assert metadata.moe_layer_ids == [0, 1]
    assert [forward.is_prefill for forward in forwards] == [True, False, False]
    assert [forward.request_count for forward in forwards] == [1, 2, 1]
    used_by_layer = [
        [set(map(int, layer_used.nonzero()[0])) for layer_used in forward.expert_used]
        for forward in forwards
    ]
    assert used_by_layer == [
        [{0, 1, 2}, {0, 2, 3}],
        [{0, 1, 3}, {1, 2}],
        [{0, 2}, {0, 3}],
    ]


@pytest.mark.parametrize("cut_bytes", [2 * 2 * TOP_K, 2])
def test_trace_reader_drops_a_forward_whose_ids_were_cut_off(
    tmp_path: Path, cut_bytes: int
) -> None:
    """A server killed while writing ids leaves a whole token or part of one missing."""
    process_dir = tmp_path / "pid1"
    write_trace(process_dir)
    expert_ids_path = process_dir / EXPERT_IDS_FILE_NAME
    expert_ids_path.write_bytes(expert_ids_path.read_bytes()[:-cut_bytes])
    _, forwards = load_trace(process_dir)
    assert len(forwards) == 2


@pytest.mark.parametrize(
    ("policy", "forward_misses", "decode_overflow"),
    [
        # lru: each layer keeps its two most recently used experts.
        ("lru", [6, 3, 4], 0.5),
        # static: layer 0 keeps experts {0, 1} and layer 1 keeps {0, 2}, the most used.
        ("static", [2, 2, 2], 0.5),
    ],
)
def test_simulation_counts_each_policys_misses(
    tmp_path: Path,
    policy: str,
    forward_misses: list[int],
    decode_overflow: float,
) -> None:
    write_trace(tmp_path / "pid1")
    metadata, forwards = load_trace(tmp_path / "pid1")
    result = simulate(
        metadata,
        forwards,
        policy=policy,
        capacity_per_layer=2,
        host_to_device_gigabytes_per_second=HOST_TO_DEVICE_GIGABYTES_PER_SECOND,
    )
    prefill_misses, *decode_misses = forward_misses
    decode_experts_used = 5 + 4
    assert result.prefill_added_milliseconds_mean == pytest.approx(prefill_misses)
    assert result.decode_added_milliseconds_per_step_mean == pytest.approx(
        sum(decode_misses) / len(decode_misses)
    )
    assert result.decode_miss_bytes_per_frame == pytest.approx(
        sum(decode_misses) * BYTES_PER_ROUTED_EXPERT / 3
    )
    assert result.decode_hit_rate == pytest.approx(
        1 - sum(decode_misses) / decode_experts_used
    )
    assert result.decode_overflow_step_fraction == pytest.approx(decode_overflow)
    assert result.resident_expert_gigabytes == pytest.approx(2 * 2 * 0.001)
