# SPDX-License-Identifier: Apache-2.0
"""Simulate a bounded per-layer routed-expert cache over a Drama 3 expert trace.

Reads the files benchmarks.drama3_expert_trace appends, then reports, per policy
and capacity, the expert bytes kept on the GPU and the bytes a decode step or a
prefill would copy in from host memory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks.drama3_expert_trace import (
    EXPERT_IDS_FILE_NAME,
    FORWARDS_FILE_NAME,
    METADATA_FILE_NAME,
)

CachePolicy = Literal["lru", "static"]
CACHE_POLICIES: tuple[CachePolicy, ...] = ("lru", "static")
BYTES_PER_GIGABYTE = 1e9
MILLISECONDS_PER_SECOND = 1e3
TOP_EXPERT_SHARE_FRACTIONS = (0.25, 0.5)


@dataclass(kw_only=True)
class TraceMetadata:
    checkpoint_path: str
    quantization: str | None
    device_name: str
    moe_layer_ids: list[int]
    routed_expert_count: int
    top_k: int
    bytes_per_routed_expert: int


@dataclass(kw_only=True)
class TracedForward:
    is_prefill: bool
    request_count: int
    # [layers, routed experts]: whether any token of this forward routed to the expert.
    expert_used: np.ndarray


@dataclass(kw_only=True)
class CacheSimulationResult:
    policy: CachePolicy
    capacity_per_layer: int
    resident_expert_gigabytes: float
    full_resident_expert_gigabytes: float
    decode_step_count: int
    decode_frame_count: int
    decode_hit_rate: float
    decode_miss_bytes_per_frame: float
    decode_miss_bytes_per_step_mean: float
    decode_added_milliseconds_per_step_mean: float
    decode_added_milliseconds_per_step_p95: float
    decode_overflow_step_fraction: float
    prefill_count: int
    prefill_added_milliseconds_mean: float
    prefill_added_milliseconds_p95: float


def load_trace(process_dir: Path) -> tuple[TraceMetadata, list[TracedForward]]:
    """Read one engine process's trace into per-forward expert-use masks."""
    metadata_fields = json.loads((process_dir / METADATA_FILE_NAME).read_text())
    metadata = TraceMetadata(
        **{
            field: metadata_fields[field]
            for field in TraceMetadata.__dataclass_fields__
        }
    )
    layer_count = len(metadata.moe_layer_ids)
    expert_ids = np.fromfile(process_dir / EXPERT_IDS_FILE_NAME, dtype=np.int16)
    # A server killed mid-write can leave part of a token's row.
    token_row_size = layer_count * metadata.top_k
    expert_ids = expert_ids[: expert_ids.size - expert_ids.size % token_row_size]
    expert_ids = expert_ids.reshape(-1, layer_count, metadata.top_k)
    layer_index = np.arange(layer_count)[None, :, None]
    forwards = []
    token_offset = 0
    with (process_dir / FORWARDS_FILE_NAME).open() as forwards_file:
        for line in forwards_file:
            forward_fields = json.loads(line)
            token_count = sum(forward_fields["token_counts"])
            forward_expert_ids = expert_ids[token_offset : token_offset + token_count]
            if forward_expert_ids.shape[0] != token_count:
                # A server killed mid-write leaves a forward line without its ids.
                break
            else:
                pass
            expert_used = np.zeros((layer_count, metadata.routed_expert_count), bool)
            expert_used[
                np.broadcast_to(layer_index, forward_expert_ids.shape),
                forward_expert_ids,
            ] = True
            forwards.append(
                TracedForward(
                    is_prefill=forward_fields["is_prefill"],
                    request_count=len(forward_fields["request_ids"]),
                    expert_used=expert_used,
                )
            )
            token_offset += token_count
    return metadata, forwards


def count_layer_misses(
    policy: CachePolicy,
    capacity_per_layer: int,
    forwards: list[TracedForward],
    static_resident: np.ndarray,
) -> list[int]:
    """Experts each forward copies in across all layers, in forward order.

    lru keeps the most recently used experts of each layer; static keeps a fixed
    set and streams every other expert in without caching it.
    """
    layer_count = forwards[0].expert_used.shape[0] if forwards else 0
    layer_caches: list[OrderedDict[int, None]] = [
        OrderedDict() for _ in range(layer_count)
    ]
    forward_miss_counts = []
    for forward in forwards:
        if policy == "static":
            forward_miss_counts.append(
                int((forward.expert_used & ~static_resident).sum())
            )
        else:
            miss_count = 0
            for layer_index, layer_cache in enumerate(layer_caches):
                for expert_id in np.flatnonzero(forward.expert_used[layer_index]):
                    expert_key = int(expert_id)
                    if expert_key in layer_cache:
                        layer_cache.move_to_end(expert_key)
                    else:
                        miss_count += 1
                        layer_cache[expert_key] = None
                while len(layer_cache) > capacity_per_layer:
                    layer_cache.popitem(last=False)
            forward_miss_counts.append(miss_count)
    return forward_miss_counts


def simulate(
    metadata: TraceMetadata,
    forwards: list[TracedForward],
    *,
    policy: CachePolicy,
    capacity_per_layer: int,
    host_to_device_gigabytes_per_second: float,
) -> CacheSimulationResult:
    expert_bytes = metadata.bytes_per_routed_expert
    layer_count = len(metadata.moe_layer_ids)
    access_counts = sum(forward.expert_used.astype(np.int64) for forward in forwards)
    static_resident = np.zeros_like(access_counts, dtype=bool)
    hottest_experts = np.argsort(-access_counts, axis=1, kind="stable")
    np.put_along_axis(
        static_resident, hottest_experts[:, :capacity_per_layer], True, axis=1
    )
    miss_counts = np.array(
        count_layer_misses(policy, capacity_per_layer, forwards, static_resident)
    )
    is_prefill = np.array([forward.is_prefill for forward in forwards])
    used_counts = np.array([int(forward.expert_used.sum()) for forward in forwards])
    overflow = np.array(
        [
            bool((forward.expert_used.sum(axis=1) > capacity_per_layer).any())
            for forward in forwards
        ]
    )
    frame_count = sum(
        forward.request_count for forward in forwards if not forward.is_prefill
    )
    added_milliseconds = (
        miss_counts
        * expert_bytes
        / (host_to_device_gigabytes_per_second * BYTES_PER_GIGABYTE)
        * MILLISECONDS_PER_SECOND
    )
    decode_misses = miss_counts[~is_prefill]
    decode_added = added_milliseconds[~is_prefill]
    prefill_added = added_milliseconds[is_prefill]
    return CacheSimulationResult(
        policy=policy,
        capacity_per_layer=capacity_per_layer,
        resident_expert_gigabytes=capacity_per_layer
        * layer_count
        * expert_bytes
        / BYTES_PER_GIGABYTE,
        full_resident_expert_gigabytes=metadata.routed_expert_count
        * layer_count
        * expert_bytes
        / BYTES_PER_GIGABYTE,
        decode_step_count=int(decode_misses.size),
        decode_frame_count=frame_count,
        decode_hit_rate=1.0
        - float(decode_misses.sum()) / max(int(used_counts[~is_prefill].sum()), 1),
        decode_miss_bytes_per_frame=float(decode_misses.sum())
        * expert_bytes
        / max(frame_count, 1),
        decode_miss_bytes_per_step_mean=(
            float(decode_misses.mean()) * expert_bytes if decode_misses.size else 0.0
        ),
        decode_added_milliseconds_per_step_mean=(
            float(decode_added.mean()) if decode_added.size else 0.0
        ),
        decode_added_milliseconds_per_step_p95=(
            float(np.percentile(decode_added, 95)) if decode_added.size else 0.0
        ),
        decode_overflow_step_fraction=(
            float(overflow[~is_prefill].mean()) if decode_added.size else 0.0
        ),
        prefill_count=int(prefill_added.size),
        prefill_added_milliseconds_mean=(
            float(prefill_added.mean()) if prefill_added.size else 0.0
        ),
        prefill_added_milliseconds_p95=(
            float(np.percentile(prefill_added, 95)) if prefill_added.size else 0.0
        ),
    )


@dataclass(kw_only=True)
class LocalitySummary:
    # Mean distinct experts one layer needs in a decode step, by the step's request count.
    decode_experts_per_layer_by_batch_size: dict[int, float]
    decode_experts_per_layer_max: int
    prefill_experts_per_layer_mean: float
    # Share of decode expert uses that go to the hottest fraction of each layer's experts.
    decode_use_share_of_hottest_experts: dict[float, float]


def summarize_locality(
    metadata: TraceMetadata, forwards: list[TracedForward]
) -> LocalitySummary:
    decode_forwards = [forward for forward in forwards if not forward.is_prefill]
    prefill_forwards = [forward for forward in forwards if forward.is_prefill]
    experts_per_layer_by_batch_size: dict[int, list[float]] = {}
    for forward in decode_forwards:
        experts_per_layer_by_batch_size.setdefault(forward.request_count, []).append(
            float(forward.expert_used.sum(axis=1).mean())
        )
    decode_use_counts = sum(
        forward.expert_used.astype(np.int64) for forward in decode_forwards
    )
    sorted_use_counts = -np.sort(-np.asarray(decode_use_counts), axis=1)
    layer_use_totals = np.maximum(sorted_use_counts.sum(axis=1), 1)
    hottest_shares = {}
    for fraction in TOP_EXPERT_SHARE_FRACTIONS:
        hottest_count = math.ceil(fraction * metadata.routed_expert_count)
        hottest_shares[fraction] = float(
            (sorted_use_counts[:, :hottest_count].sum(axis=1) / layer_use_totals).mean()
        )
    return LocalitySummary(
        decode_experts_per_layer_by_batch_size={
            batch_size: float(np.mean(values))
            for batch_size, values in sorted(experts_per_layer_by_batch_size.items())
        },
        decode_experts_per_layer_max=max(
            (int(forward.expert_used.sum(axis=1).max()) for forward in decode_forwards),
            default=0,
        ),
        prefill_experts_per_layer_mean=(
            float(
                np.mean(
                    [
                        forward.expert_used.sum(axis=1).mean()
                        for forward in prefill_forwards
                    ]
                )
            )
            if prefill_forwards
            else 0.0
        ),
        decode_use_share_of_hottest_experts=hottest_shares,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_dir", type=Path, help="one process directory (pid<N>) of a trace"
    )
    parser.add_argument(
        "--host-to-device-gigabytes-per-second",
        type=float,
        required=True,
        help="pinned host-to-device copy bandwidth measured on the target GPU",
    )
    parser.add_argument(
        "--capacity-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="experts kept per layer, as a fraction of the routed experts",
    )
    parser.add_argument("--output", type=Path, help="write the report as JSON here")
    arguments = parser.parse_args()
    metadata, forwards = load_trace(arguments.trace_dir)
    if not forwards:
        raise ValueError(f"{arguments.trace_dir} holds no traced forward")
    else:
        pass
    locality = summarize_locality(metadata, forwards)
    results = [
        simulate(
            metadata,
            forwards,
            policy=policy,
            capacity_per_layer=math.ceil(fraction * metadata.routed_expert_count),
            host_to_device_gigabytes_per_second=arguments.host_to_device_gigabytes_per_second,
        )
        for fraction in arguments.capacity_fractions
        for policy in CACHE_POLICIES
    ]
    print(
        f"{metadata.checkpoint_path} ({metadata.quantization}), "
        f"{len(metadata.moe_layer_ids)} MoE layers x {metadata.routed_expert_count} "
        f"experts, top-{metadata.top_k}, "
        f"{metadata.bytes_per_routed_expert / 2**20:.2f} MiB per expert"
    )
    print(f"locality: {json.dumps(asdict(locality))}")
    print(
        "policy capacity resident_GB/full_GB decode_hit miss_MB/frame "
        "decode_ms_mean decode_ms_p95 overflow prefill_ms_p95"
    )
    for result in results:
        print(
            f"{result.policy:6} {result.capacity_per_layer:8} "
            f"{result.resident_expert_gigabytes:6.2f}/"
            f"{result.full_resident_expert_gigabytes:<6.2f} "
            f"{result.decode_hit_rate:10.3f} "
            f"{result.decode_miss_bytes_per_frame / 1e6:13.2f} "
            f"{result.decode_added_milliseconds_per_step_mean:14.2f} "
            f"{result.decode_added_milliseconds_per_step_p95:13.2f} "
            f"{result.decode_overflow_step_fraction:8.3f} "
            f"{result.prefill_added_milliseconds_p95:14.2f}"
        )
    if arguments.output is None:
        pass
    else:
        arguments.output.write_text(
            json.dumps(
                {
                    "metadata": asdict(metadata),
                    "host_to_device_gigabytes_per_second": (
                        arguments.host_to_device_gigabytes_per_second
                    ),
                    "locality": asdict(locality),
                    "results": [asdict(result) for result in results],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
