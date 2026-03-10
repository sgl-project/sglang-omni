# SPDX-License-Identifier: Apache-2.0
"""
Relay benchmark helper.

Usage:

```bash
# benchmark nccl
python benchmark_relay.py --backend-type nccl --start-size 16 --end-size 1024 --factor 2 --warmup-iters 10 --num-iters 20 --pool-credits 4 --verbose --output-dir ./results
```
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import os
import socket
import time
import traceback

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Relay Benchmark")
    parser.add_argument(
        "-b",
        "--backend-type",
        type=str,
        default="all",
        choices=["nixl", "shm", "nccl", "mooncake", "all"],
        help="The type of communication backend for benchmarking",
    )
    parser.add_argument(
        "-s", "--start-size", type=int, default=16, help="Starting data size in MB"
    )
    parser.add_argument(
        "-e", "--end-size", type=int, default=1024, help="Ending data size in MB"
    )
    parser.add_argument(
        "-f", "--factor", type=int, default=2, help="Factor to multiply the data size"
    )
    parser.add_argument(
        "-w", "--warmup-iters", type=int, default=10, help="Warm-up iterations"
    )
    parser.add_argument(
        "-i", "--num-iters", type=int, default=20, help="Measurement iterations"
    )
    parser.add_argument(
        "-c", "--pool-credits", type=int, default=4, help="Number of concurrent slots"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for benchmarking results",
    )
    return parser.parse_args()


# ================= Configuration =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def sender_process(
    args,
    meta_queue,
    barrier,
    relay_type: str,
    data_size_mb: int,
    master_addr: str,
    master_port: str,
):
    """Sender Process: Rank 0 for NCCL."""
    try:
        if relay_type.lower() == "nccl":
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            barrier.wait()

        relay_kwargs = {
            "engine_id": "sender_engine",
            "slot_size_mb": data_size_mb,
            "credits": args.pool_credits,
            "device": (
                "cuda:0" if relay_type.lower() in ["nccl", "mooncake"] else DEVICE
            ),
        }

        # NCCL-specific parameters (ignored by Shm/Nixl)
        relay_kwargs.update(
            {"rank": 0, "world_size": 2, "send_to_ranks": [1], "recv_from_ranks": []}
        )

        if relay_type.lower() == "nixl":
            from sglang_omni.relay.nixl import NixlRelay as RelayCls
        elif relay_type.lower() == "shm":
            from sglang_omni.relay.shm import ShmRelay as RelayCls
        elif relay_type.lower() == "nccl":
            from sglang_omni.relay.nccl import NcclRelay as RelayCls
        elif relay_type.lower() == "mooncake":
            from sglang_omni.relay.mooncake import MooncakeRelay as RelayCls
        else:
            raise ValueError(f"Unknown relay type: {relay_type}")

        if relay_type.lower() != "nccl":
            for k in ["rank", "world_size", "send_to_ranks", "recv_from_ranks"]:
                relay_kwargs.pop(k, None)

        relay = RelayCls(**relay_kwargs)
        print("[Sender] Relay initialized.")

        element_size = 2 if "cuda" in str(relay_kwargs["device"]) else 4
        num_elements = (data_size_mb * 1024 * 1024) // element_size
        dtype = (
            torch.float16 if "cuda" in str(relay_kwargs["device"]) else torch.float32
        )

        async def _sender_loop():
            print(f"[Sender] Warm-up {args.warmup_iters} iters...")
            for i in range(args.warmup_iters):
                src_tensor = torch.zeros(
                    num_elements, dtype=dtype, device=relay_kwargs["device"]
                )
                src_tensor[0] = i + 0.5
                if "cuda" in str(relay_kwargs["device"]):
                    torch.cuda.synchronize()

                # NCCL requires dst_rank; other relays ignore it.
                op = await relay.put_async(src_tensor, dst_rank=1)
                meta_queue.put(op.metadata)
                await op.wait_for_completion()

            print(f"[Sender] Measuring {args.num_iters} iters...")
            for i in range(args.num_iters):
                src_tensor = torch.zeros(
                    num_elements, dtype=dtype, device=relay_kwargs["device"]
                )
                src_tensor[0] = i + 1.0
                if "cuda" in str(relay_kwargs["device"]):
                    torch.cuda.synchronize()

                op = await relay.put_async(src_tensor, dst_rank=1)
                meta_queue.put(op.metadata)
                await op.wait_for_completion()

            print("[Sender] Done.")
            meta_queue.put(None)  # EOS
            await asyncio.sleep(0.5)

        asyncio.run(_sender_loop())

    except Exception:
        traceback.print_exc()
    finally:
        if "relay" in locals() and hasattr(relay, "close"):
            relay.close()


def receiver_process(
    args,
    meta_queue,
    barrier,
    relay_type: str,
    data_size_mb: int,
    master_addr: str,
    master_port: str,
    results_pipe: multiprocessing.Pipe,
):
    """Receiver Process: Rank 1 for NCCL."""
    try:
        if relay_type.lower() == "nccl":
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            barrier.wait()

        relay_kwargs = {
            "engine_id": "receiver_engine",
            "slot_size_mb": data_size_mb,
            "credits": args.pool_credits,
            "device": (
                "cuda:1" if relay_type.lower() in ["nccl", "mooncake"] else DEVICE
            ),
        }

        relay_kwargs.update(
            {"rank": 1, "world_size": 2, "send_to_ranks": [], "recv_from_ranks": [0]}
        )

        if relay_type.lower() == "nixl":
            from sglang_omni.relay.nixl import NixlRelay as RelayCls
        elif relay_type.lower() == "shm":
            from sglang_omni.relay.shm import ShmRelay as RelayCls
        elif relay_type.lower() == "nccl":
            from sglang_omni.relay.nccl import NcclRelay as RelayCls
        elif relay_type.lower() == "mooncake":
            from sglang_omni.relay.mooncake import MooncakeRelay as RelayCls
        else:
            raise ValueError(f"Unknown relay type: {relay_type}")

        if relay_type.lower() != "nccl":
            for k in ["rank", "world_size", "send_to_ranks", "recv_from_ranks"]:
                relay_kwargs.pop(k, None)

        relay = RelayCls(**relay_kwargs)

        element_size = 2 if "cuda" in str(relay_kwargs["device"]) else 4
        num_elements = (data_size_mb * 1024 * 1024) // element_size
        dtype = (
            torch.float16 if "cuda" in str(relay_kwargs["device"]) else torch.float32
        )

        latencies_ms: list[float] = []
        throughputs_gbs: list[float] = []

        async def _receiver_loop():
            if args.verbose:
                print("[Receiver] Ready.")

            for _ in range(args.warmup_iters):
                remote_meta = meta_queue.get()
                if remote_meta is None:
                    return
                dest_tensor = torch.zeros(
                    num_elements, dtype=dtype, device=relay_kwargs["device"]
                )
                op = await relay.get_async(remote_meta, dest_tensor)
                await op.wait_for_completion()
                if "cuda" in str(relay_kwargs["device"]):
                    torch.cuda.synchronize()

            if args.verbose:
                print("[Receiver] Measuring...")
            dest_tensor = torch.zeros(
                num_elements, dtype=dtype, device=relay_kwargs["device"]
            )
            for i in range(args.num_iters):
                remote_meta = meta_queue.get()
                if remote_meta is None:
                    break

                if "cuda" in str(relay_kwargs["device"]):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                op = await relay.get_async(remote_meta, dest_tensor)
                await op.wait_for_completion()

                if "cuda" in str(relay_kwargs["device"]):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                lat_ms = (t1 - t0) * 1000
                bw_gbs = (data_size_mb / 1024) / (t1 - t0)

                latencies_ms.append(lat_ms)
                throughputs_gbs.append(bw_gbs)

                val = float(dest_tensor[0].item())
                expected = i + 1.0

                if abs(val - expected) < 0.1:
                    if args.verbose:
                        print(f"[Iter {i}] {lat_ms:.2f} ms | {bw_gbs:.2f} GB/s")
                else:
                    raise ValueError(
                        f"[Iter {i}] Mismatch! Exp: {expected}, Got: {val}"
                    )

            if latencies_ms:
                results_pipe.send((latencies_ms, throughputs_gbs))
                results_pipe.close()

        asyncio.run(_receiver_loop())

    except Exception:
        traceback.print_exc()
    finally:
        if "relay" in locals() and hasattr(relay, "close"):
            relay.close()


def benchmark_relay(args, backend_type: str):
    # the benchmark sizes should be a multiple of the factor and the start size
    benchmark_sizes = []
    start_size = args.start_size
    benchmark_sizes.append(start_size)
    while start_size < args.end_size:
        start_size *= args.factor
        benchmark_sizes.append(start_size)
    if benchmark_sizes[-1] != args.end_size:
        benchmark_sizes.append(args.end_size)

    latency_list = []
    throughput_list = []

    for size in tqdm(
        benchmark_sizes, desc=f"Benchmarking {backend_type.upper()} Relay"
    ):
        meta_queue = multiprocessing.Queue()
        init_barrier = multiprocessing.Barrier(2)
        receiver, sender = multiprocessing.Pipe()

        master_addr = "127.0.0.1"
        master_port = find_free_port()

        p_sender = multiprocessing.Process(
            target=sender_process,
            args=(
                args,
                meta_queue,
                init_barrier,
                backend_type,
                size,
                master_addr,
                master_port,
            ),
        )
        p_receiver = multiprocessing.Process(
            target=receiver_process,
            args=(
                args,
                meta_queue,
                init_barrier,
                backend_type,
                size,
                master_addr,
                master_port,
                sender,
            ),
        )
        p_sender.start()
        p_receiver.start()

        p_sender.join()
        p_receiver.join()
        latencies_ms, throughputs_gbs = receiver.recv()

        if args.verbose:
            print(
                f"--- Benchmarking {backend_type.upper()} Relay with {size} MB data ---"
            )
            print(f"Avg Latency:    {np.mean(latencies_ms):.2f} ms")
            print(f"Avg Throughput: {np.mean(throughputs_gbs):.2f} GB/s")
            print("=" * 50)

        latency_list.append(np.mean(latencies_ms))
        throughput_list.append(np.mean(throughputs_gbs))

    return benchmark_sizes, latency_list, throughput_list


def tabulate_and_save(
    backend_type, output_dir, benchmark_sizes, latency_list, throughput_list
):
    results = pd.DataFrame(
        {
            "Size (MB)": benchmark_sizes,
            "Latency (ms)": latency_list,
            "Throughput (GB/s)": throughput_list,
        }
    )
    table = tabulate(results, headers="keys", tablefmt="grid")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"benchmark_relay_{backend_type}_{timestamp}.txt"), "w"
    ) as f:
        f.write(table)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    if args.backend_type == "all":
        backend_types = ["nixl", "shm", "nccl", "mooncake"]
    else:
        backend_types = [args.backend_type]

    for backend_type in backend_types:
        benchmark_sizes, latency_list, throughput_list = benchmark_relay(
            args, backend_type
        )
        tabulate_and_save(
            backend_type,
            args.output_dir,
            benchmark_sizes,
            latency_list,
            throughput_list,
        )
