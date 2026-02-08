# SPDX-License-Identifier: Apache-2.0
"""
Relay benchmark helper.

This is intentionally guarded by `if __name__ == "__main__"` so it won't run under pytest.
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
import torch

# ================= Configuration =================
POOL_CREDITS = 4  # Number of concurrent slots
NUM_ITERS = 20  # Measurement iterations
WARMUP_ITERS = 10  # Warm-up iterations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def sender_process(
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
            "credits": POOL_CREDITS,
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
            print(f"[Sender] Warm-up {WARMUP_ITERS} iters...")
            for i in range(WARMUP_ITERS):
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

            print(f"[Sender] Measuring {NUM_ITERS} iters...")
            for i in range(NUM_ITERS):
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
    meta_queue,
    barrier,
    relay_type: str,
    data_size_mb: int,
    master_addr: str,
    master_port: str,
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
            "credits": POOL_CREDITS,
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
            print("[Receiver] Ready.")

            for _ in range(WARMUP_ITERS):
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

            print("[Receiver] Measuring...")
            dest_tensor = torch.zeros(
                num_elements, dtype=dtype, device=relay_kwargs["device"]
            )
            for i in range(NUM_ITERS):
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
                    print(f"[Iter {i}] {lat_ms:.2f} ms | {bw_gbs:.2f} GB/s")
                else:
                    print(f"[Iter {i}] Mismatch! Exp: {expected}, Got: {val}")

            if latencies_ms:
                print("\n" + "=" * 50)
                print(f"Result for {relay_type.upper()} Relay ({data_size_mb} MB)")
                print(f"Avg Latency:    {np.mean(latencies_ms):.2f} ms")
                print(f"Avg Throughput: {np.mean(throughputs_gbs):.2f} GB/s")
                print("=" * 50)

        asyncio.run(_receiver_loop())

    except Exception:
        traceback.print_exc()
    finally:
        if "relay" in locals() and hasattr(relay, "close"):
            relay.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relay Benchmark")
    parser.add_argument(
        "--type", type=str, default="nccl", choices=["nixl", "shm", "nccl", "mooncake"]
    )
    parser.add_argument("--size", type=int, default=10, help="Data size in MB")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"--- Benchmarking {args.type.upper()} Relay with {args.size} MB data ---")

    meta_queue = multiprocessing.Queue()
    init_barrier = multiprocessing.Barrier(2)

    master_addr = "127.0.0.1"
    master_port = find_free_port()

    p_sender = multiprocessing.Process(
        target=sender_process,
        args=(meta_queue, init_barrier, args.type, args.size, master_addr, master_port),
    )
    p_receiver = multiprocessing.Process(
        target=receiver_process,
        args=(meta_queue, init_barrier, args.type, args.size, master_addr, master_port),
    )

    p_sender.start()
    p_receiver.start()

    p_sender.join()
    p_receiver.join()
