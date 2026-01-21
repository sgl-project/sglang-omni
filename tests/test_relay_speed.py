import asyncio  # [New] Needed for async relay
import multiprocessing
import time

import numpy as np
import torch

from sglang_omni.relay.nixl import NIXL_AVAILABLE, NixlRelay

DATA_SIZE_MB = 1024  # Data size per transfer (MB)
# Configure Pool based on Slots and Credits
SLOT_SIZE_MB = 1024  # Size of one slot (must be >= DATA_SIZE_MB)
POOL_CREDITS = 10  # Number of concurrent slots (Total Pool = 10 * 1024 = 10GB)

NUM_ITERS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sender_process(meta_queue):
    """
    Sender process (Async):
    1. Initialize Relay (Allocates Slots * Credits)
    2. Async Loop:
       - Generate data
       - await put_async (Acquire Credit -> D2D Copy -> Send Meta)
       - Send metadata to queue
       - await op.wait_for_completion (Wait for 'done' -> Release Credit)
    """
    try:
        print(
            f"[Sender] Initializing Relay (Slot: {SLOT_SIZE_MB} MB, Credits: {POOL_CREDITS}) on {DEVICE}..."
        )
        # [Mod] New Init Signature
        relay = NixlRelay(
            "sender_engine",
            slot_size_mb=SLOT_SIZE_MB,
            credits=POOL_CREDITS,
            device=DEVICE,
        )

        element_size = 2 if DEVICE == "cuda" else 4
        num_elements = (DATA_SIZE_MB * 1024 * 1024) // element_size
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        async def _sender_loop():
            print("[Sender] Starting Async Loop...")

            # Pre-allocate source tensor to avoid malloc overhead inside loop
            src_tensor = torch.zeros(num_elements, dtype=dtype, device=DEVICE)

            for i in range(NUM_ITERS):
                val = float(i + 1)
                src_tensor.fill_(val)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

                # [Mod] Async Put
                # 1. Waits for a free credit if pool is full
                # 2. Copies data to pool
                # 3. Returns operation handle
                op = await relay.put_async(src_tensor)

                # Metadata handling
                metadata = op.metadata
                if callable(metadata):
                    metadata = metadata()  # Compatibility check

                meta_queue.put(metadata)

                # [Mod] Wait for Receiver
                # This ensures the receiver has finished reading this slot.
                # Once returned, the credit is automatically released back to the pool.
                await op.wait_for_completion()

            print("[Sender] All finished.")
            # Give receiver a moment to print final stats
            await asyncio.sleep(1)

        # [Mod] Run Async Loop
        asyncio.run(_sender_loop())

    except Exception as e:
        print(f"[Sender] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "relay" in locals():
            relay.close()


def receiver_process(meta_queue):
    """
    Receiver process (Async):
    1. Initialize Relay
    2. Async Loop:
       - Receive metadata from queue
       - await get_async (Acquire Local Credit -> RDMA Read)
       - await op.wait_for_completion (Wait RDMA -> D2D Copy -> Release Credit)
       - Verify
    """
    try:
        print(
            f"[Receiver] Initializing Relay (Slot: {SLOT_SIZE_MB} MB, Credits: {POOL_CREDITS}) on {DEVICE}..."
        )
        relay = NixlRelay(
            "receiver_engine",
            slot_size_mb=SLOT_SIZE_MB,
            credits=POOL_CREDITS,
            device=DEVICE,
        )

        element_size = 2 if DEVICE == "cuda" else 4
        num_elements = (DATA_SIZE_MB * 1024 * 1024) // element_size
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        dest_tensor = torch.zeros(num_elements, dtype=dtype, device=DEVICE)
        latencies = []

        async def _receiver_loop():
            print("[Receiver] Waiting for first transfer...")

            for i in range(NUM_ITERS):
                # Blocking queue get is fine here for benchmark sync
                remote_meta = meta_queue.get()

                t0 = time.perf_counter()

                # [Mod] Async Get
                # 1. Allocates local slot (Credit)
                # 2. Triggers RDMA Read
                op = await relay.get_async(remote_meta, dest_tensor)

                # [Mod] Wait for Completion
                # 1. Waits for RDMA to finish
                # 2. Copies from Pool to dest_tensor
                # 3. Releases local credit
                await op.wait_for_completion()

                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

                # Verification
                if NIXL_AVAILABLE:
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()

                    # check first element
                    val = dest_tensor[0].item()
                    expected_val = float(i + 1)

                    if val == expected_val:
                        print(
                            f"✅ [Iter {i}] Verified. Latency: {latencies[-1]:.2f} ms"
                        )
                    else:
                        print(f"❌ [Iter {i}] Failed! Exp: {expected_val}, Got: {val}")
                else:
                    print(f"✅ [Iter {i}] Mock Verified")

            print("\n" + "=" * 40)
            print(f"Async Credit-based Relay Benchmark")
            print(f"Avg Latency: {np.mean(latencies):.2f} ms")
            print("=" * 40)

        # [Mod] Run Async Loop
        asyncio.run(_receiver_loop())

    except Exception as e:
        print(f"[Receiver] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "relay" in locals():
            relay.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    meta_queue = multiprocessing.Queue()

    p_sender = multiprocessing.Process(target=sender_process, args=(meta_queue,))
    p_receiver = multiprocessing.Process(target=receiver_process, args=(meta_queue,))

    p_sender.start()
    time.sleep(2)
    p_receiver.start()

    p_receiver.join()
    p_sender.terminate()
