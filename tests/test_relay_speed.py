import multiprocessing
import time

import numpy as np
import torch

from sglang_omni.relay.nixl import NIXL_AVAILABLE, NixlRelay

DATA_SIZE_MB = 1024  # Data size per transfer (MB)
POOL_SIZE_MB = 1024 * 10  # Total pool size (MB), must be greater than DATA_SIZE_MB
NUM_ITERS = 20  # Number of iterations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sender_process(meta_queue):
    """
    Sender process:
    1. Initialize memory pool (one-time registration)
    2. Loop:
       - Generate data
       - Put (D2D Copy into Pool)
       - Send metadata (Offset)
       - Wait for notification (Receiver finished reading)
       - Reset Pool (simulate reuse)
    """
    try:
        print(f"[Sender] Initializing Pool ({POOL_SIZE_MB} MB) on {DEVICE}...")
        # Initialize Relay, triggers one-time large memory registration
        relay = NixlRelay("sender_engine", pool_size_mb=POOL_SIZE_MB, device=DEVICE)

        # Prepare source data container (reusable since data is copied into Pool)
        element_size = 2 if DEVICE == "cuda" else 4
        num_elements = (DATA_SIZE_MB * 1024 * 1024) // element_size
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        print("[Sender] Starting Pool-based loop...")

        for i in range(NUM_ITERS):
            val = float(i + 1)
            src_tensor = torch.zeros(num_elements, dtype=dtype, device=DEVICE)
            src_tensor.fill_(val)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

            # Put: Copy data into Pool (fast D2D copy, no pin memory overhead)
            op = relay.put(src_tensor)

            meta_queue.put(op.metadata())

            # Wait for Receiver to finish reading
            # Must wait! Otherwise after reset_pool, Receiver might read overwritten data
            op.wait_for_completion()

            # Reset Pool allocator (simulates reusing the same Pool space)
            relay.reset_pool()

        print("[Sender] All finished.")
        time.sleep(1)

    except Exception as e:
        print(f"[Sender] Error: {e}")
        import traceback

        traceback.print_exc()


def receiver_process(meta_queue):
    """
    Receiver process:
    1. Initialize memory pool
    2. Loop:
       - Receive metadata
       - Get (RDMA Read -> Pool -> D2D Copy to Dest)
       - Verify
       - Reset Pool
    """
    try:
        print(f"[Receiver] Initializing Pool ({POOL_SIZE_MB} MB) on {DEVICE}...")
        relay = NixlRelay("receiver_engine", pool_size_mb=POOL_SIZE_MB, device=DEVICE)

        element_size = 2 if DEVICE == "cuda" else 4
        num_elements = (DATA_SIZE_MB * 1024 * 1024) // element_size
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        # Prepare destination container
        dest_tensor = torch.zeros(num_elements, dtype=dtype, device=DEVICE)

        latencies = []

        # First receive includes handshake overhead, treat as warmup
        print("[Receiver] Waiting for first transfer...")

        for i in range(NUM_ITERS):
            remote_meta = meta_queue.get()

            t0 = time.perf_counter()

            # Get: Pull data (includes Alloc(Local) -> RDMA Read -> D2D Copy)
            # Blocks until transfer completes
            relay.get(remote_meta, dest_tensor)

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            if NIXL_AVAILABLE:
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                expected_val = float(i + 1)
                if dest_tensor[0].item() == expected_val:
                    print(f"✅ [Iter {i}] Verified. Latency: {latencies[-1]:.2f} ms")
                else:
                    print(
                        f"❌ [Iter {i}] Failed! Exp: {expected_val}, Got: {dest_tensor[0].item()}"
                    )
            else:
                print(f"✅ [Iter {i}] Mock Verified")

            relay.reset_pool()

        print("\n" + "=" * 40)
        print(f"Pool-based Relay Benchmark")
        print(f"Avg Latency: {np.mean(latencies):.2f} ms")
        print("Note: This should be much faster due to Zero-Copy & Pre-registration.")
        print("=" * 40)

    except Exception as e:
        print(f"[Receiver] Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    meta_queue = multiprocessing.Queue()

    p_sender = multiprocessing.Process(target=sender_process, args=(meta_queue,))
    p_receiver = multiprocessing.Process(target=receiver_process, args=(meta_queue,))

    p_sender.start()
    time.sleep(
        2
    )  # Give Sender more initialization time (large memory registration is slow)
    p_receiver.start()

    p_receiver.join()
    p_sender.terminate()
