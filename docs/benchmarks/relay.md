# Benchmark Relay

Relay is the core component of SGLang-Omni. It is responsible for transferring data between stages. We provide a benchmark script to measure the performance of different communication backends.

## Benchmark Script

```bash
python benchmarks/benchmark_relay.py \
    --backend-type all \
    --start-size 16 \
    --end-size 1024 \
    --factor 2 \
    --output-dir ./results
```

## Backend Availability

With `--backend-type all`, backends whose dependencies are missing on the host are skipped with a notice instead of failing the run:

- `shm`: host shared memory, runs on any platform (CPU tensors between same-node processes)
- `nccl`: requires at least 2 CUDA devices
- `nixl`: requires the `nixl` package
- `mooncake`: requires `mooncake-transfer-engine`

On CPU-only or macOS hosts only the `shm` backend is available; a single backend can also be selected explicitly with `--backend-type <name>`.
