# Benchmark Relay

Relay is the core component of SGLang-Omni. It is responsible for transferring data between stages. We provide a benchmark script to measure the performance of different communication backends.

## Benchmark Script

```bash
python benchmark_relay.py \
    --backend-type all \
    --start-size 16 \
    --end-size 1024 \
    --factor 2 \
    --output-dir ./results
```
