# 📊 Benchmarks

## 💬 Benchmark Relay

This benchmark measures the latency and throughput of the various backends for relay. You can use the sample command below to benchmark all the backends. If you want to benchmark a specific backend, you can use the `--backend-type` argument (check `python benchmark_relay.py --help` for more details).

```bash
python benchmark_relay.py \
    --backend-type all \
    --start-size 16 \
    --end-size 1024 \
    --factor 2 \
    --output-dir ./results
```

You can check the benchmark results in the `results` directory.
