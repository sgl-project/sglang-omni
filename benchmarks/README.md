# Benchmarks

## benchmark_tts_speed.py

Benchmark online serving latency and throughput for TTS models via the `/v1/audio/speech` HTTP API.

Supports two input modes:

- **Voice cloning** with a seed-tts-eval `meta.lst` testset (`--testset`)
- **Plain text** prompts (`--prompts`)

Metrics reported: latency (mean/median/p95/p99), real-time factor (RTF), audio duration, and throughput (req/s).
