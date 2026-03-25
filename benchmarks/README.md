# SGLang Omni Benchmarks

Comprehensive benchmark suite for SGLang Omni, covering both performance (latency, throughput, RTF etc.) and accuracy (quality metrics) across all supported modality combinations. Omni models operate on a `{video, audio, text} x {video, audio, text}` input-output matrix. The table below tracks benchmark coverage.

## Performance Benchmarks

### TTS Voice Cloning

[`performance/tts/benchmark_tts_speed.py`](performance/tts/benchmark_tts_speed.py): Benchmarks online serving latency and throughput for TTS models via the `/v1/audio/speech` HTTP API.
