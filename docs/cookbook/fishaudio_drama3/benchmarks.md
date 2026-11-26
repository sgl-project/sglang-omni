---
orphan: true
---

# Fish Audio Drama 3 Benchmarks

[Back to the Fish Audio Drama 3 quick start](../fishaudio_drama3.md).

The measurements behind the settings on the [Fish Audio Drama 3](../fishaudio_drama3.md) and [Tuning](tuning.md) pages. Each table names its hardware, workload and checkpoint; compare rows within a table rather than across tables.

## Streaming Performance

All numbers below were measured on NVIDIA H200 GPUs with streaming requests on Seed-TTS English prompts. Each request clones a voice from one pre-encoded reference (`vq_codes` with its transcript) and uses `temperature` 1.0, `top_p` 0.9, `top_k` 30 and `max_new_tokens` 500. Throughput counts generated acoustic frames per second. TTFA is the time to first audio. RTF is the time from sending a request to its last audio chunk, divided by the duration of the returned audio.

| Concurrency | Throughput (frames/s) | TTFA p50 | TTFA p95 | RTF (mean) |
|---:|---:|---:|---:|---:|
| 1 | 166.7 | 88.9 ms | 90.8 ms | 0.131 |
| 2 | 287.0 | 96.2 ms | 130.1 ms | 0.152 |
| 4 | 467.6 | 108.2 ms | 156.4 ms | 0.186 |
| 8 | 678.1 | 153.2 ms | 210.9 ms | 0.256 |
| 16 | 927.6 | 210.1 ms | 287.2 ms | 0.373 |
| 32 | 1,244.7 | 304.6 ms | 396.1 ms | 0.551 |
| 64 | 1,687.3 | 450.9 ms | 1,020.1 ms | 0.801 |

*Table 1. Drama 3 streaming performance on one H200. Each cell is the mean of four runs.*

A single stream gets its first audio in under 100 ms. At concurrency 16, audio is still generated about 2.7× faster than it plays back, and at concurrency 64 mean RTF stays at 0.80. In SGLang-Omni, how the pipeline stages are grouped into processes and placed on GPUs is deployment-time configuration rather than per-model code. Table 2 uses this to compare two vocoder placements at higher concurrency, with the vocoder in its own process and its default GPU share, 1,088 requests per concurrency level, and the average of six server starts. Its throughput also counts the end-of-speech frame that ends each stream.

| Vocoder placement | Concurrency | Throughput (frames/s) | TTFA p50 | TTFA p95 |
|---|---:|---:|---:|---:|
| Engine's GPU | 128 | 2,177 | 685 ms | 1,412 ms |
| Engine's GPU | 256 | 2,422 | 3,403 ms | 4,255 ms |
| Second GPU | 128 | 3,692 | 401 ms | 995 ms |
| Second GPU | 256 | 4,160 | 2,146 ms | 2,469 ms |

*Table 2. Drama 3 streaming performance by vocoder placement. Each cell is the mean of six server starts.*

With the vocoder on the engine's GPU, at 256 streams the vocoder is fully loaded and each new stream's first audio waits in its queue. Moving it to a second GPU, as described under [Deployment Performance](tuning.md#deployment-performance), gave about 72% more throughput and a 37% lower median TTFA at 256 streams. Use this layout when you need many concurrent streams and a short time to first audio together.

## FP8 and Int4 Checkpoints

Table 3 compares the two checkpoints on one H200 with the default server settings: the 1,088 Seed-TTS English prompts, each cloning the voice of its own reference clip, sent as non-streaming requests at concurrency 8. WER is computed on the generated audio transcribed by Qwen3-ASR-1.7B.

| Checkpoint | Weights | Latency mean | Latency p95 | RTF mean | Throughput (req/s) | WER (corpus) | WER without samples above 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fishaudio/drama-3` (FP8 experts) | 37.0 GB | 1.235 s | 1.902 s | 0.281 | 6.47 | 1.99% | 1.32% |
| `fishaudio/drama-3-int4` | 20.8 GB | 1.334 s | 2.178 s | 0.320 | 5.99 | 1.63% | 1.42% |

*Table 3. Drama 3 FP8 and int4 checkpoints on Seed-TTS English, concurrency 8, one H200.*

The int4 checkpoint is about 8% slower per request and its WER stays at the FP8 level: the gap between the two is about as large as the run-to-run spread of one checkpoint (a second FP8 run measured 1.365 s mean latency and 2.02% WER).

## Tensor-Parallel Layouts

Table 4 compares the layouts under [Tensor Parallelism](tuning.md#tensor-parallelism) with one H200 on the Seed-TTS English workload of Table 3, all measured with checkpoint revision `78e5ab7`; compare its rows with each other rather than with Table 3.

| Setup | Latency mean | Latency p95 | RTF mean | Throughput (req/s) | WER (corpus) | Samples above 50% WER |
|---|---:|---:|---:|---:|---:|---:|
| One H200 | 1.048 s | 1.551 s | 0.237 | 7.62 | 0.85% | 0 |
| Two RTX 5090, tensor parallel | 1.255 s | 1.876 s | 0.283 | 6.37 | 0.90% | 0 |
| Four RTX 5090, tensor parallel | 1.403 s | 2.081 s | 0.316 | 5.69 | 0.87% | 0 |

*Table 4. Drama 3 FP8 checkpoint on Seed-TTS English, non-streaming requests at concurrency 8.*

## Native MLX Parity and Performance

The numerical reference for the [native MLX server](platforms.md) is mlx-audio 0.6.2 (`f6d5da9`: `mlx_audio/tts/models/fish_drama3` and `mlx_audio/codec/models/fish_s1_dac`) on the same checkpoint and MLX 0.32.3; `native/licenses/fishaudio_drama3-MIT.txt` holds its license. The native runtime runs the gated delta rule on MLX's `mx::fast::gated_delta_update`, where mlx-audio runs its own Metal kernel, and the two round their bf16 outputs differently. With mlx-audio's gated delta step on the same MLX primitive and the same seed (given to `fishaudio_drama3_synthesize`), the native runtime samples the same codes and decodes the same PCM16 samples on every seeded test request.

Measured on an Apple M4 Max (64 GB), warm, in ABBA order (native, reference, reference, native). The server takes no seed, so each side ran each request three times; the table gives each round's median real-time factor (generation time over audio duration) and the median time to the first streamed audio. Native times include the HTTP request and PCM encoding. The 4-bit checkpoint:

| Request | Real-time factor, native | Real-time factor, mlx-audio 0.6.2 | First streamed audio, native | First streamed audio, mlx-audio |
|---|---:|---:|---:|---:|
| English, no reference | 0.376 / 0.377 | 0.372 / 0.372 | 0.18 s | 0.18 s |
| English, cloned voice | 0.375 / 0.373 | 0.375 / 0.372 | 0.18 s | 0.18 s |
| Chinese, cloned voice | 0.367 / 0.370 | 0.366 / 0.366 | 0.18 s | 0.18 s |
| Two speakers, cloned voice | 0.371 / 0.373 | 0.368 / 0.369 | 0.19 s | 0.18 s |
| 26 s passage, two chunks | 0.374 / 0.371 | 0.369 / 0.377 | 0.22 s | 0.22 s |

Loading takes 4.1 s native and 4.5 s for the reference, from a warm file cache. Peak MLX memory (`mx.get_peak_memory()`) is 28.2 GB native on the passage and 28.7 GB in the reference; the server's process footprint stays near 32 GB under mixed load, against 23 GB idle.

The 8-bit checkpoint, measured the same way:

| Request | Real-time factor, native | Real-time factor, mlx-audio 0.6.2 | First streamed audio, native | First streamed audio, mlx-audio |
|---|---:|---:|---:|---:|
| English, no reference | 0.513 / 0.513 | 0.513 / 0.514 | 0.25 s | 0.25 s |
| English, cloned voice | 0.507 / 0.506 | 0.510 / 0.506 | 0.25 s | 0.25 s |
| Chinese, cloned voice | 0.500 / 0.499 | 0.502 / 0.501 | 0.25 s | 0.25 s |
| Two speakers, cloned voice | 0.504 / 0.500 | 0.508 / 0.505 | 0.26 s | 0.26 s |
| 26 s passage, two chunks | 0.496 / 0.495 | 0.498 / 0.497 | 0.30 s | 0.30 s |

Loading takes 6.4 to 6.8 s native and 6.9 to 7.2 s for the reference. Peak MLX memory is 47.1 GB in the reference, and the server's process footprint reaches 48 GB. Its routed experts are twice the size of the 4-bit ones and its other projections are bf16, and it runs about 1.35 times slower.
