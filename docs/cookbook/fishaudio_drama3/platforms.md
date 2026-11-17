---
orphan: true
---

# Fish Audio Drama 3 Platform Deployment

[Back to the Fish Audio Drama 3 quick start](../fishaudio_drama3.md).

The CUDA server on the [Fish Audio Drama 3](../fishaudio_drama3.md) page is the primary path. This page covers the native Apple Silicon runtime.

## Apple Silicon: native MLX

The standalone `fishaudio_drama3_server` runs entirely in C++ on MLX under `sglang_omni_mlx/native/` and serves `/v1/audio/speech` with the CUDA server's request fields; reference audio is sent as base64 WAV (see the table below). It serves two community MLX conversions of `fishaudio/drama-3`, made by mlx-audio's `fish_drama3/convert.py` from the official FP8 checkpoint:

| Checkpoint | Routed experts | Other projections | On disk | Peak MLX memory, mlx-audio | Unified memory |
|---|---|---|---:|---:|---:|
| `mlx-community/fish-audio-drama-3-4bit` | 4-bit | 8-bit | 24 GB | 28.7 GB | 48 GB or more |
| `mlx-community/fish-audio-drama-3-8bit` | 8-bit | bf16 | 42 GB | 47.1 GB | 64 GB or more |

Both quantize with MLX affine groups of 64 and keep the embeddings in bf16, the norm gains and the codec in fp32; the 8-bit checkpoint quantizes only the routed experts. The 4-bit checkpoint peaks above the GPU working set of a 32 or 36 GB Mac; the 8-bit one fits the 55.7 GB working set of a 64 GB Mac, with nothing else holding much memory. Both were measured on a 64 GB Mac. Download a pinned checkpoint, then build with the native runtime's MLX 0.32.3 toolchain:

```bash
python sglang_omni_mlx/native/ci/provision_models.py "$DRAMA3_DATA" \
  --model mlx-community/fish-audio-drama-3-4bit
sglang_omni_mlx/native/scripts/build_runtime.sh "$DRAMA3_RUNTIME" "$DRAMA3_BUILD"
"$DRAMA3_RUNTIME/bin/fishaudio_drama3_server" \
  --model-path "$DRAMA3_DATA/models/mlx-community_fish-audio-drama-3-4bit" \
  --model-name fishaudio/drama-3 --port 8000
```

Set `DRAMA3_DATA`, `DRAMA3_RUNTIME` and `DRAMA3_BUILD` to separate local directories; for the 8-bit checkpoint, replace `4bit` with `8bit` in both commands. Serving needs neither Python nor SGLang's CUDA package. Voice cloning sends the reference clip, a mono 44.1 kHz WAV, as base64:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"Get the trust fund to the bank early.\",
       \"references\": [{\"data\": \"$(base64 < reference_44k.wav | tr -d '\n')\",
                       \"text\": \"We asked over twenty different people, and they all said it was his.\"}]}" \
  --output output_voice_cloning.wav
```

The server reads the request fields the CUDA server reads and refuses the same unsupported ones (`seed` included). Where they differ:

| | CUDA server | Native MLX server |
|---|---|---|
| Reference audio | A path, URL or base64 data in any format the server decodes, resampled to 44.1 kHz; up to 60 s each by default | Base64 mono 44.1 kHz PCM16 or float32 WAV, as `references[i].data`, a `data:` URL in `references[i].audio`, or `ref_audio` with `ref_text`; paths and remote URLs are refused |
| References | Any number, within the prompt row limit (`max_prompt_rows`) | Up to 4, together at most 30 s (646 frames of `vq_codes`) and 1024 transcript characters |
| Voices | `default` (or `speaker`) or an uploaded voice | `default` (or `speaker`) only |
| Long input | One prompt; `max_new_tokens` defaults to the remaining context, up to 2048 frames | Split into chunks of about 300 bytes, each seeing the first and the previous chunk; `max_new_tokens` (default 1024, at most 4096) caps each chunk; `input` holds at most 4096 characters |
| Requests | Batched | One at a time; `--max-queued-requests` more may wait (2 by default), the rest get 503; a request not done `--request-timeout-s` seconds after it arrives (600 by default) gets 504, and a disconnected client's request is cancelled |
| Streaming | `"stream": true` with `"response_format": "pcm"` | The same, with audio every 7 frames (0.33 s) |

`fishaudio_drama3_synthesize` runs one request from the command line and prints its timings; with `--codes-output` it also writes the first chunk's prompt ids and every frame's codes, for parity checks. Parity with mlx-audio and the measured speed of both checkpoints are in [Benchmarks](benchmarks.md#native-mlx-parity-and-performance).

### Optional native expert offload

Both native binaries accept `--expert-offload 2gib`. Without it, the existing resident loading and execution path is unchanged. Offload removes routed backbone experts before checkpoint evaluation and loads one expert's packed up/gate/down weights, scales and biases on demand. Router, shared experts, attention and tower weights stay resident; the codec loads separately.

The model owns one serial LRU cache with a 2,147,483,648-byte limit on live packed expert arrays, including a pending load. A pinned expert cannot be evicted; every expert computation completes before its lease is released. Prefill groups selected rows by expert and processes groups serially without changing top-k ids or scores. Small groups may repeat rows to preserve MLX 0.32.3's sorted gather kernel; repeated outputs are discarded. This path prioritizes correctness over throughput and may perform many file reads.

For N input rows and top-k K, one expert group processes at most max(N, 16) rows. Per-expert result batches hold at most (N*K + 15*E) hidden-width rows before assembly, where E is the number of selected experts. Assembly additionally uses tensors shaped up to [N, K, hidden] and a final [N, hidden] result; backend workspace has no separate byte limit. Padding outputs and operation graphs are released after the final weighted sum is evaluated. Cached expert arrays remain until LRU eviction, explicit release before non-streaming codec decoding, or model destruction. During offloaded codec decoding only, the native runtime sets a target of min(the caller’s existing limit, 256 MiB) for MLX idle allocator buffers and clears old idle buffers, then restores the caller's previous cache limit on return or exception. This process-wide setting relies on the native server worker and CLI serializing MLX execution; concurrent embedded MLX jobs need their own coordination. It does not alter the 2 GiB expert capacity or AR-phase allocator limit. The idle-buffer target is soft and does not cap active tensors, workspace, process memory, or the operating system's page cache.

The limit is not a process or GPU memory cap. Resident model weights, output tensors, current operation workspace, MLX allocator reuse and CPU/GPU copies, and the operating system's file cache are separate. No duplicate host payload cache is retained. Packed weights remain U32; F16/BF16/F32 quantization scales and biases retain their original dtype. Unquantized expert weights are unsupported by this offload path.

`fishaudio_drama3_experts_test` is an explicit CMake target; pass a disposable fixture directory as its only argument. It generates small safetensors files and checks actual MLX loading, excluded payload reads, cache limits, pinning, eviction, failed reads, and resident/cached quantized execution. These fixtures do not establish complete checkpoint/audio parity or real-model throughput; run the checkpoint parity suite separately before relying on the option in production.

`fishaudio_drama3_codec_test` is an optional local CMake target taking a checkpoint directory, saved codes JSON, output prefix, block frame count, `budget` or `resident`, and `success` or `failure`. It exercises the production codec decoder with actual weights, checks the cache target during decoding and restoration afterward, and can raise a callback exception after actual MLX evaluation. Codec-only checks do not establish full-model streaming, long-runtime, or request-cancellation behavior.
