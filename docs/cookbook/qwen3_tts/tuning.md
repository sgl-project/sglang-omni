---
orphan: true
---

# Qwen3-TTS Tuning

[Back to the Qwen3-TTS quick start](../qwen3_tts.md).

## Server Readiness

Before it reports ready, the server sends itself a streaming speech request in its checkpoint's mode, once per replica of its most replicated process, one after another: CustomVoice uses its first built-in speaker, VoiceDesign a short voice description, and Base a one-second tone as its reference clip. `/health` returns 503 with status `starting` until these requests finish, so the first requests after that do not wait for the kernel loading and first-use setup of the speech path. The sampler is compiled for single and batched requests while the model loads, so the first concurrent requests need no warmup of their own. A failed warmup stops the server. `SGLANG_OMNI_WARMUP_TIMEOUT` (seconds, default 600) bounds each request, and `--skip-server-warmup` reports ready without them.

## Overload / admission policy

Two SGLang generation-stage knobs bound how the server behaves past saturation:

| Knob | Meaning | Qwen3-TTS default |
|---|---|---|
| `--tts_engine.engine.max_running_requests` | Concurrent running slots | `64` |
| `--tts_engine.engine.max_queued_requests` | Waiting-queue depth before fast-reject | `64` |

Every request enters the waiting queue first, so `max_queued_requests` must be **≥ 1**. Capacity is about `running + queued`. Extra arrivals get HTTP **503** (`The request queue is full.`) before preprocessing, or later if the AR waiting queue or request-build backlog is full. Qwen3-TTS defaults to 4 request-build workers with pending depth 16.

A finished request no longer counts toward that capacity, but its streamed audio waits in the API process until the client reads it. `--stream_delivery.max_total_backlog_bytes` caps the undelivered stream bytes all requests may hold together, per API process (default `1073741824`, 1 GiB; `none` turns it off).

A client that reads at least as fast as audio is generated holds almost nothing in the API process. A slower client holds the audio that is generated but not yet in its socket buffers; 24 kHz float32 audio is about 96 kB per second. This includes a client that reads at playback speed while generation runs faster than real time, once its socket buffers are full.

Over the cap, the streams whose readers have been idle longest fail first, until the total fits. A failed stream's generation is aborted if it is still running. SSE (`stream_format=sse`) and WebSocket clients get an error event with `The request queue is full.`; a raw PCM stream is closed early. New requests are not refused because of the cap. `/health` reports the backlog under `stream_delivery`, including the held bytes, open streams and failed streams.

To change the ceiling, set `max_running_requests` and `max_queued_requests` together:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --tts_engine.engine.max_running_requests 32 \
  --tts_engine.engine.max_queued_requests 16 \
  --port 8000
```

Stepped `--concurrencies` is a closed-loop client: it never holds more than N in-flight requests, so past-ceiling load is a burst that drains. Keep offered load above `max_running_requests + max_queued_requests` for a duration with open-loop sustained overshoot:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server --stream \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --max-running-requests 32 \
  --max-queued-requests 16 \
  --sustained-overshoot \
  --overshoot-duration-s 10 \
  --max-samples 64
```

Arrivals default to `2 × capacity` (`--request-rate` overrides). Stats are on successes only; artifacts land in `<output-dir>/overshoot/`.

A closed-loop `--concurrencies 16,32,48,64` sweep is still available for comparing healthy vs past-ceiling points, but it does not hold overshoot. Each concurrency writes inspectable artifacts under `<output-dir>/c<N>/`.

## Prefill CUDA graphs

Qwen3-TTS replays prefills as CUDA graphs. Base and CustomVoice use the full backend, VoiceDesign the breakable backend.

| Knob | Meaning | Default |
|---|---|---|
| `--tts_engine.engine.cuda_graph_backend_prefill` | Prefill graph backend (`full`, `breakable` or `disabled`) | `full` on Base and CustomVoice, `breakable` on VoiceDesign |
| `--tts_engine.engine.cuda_graph_bs_prefill` | Prefill token counts to capture | a ladder from `1` up to the chunked prefill size |
| `--tts_engine.engine.cuda_graph_max_bs_prefill` | Cap for the ladder | the chunked prefill size |

`full` captures the whole prefill transformer, attention included, as one graph per token count. `breakable` captures the layers in segments and runs attention between them. `full` needs the `fa3`, `fa4` or `flashinfer` attention backend. With another attention backend, Base and CustomVoice use `breakable`, and an explicit `full` fails at startup.

On Hopper GPUs the full graph runs its prefill attention on `fa4` and holds every running request, unless you set `--tts_engine.engine.attention_backend`, `--tts_engine.engine.prefill_attention_backend` or an FP8 `--tts_engine.engine.kv_cache_dtype`. Decode keeps its own attention backend.

The ladder reaches the chunked prefill size, which the server picks from the GPU's memory (8192 tokens on an 80 GB GPU), so prefills batched together under load still replay. It starts at `1` because a prompt that is already cached prefills a single token.

Opt out with `--tts_engine.engine.cuda_graph_backend_prefill disabled`. The default adds a few seconds of graph capture at startup. Setting `cuda_graph_max_bs_prefill` rebuilds the ladder up to the new cap, and declaring `cuda_graph_bs_prefill` yourself keeps your list as is.

## Prefill Admission Coalescing

Under concurrent load, the `tts_engine` stage can coalesce prefill admission: instead of admitting each prepared request into its own prefill batch, the scheduler can briefly hold admission so that multiple ready requests are prefilled together.

A prefill step has a largely fixed scheduler cost, so fuller batches can reduce prefill overhead. The end-to-end benefit depends on whether that saving outweighs the extra admission delay and any resulting reduction in decode occupancy.

Coalescing is **off by default** and opt-in through the `tts_engine` factory configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.prefill_coalesce_requests 2 \
  --tts_engine.factory.prefill_coalesce_wait_ms 30 \
  --port 8000
```

or per-stage in YAML:

```yaml
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 2
      prefill_coalesce_wait_ms: 30.0
```

The gate engages only when `prefill_coalesce_requests >= 2`. Once engaged, admission is released as soon as any of the following holds:

- decode is idle, so a ready request can start immediately;
- the waiting queue reaches `prefill_coalesce_requests`;
- the oldest waiting request has waited `prefill_coalesce_wait_ms`.

`prefill_coalesce_wait_ms` is therefore an upper bound on the added admission wait. Admission may be released earlier if the target queue size is reached.

The values above are an example for the Qwen3-TTS workload and are not intended as universal defaults. Match both `prefill_coalesce_requests` and `prefill_coalesce_wait_ms` to the workload you actually serve. Coalescing is most useful when natural prefill batches are small and a short hold can increase batching without materially reducing decode occupancy. If the wait is too long, the reduced decode occupancy can offset the prefill savings.

Leave coalescing disabled for latency-sensitive traffic or workloads where the added wait does not produce enough additional batching.

## Process topology

See [TTS Process Topology](../../basic_usage/tts_process_topology.md) for placement rules. The following recipe isolates Qwen3-TTS reference preprocessing.

 By default all three stages share one process. Per-request reference preprocessing (speech-tokenizer encode, speaker embedding, prompt embedding) then competes with the AR scheduler and the vocoder for the same interpreter, which caps single-replica throughput once concurrency passes ~32. Moving the preprocessing stage to its own process removes that contention: the stage loads a prompt-only frontend (embedding tables, text projection, predictor codec embeddings, speaker encoder) plus the speech tokenizer, together about 2.2 GB of GPU memory for the extra process on a 1.7B checkpoint, and ships the prepared prompt tensors to the engine through the payload. Every GPU stage must then declare a memory fraction, and the engine's static fraction has to agree with the one it declares:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --preprocessing.process tts_frontend \
  --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --vocoder.gpu_memory_fraction 0.12
```

`--vocoder.process vocoder` composes with it (lower `tts_engine` to 0.72 and give the vocoder 0.15). Six SeedTTS samples at a fixed seed produced identical PCM in both layouts; the extra process costs its CUDA context plus the frontend weights.

## Codec decoding defaults

Streaming decodes run on the stateful incremental codec by default: each follow-up chunk decodes only its fresh frames against per-stream state held in a preallocated arena, and the follow-up workers collect for 4 ms. On NVIDIA, AMD and MUSA GPUs steady-state cohorts also decode asynchronously on a priority stream and replay CUDA graphs whose decode step is `torch.compile`d, which costs about a minute of startup compiling the steady shapes. The left-context decoder remains available as a rollback:

```yaml
stages:
  vocoder:
    factory:
      enable_stateful_codec_decoder: false
```

`incremental_codec_cuda_graph`, `incremental_codec_compile`, `async_decode` and `followup_batch_wait_ms` are the individual switches. Measured on one H100 80GB at 20 requests per second, three client seeds of roughly 1200 requests each: the default path holds 0.6% to 2.3% of streams underrun against 20.9% for the left-context decoder, with first playable audio at 55 to 58 ms against 82 to 89 ms.

Each stream holds one of `codec_state_slots` (default 64, matching the default `max_running_requests`) while it decodes; streams past the slots decode eagerly on the left-context path. At startup the vocoder logs the decode graphs it captured and warns when the slots are fewer than the streams it may have to hold at once: the engine's `max_running_requests` when the vocoder shares its process, and that times the engine replicas when the vocoder runs in its own process, since requests bind to its replicas round-robin. Raise `--vocoder.factory.codec_state_slots` together with `max_running_requests`.

Non-streaming requests decode on the same stateful decoder. Each finished request holds one of a second, equal set of `codec_state_slots` reserved for them and steps through the window graphs, with the first decode of a Base request on the tail graphs. A request past the reserved slots waits for one to free instead of decoding eagerly. The startup line reports both sets, so the arena is twice the stream slots' memory. A server whose window graphs fail to capture, or one started with `enable_stateful_codec_decoder: false`, decodes non-streaming requests whole on the whole-utterance graphs instead.

Intel XPU, CPU and Apple MPS decode synchronously by default. For NPU defaults and configuration, see [Ascend NPU baseline](platforms.md#ascend-npu-baseline). To opt in on Intel GPUs, follow the [Qwen3-TTS XPU recipe](../../other_platforms/installation_xpu.md#qwen3-tts-text-to-speech-single-xpu).

Each follow-up batch sorts the stateful decoder's cohorts, and the rows within each, by playback deadline: the nearest cohort launches first and its nearest rows fill the first graph-sized group. Left-context decodes keep queue order. The follow-up workers also commit each in-flight cohort as soon as its decode has finished, checked at the top of the worker loop and before each launch, instead of at the next drain.

## First-audio chunk ramp

For latency-sensitive deployments the whole early chunk schedule can be configured server-side with `stream_chunk_ramp` on the vocoder stage: entry `i` sizes streaming decode chunk `i + 1` in codec frames, and past the ramp the steady stride takes over, so `[2, 4, 8]` yields a `2 -> 4 -> 8 -> 8 -> ...` schedule. Set it through a pipeline config file:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config qwen3_tts_ramp.yaml
```

Smaller early chunks lower time-to-first-audio but start playback with less buffered audio, so the continuity cost grows with concurrency. The default `[1, 2, 4]` is the most aggressive schedule and suits low concurrency; prefer `[2, 4, 8]` at moderate concurrency and `[4, 8]` for saturated serving, where the wider first chunk buys back the playback cushion. The ramp is mutually exclusive with the legacy `initial_chunk_frames` / `stream_initial_followup_stride` options, its first entry must not exceed the steady stride, and a per-request `initial_codec_chunk_frames` still overrides only the first chunk.

## Deterministic Inference

Dynamic batching can change Qwen3-TTS codec and waveform outputs even when the prompt, reference audio, and seed are unchanged. Both the 0.6B and 1.7B Base checkpoints provide an opt-in deterministic mode:

```yaml
enable_deterministic_inference: true
```

When enabled, the same prompt, reference audio, and seed produce byte-identical PCM across runtime batch sizes. This mode reduces throughput because it serializes reference preprocessing and vocoder decoding and disables both the initial and follow-up vocoder graph-capture paths (`initial_cuda_graph` and `followup_cuda_graph`), so it is disabled by default.

## Optional CuTeDSL decode

Install the optional dependency with `pip install -e '.[qwen3-tts-cutedsl]'`, then set `SGLANG_OMNI_QTTS_BACKEND=cutedsl` before launching the server. Leave it unset, or set it to `default`, for the existing implementation. Both talker and code predictor use the same selector.

The kernels fuse input RMSNorm, QKV projection, Q/K normalization, RoPE and cache stores before attention, and output projection, residual addition, RMSNorm and the MLP after attention. Weights are packed once after loading; these copies add GPU memory because the originals remain for prefill and fallback. Online weight updates are rejected while this backend is active. K/V writes use the existing cache directly.

Compiled attention kernels are shared within each 16/32-row tile bucket; live batch size, K/V capacity and RoPE table length are symbolic. Sampling and embedding kernels also reuse binaries across batch sizes. Model dimensions, dtypes, layouts, tail masking, fusion flags and scheduling choices remain specialization keys. CUDA graphs and per-batch workspaces are separate from this compilation cache.

Warm each batch size before CUDA graph capture so workspace initialization stays outside replay. Norm weights are borrowed immutable tensors, and input/Q/K norms use the model's common RMSNorm epsilon. Gumbel draws deliberately remain FP64, including endpoint handling, to preserve the default CUDA sampler's behavior even on Blackwell; the faster FP32 variant changes seeded draws.

On Hopper and datacenter Blackwell, the same selector also enables predictor sampling/embedding fusion and BF16 SnakeBeta in the vocoder. Greedy and top-k buckets 50/64/128 fuse token selection, code-buffer writes, embedding accumulation and optional projected lookup; other sampling settings retain their filtering/tie order and use separate embedding lookup. Prefiltered sampling supports up to 2048 candidates. SnakeBeta prepares immutable coefficients after weight loading and preserves eager BF16 rounding, including an exact BF16 sine table for large tensors. Unsupported layouts, dtypes and devices retain the existing implementations. Attention, preprocessing and scheduling are unchanged.

With fused vocoder activation enabled, BF16 full/window decoding also uses causal Conv1D + bias + SnakeBeta and Conv1D + bias + residual kernels. Supported convolutions have stride/groups one and channel counts divisible by 32; transposed convolutions and the final one-channel convolution use channels-last cuDNN. Weights are packed after loading and remain immutable. Batch size and audio length are runtime parameters, so graph buckets reuse compiled kernels. Stateful incremental decoding uses CuTeDSL pointwise Conv1D and retains the default fused bias/residual/SnakeBeta activation. Convolution history and KV rings stay in place; remaining BF16 state gathers/scatters are batched into one kernel each. BF16 convolution reduction order can change waveform values; this is not bitwise-equivalent decoding.

The transformer blocks support BF16, TP1, 0.6B/1.7B Qwen3-TTS, decode batches of 1–32 on full SM90/SM100/SM103 GPUs with at least 96 SMs. Devices below this minimum, including H20, keep the default transformer blocks; eligible auxiliary and vocoder kernels can still use CuTeDSL. The predictor's two-token prefix also uses these blocks for batches up to 16, with causal attention. The post-attention kernels require a cooperative grid of 24 resident four-CTA clusters; MIG and SM-partitioned contexts are not supported. Talker prefill, larger batches, unsupported weight layouts and other GPU architectures keep the default transformer blocks. Deterministic and batch-invariant modes also keep the default transformer blocks. The custom kernels are imported only when this backend is selected.

The predictor follows the default fused predictor's rounding, norm accumulation and split-K order, and reuses its attention kernel. The 0.6B talker preserves the FP32 residual sum through RMSNorm and stores a separate BF16 residual carry; the 1.7B talker retains BF16-rounded residuals. GEMM accumulation and convolution reduction can still differ, so generated tokens and waveforms need not match exactly. Validate speech quality for your workload before deployment.
