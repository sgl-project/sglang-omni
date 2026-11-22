---
orphan: true
---

# Fish Audio Drama 3 Tuning

[Back to the Fish Audio Drama 3 quick start](../fishaudio_drama3.md).

The [Fish Audio Drama 3](../fishaudio_drama3.md) defaults serve one GPU as they are. This page covers the settings to change when you have more GPUs, need more concurrent streams, or want a different decode path. The measurements these settings were chosen from are in [Benchmarks](benchmarks.md).

## Deployment Performance

On one GPU the default placement works as is: every stage runs on GPU 0, and the vocoder runs in the engine's process, decoding on its own high-priority CUDA stream. With two GPUs, put the vocoder in its own process on the second one, so its decoding no longer shares a GPU with the engine (one process cannot span two GPUs):

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --tts_engine.gpu 0 \
  --vocoder.process vocoder \
  --vocoder.gpu 1
```

When the vocoder runs in its own process on the engine's GPU (`--vocoder.process vocoder`), `--mps on` lets the two processes share it through CUDA MPS (see [Same-GPU Data Parallelism with CUDA MPS](../../basic_usage/mps_dp.md)). It can raise throughput under high concurrency but may make a single stream slower; measure it with your own load before you keep it.

### Tensor Parallelism

When the FP8 checkpoint does not fit one GPU, `--tts_engine.tp_size` splits the engine across 2 or 4 GPUs: SGLang shards the backbone, and each GPU holds its share of the codebook tower's attention heads and MLP width; with 4 GPUs, pairs of GPUs share each of the tower's two key/value heads, as they share the backbone's. The FP8 routed experts are stored in 128-wide blocks, so their 512-wide intermediate splits across at most 4 GPUs. On two 32 GB GPUs, such as two RTX 5090:

```bash
sgl-omni serve \
  --model-path fishaudio/drama-3 \
  --tts_engine.tp_size 2 \
  --tts_engine.gpu "[0, 1]" \
  --tts_engine.gpu_memory_fraction 0.75 \
  --preprocessing.gpu 0 \
  --vocoder.gpu 1 \
  --vocoder.gpu_memory_fraction 0.2
```

Each GPU then holds 18.5 GB of engine weights, and 9.7 GB with `--tts_engine.tp_size 4 --tts_engine.gpu "[0, 1, 2, 3]"`. The vocoder's default share of 0.10 is sized for an H200: on a 32 GB GPU it cannot hold one non-streaming decode of `max_new_tokens` frames, so the command raises it to 0.2. The fused codebook tower layers stay off under tensor parallelism. [Benchmarks](benchmarks.md#tensor-parallel-layouts) compares these layouts with one H200.

### High Concurrency

The defaults run at most 16 requests at a time on the engine and give it 0.70 of the GPU's memory. For more streams at once, raise the request cap; the measurements in [Benchmarks](benchmarks.md#streaming-performance) also raise the engine's share:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --tts_engine.gpu_memory_fraction 0.85 \
  --tts_engine.factory.max_running_requests 256
```

These values are sized for an H200: the engine's and the vocoder's shares and the preprocessing reserve must fit one GPU together, which the server checks at startup.

### Non-Streaming Requests

The vocoder decodes a non-streaming request's audio in one piece after the engine finishes it, one request at a time, replaying a CUDA graph for requests of up to 512 frames (`--vocoder.factory.non_streaming_cuda_graph_max_frames`). With `--vocoder.factory.max_batch_size` above 1 it instead batches requests that finish together, decoding them eagerly. Either way a decode takes at most a frame limit it sets at startup from the room its process leaves in the process's memory share: the engine's and the vocoder's `gpu_memory_fraction` together (0.65 and 0.15 by default) while the vocoder runs in the engine's process, the vocoder's alone in its own process. It logs:

```text
Drama 3 vocoder: a non-streaming decode takes up to <frames> padded frames of <bytes> bytes each; share <share> GiB, held <held> GiB
```

The limit does not count the memory the vocoder's process holds outside PyTorch's allocator, so it cannot guarantee that the process stays within its share. With the engine's share raised to 0.85 on the same GPU, concurrent non-streaming requests can run out of GPU memory. To avoid that, stream those requests (`"stream": true`), keep the engine's default share of 0.70 on the GPU that runs the vocoder, or put the vocoder on a second GPU, as under [Deployment Performance](#deployment-performance).

## Defaults Parameters

| Default | What it does | Change it with |
|---|---|---|
| Decode CUDA graph | Replays decode steps as CUDA graphs. | `--tts_engine.engine.cuda_graph_backend_decode disabled`; `--tts_engine.engine.disable_cuda_graph true` turns off every engine CUDA graph |
| Breakable prefill CUDA graph, up to 2048 tokens | Replays prefill batches of up to 2048 tokens, backbone and codebook tower together, as one breakable CUDA graph; larger batches run eagerly. Shortens the time to the first audio at the cost of startup time and GPU memory; the KV cache keeps its size. | `--tts_engine.engine.cuda_graph_backend_prefill disabled`; the token cap with `--tts_engine.engine.cuda_graph_max_bs_prefill` |
| GDN extend inside the breakable prefill graph | The breakable prefill CUDA graph also runs the extend of the linear-attention (GDN) layers for prefills of up to `max_running_requests` requests, instead of breaking the graph at each of those layers and launching their kernels from Python. Its outputs and states equal the eager extend bit for bit. | `--tts_engine.factory.gdn_prefill_cuda_graphs false` |
| First-frame CUDA graphs, up to 16 requests | Replays the step that samples each request's first frame as a CUDA graph for prefills of up to 16 requests (at most the decode CUDA graph's largest batch); larger prefills sample eagerly. Needs the decode CUDA graph; shortens each prefill at the cost of startup time and GPU memory. | `--tts_engine.factory.first_frame_cuda_graphs false`; the request cap with `--tts_engine.factory.first_frame_cuda_graph_max_requests` |
| `prefill_decode_interval` 4 | After a prefill, new prefills wait for the next 4 scheduling rounds, so running streams keep decoding. | `--tts_engine.engine.prefill_decode_interval 0` schedules prefills without the wait |
| Sampling: `temperature` 1.0, `top_p` 0.9, no `top_k` cap | Without a request `top_k`, the main token is drawn with no top-k cap; a cap of 30 there lets a stream fall silent after its first phrase. The codebook heads draw with top-k 30 at most whatever the request asks, since wider caps there cost speaker similarity. | The request's `temperature`, `top_p` and `top_k`; a request `top_k` caps the main token, and the codebook heads up to 30 |
| Triton sampling | Samples each frame's main token and codes with Triton kernels instead of the PyTorch sampler, under the same top-k, top-p and repetition-aware rules. Using the same seed does not guarantee the same tokens as the PyTorch sampler. Rounding can keep or drop a candidate whose cumulative probability lies at top_p. | `--tts_engine.factory.triton_sampling false` |
| Reference sampling, off | Samples each frame with Fish Audio's reference sampling for Drama 3, in PyTorch, instead of the samplers above. The main token's first draw keeps only candidates whose probability reaches the request's `epsilon_cutoff` (1e-3 when unset, 0 turns it off) and top-p drops the token that crosses it; a first draw that repeats one of the last 16 main tokens, end of speech included, becomes the most probable candidate. The codebook heads take the request's `temperature`, `top_k` and `top_p` with no epsilon cutoff and never return a code the codec's 1024-entry residual codebooks lack. Its recommended request parameters are `temperature` 1.0, `top_p` 1.0, no `top_k` and `epsilon_cutoff` 1e-3. With it off, a request that sets a non-zero `epsilon_cutoff` is refused. | `--tts_engine.factory.reference_sampling true`; `--tts_engine.factory.reference_sampling_fallback ras` replaces a repeated first draw with a second draw at temperature 1.0, top-p 0.95 and top-k 30 instead |
| Fused codebook tower layers, up to 32 rows | Runs the codebook tower layers of decode and first-frame steps of up to 32 rows as fused Triton kernels (row tiles of 16 and 32), eagerly or inside the CUDA graphs; larger steps and the tower's prefill keep the module path. The kernels add in a different order, so the tower's outputs can differ in the last bits and the sampled frames can differ. | `--tts_engine.factory.tower_fused_layers false` |
| Vocoder CUDA graphs, up to 512 frames | Replays streaming window decodes, and the decode of non-streaming requests of up to 512 frames padded to a multiple of 8, as CUDA graphs; a non-streaming request replays at batch 1, and non-streaming batches and longer requests decode eagerly. Beside incremental streams the vocoder halves the 512-frame cap until one request of `max_new_tokens` frames still fits its process's GPU memory share, and logs each halving: with the default shares on an 80 GB GPU the cap is 256 frames. Under overlap or stateful decoding the cap does not halve, and a share too small for it stops startup with the flags to change. Costs startup time and vocoder GPU memory. | `--vocoder.factory.window_cuda_graphs false`; the non-streaming cap with `--vocoder.factory.non_streaming_cuda_graph_max_frames` (0 keeps only the window graphs) |
| Incremental streaming decode | Each streaming window decodes only its new frames, with the stream's decoder state kept in one of 256 slots, and windows of one length from up to 16 streams share a decode. Needs fp16 decoding on a CUDA device with FA3 (compute capability 8.x or 9.x, CUDA 12.3 or newer) and `stream_crossfade_samples` 0; elsewhere, or when the state does not fit its process's GPU memory share, the vocoder uses overlap decoding, where each window decodes again the `stream_overlap_tokens` frames before it. | `--vocoder.factory.stream_decode overlap`; the slots with `--vocoder.factory.stream_slot_count` and the streams per decode with `--vocoder.factory.max_stream_batch_size` |
| Batched streaming windows, up to 4 | Under overlap decoding, same-length streaming windows already queued from different requests decode in one CUDA graph replay, shortest windows first; no window waits for others to arrive. At startup the vocoder times each batch size and keeps only those that save at least 5% of the GPU time of one-at-a-time replays, and logs the ratios. A batched decode can differ from a batch-1 decode in the last fp16 bits. Beside `stateful_decode`, windows do not batch. | `--vocoder.factory.stream_window_batch_size 1` replays each window alone, still shortest first |
| Vocoder fp16 decoding | Decodes frames into audio in fp16. | `--vocoder.factory.decode_dtype float32` |
| Vocoder in the engine's process | The vocoder runs in the engine's process and decodes on its own CUDA stream, at a higher priority than the engine's stream, so its windows do not queue behind decode steps, and the engine hands it codes without a process hop. A tensor-parallel engine or vocoder holds its process alone, and `stateful_decode` needs a dedicated vocoder process, so with any of these the vocoder runs in its own. | `--vocoder.process vocoder` runs it in its own process |
| Reference encode CUDA graphs, up to 20 s | Encodes each reference by replaying a CUDA graph captured for its length rounded up to a multiple of 22 codec frames (1.02 s); the longest of the 20 graphs holds 440 frames (20.43 s), and longer references encode eagerly. The encoder is causal, so the zeros after a reference feed none of its codes, but the padded encode rounds differently from an unpadded one: on H100 about 4.5% of a reference's code elements differ in bf16 and under 0.3% in fp32. Costs startup time and preprocessing GPU memory. | `--preprocessing.factory.reference_encode_graph_max_seconds 0` encodes every reference eagerly; another value moves the cap |

The same settings in a config file:

```yaml
stages:
  tts_engine:
    engine:
      cuda_graph_backend_prefill: disabled
      prefill_decode_interval: 0
  vocoder:
    factory:
      window_cuda_graphs: false
      decode_dtype: float32
```

### Kernel fallbacks

Triton sampling and the fused codebook tower layers have been validated on Hopper GPUs only, so they run only on compute capability 9.0; on any other GPU no setting turns them on. Both also fall back without Triton or a CUDA device. Triton sampling falls back to the PyTorch sampler when its kernels do not cover the codebook size or the main-token candidate count, and the fused layers fall back to the module path with deterministic inference, under tensor parallelism, or when their kernels do not cover the tower weights or the KV cache. The startup log names the path each one takes and, for a fallback, the reason, for example:

```text
Drama 3 codebook sampling: PyTorch, untested GPU sm_120
Drama 3 tower fused layers: off, untested GPU sm_120
```

## Incremental streaming codec

The incremental codec is opt-in: set `--vocoder.factory.stateful_decode true` to reuse per-request convolution and attention history. While it is on, it replaces the vocoder's default batched incremental stream decoding (`--vocoder.factory.stream_decode incremental`). GEMM upsampling, folded codebooks, Snake/history fusion and compiled attention follow this setting unless individually overridden. Slot capacity follows `tts_engine.engine.max_running_requests` when set, otherwise `tts_engine.factory.max_running_requests`; it does not add an HTTP admission limit. Streams that cannot acquire a slot use window decoding for their entire lifetime. Fixed routing, single replicas, a dedicated TP1 vocoder process and zero crossfade are required. Enabling the feature adds state memory and compilation/capture startup cost. It targets sustained concurrent workloads, especially shared-GPU deployments; benefits with a dedicated vocoder GPU are workload-dependent.
