# MOSS-Transcribe-Diarize

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) is a multi-speaker ASR and diarization model from the OpenMOSS team.

MOSS-Transcribe-Diarize does not support `/v1/audio/translations`; that endpoint returns HTTP 400. Use `/v1/audio/transcriptions`.

![Model Architecture](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/main/Model_Architecture.png)

It transcribes speech, assigns speakers, and predicts timestamps in a single generation pass. With 128K context, it supports up to ~90-minute audio, handles meetings, interruptions, long conversations, and overlapping speech, and adds hotword boosting for names, companies, product terms, and domain vocabulary. MOSS-Transcribe-Diarize is served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint.
Transcriptions support `response_format=srt` or `response_format=vtt` with real segment timestamps for this model.

| Component | Spec |
|---|---|
| Architecture | `MossTranscribeDiarizeForConditionalGeneration` |
| Audio encoder | Whisper encoder (24 L, d_model=1024) |
| Text decoder | Qwen3 (28 L, hidden=1024, GQA 16/8) |
| Output | Speaker-labelled transcript with start/end timestamps |
| Endpoint | `/v1/audio/transcriptions` |

## Architecture and Optimization

MOSS-TD (short for MOSS-Transcribe-Diarize) is served through SGLang-Omni for two reasons. First, SGLang-Omni's multi-stage pipeline is a natural fit — ASR follows the same encoder → prefill → decode pattern that the framework already orchestrates for TTS. Second, ASR is a category of multimodal-input models, and many of the optimizations we have built in SGLang-Omni (CUDA Graph capture, async decode, continuous batching, KV cache management) transfer directly to the ASR setting.

### Inference Pipeline

![ASR Pipeline](../_static/image/moss-td-asr-pipeline.svg)

MOSS-TD follows the Audio LLM pattern: a Whisper encoder produces continuous embeddings, which are projected into a decoder-only LLM that autoregressively generates the transcript.

1. **Encoder.** Waveform → log-mel spectrogram (80 bins) → 24-layer Whisper Transformer → 4× Time Merge (concatenate every 4 frames) → VQAdaptor MLP (4096→1024). The output is a sequence of continuous float vectors in the LLM's embedding space. (The "VQ" name is a misnomer — no vector quantization is involved.)
2. **LLM Prefill.** The audio embeddings replace `<|audio_pad|>` tokens in the prompt. Qwen3 processes the full prompt in one parallel forward to build KV cache.
3. **AR Decode.** Qwen3 generates text tokens (transcript with speaker labels and timestamps) one at a time until EOS.

For long audio (up to ~90 min), the token sequence after encoding can reach tens of thousands of tokens. **Chunked Prefill** splits it into 4096-token chunks, processing one per scheduling step and interleaving decode steps for other requests between chunks. Stream output is suppressed during chunked prefill to avoid emitting intermediate states.

### ASR vs TTS

ASR and TTS share much of the same serving infrastructure in sglang-omni — both use `OmniScheduler` for scheduling, share CUDA Graph / KV Cache management / continuous batching, and are built on the same Qwen3 LLM backbone. The key differences lie in what they encode, what they generate, and how their pipelines are structured:

| Dimension | ASR (MOSS-TD) | TTS (Higgs / MOSS-TTS) |
|---|---|---|
| Audio representation | Continuous features (mel spectrogram → encoder hidden states) | Discrete codec tokens (RVQ multi-codebook encoding) |
| Data flow | Audio → text | Text → audio |
| Decoder / Vocoder | Not needed — output is plain text | Vocoder required to reconstruct waveform from codec tokens |
| Typical input length | Can be very long (MOSS-TD supports ~90 min) | Usually short (reference voice: a few seconds) |
| Pipeline stages | Single stage (encoder + LLM) | Multi-stage (preprocessing → AR engine → vocoder) |
| Streaming | Stream output (incremental text); streaming input is possible via cumulative chunking but not yet optimized — a new encoder architecture with native streaming in/out support is in development | Stream output (incremental audio) + streaming vocoder |

These differences shape optimization priorities: ASR optimization focuses on the AR decode loop (which dominates latency) and long-sequence memory management, while TTS optimization additionally targets vocoder batching/streaming and multi-codebook generation strategies. For the full TTS optimization story, see [Optimizing TTS Inference](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md).

### Where Time Is Spent

AR Decode dominates at low concurrency, but at high concurrency decode gets amortized by batching and encoder share rises — especially for short audio.

Profiling on a single H100 (CUDA Graph, bf16), showing the percentage breakdown across all three stages:

| Audio length | Concurrency | Encoder | LLM Prefill | AR Decode |
|---:|---:|---:|---:|---:|
| 5 s | 1 | 8.9% | 14.7% | 76.4% |
| 5 s | 4 | 20.0% | 22.3% | 57.7% |
| 5 s | 16 | 38.2% | 29.7% | 32.1% |
| 60 s | 1 | 4.0% | 2.1% | 94.0% |
| 60 s | 4 | 5.0% | 4.6% | 90.4% |
| 60 s | 16 | 13.7% | 9.5% | 76.8% |
| 20 min | 1 | 4.7% | 0.8% | 94.5% |
| 20 min | 4 | 9.2% | 1.9% | 88.9% |
| 20 min | 16 | 11.6% | 2.6% | 85.7% |

![Profiling Breakdown](../_static/image/moss-td-profiling.svg)

At c=1 with longer audio, AR Decode takes 94%+ of total time — the leverage is almost entirely in the decode loop. At c=16 with short audio, encoder + prefill together account for 68%, making encoder-side optimizations (CUDA Graph capture, Torch Compile, caching) worthwhile.

### Optimization Strategies

![Optimization Overview](../_static/image/moss-td-optimization.svg)

The optimization stack mirrors [what we built for TTS](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md), sharing the same core infrastructure with ASR-specific adaptations.

**CUDA Graph.** The LLM decode step pads batch size to predefined buckets (1, 2, 4, 8, …) and replays a captured CUDA graph, eliminating kernel launch overhead on every token. This is the single biggest optimization for AR Decode. Breakable prefill graphs bucket over token count instead, and the ladder starts at 1 and 2: a fully cached prefix still re-prefills its last token, and that 1-token extend would otherwise pad to the 4-token floor and exceed SGLang's 2x padding guard, falling back to eager. The 2-token bucket sits exactly on that guard, so it replays either way and only saves the padding. The Whisper encoder gets the same treatment, bucketed over chunk count (`encoder_chunk_buckets`, default `1..8` ≈ 4 min of audio).

**Decoder Torch Compile.** The default pipeline compiles Qwen3 decoder shapes through batch size 4 and uses the eager decoder above that cap. Override the cap with `--torch-compile-max-bs`, or disable decoder compilation with `--torch-compile off`. Compilation runs once per captured decode bucket at startup (`max-autotune-no-cudagraphs`), so cold start pays an autotuning cost before the server accepts traffic. This setting is independent of the encoder compile option below.

**Encoder Torch Compile (opt-in).** `encoder_torch_compile=True` swaps the encoder CUDA graph for `torch.compile` (default mode) with kernel fusion. The two are mutually exclusive. Reduce-overhead mode must not be used: its cudagraph trees corrupt memory alongside the decode CUDA graphs that always run in this process (illegal memory access after ~60s of serving). The cost is a one-time per-bucket compile at startup; `dynamic=False` means only the warmed chunk counts are accelerated, anything else runs eager.

**Async Decode.** Same one-step lookahead as TTS: launch the current decode step's GPU work, then resolve the previous step's host-side work (D2H copy, finish detection, result dispatch) in parallel. MOSS-TD enables lookahead starting at batch size 1 by default. Set `--async-lookahead-min-batch-size 2` to keep batch-size-1 decode synchronous, or use `--decode-mode sync` to disable lookahead for the stage. Two alternating pinned host buffers prevent read/write races between the GPU's async D2H write and the CPU's read. For the full mechanism and code pointers, see [Asynchronous Decode + Lookahead](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-optimization.md#asynchronous-decode--lookahead) in the TTS optimization guide.

**LRU Encoder Cache.** The Whisper encoder forward is deterministic for identical input audio — same waveform always produces the same embeddings. We exploit this with an LRU cache (max 64 entries, 4 GB budget) that stores encoder outputs on CPU, keyed by a content hash of the input waveform. On a cache hit the stored tensor is transferred back to GPU asynchronously, skipping the encoder entirely. On a miss the encoder runs normally and the result is moved to CPU for storage. The cache evicts by both entry count and total bytes, always dropping the least-recently-used entry first.

Unlike TTS where the same reference voice is reused across many prompts (high hit rate), ASR inputs are typically unique in production. The cache is most useful for request retries, A/B testing with different decoding parameters, and development iteration.

**Stream Output.** Emits transcript text incrementally during AR decode via SSE, so users see partial results as they are generated rather than waiting for the full sequence. Three mechanisms control when to emit:

1. **Rate limiting** (default 50 ms): tokens accumulate in a per-request buffer and are flushed only when enough time has elapsed since the last emit. The very first token goes out immediately; EOS always triggers a flush regardless of timing.
2. **Chunked prefill suppression**: during chunked prefill (prompt chunks still being processed), all emission is suppressed to prevent intermediate states from being misinterpreted as transcript output.
3. **Incomplete UTF-8 handling**: accumulated tokens are decoded together. If the result ends with the Unicode replacement character (indicating an incomplete multi-byte sequence split across token boundaries), emission is held until the next token completes the sequence.

## Model Usage

### Launching Commands

Install `sglang-omni` with [Installation](../get_started/installation.md) on CUDA or
[Intel XPU installation](../other_platforms/installation_xpu.md) on Intel GPUs, then download the model:

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize
```

MOSS-TD briefly holds newly built LM requests to admit larger prefills. The
default target is 4 requests with a 12 ms oldest-request deadline. While more
request builds are pending, the scheduler waits for either limit; after build
work drains, it releases immediately only when decode is idle. During active
decode, it continues coalescing until the target or deadline. With a draft
(DFlash or DFlash2), a ready prefill is released as soon as build work drains,
whether or not decode is active; set
`--asr.factory.prefill_coalesce_after_builds_during_decode true` to keep
coalescing during decode. Override the two
limits with `--prefill-coalesce-requests` and `--prefill-coalesce-wait-ms`, or
set the request target to `0` to disable coalescing.

Serve the model:

**CUDA GPU**

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --port 8000 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80
```

On one CUDA GPU this serves the DFlash draft by default; see [Speculative Decoding](#speculative-decoding).

**Intel GPU**

MOSS-TD with LLM and Whisper encoder XPUGraph capture has been validated in
BF16 on one Intel Arc Pro B60 with 24 GB of memory:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --port 8000 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --asr.engine.enable_torch_compile false \
  --mem-fraction-static 0.70
```

The graph configuration names retain `cuda_graph` for compatibility, but
SGLang selects the XPUGraph backend on XPU. The B60 validation uses
`mem_fraction_static=0.70` to leave enough memory for all eight encoder graph
buckets; `0.80` leaves too little graph-capture headroom on a 24 GB card.

Before it reports ready, the server transcribes one second of audio, at most 8
tokens. `/health` returns 503 with status `starting` until that request has
finished, so the one-time kernel builds of the transcription path happen before
the first user request. The first concurrent batch can still build batch-size
kernel variants. A failed warmup stops the server. `SGLANG_OMNI_WARMUP_TIMEOUT`
(seconds, default 600) bounds the request, and `--skip-server-warmup` reports
ready without it.

### Sending Requests

Use `response_format=verbose_json` when you need parsed speaker segments. `json` returns the raw transcript text only.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=verbose_json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "response_format": "verbose_json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
payload = resp.json()
print(payload["text"])
for segment in payload.get("segments", []):
    print(
        f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}"
    )
```

When a request omits `max_new_tokens`, the server sizes the output budget from the audio duration in both directions: `max(512, 10 tokens per audio second)` with the stock config, so a 60 minute recording gets a 36000 token budget without any client changes, while a 6 second clip is bounded at 512 tokens instead of inheriting the old fixed 5120 default — this keeps greedy decoding from looping for thousands of tokens on short non-speech audio (#975) without truncating dense, timestamped multi-speaker transcripts. A zero-duration input uses a tighter 128-token fallback because it has no legitimate transcript to preserve. Neither bound exceeds an operator-pinned smaller default. The form also accepts `repetition_penalty` (0 < x <= 2, default 1.0 = off) to damp repetition loops on noisy audio without touching the greedy default. Operators can pin a fixed `max_new_tokens` in the stage config, which disables duration scaling for requests that omit the field. An explicit `max_new_tokens` in the request always wins over both defaults. The scheduler clamps the final value to the context remaining after the audio prompt, so large explicit values are safe to send. Set the field explicitly when you want a hard cap or a larger budget than the default, as in this example with a clip from the repo that has two speakers:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@docs/_static/audio/gaokao-listening.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=65536
```

### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional language hint |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | model default (`0.0`) | Sampling temperature. A negative value returns HTTP 422 |
| `max_new_tokens` | int | duration scaled | Max generated tokens. With the stock config, omitted requests use `max(512, 10 * audio seconds)`; empty audio uses a 128-token fallback. A fixed stage value disables duration scaling. Explicit values always win and are clamped to the remaining model context |
| `prompt` | string | unset | Optional instruction override; omit to use the built-in transcribe+diarize prompt |

`verbose_json` parses the model markup into OpenAI-style `segments` with
`start`, `end`, and speaker-prefixed `text` (for example `[S01]...`).
`json` / `text` return the full transcript string without segment parsing.

### Speculative Decoding

On one CUDA GPU, MOSS-TD serves with its DFlash draft, [`wilsonzheng0327/MOSS-Transcribe-Diarize-dflash`](https://huggingface.co/wilsonzheng0327/MOSS-Transcribe-Diarize-dflash), by default, at the revision `sglang_omni/speculative/draft_registry.py` pins. This holds when the model path that launches, from `--model-path` or a config file, is `OpenMOSS-Team/MOSS-Transcribe-Diarize`, with or without an `@<revision>` pin, or a Hugging Face cache snapshot of it. Intel GPUs and other platforms serve without the draft, and so do a stage with `tp_size` above 1 and a copy outside the Hugging Face cache. The [two-replica H100 MPS profile](../basic_usage/mps_dp.md) turns it off, since it was validated without it. To serve without the draft, add `--asr.engine.disable_speculative_decoding true`:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --asr.engine.disable_speculative_decoding true \
  --port 8000
```

The DFlash2 draft [`wilsonzheng0327/MOSS-Transcribe-Diarize-dflash2`](https://huggingface.co/wilsonzheng0327/MOSS-Transcribe-Diarize-dflash2) continues that draft's training with DFlash2's two additions: small convolutions in each layer, and a candidate selector that picks each proposed token from the target head's top 16, so the draft has no output head of its own. It is 1% to 6% faster than the DFlash draft synchronously, with the same CER and DER (table below). Serve it by its path:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --asr.engine.speculative_algorithm DFLASH \
  --asr.engine.speculative_draft_model_path wilsonzheng0327/MOSS-Transcribe-Diarize-dflash2 \
  --port 8000
```

The selector takes its top 16 with FlashInfer's top-k kernel. The Docker image ships that kernel prebuilt. Without FlashInfer's prebuilt kernels, the first start compiles it, which took about 13 minutes on an H200 host, and later starts reuse it. Install them with `uv pip install flashinfer-jit-cache==0.7.2 --index-url https://flashinfer.ai/whl/cu130` to skip the compile.

The notes below apply to both drafts:

- **Acceptance length.** Send the form field `return_spec_tokens_details=true` to get `sglext.spec_tokens_details` (`spec_accept_length`, `spec_accept_rate`, the draft counts and the per-verify histogram) on `json` and `verbose_json` responses and on the streamed `transcript.text.done` event. Audio long enough to be transcribed in chunks returns none. `spec_accept_rate` and `spec_num_proposed_drafts` are left out when no drafts were proposed or the draft width is adaptive.
- **Transcripts.** The target verifies every proposed token, so accuracy stays at the target's level, but the text is not byte-identical to a run without the draft: MOSS-TD has many near-tied decisions, mostly timestamp hundredths, which any numeric change, including batching, can flip.
- **Context.** Every output budget leaves room for the few tokens each speculative step reserves, 16 with the default settings: a `max_new_tokens` that would not leave them is lowered to the remaining context less that reserve, where the server without a draft would take up to the remaining context.
- **Capacity.** The draft's KV shares the pool, one slot per token, and is kept in fp8 (`speculative_draft_kv_cache_dtype fp8_e4m3`), which halves the 5-layer draft's share. With the memory of one H100 80GB at `mem_fraction_static=0.80` (measured on an H200 at 0.454), the pool holds 524k tokens, against 573k without the draft and 484k with the draft's KV in bf16. That decides whether 16 `aishell4_long` meetings of 37 to 40 minutes fit at once: with a bf16 draft KV at most 14 ran, and first output p95 was 48 to 51 s against 18 s without the draft; with fp8, 15 ran and first output p95 was 13 to 15 s, at the same throughput, acceptance, CER and DER. Given the same pool, the draft reaches first output sooner than the server without it (48 s against 148 s at 484k tokens), because it finishes meetings sooner. Long audio at high concurrency can still fill the pool; the scheduler then retracts requests and finishes them later, and every file completed.
- **Compilation.** Torch compile is turned off under the draft: on SGLang 0.5.23 the compiled path makes each verify step about four times slower. Without a draft it stays on.
- **Sampling.** `repetition_penalty` applies as ordinary decoding does: each verified position is penalized over the transcript so far and the drafted tokens before it. On one H200 with every `movies800times` clip sent at 1.2, no request failed, and transcripts differed from a synchronous server without the draft about as often as they do with no penalty (79 and 35 of 400 at 1 and 16 concurrent requests, against 100 and 28), while the penalty itself changes 392 of 400. Throughput fell 4% at one request and 10% at 16, and the logged acceptance length 6%. Requests with `frequency_penalty`, `presence_penalty`, `min_p` or logprobs are rejected, because the draft worker applies those penalties once per verify block, drops `min_p`, and skips the sampler that records logprobs.
- **Scheduling.** By default each verify round overlaps with host work (speculative overlap), and the scheduler processes every prefill's result before it chooses the next batch, so a new request's first token does not wait for the verify round launched after its prefill. Decode rounds still overlap. Add `--asr.factory.enable_async_decode false` to run the synchronous verify loop instead; plain fallback always runs it. `--asr.factory.prefill_drain_below_batch_size N` drains a prefill only while fewer than N requests are decoding, and 0 never drains. It applies only with the draft and overlap; otherwise the scheduler ignores it without a warning. On one H200 with the DFlash draft on streamed `movies800times`, the default ran at 1.10x to 1.17x the synchronous verify loop's throughput from 1 to 32 requests, with first output p95 0.0 to 1.2 ms later than synchronous and 3.7 to 3.9 ms earlier than the server without a draft at 16 requests. Overlap without the drain (N = 0) had up to 4% more throughput, but its first output p95 was 2.8 to 3.0 ms later than the default at 16 requests. Async prefill stays off under the draft. Setting `enable_mixed_chunk` is rejected.

Gain over the same server without a draft, on one H200 with SGLang 0.5.23, the default server config and the prefix cache off (`python -m benchmarks.eval.speculative_sweep`). Each draft ran on its own GPU of one node, against its own server without a draft. `movies800times` is the mean of 3 passes per load and `aishell4_long` one pass over its first 8 recordings (`--max-samples 8`):

| Dataset | Draft | Scheduling | Concurrency 1 | 4 | 16 | 64 |
|---|---|---|---:|---:|---:|---:|
| `movies800times` | DFlash2 | Synchronous | 1.69x | 1.60x | 1.42x | 1.22x |
| `movies800times` | DFlash | Synchronous | 1.63x | 1.54x | 1.41x | 1.21x |
| `movies800times` | DFlash2 | Overlapped | 1.93x | 1.74x | 1.57x | 1.41x |
| `movies800times` | DFlash | Overlapped | 1.87x | 1.68x | 1.51x | 1.37x |
| `aishell4_long` | DFlash2 | Synchronous | 2.57x | 3.31x | 3.70x | |
| `aishell4_long` | DFlash | Synchronous | 2.50x | 3.15x | 3.51x | |
| `aishell4_long` | DFlash2 | Overlapped | 3.29x | 3.82x | 4.15x | |
| `aishell4_long` | DFlash | Overlapped | 3.13x | 3.58x | 3.85x | |

In its own throughput DFlash2 is 1% to 6% faster than DFlash synchronously, 5% to 8% faster overlapped on `aishell4_long`, and within 1% overlapped on `movies800times` from 4 requests up, where the server without a draft ran 3% to 4% slower on the DFlash2 GPU. The logged acceptance length is 5.16 with DFlash2 and 4.82 with DFlash on `movies800times`, and 5.80 and 5.28 on `aishell4_long`. Neither draft or mode is slower than the server without a draft at any load measured, and CER and DER do not change. On `googletime`, measured earlier with the DFlash draft only, the gain was 2.97x, 4.14x and 4.27x at 1, 4 and 16 concurrent requests synchronously, and 3.73x, 4.68x and 4.57x overlapped.

### Per-request plain fallback

Set `asr.factory.enable_plain_fallback` to route requests with unsupported sampling options through ordinary decoding. Requests that fit the target context but leave too little room for draft KV also use ordinary decoding.

```bash
sgl-omni serve \
  --config examples/configs/moss_td_plain_fallback.yaml \
  --port 8000
```

The [Qwen3-ASR fallback restrictions](qwen3_asr.md#per-request-plain-fallback) also apply to MOSS-TD.

When both paths have work, the scheduler gives speculative batches one to two turns per plain turn, according to their live request counts. This reduces the cost of sparse fallback traffic on speculative requests, but increases latency for plain requests. The two-turn cap prevents plain requests from starving; it does not bound wall-clock waiting time because forward durations vary. Fallback requires synchronous decoding, while the default DFlash server overlaps draft execution, so even 0% fallback can cost throughput relative to it: the synchronous verify loop had 12% less throughput at 16 requests on `movies800times` on one H200. Benchmark both request cohorts before enabling this option.

## Benchmarking

Thanks to the Moss team for providing the benchmark datasets, we prepare movies800times, aishell4_long, and googletime as benchmark datasets for multi-speaker ASR. movies800times is a short-sequence dataset with 800 dialog clips, aishell4_long with googletime are long-sequence dataset of long-form meeting audio, and English podcasts. These datasets are right now under private license, and you can contact the Moss team for access.


```bash
# Short-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --concurrency 16 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --output-dir results/moss_transcribe_diarize_movies800times

# note (Xinyu): Add one dedicated request-event profiling pass after the measured
# evaluation. This pass is excluded from reported accuracy and speed metrics.
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --concurrency 16 \
  --profile-events \
  --profile-event-dir /tmp/moss_td_bench_profile \
  --output-dir results/moss_transcribe_diarize_movies800times_profile

# Long-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset aishell4_long \
  --concurrency 16 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --max-new-tokens 65536 \
  --request-timeout-s 1800 \
  --output-dir results/moss_transcribe_diarize_aishell4_long

# Long-sequence English podcast ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset googletime \
  --concurrency 16 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --max-new-tokens 65536 \
  --request-timeout-s 1800 \
  --output-dir results/moss_transcribe_diarize_googletime
```

`--profile-events` starts request-level event recording through the serve
profiler endpoints, runs one extra pass, and adds its `stage_breakdown`,
`hop_breakdown`, and speed metrics under `profile` in both
`transcribe_diarize_results.json` and `transcribe_diarize_speed_results.json`.
The event directory is a server-side path, so report generation requires the
benchmark process to see the same filesystem. For router or DP deployments,
pass every worker serve URL with a comma-separated `--profile-urls` value.

## Benchmark Results

Here we provide the benchmark results of movies800times on a single H200 141GB GPU with SGLang 0.5.23, and of aishell4_long on a single H100 80GB GPU. Each row is the **mean of 3 runs** against a server without the draft, with `max_running_requests=16`, `cuda_graph_max_bs=16`, and `mem_fraction_static=0.80`; the draft's gains are under [Speculative Decoding](#speculative-decoding).

### movies800times

| Concurrency | Throughput (req/s) | Mean latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 7.75 | 0.129 | 0.0120 | 89.73 |
| 2 | 12.98 | 0.154 | 0.0149 | 150.15 |
| 4 | 24.19 | 0.165 | 0.0160 | 279.93 |
| 8 | 43.79 | 0.182 | 0.0175 | 506.76 |
| 16 | 73.78 | 0.215 | 0.0207 | 853.73 |

### aishell4_long

| Concurrency | Throughput (req/s) | Mean latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 0.022 | 45.2 | 0.0197 | 50.64 |
| 2 | 0.032 | 60.7 | 0.0265 | 74.25 |
| 4 | 0.036 | 105.6 | 0.0461 | 81.64 |
| 8 | 0.040 | 172.6 | 0.0754 | 90.62 |
| 16 | 0.043 | 282.8 | 0.1237 | 98.83 |


- **Concurrency** — Maximum number of in-flight client requests (`--concurrency`).
- **Throughput (req/s)** — Completed requests divided by total benchmark wall-clock time.
- **Mean latency** — Average end-to-end time per request (send to full response received).
- **RTF mean** — Average ratio of processing time to input audio duration per request. `<1` is faster than real time.
- **audio_s/s** — Total seconds of input audio processed divided by total benchmark wall-clock time.

To reproduce the results, follow the commands above or the entry point in [`benchmark_asr_transcribe_diarize.py`](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_asr_transcribe_diarize.py).

## Acknowledgments

Thanks for the joint effort of the OpenMOSS team and SGLang Omni team.

MOSS Team: Donghua Yu, Zhengyuan Lin, Hanfu Chen, Yiyang Zhang, Yang Gao, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu

SGLang Omni Team: Yijiang Tian, Xinli Jing, Xiangrui Ke, Zhihao Guo, Ruoqi Zhang, Lifan Shen, Jintao Qu, Xuxiang Tian, Kaige Li, Ratish P, Haoguang Cai, Zijie Xia, Chenchen Hong, Xuesong Ye, Jingwen Gu,  Jiaxin Deng, Jiaxuan Luo, Xinyu Lu, Hao Jin, Chenyang Zhao, Yichi Zhang
