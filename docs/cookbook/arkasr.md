# ARK-ASR-3B

[ARK-ASR-3B](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) (AutoArk-AI, Apache-2.0)
is a multilingual open ASR model served through the OpenAI-compatible
`/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per
request and returns text. Architecturally it is a Whisper-style audio tower
(RoPE self-attention) plus an MLP frame-merge adapter feeding a dense Qwen2 LM,
so it runs on the same single-stage batched ASR pipeline as Qwen3-ASR and reuses
SGLang's native Qwen2 decoder. The checkpoint's tokenizer and config are loaded
with `trust_remote_code=True` (the served `ServerArgs` also sets it), so the
first launch will prompt to execute the checkpoint's bundled code.

ARK-ASR does not support `/v1/audio/translations`; that endpoint returns HTTP 400. Use `/v1/audio/transcriptions`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download AutoArk-AI/ARK-ASR-3B
```

## Server Configuration

ARK-ASR runs a single ASR stage on one GPU, in `bfloat16` by default. On CUDA with TP=1, serving `AutoArk-AI/ARK-ASR-3B`, a pinned `AutoArk-AI/ARK-ASR-3B@<revision>` specification, or its Hugging Face cache snapshot enables synchronous DFlash with the registry-pinned draft. Other devices, TP sizes and unregistered checkpoints keep plain decoding unless a draft is explicitly configured.

With DFlash disabled, async decode is enabled by default for decode batches of at least two requests, allowing the shared one-step-lookahead path to overlap host-side result processing with the next GPU decode forward. Use `--asr.factory.enable_async_decode false` to disable it, or tune the crossover with `--async-lookahead-min-batch-size`.
Request concurrency and audio-encoder batching are controlled separately:

- `max_running_requests` defaults to `32` and limits requests admitted by the
  ASR scheduler.
- `encoder_max_batch_size` defaults to `8` and limits the number of uncached
  audio items processed by one encoder forward. Larger cache-miss batches are
  processed as sequential encoder microbatches, so request concurrency does
  not directly create an unbounded encoder batch.

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --port 8000
```

The encoder activation memory is in addition to the model weights and KV
cache. For long clips or high concurrency, leave additional runtime headroom
by lowering SGLang's static memory fraction, for example:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --mem-fraction-static 0.75 \
  --port 8000
```

`mem_fraction_static` controls the SGLang memory budget for model weights and
the KV-cache pool; it does not replace `encoder_max_batch_size`. The two
settings protect different parts of the serving path. ARK does not override
SGLang's default static memory fraction; use `0.75` when long clips or high
concurrency still need additional encoder headroom and the reduced KV-cache
capacity is acceptable.

To force synchronous decode while comparing modes, use:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.factory.enable_async_decode false \
  --port 8000
```

### Speculative Decoding

The default decoder verifies tokens proposed by the 19-language DFlash draft [`wilsonzheng0327/ARK-ASR-3B-dflash`](https://huggingface.co/wilsonzheng0327/ARK-ASR-3B-dflash) on one CUDA GPU. Its revision is pinned in the draft registry. Use the stage's disable switch to serve without a draft:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.engine.disable_speculative_decoding true \
  --port 8000
```

The explicit form is below; use the same draft flags when serving a local checkpoint outside the Hugging Face cache:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.engine.speculative_algorithm DFLASH \
  --asr.engine.speculative_draft_model_path wilsonzheng0327/ARK-ASR-3B-dflash \
  --port 8000
```

Scheduling is synchronous by default. Add `--asr.factory.enable_async_decode true` to overlap verification with host work. The draft turns torch.compile off for the stage: on SGLang 0.5.23 the compiled path makes each verify step several times slower, and a draft that accepted 6.8 tokens per round gained nothing at one request at a time. Before the admission tuning described below, one H200 sweep with greedy decoding and the prefix cache off measured these throughput ratios against no draft:

| Concurrent requests | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SeedTTS English, synchronous | 2.02x | 1.92x | 1.62x | 1.45x | 1.15x | 1.13x | 1.86x |
| SeedTTS English, overlapped | 2.08x | 2.05x | 1.75x | 1.51x | 1.20x | 1.17x | 1.83x |
| SeedTTS Chinese, synchronous | 1.74x | 1.60x | 1.56x | 1.39x | 1.18x | 1.21x | 2.00x |
| SeedTTS Chinese, overlapped | 1.84x | 1.65x | 1.65x | 1.44x | 1.21x | 1.20x | 1.74x |

WER stays at 1.11-1.13% on English and CER at 0.86-0.89% on Chinese, and p95 latency drops at every load in both modes. The larger gain at 64 comes from the default cap of 32 running requests, under which the server without a draft slows from 32 to 64 concurrent requests; with `--asr.engine.max_running_requests 64 --asr.engine.cuda_graph_max_bs 64` on both sides, the gain at 16, 32 and 64 is 1.12x, 1.18x and 1.06x synchronous and 1.30x, 1.27x and 1.13x overlapped. Overlap delays streamed first text: first-output p95 against the server without a draft is -4%, +6% and +15% at 1, 4 and 16 concurrent requests synchronous, and +8%, +23% and +31% overlapped. The draft reserves 13% of the KV pool. The draft checkpoint sets the default block size; `--asr.engine.speculative_num_draft_tokens` overrides it. The default context budget includes speculative KV reserve; an explicit `--context-length` must leave room for that reserve as well as the prompt and output.

### Measure First Text at Concurrency 16

Run the [P9.3](https://github.com/sgl-project/sglang-omni/issues/2941#p9--ark-asr-3b-dflash) comparison from the repository root on a dedicated H200. Use SGLang 0.5.23 when comparing with the original measurements.

```bash
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/arkasr-en-c16.XXXXXX")
target_model=$(hf download AutoArk-AI/ARK-ASR-3B \
  --revision 1e28271b79edc97635783bea65abc89195a09ed3)
cat > "$run_dir/arkasr-dflash.yaml" <<EOF
config_cls: ArkasrPipelineConfig
model_path: "$target_model"
stages:
  asr:
    engine:
      max_running_requests: 32
      speculative_algorithm: DFLASH
      speculative_draft_model_path: wilsonzheng0327/ARK-ASR-3B-dflash
      speculative_draft_model_revision: b0789bb314fab990e0f2377054c4b1947d204273
EOF

CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.speculative_sweep \
  --draft-config "$run_dir/arkasr-dflash.yaml" \
  --workload seedtts-en-unique-clips \
  --concurrencies 16 --repeats 3 --order ABBA \
  --output-dir "$run_dir/sweep" -- --stream
```

The pinned English dataset has 1,088 rows. This workload sends its 666 distinct audio clips once per pass. The sweep starts a fresh server for each leg, with the draft off for A and synchronous DFlash for B. Both sides disable the prefix cache and run one discarded warmup followed by three passes. It reports client-side p95 time to the first SSE text fragment, throughput, WER and failed requests in `$run_dir/sweep/sweep.json`. P9.3 passes when the draft's first-text p95 is at most `max(5 ms, 5% of the baseline p95)` slower than the baseline.

To record the request build, queue and prefill timings, append `--profile-events --profile-event-buffer-capacity 32768 --profile-event-dir "$run_dir/events"` after `--stream`. Each leg then runs an additional profiled pass and attaches its breakdown to that leg's `client.json`. The three measured passes finish before profiling starts. Check that every `run_status_*.json` in the event directory reports `complete: true` before interpreting the breakdown. The default 1,024-event buffer dropped events on this workload; 32,768 events per producer retained the full pass.

### Prefill Coalescing

ARK-ASR holds newly built requests briefly so the scheduler can admit a larger prefill batch. Synchronous DFlash defaults to four request-build workers and a 1 ms coalescing deadline. No draft and opt-in speculative overlap retain two workers and a 32 ms deadline. Both use a 16-request threshold. These defaults are resolved after selecting the decoding mode; explicit `request_build_max_workers` and `prefill_coalesce_wait_ms` settings take precedence.

With four workers and the earlier 4 ms DFlash deadline, a pinned SeedTTS ABBA sweep on one H200 passed the TTFT, throughput and end-to-end p95 gates at c1, c2, c4, c8 and c16 on both English and Chinese. Selected loads with the prefix cache off and the running-request cap at 32:

| Language | Concurrency | No-draft TTFT p95, ms | DFlash TTFT p95, ms | Delta, ms | Throughput ratio |
|---|---:|---:|---:|---:|---:|
| EN | 4 | 32.72 | 32.97 | +0.25 | 1.76x |
| EN | 16 | 69.97 | 71.20 | +1.23 | 1.26x |
| ZH | 4 | 28.25 | 31.28 | +3.03 | 1.62x |
| ZH | 16 | 67.07 | 65.90 | -1.17 | 1.20x |

Each p95 is the mean of six measured passes per side, with one discarded warmup per server and concurrency. The sweep used SGLang 0.5.23, PyTorch 2.14.1+cu130, BF16 and greedy streaming. There were no failed or skipped requests; mean WER was 1.15–1.17% on English and mean CER was 0.87–0.89% on Chinese.

Paired checks also passed at cap 64 with c4/c16, and with the prefix cache enabled at c1–c16, on both languages. A later English c16 ABBA measured a +0.67 ms TTFT delta. Chinese cached c4 was the closest case: +4.60 ms in the full sweep and +4.08 ms in an independent ABBA repeat, both below the 5 ms limit. See [the validation results](https://github.com/sgl-project/sglang-omni/pull/4109) for the complete comparisons.

For example, to set the original coalescing deadline explicitly:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.factory.prefill_coalesce_requests 16 \
  --asr.factory.prefill_coalesce_wait_ms 32 \
  --port 8000
```

Admission is released after either the request threshold or wait deadline is
reached; it can release earlier when pending request-build work drains. Set
`--asr.factory.prefill_coalesce_requests 0` to disable coalescing. Tune these values for the
target request distribution and latency requirements.

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=AutoArk-AI/ARK-ASR-3B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "AutoArk-AI/ARK-ASR-3B",
            "language": "en",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Stream Transcription

Set the multipart `stream` field to `true` and keep `response_format` as
`json` or `text` to receive Server-Sent Events (SSE):

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=AutoArk-AI/ARK-ASR-3B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

The response contains zero or more `transcript.text.delta` events, followed
by one `transcript.text.done` event with the complete post-processed
transcript, then `data: [DONE]`. Streaming primarily reduces time to first
text; it does not change the final transcript.

Treat `transcript.text.done` as the authoritative transcript: it is the same
post-processed string as the non-stream `text` field. Incremental
`transcript.text.delta` events are a live preview. Concatenating them may
differ from `done` by leading or trailing whitespace (the final adapter
`.strip()`s the full decode). Persist `done.text`, not `"".join(deltas)`.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `en` | Language hint recorded on the request. The transcription instruction is a fixed English prompt (`Please transcribe this audio.`); ARK-ASR auto-detects the spoken language, so this field does not switch prompt templates. |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `stream` | boolean | `false` | Emit SSE text deltas. Streaming accepts only `json` or `text` response formats |
| `temperature` | float | `0` (greedy) | Sampling temperature. When unset or `0`, SGLang's sampling normalization selects greedy decoding (`top_k=1`); no non-zero temperature is substituted. A negative value returns HTTP 422. |

### Response Formats

The three `response_format` values return different shapes:

- **`text`** — the raw transcript as `text/plain`.
- **`json`** — `{"text": "...", "usage": {"type": "duration", "seconds": <int>}}`.
- **`verbose_json`** — the OpenAI verbose shape, built by the default transcription
  adapter (whole transcript as a single segment):

  ```json
  {
    "task": "transcribe",
    "language": "en",
    "duration": 12.34,
    "text": "...",
    "segments": [{"id": 0, "start": 0.0, "end": 12.34, "text": "..."}],
    "usage": {"type": "duration", "seconds": 13}
  }
  ```

`language` and `usage` are omitted when unknown (`exclude_none`). If the uploaded
audio's duration cannot be probed, `duration` and the segment end time are `0.0`.

## Audio Handling

- Audio is resampled to **16 kHz** before feature extraction.
- Features use a 128-bin log-mel front end (stock `WhisperFeatureExtractor`).
- The feature extractor truncates at the **30-second Whisper boundary**
  (`n_samples = 480000`); audio longer than 30 s is truncated to its first 30 s.
  Mel padding is `"longest"`, so short clips do not pay the full 30 s of FFT.
- The audio-token count fed to the LM is `(mel_frames + 1) // 2 // merge_factor`
  with `merge_factor = 4` (conv2 stride-2 down-sampling, then a 4-frame merge).

## dtype

- Default serving dtype is **`bfloat16`**. This is the validated path: the native
  audio-encoder reimplementation was checked for parity against the reference
  `transformers` implementation on identical mel inputs.
- A `float16` path is also exposed. The encoder layers clamp the post-residual
  activations under fp16 (matching the reference `modeling_audio.py`) so large
  activations stay finite; this clamp is a no-op under `bfloat16`.

## Apple Silicon

On macOS arm64, ARK-ASR-3B runs through the existing Torch/MPS compatibility
path without a converted checkpoint or model-specific code changes:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --model-name AutoArk-AI/ARK-ASR-3B \
  --asr.engine.max_running_requests 1 \
  --port 8000
```

This path was validated on a MacBook Air with an M4 and 16 GB of unified
memory using the official checkpoint and `tests/data/query_to_cars.wav`. The
server returned the expected transcript through `/v1/audio/transcriptions`.

## Ascend NPU

ARK-ASR-3B runs on Ascend 910 with the provided NPU configuration:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --config examples/configs/arkasr_npu.yaml \
  --port 8000
```

See the [NPU installation guide](../other_platforms/installation_npu.md) for environment setup.

## Marker-Token Suppression

The stock checkpoint ships no `bad_words_ids`, so plain `skip_special_tokens=True`
decoding can leak non-special added markers (e.g. `<tool_call>`, `<|audio|>`) on
adversarial / OOD audio. The request builder defensively suppresses every reserved
marker (all special + `<...>`-added ids except EOS) at sampling via `logit_bias`
and strips them on decode. This is a verified no-op on clean speech.

## Pre-LM Audio Encoder

Audio encoding runs **before LM admission**, not inside the LM forward. The
audio encoder executes at request-build time on a dedicated worker thread and
CUDA stream, and a request is admitted only once its complete LM-ready
embedding is attached (`MultimodalDataItem.precomputed_embeddings`). Without
this, every admission stalls the running decode batch for the whole encoder
forward on the scheduler thread and the default stream.

Encoded embeddings are cached in a bounded CPU LRU keyed on the audio
fingerprint plus a namespace digest of the encoder pipeline (checkpoint path,
model config including `merge_factor`, mel front-end fields, dtype, attention
backend). Changing any of those re-keys the cache rather than serving a stale
embedding. Concurrent requests for identical audio are deduplicated
single-flight, so the clip is encoded once.

Request building submits encoder work without waiting for the GPU. The
scheduler holds the built LM request outside its waiting queue until that
request's encoder future completes, then performs normal admission on the
scheduler thread. A bounded encoder queue applies backpressure before mel
tensors can accumulate without limit.

| knob | default | meaning |
|---|---|---|
| `enable_pre_lm_encoder` | `true` | Off falls back to encoding inside the LM forward. |
| `pre_lm_cache_max_entries` | `4096` | Max cached embeddings. |
| `pre_lm_cache_size_bytes` | `2 GiB` | Byte budget; LRU evicts past it. |
| `pre_lm_max_batch_size` | `8` | Max queued requests drained into one `get_audio_feature` call. |
| `pre_lm_max_batch_wait_ms` | `0` | Batch-formation window. `0` is a greedy drain: items queued while the previous group encoded are taken instantly, so an idle-arrival request pays no batching latency. |
| `pre_lm_max_pending` | `32` | Max encoder items waiting behind the active batch. |

### How this relates to `encoder_max_batch_size`

The two batch knobs act at different levels and both stay in force:

- `pre_lm_max_batch_size` decides **how many queued requests are handed to one
  `get_audio_feature` call**.
- `encoder_max_batch_size` decides **how that call is executed** — it pads and
  masks the group, then splits it into sequential microbatches to bound
  encoder activation memory.

Both default to `8`, so one drained group is exactly one encoder microbatch.
Raising `pre_lm_max_batch_size` above `encoder_max_batch_size` turns a group
into several bounded forwards; it never widens a single forward, so it does
not change peak encoder activation memory.

`request_build_max_workers` defaults to **4** for synchronous DFlash and **2**
otherwise; `request_build_max_pending` defaults to **16**. These workers only perform CPU request
construction; encoder concurrency and backpressure are owned by the separate
pre-LM queue.

## Encoder CUDA Graph

The audio encoder CUDA Graph is enabled by default. At startup it captures
batch buckets derived from `encoder_max_batch_size` (powers of two, plus the
limit itself; the default of 8 yields `1/2/4/8`) and mel-frame buckets in
64-frame steps up to about 10 s (1024 frames). Longer clips, any uncaptured
bucket, and a failed capture or replay use the eager encoder; requests never
trigger capture.

The graphs are captured after SGLang's generation CUDA graphs and before the
pre-LM encoder service. To profile eager encoder execution:

```bash
sgl-omni serve --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.factory.enable_encoder_cuda_graph false
```

Or in a pipeline config:

```yaml
stages:
  asr:
    factory:
      enable_encoder_cuda_graph: false
```

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. Pass
`--model-path AutoArk-AI/ARK-ASR-3B`; the shared request and metric logic lives in
`benchmarks.tasks.asr`.

```bash
# Download the test set once:
python -m benchmarks.dataset.prepare --dataset seedtts

# Launch ARK-ASR:
sgl-omni serve --model-path AutoArk-AI/ARK-ASR-3B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup

# Quick smoke on a 20-sample subset:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --max-samples 20 --concurrencies 2 --repeats 1

# Measure text TTFT and inter-chunk latency through the SSE endpoint:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

The script reports corpus WER, throughput, and latency per concurrency level.
Transcription accuracy tracks the official `transformers` checkpoint on the same
audio.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  ARK-ASR currently ignores it (the transcription instruction is fixed).
- Audio is resampled to 16 kHz and truncated at 30 s before transcription.
