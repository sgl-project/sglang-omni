# Whisper ASR

Whisper ASR checkpoints are served through the OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/translations` endpoints. The
path remains experimental in the current SGLang-Omni tree; validate
checkpoint-specific accuracy and operational behavior before production
deployment. The single-GPU profile below was validated on one RTX 4090
(24 GB) against the pinned revision documented in the benchmark results.

## Prerequisites

Install `sglang-omni` by following
[Installation](../get_started/installation.md), then download the validated
multilingual checkpoint revision:

```bash
MODEL_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1
MODEL_PATH=$(hf download openai/whisper-large-v3 \
  --revision "${MODEL_REVISION}")
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

### RTX 4090 (24 GB)

The consumer profile uses BF16, disables `torch.compile`, caps running
requests at 16, and reserves 65% of device memory for static allocations:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/whisper_asr_rtx4090.yaml \
  --model-path "${MODEL_PATH}" \
  --model-name openai/whisper-large-v3 \
  --port 8000
```

## Encoder CUDA Graph

The encoder CUDA Graph is enabled by default. With pre-LM encoding (the default), capture buckets follow `pre_lm_max_batch_size` (8), so batches **1/2/4/8** are captured. `request_build_max_workers` defaults to 8, matching Qwen3-ASR and Fun-ASR. When `enable_pre_lm_encoder` is false, buckets follow the atomic prefill budget (`6144 // 1500 = 4`). To use eager encoder execution, override the pipeline configuration:

```yaml
config_cls: WhisperASRPipelineConfig
name: whisper
model_path: openai/whisper-large-v3-turbo

runtime_overrides:
  asr:
    enable_encoder_cuda_graph: false
```

The graph is captured after SGLang's generation graphs. With pre-LM off, raise `max_prefill_tokens` before configuring larger LM-side buckets (12/16). Each request uses the smallest captured bucket that fits its batch. Requests larger than every captured bucket, with a different feature shape, or without a successful capture run eagerly. Startup and first-replay logs identify the captured and executed buckets.

## Prefill Coalescing

Whisper builds requests with eight worker threads by default, matching other pre-LM ASR pipelines. The coalescing gate targets two requests, while the default 6,144-token atomic budget lets the LM scheduler admit up to four 1,504-token Whisper requests together. A partial batch waits for at most 6 ms only while another request build is pending; a single request and a partial batch with no remaining build work are released immediately.

`request_build_max_pending` bounds submitted request-build futures, not the request backlog. When `max_queued_requests` is unset, requests beyond that pending-build limit remain queued for later construction. Setting `max_queued_requests` retains the configured finite-queue rejection behavior.

Use `prefill_coalesce_requests` and `prefill_coalesce_wait_ms` to tune the gate. Set `prefill_coalesce_requests: 0` to disable only coalescing, or also set `request_build_max_workers: 1` to restore the pre-optimization request-build path:

```yaml
runtime_overrides:
  asr:
    request_build_max_workers: 1
    prefill_coalesce_requests: 0
```

## Async Decode

Whisper enables the shared one-step-lookahead decode path at batch size 2 and above. It overlaps the current decode step's GPU work with the previous step's host-side result processing, while batch size 1 remains on the synchronous path. The default running-request limit is 64. Use the shared decode-mode option to compare against synchronous decode or diagnose a request lifecycle issue:

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --decode-mode sync \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Translate Audio

Whisper multilingual checkpoints can translate source speech to English via
`/v1/audio/translations`. Use a multilingual, non-turbo checkpoint: `*.en`
checkpoints have no translate task, and `whisper-large-v3-turbo` was distilled
without it.

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F language=fr \
  -F response_format=json
```

For this endpoint, `language` is an optional source-language hint and a
**SGLang-Omni extension**. OpenAI's official audio translations request schema
does not include `language`; the translation target is English in both APIs.
See the [audio translation support matrix](../basic_usage/audio_translations.md)
for response formats and other ASR models.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional source-language hint; on translations this is a SGLang-Omni extension |
| `prompt` | string | unset | Optional text used as Whisper prev-context conditioning |
| `response_format` | string | `json` | `json`, `verbose_json`, or raw `text`; translation `srt`/`vtt` require segment timestamps and return HTTP 400 |
| `temperature` | float | `0.0` | Sampling temperature; defaults to greedy decoding |

The serving route selects the internal `task` from the endpoint (`transcribe`
or `translate`); it is not a public form field. The route uses the ASR stage
default unless the pipeline is configured another way. For smoke tests, keep
the request minimal and use `response_format=json`.

## Long Audio

Whisper reads at most 30 seconds of audio in one request: the feature extractor works on a fixed 30-second mel window and drops everything past it.
In SGLang-Omni, we transcribe longer uploads in chunks by splitting the audio at the quietest point near each 30-second boundary,
running each chunk as its own engine request, and joining the transcripts back in order. The behavior follows these values, which Whisper
declares in code (`WhisperASRPipelineConfig.audio_chunking`). They are fixed model defaults in this release:

| Name | Value | Meaning                                                                                                                                                                     |
|---|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `max_audio_clip_s` | `30` | Longest clip we send to the engine in one request, and therefore the chunk length. Unlike Qwen3-ASR this is not a scheduling choice: 30s is the hard edge of the model's mel window. |
| `max_native_clip_s` | `30` | Same as the chunk length. Streaming cannot chunk, so `stream=true` takes audio up to 30s and gets HTTP 400 above that.                                                      |
| `max_total_audio_s` | `3600` | Upper limit on the whole upload; you get HTTP 400 above it. This is a memory guard: we keep the decoded waveform in memory while its chunks run.                            |
| `max_concurrent_chunks` | `8` | How many chunks of one request run in the engine at once. A per-request cap so one long upload can't crowd out everyone else's requests.                                    |
| `min_tail_s` | `1` | Shortest final chunk worth transcribing; if the tail would be shorter, we move the previous cut earlier to absorb it, which keeps Whisper from hallucinating on very short clips.      |

## Benchmarking

Use the shared SeedTTS benchmark for end-to-end concurrency, WER, latency, and throughput:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts

# Set this to the server's GPU worker PID reported by `nvidia-smi`.
SERVER_HOST_PID=12345

python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 --lang en \
  --model-revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --concurrencies 1,2,4,8,16,32 --repeats 3 --warmup \
  --output whisper_concurrency.json

python -m benchmarks.eval.benchmark_asr_stability \
  --port 8000 --model-path openai/whisper-large-v3 \
  --model-revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --duration-s 1800 --concurrencies 1,4,8,16 \
  --request-timeout-s 120 --no-check-audio-boundary \
  --include-translation --translation-source-language zh \
  --dtype bfloat16 --attention-backend flashinfer \
  --cuda-graph --no-torch-compile \
  --max-running-requests 16 --mem-fraction-static 0.65 \
  --gpu-process-pid "${SERVER_HOST_PID}" \
  --min-free-memory-mib 2048 --max-retained-memory-mib 256 \
  --output whisper_stability.json

python -m benchmarks.dataset.prepare --dataset covost2-zh-en
python -m benchmarks.eval.benchmark_whisper_translation \
  --backend server --port 8000 \
  --model-path openai/whisper-large-v3 \
  --model-revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --dataset-id lmms-lab/covost2 --dataset-config zh_en \
  --dataset-split test \
  --dataset-revision e38a7a7fba8adcd1563b2169afc3bc7eed202a25 \
  --source-language zh --concurrency 8 --warmup-samples 8 \
  --request-timeout-s 120 --gpu-process-pid "${SERVER_HOST_PID}" \
  --dtype bfloat16 --attention-backend flashinfer \
  --cuda-graph --no-torch-compile \
  --max-running-requests 16 --mem-fraction-static 0.65 \
  --output whisper_covost2.json

# Stop the server before running the same-revision Transformers reference.
python -m benchmarks.eval.benchmark_whisper_translation \
  --backend transformers --port 8000 \
  --model-path openai/whisper-large-v3 \
  --model-revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --dataset-id lmms-lab/covost2 --dataset-config zh_en \
  --dataset-split test \
  --dataset-revision e38a7a7fba8adcd1563b2169afc3bc7eed202a25 \
  --source-language zh --max-samples 20 --warmup-samples 8 \
  --request-timeout-s 120 --dtype bfloat16 \
  --output whisper_covost2_transformers_20.json
```

To reproduce the async-decode comparison below, resolve the pinned checkpoint and start each mode separately on the same GPU:

```bash
MODEL_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1
MODEL_PATH=$(hf download openai/whisper-large-v3 --revision "$MODEL_REVISION")

CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static 0.30 \
  --port 8000

# Replace the command above with this one for the synchronous baseline.
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static 0.30 \
  --decode-mode sync \
  --port 8000
```

Run the same client command once per mode, changing only the output filename:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --model-path openai/whisper-large-v3 \
  --model-revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --max-samples 128 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 \
  --warmup \
  --dtype float16 \
  --cuda-graph \
  --torch-compile \
  --max-running-requests 64 \
  --mem-fraction-static 0.30 \
  --fingerprint \
  --output whisper_async.json
```

## Benchmark Results

### Historical H200 references

The following W-PR1 results used the 20-sample SeedTTS EN subset on a single H200 with `openai/whisper-base` in FP16. Each mode ran one discarded warmup and three measured repeats per concurrency.

| Concurrency | Eager req/s | CUDA Graph req/s | Throughput gain | Eager mean latency (s) | CUDA Graph mean latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.57 | 20.29 | 3.7% | 0.051 | 0.049 | 0.0415 |
| 2 | 28.41 | 30.87 | 8.7% | 0.070 | 0.065 | 0.0415 |
| 4 | 37.90 | 41.70 | 10.0% | 0.104 | 0.094 | 0.0415 |
| 8 | 42.10 | 49.00 | 16.4% | 0.185 | 0.158 | 0.0415 |

All 480 W-PR1 measured requests completed successfully. Corpus WER was unchanged across eager and CUDA Graph modes at every concurrency.

The following W-PR2 results were measured separately on the same H200 and 20-sample subset with five measured repeats plus one discarded warmup per concurrency. The baseline used one request-build worker with coalescing disabled; the attribution run used two workers with coalescing disabled; the optimized run used two workers, a batch target of two, and a pending-build-aware 6 ms deadline.

| Concurrency | Baseline req/s | Two workers req/s | Coalesced req/s | Total gain | Gate gain | Baseline latency (s) | Coalesced latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.04 | 22.51 | 22.46 | 6.8% | -0.3% | 0.047 | 0.044 | 0.0415 |
| 2 | 30.45 | 36.68 | 41.96 | 37.8% | 14.4% | 0.066 | 0.047 | 0.0415 |
| 4 | 40.24 | 55.62 | 62.83 | 56.2% | 13.0% | 0.097 | 0.063 | 0.0415 |
| 8 | 48.03 | 75.93 | 82.15 | 71.0% | 8.2% | 0.161 | 0.092 | 0.0415 |

All 1,200 measured requests completed successfully. Corpus WER remained 0.0415 in all three modes and at every concurrency. Logs from the optimized run showed `Replaying Whisper encoder CUDA graph batch=2 request_batch=2` and prefill batches with two sequences and 3,008 new tokens.

The async-decode comparison used the 128-sample SeedTTS EN subset on the same H200 with `openai/whisper-large-v3` in FP16, one discarded warmup, and three measured repeats per concurrency. The baseline disabled async decode; all other serving settings, including the 6,144-token prefill budget, were identical.

| Concurrency | Sync req/s | Async req/s | Throughput change | Sync P95 (s) | Async P95 (s) | P95 change | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.26 | 11.44 | +1.6% | 0.117 | 0.115 | -1.7% | 0.0084 |
| 2 | 18.45 | 19.53 | +5.8% | 0.140 | 0.133 | -5.4% | 0.0084 |
| 4 | 27.40 | 29.35 | +7.1% | 0.197 | 0.185 | -6.2% | 0.0084 |
| 8 | 38.77 | 40.88 | +5.4% | 0.285 | 0.268 | -6.2% | 0.0084 |
| 16 | 55.59 | 57.90 | +4.2% | 0.396 | 0.366 | -7.6% | 0.0084 |
| 32 | 66.47 | 69.91 | +5.2% | 0.691 | 0.639 | -7.6% | 0.0084 |

All 4,608 measured requests across both modes completed successfully, and all 2,304 paired transcripts matched exactly. Batch size 1 uses the synchronous fast path, so its 1.6% difference is run-to-run noise rather than async work. At concurrency 32, request-stage profiling measured 614.3 ms synchronous versus 585.5 ms asynchronous P95 from prefill completion to request completion. A separate async-only `openai/whisper-base` budget comparison showed why 6,144 is the default: relative to 4,096, scheduler queue P95 fell from 92.2 ms to 52.2 ms and throughput rose from 134.83 to 166.69 req/s.

### RTX 4090 validation on base `cd45a47a`

The PR stack based on `cd45a47a` was validated in the digest-pinned ASR CI
image on one RTX 4090 (24,564 MiB), Linux 6.8, driver 590.48.01, driver CUDA
13.1, PyTorch 2.11.0+cu130, SGLang 0.5.16, SGLang-Omni 0.1.2, and Transformers
5.12.1. The outer image was
`hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101`.
The model and dataset revisions are those in the commands above. Each SeedTTS
row is the mean of three measured repeats after one discarded warmup; all
samples completed with zero skips.

SeedTTS EN (1,088 clips):

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Audio s/s | Corpus WER |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.65 | 0.103 | 0.125 | 45.70 | 0.0140 |
| 2 | 14.29 | 0.140 | 0.177 | 67.66 | 0.0140 |
| 4 | 21.30 | 0.187 | 0.253 | 100.88 | 0.0140 |
| 8 | 29.41 | 0.271 | 0.370 | 139.28 | 0.0140 |
| 16 | 37.62 | 0.424 | 0.562 | 178.15 | 0.0140 |
| 32 | 34.02 | 0.934 | 1.144 | 161.11 | 0.0140 |

SeedTTS ZH (2,020 clips; normalized error is character-spaced):

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Audio s/s | Corpus error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.96 | 0.125 | 0.158 | 37.28 | 0.0646 |
| 2 | 13.99 | 0.143 | 0.177 | 65.48 | 0.0650 |
| 4 | 20.65 | 0.193 | 0.248 | 96.66 | 0.0649 |
| 8 | 28.67 | 0.279 | 0.362 | 134.24 | 0.0650 |
| 16 | 36.38 | 0.439 | 0.585 | 170.33 | 0.0652 |
| 32 | 31.22 | 1.021 | 1.239 | 146.18 | 0.0647 |

The 30-minute mixed transcription/translation workload completed 43,515
normal requests with zero unexpected errors. All 28 cancel/reconnect and 32
malformed-audio events passed, final `/health` returned HTTP 200, sampled free
GPU memory stayed above 6,386 MiB, and retained device memory after cooldown
was 234 MiB (limit: 256 MiB). Whisper emits terminal-only streaming text, so
the cancellation check disconnects while the first event is pending and then
verifies a clean streaming reconnect.

The pinned CoVoST2 `zh_en/test` server run evaluated 4,898/4,898 samples with
zero skips in 258.88 seconds: 18.92 requests/s, 114.77 input-audio seconds/s,
0.422-second mean latency, and 0.666-second p95 latency. A same-revision
Transformers reference completed 20/20 samples with zero skips. The validated
CI environment did not include the optional `sacrebleu` package, so BLEU and
chrF are intentionally absent; these runs establish execution, coverage, and
stability, not a translation-quality parity claim.

The fixed validation base was
`cd45a47a1838017c89fb2178f167aac0cd7412a3`, and every lane used clean
integrated candidate `e54422c2de28e72c17dd2b744908e69c1b7ffd20`. The final
14-patch integrated stack was then rebased to
`b3bbff6ea1f48d50c78a4c12d059af8705f6f4f0` with a 14/14 exact `=`
range-diff. The publication history has 15 commits because the final test
cleanup was split between its PR #2 and PR #4 owners; its tree is
byte-identical to the integrated stack. The two intervening upstream commits
change only Qwen3-TTS/Moss-TTS runtime and tests plus `mps_dp.md`, not the
Fun-ASR, Whisper, benchmark, or focused-test paths validated here.

The [issue #1170](https://github.com/sgl-project/sglang-omni/issues/1170)
validation record identifies the provider run as `vastai-48058271` and the
artifact set as `run-artifacts-e54422c2`. Its 233-entry remote `SHA256SUMS`
manifest has SHA-256
`de49284660acd6a312c4f540a90232ffd3757a928578d160f5d87d81441522c8`;
the collected 248-file tree also has a verified 247-entry local manifest. The
set contains exact commands, per-sample output, dependency inventory, and
checksums. These clean pinned-base runs supersede the issue's older RTX 4090
numbers.

No Nsight Compute hardware-counter profile was run; the tables report
application metrics and NVML resource telemetry only.

These results do not directly execute upstream commits added after
`cd45a47a`; rerun the commands above when a later upstream change touches a
validated path or when changing the serving revision or stack delta.

## Known Limitations

- Whisper ASR remains experimental. Validate checkpoint-specific accuracy and
  operational behavior before production deployment.
- `verbose_json` returns a single segment spanning the audio duration; `srt`
  and `vtt` are not supported and return HTTP 400.
- Encoder CUDA Graph is enabled by default and requires SGLang generation CUDA
  Graph. Validate the selected buckets before production use.
- Audio encoding runs before LM admission by default
  (`pre_lm_max_batch_size=8`, `request_build_max_workers=8`). Set
  `enable_pre_lm_encoder: false` under `runtime_overrides.asr` to run the
  encoder inside prefill again.
- The pre-LM encoder cache (`pre_lm_cache_max_entries=1024`) keeps its
  entries in page-locked (pinned) host memory so device-to-host and
  host-to-device copies run asynchronously on the DMA path instead of
  blocking a worker thread through a pageable bounce buffer. The whole
  budget (`entries × 3.84 MB` for large-v3, `≈3.9 GB`) is locked at start-up
  and cannot be swapped; size container memory limits accordingly, or set
  `pre_lm_cache_pin_host_memory: false` to fall back to pageable memory.
- Prefill budget defaults to 6,144 tokens (`⌊6144/1500⌋=4`) under atomic
  admission (`chunked_prefill_size=0`). This caps LM-side prefill batching
  independently of the pre-LM encoder batch limit.
- Chunked prefill stays disabled because the Whisper encoder prefix must be
  admitted atomically. Requests that exceed the current prefill budget wait
  for the next batch instead of splitting the encoder prefix.
- First startup can take several minutes.
- The endpoint accepts one uploaded file per request.
- Streaming transport is supported, but Whisper currently emits terminal-only
  transcript text rather than low-latency text deltas.
- Audio is resampled to 16 kHz before transcription.
- `prompt` conditions decoding via Whisper prev-context tokens. Only the last
  223 prompt tokens are kept (224 prev-context tokens including
  `<|startofprev|>`) — fewer when `max_new_tokens` is large, since prompt,
  task prefix, and output share Whisper's 448-token decoder context.
  `max_new_tokens` is likewise clamped to that context. The prompt must not
  contain Whisper special tokens.
