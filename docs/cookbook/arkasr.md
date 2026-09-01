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

ARK-ASR runs a single ASR stage on one GPU, in `bfloat16` by default.
Async decode is enabled by default for decode batches of at least two requests,
allowing the shared one-step-lookahead path to overlap host-side result
processing with the next GPU decode forward. Use `--asr.factory.enable_async_decode false` to disable
it, or tune the crossover with `--async-lookahead-min-batch-size`.
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

### Prefill Coalescing

ARK-ASR holds newly built requests briefly by default so the scheduler can
admit a larger prefill batch. The tuned defaults are 16 requests / 32 ms:

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --prefill-coalesce-requests 16 \
  --prefill-coalesce-wait-ms 32 \
  --port 8000
```

Admission is released after either the request threshold or wait deadline is
reached; it can release earlier when pending request-build work drains. Set
`--prefill-coalesce-requests 0` to disable coalescing. Tune these values for the
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

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `en` | Language hint recorded on the request. The transcription instruction is a fixed English prompt (`Please transcribe this audio.`); ARK-ASR auto-detects the spoken language, so this field does not switch prompt templates. |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0` (greedy) | Sampling temperature. When unset or `0`, SGLang's sampling normalization selects greedy decoding (`top_k=1`); no non-zero temperature is substituted. |

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

`request_build_max_workers` defaults to **2** and
`request_build_max_pending` to **16**. These workers only perform CPU request
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
sgl-omni serve --model-path AutoArk-AI/ARK-ASR-3B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup
```

The script reports corpus WER, throughput, and latency per concurrency level.
Transcription accuracy tracks the official `transformers` checkpoint on the same
audio.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  ARK-ASR currently ignores it (the transcription instruction is fixed).
- Audio is resampled to 16 kHz and truncated at 30 s before transcription.
