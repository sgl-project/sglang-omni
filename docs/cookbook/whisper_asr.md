# Whisper ASR

Whisper ASR checkpoints are served through the OpenAI-compatible
`/v1/audio/transcriptions` and `/v1/audio/translations` endpoints. The
configuration below is validated on one RTX 4090 (SM89, 24 GB) on Linux.

## Prerequisites

Install `sglang-omni` by following
[Installation](../get_started/installation.md). Install FFmpeg when accepting
compressed inputs such as MP3, then download the pinned checkpoint:

```bash
apt-get install -y ffmpeg

MODEL_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1
MODEL_PATH=$(hf download openai/whisper-large-v3 \
  --revision "${MODEL_REVISION}" \
  added_tokens.json config.json generation_config.json merges.txt \
  model.safetensors normalizer.json preprocessor_config.json \
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json)
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

### RTX 4090 (24 GB)

The consumer profile uses BF16, FlashInfer decoder attention, PyTorch SDPA for
the encoder, CUDA Graph batches through 16, `max_running_requests=16`,
`mem_fraction_static=0.65`, and no `torch.compile`.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/whisper_asr_rtx4090.yaml \
  --model-path "${MODEL_PATH}" \
  --model-name openai/whisper-large-v3 \
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

## Translate Audio into English

`/v1/audio/translations` accepts the standard OpenAI translation fields. An
optional `language` extension supplies the source language because automatic
Whisper language detection is not yet implemented in the SGLang request path.

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F model=openai/whisper-large-v3 \
  -F file=@chinese_speech.mp3 \
  -F language=zh \
  -F response_format=json
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional transcription language or translation source-language hint |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.0` | Sampling temperature; defaults to greedy decoding |
| `max_new_tokens` | integer | stage default | Transcription-only extension, bounded by the configured stage maximum |
| `stream` | boolean | `false` | Transcription-only SSE output |

The translation route sets Whisper's internal task to `translate`; callers do
not send a `task` field.

## Benchmarking

Use the pinned SeedTTS harness for EN/ZH transcription and CoVoST2 for
Chinese-to-English translation:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.dataset.prepare --dataset covost2-zh-en

python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path openai/whisper-large-v3 --port 8000 --lang en \
  --concurrencies 1,2,4,8,16,32 --repeats 3 --warmup

python -m benchmarks.eval.benchmark_whisper_translation \
  --port 8000 --source-language zh --concurrency 8

python -m benchmarks.eval.benchmark_asr_stability \
  --model-path openai/whisper-large-v3 --port 8000 \
  --duration-s 1800 --concurrencies 1,4,8,16 \
  --include-translation --translation-source-language zh
```

## RTX 4090 Validation

Environment: Linux 6.8, one RTX 4090 24,564 MiB (SM89), driver 580.126.20,
CUDA 13.0, PyTorch 2.11.0, SGLang 0.5.12.post1, Transformers 5.6.0, and
model revision `06f233fe06e710322aca913c1bc4249a0d71fce1`.

The full SeedTTS EN set completed without skips at every level. Corpus WER was
0.0140. Mean results across three measured runs:

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Mean RTF | Audio s/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.42 | 0.451 | 1.265 | 0.0980 | 11.46 |
| 2 | 9.79 | 0.204 | 0.260 | 0.0445 | 46.37 |
| 4 | 12.54 | 0.318 | 0.423 | 0.0692 | 59.41 |
| 8 | 14.92 | 0.535 | 0.718 | 0.1160 | 70.67 |
| 16 | 16.46 | 0.971 | 1.311 | 0.2100 | 77.94 |
| 32 | 16.50 | 1.928 | 2.380 | 0.4203 | 78.16 |

The full SeedTTS ZH set also completed without skips. Corpus CER was
0.0646–0.0654:

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Mean RTF | Audio s/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.60 | 0.151 | 0.191 | 0.0327 | 30.90 |
| 2 | 8.53 | 0.234 | 0.286 | 0.0507 | 39.92 |
| 4 | 11.94 | 0.335 | 0.462 | 0.0722 | 55.90 |
| 8 | 14.91 | 0.536 | 0.730 | 0.1157 | 69.81 |
| 16 | 16.87 | 0.948 | 1.302 | 0.2043 | 78.97 |
| 32 | 16.19 | 1.970 | 2.422 | 0.4258 | 75.79 |

The complete pinned CoVoST2 `zh_en/test` translation set evaluated
4,898/4,898 samples with zero skips at concurrency 8: 13.09 requests/s,
0.611 s mean latency, 1.055 s p95, BLEU 14.02, and chrF 41.07. These are
absolute consumer results; no matching same-revision H100 reference is
currently published. On the fixed first-20 subset, the SGLang path scored
BLEU 18.98 / chrF 44.47 versus BLEU 22.10 / chrF 46.37 for the same BF16
checkpoint through Hugging Face Transformers. Translation is functional, but
does not yet meet a strict same-revision quality-parity gate.

The 30-minute mixed transcription/translation soak completed 18,390 requests
with zero unexpected errors. All 28 malformed-input and 28
cancel/reconnect events passed, `/health` remained available, and cooldown
retained 0 MiB.

Startup used 16.04 GiB after static allocation and 16.36 GiB after CUDA Graph
capture. A warmed mixed workload used about 21.17 GiB process memory and kept
about 2.9 GiB free; the sampled translation peak left 2.46 GiB free.

## Known Limitations

- First startup can take several minutes.
- The endpoint accepts one uploaded file per request.
- Each request is limited to 30 seconds; split longer recordings before upload.
- Audio is resampled to 16 kHz before transcription.
- Translation outputs English only. Pass the optional source `language` for
  reproducible quality until automatic language detection is implemented.
- The custom SGLang Whisper path trails the same-revision Transformers
  translation reference on the measured CoVoST2 subset; treat the published
  translation score as a baseline rather than parity.
- Nsight Compute 2024.3.2 was present on the validation host, but kernel-counter
  collection was blocked by `ERR_NVGPUCTRPERM`. NVML resource sampling and
  SGLang scheduler metrics remain included in the results.
- SSE transport is functional, but this model currently emits only a terminal
  transcript event rather than low-latency text deltas.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  Whisper ASR currently does not pass it into decoding.
