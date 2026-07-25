# ARK-ASR-3B

[ARK-ASR-3B](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) (AutoArk-AI, Apache-2.0)
is a multilingual open ASR model served through the OpenAI-compatible
`/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per
request and returns text. Architecturally it is a Whisper-style audio tower
(RoPE self-attention) plus an MLP frame-merge adapter feeding a dense Qwen2 LM,
so it runs on the same single-stage batched ASR pipeline as Qwen3-ASR and reuses
SGLang's native Qwen2 decoder. Serving requires no `trust_remote_code`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download AutoArk-AI/ARK-ASR-3B
```

## Server Configuration

ARK-ASR runs a single ASR stage on one GPU, in `bfloat16` by default.

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --port 8000
```

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
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |

`verbose_json` is accepted, but currently returns the same minimal JSON shape as
`json`: `{"text": "..."}`.

## Audio Handling

- Audio is resampled to **16 kHz** before feature extraction.
- Features use a 128-bin log-mel front end (stock `WhisperFeatureExtractor`).
- The feature extractor truncates at the **30-second Whisper boundary**
  (`n_samples = 480000`); audio longer than 30 s is truncated to its first 30 s.
  Mel padding is `"longest"`, so short clips do not pay the full 30 s of FFT.
- The audio-token count fed to the LM is `(mel_frames + 1) // 2 // merge_factor`
  with `merge_factor = 4` (conv2 stride-2 down-sampling, then a 4-frame merge).

## dtype

- Default serving dtype is **`bfloat16`**. This is the validated path (the audio
  encoder reimplementation is numerically equivalent to the reference, max abs
  diff `0.0` on identical mel inputs).
- A `float16` path is also exposed. The encoder layers clamp the post-residual
  activations under fp16 (matching the reference `modeling_audio.py`) so large
  activations stay finite; this clamp is a no-op under `bfloat16`.

## Marker-Token Suppression

The stock checkpoint ships no `bad_words_ids`, so plain `skip_special_tokens=True`
decoding can leak non-special added markers (e.g. `<tool_call>`, `<|audio|>`) on
adversarial / OOD audio. The request builder defensively suppresses every reserved
marker (all special + `<...>`-added ids except EOS) at sampling via `logit_bias`
and strips them on decode. This is a verified no-op on clean speech.

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

On the full SeedTTS EN set (1088 clips) the served model reports a corpus WER of
about **1.1%**, matching the official `transformers` checkpoint.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  ARK-ASR currently ignores it (the transcription instruction is fixed).
- Audio is resampled to 16 kHz and truncated at 30 s before transcription.
