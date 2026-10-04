# Parakeet ASR

NVIDIA Parakeet checkpoints serve the OpenAI-compatible `/v1/audio/transcriptions` endpoint. Parakeet pairs a FastConformer encoder with a CTC, RNN-T, or TDT head and has no language-model decoder, so it runs as a single batched stage on the Hugging Face Transformers implementation rather than through the SGLang engine. The same stage serves NVIDIA GPUs, Apple Silicon (MPS), and CPU.

## Supported Checkpoints

Any checkpoint published in Hugging Face Transformers format (a `config.json` whose `architectures` is `ParakeetForCTC`, `ParakeetForRNNT`, or `ParakeetForTDT`) is supported. NeMo-only repositories that ship just a `.nemo` file, such as `nvidia/parakeet-tdt-0.6b-v2`, are not.

| Checkpoint | Head | Languages | Output |
|---|---|---|---|
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | TDT | 25 European languages, auto-detected | Punctuated, cased |
| [nvidia/parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) / [1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) | RNN-T | English | Lowercase, unpunctuated |
| [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) / [1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CTC | English | Lowercase, unpunctuated |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md) (on macOS, `./install.sh`), then download a checkpoint:

```bash
hf download nvidia/parakeet-tdt-0.6b-v3
```

## Server Configuration

```bash
sgl-omni serve \
  --model-path nvidia/parakeet-tdt-0.6b-v3 \
  --port 8000
```

The stage places the model on the platform's accelerator: CUDA on NVIDIA GPUs and MPS on Apple Silicon. Startup runs one second of silence through the model so the first request does not pay for kernel setup.

Requests that arrive together are batched. A batch holds up to `max_batch_size` requests (default 16) and waits at most `max_batch_wait_ms` (default 5 ms) for company. Inside a batch, requests are sorted by length and split so that no forward pass pads more than `max_batch_audio_s` seconds of audio in total (default 600). A request that fails to decode fails alone; the rest of its batch still completes.

Weights run in `float32` by default. To trade a little accuracy for speed and memory, set `dtype`:

```bash
sgl-omni serve \
  --model-path nvidia/parakeet-tdt-0.6b-v3 \
  --asr.factory.dtype bfloat16 \
  --asr.factory.max_batch_size 32 \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=nvidia/parakeet-tdt-0.6b-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={"model": "nvidia/parakeet-tdt-0.6b-v3", "response_format": "json"},
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

`response_format` accepts `json`, `text`, and `verbose_json`; `stream=true` returns the transcript as a single final event. `srt` and `vtt` return HTTP 400 because the pipeline does not emit segment timestamps yet.

## Request Parameters

Parakeet decodes greedily and has no prompt or length budget, so the following return HTTP 400:

- `temperature` other than 0
- a non-empty `prompt`
- `max_new_tokens`
- `/v1/audio/translations` (Parakeet does not translate)

`language` is accepted and echoed back in `verbose_json`, but it does not steer decoding: `parakeet-tdt-0.6b-v3` detects the spoken language itself, and the CTC and RNN-T checkpoints are English-only.

## Long Audio

The FastConformer encoder has no fixed input window, but its attention cost grows quadratically with length. Uploads longer than 120 seconds are split at the quietest point near each boundary and transcribed as independent chunks; `verbose_json` returns one segment per chunk. Tune the policy with the shared `audio_chunking` settings:

```bash
sgl-omni serve \
  --model-path nvidia/parakeet-tdt-0.6b-v3 \
  --audio_chunking.max_audio_clip_s 300 \
  --port 8000
```
