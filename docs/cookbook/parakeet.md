# Parakeet ASR

NVIDIA Parakeet checkpoints serve the OpenAI-compatible `/v1/audio/transcriptions` endpoint on **macOS Apple Silicon**. Parakeet pairs a FastConformer encoder with a CTC, RNN-T, or TDT head and has no language-model decoder, so it runs as a single batched stage rather than through the SGLang engine. Two backends read the same checkpoint: the Hugging Face Transformers implementation on Torch MPS (default), and a native MLX implementation selected with `SGLANG_USE_MLX=1`.

Parakeet is supported on macOS arm64 only. On other platforms (NVIDIA, AMD, Intel, CPU-only hosts) the server refuses to start the Parakeet stage.

## Supported Checkpoints

Any checkpoint published in Hugging Face Transformers format (a `config.json` whose `architectures` is `ParakeetForCTC`, `ParakeetForRNNT`, or `ParakeetForTDT`) is supported. NeMo-only repositories that ship just a `.nemo` file, such as `nvidia/parakeet-tdt-0.6b-v2`, are not.

| Checkpoint | Head | Languages | Output |
|---|---|---|---|
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | TDT | 25 European languages, auto-detected | Punctuated, cased |
| [nvidia/parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) / [1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) | RNN-T | English | Lowercase, unpunctuated |
| [nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) / [1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CTC | English | Lowercase, unpunctuated |

## Prerequisites

On an Apple Silicon Mac, install `sglang-omni` with `./install.sh` (see "Option B: macOS Apple Silicon installer" in [Installation](../get_started/installation.md)), then download a checkpoint:

```bash
hf download nvidia/parakeet-tdt-0.6b-v3
```

## Server Configuration

```bash
sgl-omni serve \
  --model-path nvidia/parakeet-tdt-0.6b-v3 \
  --port 8000
```

The stage loads the model on the MPS device. Startup runs one second of silence through the model so the first request does not pay for Metal kernel setup.

Requests that arrive together are batched. A batch holds up to `max_batch_size` requests (default 16) and waits at most `max_batch_wait_ms` (default 5 ms) for company. Inside a batch, requests are sorted by length and split so that no forward pass pads more than `max_batch_audio_s` seconds of audio in total (default 600). A request that fails to decode fails alone; the rest of its batch still completes.

Weights run in `float32` by default. To trade a little accuracy for speed and memory, set `dtype`:

```bash
sgl-omni serve \
  --model-path nvidia/parakeet-tdt-0.6b-v3 \
  --asr.factory.dtype bfloat16 \
  --asr.factory.max_batch_size 32 \
  --port 8000
```

## MLX Backend

Set `SGLANG_USE_MLX=1` to run Parakeet on native MLX instead of Torch MPS:

```bash
SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path nvidia/parakeet-tdt-0.6b-v3 \
  --port 8000
```

The MLX backend loads the same Transformers-format checkpoint as the Torch path, so no MLX-converted repository (such as `mlx-community/parakeet-tdt-0.6b-v3`) is needed. It supports all three heads (CTC, RNN-T, TDT), the same request parameters, batching, and `dtype` setting, and computes log-mel features with the checkpoint's own Transformers feature extractor, so both backends see identical inputs.

On an M5 Pro with `parakeet-tdt-0.6b-v3`, server-side end-to-end latency (single-request rows average three requests; the burst row is the median of three warm bursts of 32 concurrent 5-second clips):

| Backend | dtype | 4.6 s clip | 210 s clip (2 chunks) | 32-request burst |
|---|---|---|---|---|
| Torch MPS | `float32` | 120 ms | 3.06 s | 1.28 s |
| Torch MPS | `bfloat16` | 86 ms | 1.89 s | 0.66 s |
| MLX | `float32` | 68 ms | 1.44 s | 0.66 s |
| MLX | `bfloat16` | 58 ms | 0.93 s | 0.50 s |

All four configurations returned identical transcripts in this test.

MLX on the GPU computes `float32` matrix products at lower precision than PyTorch, so `float32` encoder outputs differ from Torch by up to about 1% relative; on the MLX CPU device they agree to within 1e-5. Transcripts matched across backends in our tests.

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
