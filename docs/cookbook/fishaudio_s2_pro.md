# Fish Audio S2-Pro

[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) is a text-to-speech model served through `/v1/audio/speech`. It supports plain TTS, voice cloning with a reference clip, and streaming audio chunks.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

Fish Audio uses the Descript DAC codec, which is not included in the base
`sglang-omni` package. From the SGLang-Omni repository root, install its
model-specific dependencies:

```bash
uv pip install \
  "descript-audiotools==0.7.2" \
  "descript-audio-codec==1.0.0"
```

Then download the model:

```bash
hf download fishaudio/s2-pro
```

## Server Configuration

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

Reference audio encoding runs on GPU 0 in the preprocessing process by default,
using the checkpoint's FP32 codec. Tokenization and audio loading remain on CPU.
The preprocessing process has a 6 GiB PyTorch allocator limit; this is a memory
budget, not a reference-duration limit or a reservation of all GPU allocations.
Long references can exceed it and fail encoding.

Set `--preprocessing.total_reserve_bytes 8GiB` to increase the budget, while
leaving enough GPU memory for the autoregressive engine and vocoder. To encode
references on CPU instead, clear both the GPU placement and its budget:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --preprocessing.gpu null \
  --preprocessing.total_reserve_bytes null \
  --port 8000
```

## Synthesize Speech

Plain TTS:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Hello, how are you?"
  }' \
  --output output.wav
```

Voice cloning:

See the [non-streaming Fish Speech voice cloning request](../basic_usage/tts.md#fish-speech-voice-cloning).

Streaming:

See the [streaming Fish Speech voice cloning request](../basic_usage/tts.md#fish-speech-voice-cloning).

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | required | Text to synthesize |
| `voice` | string | `default` | Voice identifier for non-reference requests |
| `response_format` | string | `wav` | Output audio format |
| `speed` | float | `1.0` | Playback speed multiplier |
| `stream` | bool | `false` | Stream raw PCM audio chunks |
| `references` | list | `null` | Reference clip for voice cloning. Each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0].audio_path` and `references[0].text` |
| `max_new_tokens` | int | `2048` | Maximum generated semantic tokens |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.8` | Top-p sampling |
| `top_k` | int | `30` | Top-k sampling. It must be `-1` or between `1` and `30` |
| `repetition_penalty` | float | `1.1` | Repetition penalty |

## Apple Silicon (Native MLX)

`fish_s2_pro_server` from the native MLX runtime (`sglang_omni_mlx/native`) serves the same `/v1/audio/speech` API on an Apple Silicon Mac, from the 8-bit [`mlx-community/fish-audio-s2-pro-8bit`](https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit) checkpoint, as one C++ process without CUDA, PyTorch or Python. It needs about 10 GB of memory, so a Mac with 16 GB or more.

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime

hf download mlx-community/fish-audio-s2-pro-8bit \
  --revision c8d4481b3f7cbfe64d855c8b7cda7739502fc3ff --local-dir fish-audio-s2-pro-8bit

~/omni-runtime/bin/fish_s2_pro_server --model-path fish-audio-s2-pro-8bit \
  --model-name fishaudio/s2-pro --port 8000
```

Plain TTS works unchanged. The prompt, the semantic sampler (repetition penalty, the hotter resampling when the last four semantic codes repeat one, `top_k` of at most 30) and the defaults (`temperature` 0.8, `top_p` 0.8, `top_k` 30, `repetition_penalty` 1.1, `max_new_tokens` 1024, capped at 2048) follow the CUDA server. A `seed` draws as the CUDA server's seeded sampler does, and the same seed repeats the same speech. The native server differs from the CUDA server in that:

- The voice to clone comes in the request body: `ref_audio` is a WAV file as base64 or a `data:audio/wav;base64,...` URL, with `ref_text` its transcript; `references` items take the WAV as `data` or `audio`, or the codes as `vq_codes`. Paths and URLs, and `audio_path`, return 400. A WAV may be 8, 16, 24 or 32-bit PCM or float, at a common sample rate (8 to 384 kHz) and any channel count; references may last 60 s in all.
- `voice` must be `default` (or empty); there are no uploaded voices.
- `response_format` is `wav` or `pcm`; `speed` must be `1.0`; `stream_format` must be `audio`. A streamed request decodes in the CUDA server's windows (40 frames, then every 45, with 20 frames of overlap and a 512-sample crossfade).
- Started with `--model-name`, it returns 404 for any other `model`; without that flag, `model` is ignored, as on CUDA.
- Errors are `{"detail": "..."}`, and every invalid field, including `top_k` outside `-1` or `1..30`, returns 400.
- Requests run one at a time and `--max-queued-requests` (default 2) more wait; the rest get 503, and `/health` shows the queue.
- A request not finished `--request-timeout-s` (default 600) seconds after it arrives, time in the queue included, is cancelled: it returns 504, or its stream ends early. A client that disconnects cancels its request.
- A request body over 16 MiB returns 413; that fits 60 s of 48 kHz stereo 16-bit WAV as base64.

Measured on an Apple M5 (24 GB), MLX 0.32.3, 15 utterances: 1.0 s load, real-time factor 1.99 (it generates about 11 of the 21.5 frames in a second of audio, so streamed speech arrives slower than it plays), and 9.8 GB peak memory footprint: 10.1 GB for an 80 s utterance, 8.4 GB with a 60 s reference, which takes 4.4 s to encode. With the same seed and MLX build, its prompt ids, reference codes, audio codes and 16-bit PCM are bit-identical to mlx-audio's Fish Speech port on the same checkpoint, driven with the CUDA server's prompt and sampler. Transcribed back with Qwen3-ASR-1.7B, 10 English sentences give 0.0% WER and 5 Chinese sentences give 0.0% CER. Mac CI checks five seeded requests' reference codes, audio codes and PCM (English, Chinese, Japanese, a cloned voice, and a `max_new_tokens` cap) against `sglang_omni_mlx/native/ci/model_golden/fish-s2-pro-8bit.json`.

## Known Limitations

- `top_k` is constrained to `-1` or `1..30`; keep requests inside this range because invalid values currently fail the S2-Pro pipeline instead of returning a clean parameter error.
- Reference quality strongly affects cloned voice quality.
- Use streaming for interactive playback; CLI inspection of raw audio responses is awkward.
