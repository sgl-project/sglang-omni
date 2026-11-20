# Fish Audio S2-Pro

[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) is a text-to-speech model that generates **44.1 kHz** speech through the OpenAI-compatible `/v1/audio/speech` API. It accepts text alone or a reference clip for voice cloning, and can stream audio during generation.

## Prerequisites

Follow the [CUDA source installation guide](../get_started/installation.md#from-source) to prepare the environment and clone the repository. From the repository root, include the Fish Audio extra in the editable install to get the Descript DAC codec. Run the server commands below from the same directory so they can find `examples/configs/s2pro_tts.yaml`.

```bash
uv pip install --prerelease=allow -e ".[fishaudio]"
```

To download the checkpoint before starting the server, run:

```bash
hf download fishaudio/s2-pro
```

<a id="server-configuration"></a>
<a id="launch-the-server"></a>

## Start the server

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

In another terminal, wait until `/health` returns HTTP 200:

```bash
curl -fS http://localhost:8000/health
```

Continue when this returns HTTP 200 with `"status": "healthy"`. During startup it may return HTTP 503 or refuse the connection; check the server logs and retry once initialization finishes.

## Synthesize speech

No reference audio is required. Send text to the running server:

```bash
curl -fS -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Hello, how are you?"
  }' \
  --output output.wav
```

The response is a complete WAV file containing 44.1 kHz mono speech. Open `output.wav` in an audio player. For Python or the OpenAI SDK, use the [TTS client examples](../basic_usage/tts.md#python) with `model` set to `fishaudio/s2-pro`. Omit the Qwen-specific `language` field, and keep `voice` as `"default"` for text-only requests.

## Voice cloning

Add `references` to clone a voice. The CUDA server accepts a reference clip with or without the optional transcript in `text`. The clip below is from [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini).

```bash
curl -fS -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

For one clip, you can use `ref_audio` and optional `ref_text`. If you also send `references`, the CUDA server appends the `ref_audio` clip after those entries. Reference quality strongly affects cloned voice quality. For reference formats and server-visible local paths or remote URLs, see [Voice Cloning](../basic_usage/tts.md#voice-cloning) and [media access rules](../basic_usage/tts.md#start-a-tts-server). S2-Pro also accepts [uploaded voices](../basic_usage/tts.md#uploaded-voices) reused by name.

## Streaming

Add `"stream": true` and `"response_format": "pcm"` to either request above to receive audio while generation is still running.

Use the shared [streaming examples](../basic_usage/tts.md#raw-wav-and-pcm) or [Python streaming client](../basic_usage/tts.md#python-streaming-client) with the S2-Pro fields above. S2-Pro outputs **44.1 kHz, signed 16-bit little-endian mono PCM**. The Python client reads the sample rate from the response headers.

A streamed request decodes in 40-frame windows, then every 45 frames, with 20 frames of overlap and a 512-sample crossfade. Streaming PCM can differ from non-streaming WAV even with the same seed. For interactive playback, the [S2-Pro playground](../basic_usage/tts.md#interactive-playground) consumes the same raw PCM stream.

<a id="request-parameters"></a>

## Common options

Add these fields to the JSON request as needed. Shared speech fields are documented in [Request Parameters](../basic_usage/tts.md#request-parameters).

| Parameter | Default | Usage |
|---|---|---|
| `max_new_tokens` | `1024` | Maximum generated semantic tokens. The engine caps the value at 2048. Reaching the limit can truncate speech. |
| `temperature` | `0.8` | Sampling temperature. |
| `top_p` | `0.8` | Top-p sampling. |
| `top_k` | `30` | Must be `-1` or an integer from `1` to `30`. Values outside that range fail the S2-Pro pipeline. |
| `repetition_penalty` | `1.1` | Semantic-history repetition penalty. |
| `seed` | unset | Integer sampling seed. Omitted requests stay unseeded. A fixed seed does not guarantee byte-identical audio across batch sizes, especially with [async decode](#async-decode). |

## CUDA tuning

For stage placement, see [TTS Process Topology](../basic_usage/tts_process_topology.md).

### Reference encoding placement

Reference audio encoding runs on GPU 0 in the preprocessing process by default, using the checkpoint's FP32 codec. Tokenization and audio loading remain on CPU. The preprocessing process has a 6 GiB PyTorch allocator limit. That value is a memory budget, not a reference-duration limit or a reservation of every GPU allocation. Long references can exceed it and fail encoding.

Set `--preprocessing.total_reserve_bytes 8GiB` to increase the budget, and leave enough GPU memory for the autoregressive engine and vocoder. To encode references on CPU instead, clear both the GPU placement and its budget:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --preprocessing.gpu null \
  --preprocessing.total_reserve_bytes null \
  --port 8000
```

### Async decode

On CUDA, enable one-step async decode to overlap CPU result collection with the next GPU decode step:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --tts_engine.factory.enable_async_decode true \
  --port 8000
```

This is opt-in. A decode batch enters the lookahead path only after its request rows stabilize; prefill and batch changes use synchronous decoding. One decode step may already be in flight when the host observes a request completion, so the effective CUDA Graph batch size can shrink one step later than in synchronous decoding. Different batch sizes can produce different floating-point results and generated codes, even with the same seed.

## Known limitations

- Keep `top_k` inside `-1` or `1..30`. Invalid values currently fail the pipeline instead of returning a clean parameter error.
- Long reference clips can exceed the preprocessing GPU allocator budget and fail encoding. Raise `--preprocessing.total_reserve_bytes` or encode on CPU as shown above.

<a id="apple-silicon-native-mlx"></a>

## Apple Silicon

`fish_s2_pro_server` from the native MLX runtime (`sglang_omni_mlx/native`) serves the same `/v1/audio/speech` API on an Apple Silicon Mac, from the 8-bit [`mlx-community/fish-audio-s2-pro-8bit`](https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit) checkpoint, as one C++ process without CUDA, PyTorch, or Python. It needs about 10 GB of memory, so use a Mac with 16 GB or more. See the [Apple Silicon installation guide](../other_platforms/installation_apple_silicon.md) and [Native MLX](../developer_reference/mlx/index.md).

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime

hf download mlx-community/fish-audio-s2-pro-8bit \
  --revision c8d4481b3f7cbfe64d855c8b7cda7739502fc3ff --local-dir fish-audio-s2-pro-8bit

~/omni-runtime/bin/fish_s2_pro_server --model-path fish-audio-s2-pro-8bit \
  --model-name fishaudio/s2-pro --port 8000
```

The text-only request above works unchanged. Sampling defaults (`temperature` 0.8, `top_p` 0.8, `top_k` 30, `repetition_penalty` 1.1, `max_new_tokens` 1024, capped at 2048) and the `top_k` range match the CUDA server. A `seed` repeats the same speech on that native server. The native server differs from the CUDA server in that:

- The voice to clone comes in the request body. `ref_audio` is a WAV file as base64 or a `data:audio/wav;base64,...` URL, with `ref_text` its transcript. `references` items take the WAV as `data` or `audio`, or the codes as `vq_codes`. Paths, HTTP URLs, and `audio_path` return 400. A WAV may be 8, 16, 24, or 32-bit PCM or float, at a common sample rate (8 to 192 kHz), with up to 254 channels. A request may hold 4 references, with 60 s of audio and 2048 transcript characters in all.
- `voice` must be `default` (or empty). There are no uploaded voices.
- `response_format` is `wav` or `pcm`; `speed` must be `1.0`; `stream_format` must be `audio`. A streamed request uses the CUDA server's windows (40 frames, then every 45, with 20 frames of overlap and a 512-sample crossfade).
- Started with `--model-name`, it returns 404 for any other `model`; without that flag, `model` is ignored, as on CUDA.
- Errors are `{"detail": "..."}`, and every invalid field, including `top_k` outside `-1` or `1..30`, returns 400.
- Requests run one at a time and `--max-queued-requests` (default 2) more wait; the rest get 503, and `/health` shows the queue.
- A request not finished `--request-timeout-s` (default 600) seconds after it arrives, time in the queue included, is cancelled: it returns 504, or its stream ends early. A client that disconnects cancels its request.
- A request body over 16 MiB returns 413; that fits 60 s of 48 kHz stereo 16-bit WAV as base64.

Validated on an Apple M5 (24 GB), MLX 0.32.3, 15 utterances: 1.0 s load, real-time factor 1.99 (it generates about 11 of the 21.5 frames in a second of audio, so streamed speech arrives slower than it plays), and 9.8 GB peak memory (10.1 GB for an 80 s utterance, 8.4 GB with a 60 s reference, which takes 4.4 s to encode). Transcribed back with Qwen3-ASR-1.7B, 10 English sentences give 0.0% WER and 5 Chinese sentences give 0.0% CER. Seeded speech differs from mlx-audio's Fish Speech port on the same checkpoint. `sglang_omni_mlx/native/ci/check_model_golden.py` checks five seeded requests against `sglang_omni_mlx/native/ci/model_golden/fish-s2-pro-8bit.json`. Mac CI runs no model, so this check is run by hand.

## Ascend NPU

Follow the [NPU installation guide](../other_platforms/installation_npu.md) and run `sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000`. [PR #1691](https://github.com/sgl-project/sglang-omni/pull/1691) also passed `--allowed-local-media-path .` for its local reference clip. The PR validated S2-Pro with the configuration below and reported the SeedTTS results, which were not re-measured for this page.

| Validation | Configuration or result |
|---|---|
| Environment | Official Ascend SGLang 0.5.18 image `sglang:v0.5.18-cann9.0.0-a3` (CANN 9.0.0) on a single Ascend 910 (`Ascend910_9362`, 64 GiB HBM) |
| NPU graph decode | Enabled with decode buckets `[1, 2, 4, 8, 16]` and `max_running_requests=16` |
| SeedTTS full set (1088 utterances), concurrency 16 | 1.428 req/s, mean latency 11.14 s, RTF 2.88, 1088/1088 successful requests |

## Next steps

Use the shared [Python requests](../basic_usage/tts.md#python), [OpenAI Python SDK](../basic_usage/tts.md#openai-python-sdk), [streaming](../basic_usage/tts.md#raw-wav-and-pcm), [SSE](../basic_usage/tts.md#sse-event-framing), [WebSocket](../basic_usage/tts.md#websocket-speech-streaming), and [S2-Pro playground](../basic_usage/tts.md#interactive-playground) examples with `fishaudio/s2-pro`. Omit `language`. Keep `voice` as `"default"` unless you are using an uploaded voice on the CUDA server.
