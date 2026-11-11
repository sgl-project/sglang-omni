# Zonos3

[Zonos3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) is Zyphra's text-to-speech model. Generate **44.1 kHz** speech with a default voice or clone a voice from reference audio through the OpenAI-compatible `/v1/audio/speech` API.

## Quick start

<a id="installation"></a>

### Installation requirements

Follow the [CUDA installation guide](../get_started/installation.md), using a checkout that includes Zonos3 support. No additional model-specific dependencies are required. Your Hugging Face account must have access to `Zyphra-staging/Zonos3-Mini-Staging-Release`; authenticate the server environment before launching.

<a id="launch-the-server"></a>

### Start the server and check readiness

```bash
sgl-omni serve \
  --model-path Zyphra-staging/Zonos3-Mini-Staging-Release \
  --port 8000
```

In another terminal, check readiness after the weights load:

```bash
curl -fS http://localhost:8000/health
```

Continue when this returns HTTP 200 with `"status": "healthy"`. During startup it may return HTTP 503 or refuse the connection; check the server logs and retry once initialization finishes.

<a id="synthesizing-speech"></a>

### Generate output.wav with the default voice

No reference audio is required. Send text to the running server:

```bash
curl -fS -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!"
  }' \
  --output output.wav
```

The response is a complete WAV file containing 44.1 kHz mono speech. Open `output.wav` in an audio player.

## Voice cloning

Add `references` to the same request to use a reference voice. **No transcript is needed.** Zonos3 accepts at most one reference clip, at least **0.5 seconds** long. By default, clips longer than **60 seconds** are truncated to their first 60 seconds.

```json
{
  "input": "SGLang-Omni is a great project!",
  "references": [{
    "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
  }]
}
```

For reference formats and the `ref_audio` shorthand, see [Voice Cloning](../basic_usage/tts.md#voice-cloning); for server-visible local paths and remote URL access, see [media access rules](../basic_usage/tts.md#start-a-tts-server).

## Streaming

Add `"stream": true` and `"response_format": "pcm"` to either request above to receive audio while generation is still running.

Use the shared [streaming examples](../basic_usage/tts.md#raw-wav-and-pcm) or [Python streaming client](../basic_usage/tts.md#python-streaming-client) with the request fields above. Zonos3 outputs **44.1 kHz, signed 16-bit little-endian mono PCM**. The Python client reads the sample rate from the response headers and supports this format.

Streaming PCM can differ from non-streaming WAV even with the same seed. The default first chunk targets 0.8 s of audio; this is playback duration, not the wait for the first chunk.

<a id="generation-parameters"></a>

## Common options

Add these fields to the JSON request as needed:

| Parameter | Default | Usage |
|---|---|---|
| `max_new_tokens` | `2048` | Positive generation budget in **delay-pattern rows**, not text tokens or seconds. |
| `temperature` | `0.75` | Nonnegative; `0` selects greedy decoding. |
| `top_k` | `80` | Top-k sampling; integer from `1` to `4098`. |
| `top_p` | `0.69` | Applied after top-k; must be in `(0, 1]`. |
| `seed` | unset | Integer sampling seed; omitted requests draw a random seed. |
| `repetition_penalty` | `2.4` | Must be at least `1.0`; `1.0` disables the penalty. |

Text, reference audio, and the generation budget must fit the server's default 8192-row context window. The generation budget is an upper limit, not a requested audio duration. Shared fields are documented in [Request Parameters](../basic_usage/tts.md#request-parameters).

## Next steps

Use the shared [Python requests](../basic_usage/tts.md#python), [OpenAI Python SDK](../basic_usage/tts.md#openai-python-sdk), and [Python streaming client](../basic_usage/tts.md#python-streaming-client) examples with the checkpoint ID above. Remove the Qwen-specific `language` field; omit `voice` in HTTP requests or use `voice="default"` in the SDK.

## Benchmark results

SeedTTS EN voice cloning (1088 samples), one H100 per server, concurrency 16, default server and sampling settings, no request seed, scored with Qwen3-ASR-1.7B. Values are means of 30 passes: six per warm server on five GPUs. SIM is WavLM-large speaker similarity to the reference, in percentage points.

| Mode | WER (corpus) | WER, samples ≤50% | Samples above 50% WER | SIM | RTF mean | Throughput (qps) | First audio (s) |
|---|---|---|---|---|---|---|---|
| Non-streaming | 3.03% | 1.26% | 21.1 | 50.8 | 0.060 | 55.67 | - |
| Streaming | 3.13% | 1.25% | 22.2 | 50.8 | 0.070 | 48.03 | 0.14 |

Corpus WER includes all scored samples. The filtered WER excludes individual samples above 50% WER and is diagnostic only; `21.1` and `22.2` are mean counts across passes. First audio measures the wait for the first audio payload, not its playback duration. See the [TTS benchmark guide](../developer_reference/tts_bench.md) for RTF, throughput, and other metric definitions.

Throughput by concurrency on the same H100 setup: one pass per level, except concurrency 16, which repeats the 30-pass means above.

| Concurrency | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Non-streaming (qps) | 9.05 | 25.61 | 55.67 | 70.32 | 74.35 |
| Streaming (qps) | 8.54 | 23.22 | 48.03 | 57.97 | 61.66 |
| Streaming first audio (s) | 0.057 | 0.078 | 0.140 | 0.221 | 0.264 |

These original measurements did not record an immutable checkpoint revision and have not been rerun for the staging checkpoint linked above.

## Apple Silicon (Native MLX)

`zonos3_server` from the native MLX runtime (`sglang_omni_mlx/native`) serves the same `/v1/audio/speech` API on an Apple Silicon Mac, from [`mlx-community/ZONOS3-Mini`](https://huggingface.co/mlx-community/ZONOS3-Mini) (1.3 GB), as one C++ process without CUDA, PyTorch or Python. It peaks at about 2 GB of memory, so any Apple Silicon Mac runs it.

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime

hf download mlx-community/ZONOS3-Mini \
  --revision c3b1320037be83fdbe57586bd6678573de1789f4 --local-dir ZONOS3-Mini

~/omni-runtime/bin/zonos3_server --model-path ZONOS3-Mini \
  --model-name mlx-community/ZONOS3-Mini --port 8000
```

The checkpoint is mlx-audio's conversion of the release, and the native server follows mlx-audio's ZONOS3 port (commit `2c4122d7`): a seeded request gives its codes and float32 samples bit for bit, streamed or not. It therefore differs from the CUDA server above:

| | CUDA server | Native MLX server |
|---|---|---|
| Backbone | bf16 | 8-bit projections and 4-bit experts, float32 activations |
| Codec decoder | full_mr head | keep2 head |
| Voice cloning | Reference codes as a prompt prefix | A 192-dimensional speaker embedding, from reference audio or given as `references[0].speaker_embedding`; no transcript needed |
| Default voice | No speaker | `jett`; `voice` takes the 27 bundled voices in any case, or `none` |
| Text | As written | Numbers, dates, times, money and units spelled out in English words unless `language` is not English |
| `max_new_tokens` | `2048` by default | `2000` by default, capped by the 32768-row context |
| Extra fields | | `speed`, `condition_buckets` or `condition_values` (`loudness`, `fmax`, `speaking_rate`), `prepend_silence` |

- Reference audio comes in the request body: `ref_audio` is a WAV file as base64 or a `data:audio/wav;base64,...` URL, and a `references` item takes it as `data` or `audio`. Paths, URLs, `audio_path` and `vq_codes` return 400. One reference per request; the first 60 s count.
- `response_format` is `wav` or `pcm`; a streamed request sends audio every 8 frames (0.53 s).
- Started with `--model-name`, it returns 404 for any other `model`; without that flag, `model` is ignored.
- Errors are `{"detail": "..."}`, and every invalid field returns 400. A `max_new_tokens` too small to complete one frame returns empty audio.
- Requests run one at a time and `--max-queued-requests` (default 2) more wait; the rest get 503. A request not finished `--request-timeout-s` (default 600) seconds after it arrives is cancelled with 504, or its stream ends early; a client that disconnects cancels its request. A request body over 16 MiB returns 413.
- An English voice reads Chinese text poorly: with the default `jett`, SeedTTS zh scored 39% CER; choose a Chinese voice such as `meilin` or `haoran`.

Measured on an Apple M4 (16 GB), MLX 0.32.3, SeedTTS text-only (first 50 samples of each split, concurrency 1), transcribed with Qwen3-ASR-1.7B-8bit:

| Language | Voice | Mode | WER / CER | Samples above 50% | RTF mean | First audio (s) |
|---|---|---|---:|---:|---:|---:|
| en | `jett` | Non-streaming | 8.69% (1.85% for samples ≤50%) | 3 | 0.182 | - |
| en | `jett` | Streaming | 10.64% (1.10% for samples ≤50%) | 3 | 0.259 | 0.197 |
| zh | `meilin` | Non-streaming | 0.64% | 0 | 0.211 | - |
| zh | `meilin` | Streaming | 0.64% | 0 | 0.276 | 0.211 |

The English samples above 50% speak the text and then keep talking before the end of audio; mlx-audio gives the same speech for the same seed. `sglang_omni_mlx/native/ci/check_model_golden.py` checks eight seeded requests' prompt ids, audio codes and PCM against `sglang_omni_mlx/native/ci/model_golden/zonos3-mini.json`, and `sglang_omni_mlx/native/ci/test_zonos3_api.py` runs the server against the real model. Mac CI runs no model, so both are run by hand.
