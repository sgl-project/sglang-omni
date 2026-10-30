# Chatterbox-Turbo

[Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) is a 350M-parameter text-to-speech model from Resemble AI. A GPT-2 backbone (T3) samples S3 speech tokens for the text, conditioned on a speaker, and S3Gen (a conformer encoder, a two-step mean-flow decoder and a HiFT vocoder) turns them into 24 kHz audio. It speaks in a built-in voice or clones one from a few seconds of reference audio.

## Apple Silicon (Native MLX)

`chatterbox_server` from the native MLX runtime (`sglang_omni_mlx/native`) serves the OpenAI-compatible `/v1/audio/speech` API on an Apple Silicon Mac, from the [`mlx-community/chatterbox-turbo-fp16`](https://huggingface.co/mlx-community/chatterbox-turbo-fp16) checkpoint and its [`mlx-community/S3TokenizerV2`](https://huggingface.co/mlx-community/S3TokenizerV2) speech tokenizer, as one C++ process without PyTorch or Python.

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh ~/omni-runtime

hf download mlx-community/chatterbox-turbo-fp16 \
  --revision b2d0a13aa7cfff0a06d9acb247ae91c8f19a6d75 --local-dir chatterbox-turbo
hf download mlx-community/S3TokenizerV2 \
  --revision e0c9886f0e1c35ae85b1f27277416fb19fc72bec --local-dir chatterbox-turbo/speech_tokenizer

~/omni-runtime/bin/chatterbox_server --model-path chatterbox-turbo \
  --model-name ResembleAI/chatterbox-turbo \
  --allowed-local-media-path docs/_static/audio --port 8000
```

Speak in the built-in voice:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Get the trust fund to the bank early.", "seed": 42}' \
  --output output.wav
```

Clone a voice from a mono or stereo WAV file longer than 5 seconds and at most 60 seconds, as a path inside `--allowed-local-media-path` or as a base64 data URL; the transcript is not needed:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{"audio_path": "docs/_static/audio/male-voice.wav"}],
    "seed": 42
  }' \
  --output output.wav
```

`ref_audio` is a shorthand for `references[0].audio_path`, and a reference may instead give base64 WAV bytes in `data`.

### Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to speak, at most 4096 characters and 1024 text tokens |
| `model` | string | served model | Must equal `--model-name` when that flag is given |
| `voice` | string | `default` | Only `default`, the built-in voice |
| `references` | list | none | One reference whose `audio_path`, `ref_audio`, `audio` or `data` holds a mono or stereo WAV file of more than 5 s and at most 60 s |
| `ref_audio` / `ref_text` | string | none | Shorthand for `references[0]`; the text is not used |
| `response_format` | string | `wav` | `wav` or `pcm` (16-bit, 24 kHz, mono) |
| `max_new_tokens` | int | `1000` | Cap on speech tokens, 25 a second, at most 1024; generation stops after exactly this many |
| `temperature` | float | `0.8` | Sampling temperature, above 0 |
| `top_k` | int | `1000` | Top-k filtering, `-1` for off |
| `top_p` | float | `0.95` | Nucleus filtering, in (0, 1] |
| `repetition_penalty` | float | `1.2` | Penalty on already sampled speech tokens, in (0, 2] |
| `seed` | int | random | With the same seed, MLX version and chip, the audio repeats bit for bit |

The server differs from the other speech servers in that:

- `stream` must be `false` and `speed` must be `1.0`; `voice` presets other than `default` and HTTP(S) reference URLs return 400.
- Local reference paths and `file://` URLs are read only inside `--allowed-local-media-path`.
- Started with `--model-name`, it returns 404 for any other `model`; without that flag, `model` is ignored.
- Errors are `{"detail": "..."}`.
- Requests run one at a time and `--max-queued-requests` (default 2) more wait; the rest get 503, and `/health` shows the queue.
- A request not finished `--request-timeout-s` (default 600) seconds after it arrives, time in the queue included, is cancelled with 504, and so is a request whose client disconnects.
- A request body over 16 MiB returns 413.

### Accuracy and performance

Measured on an Apple M5 (24 GB, macOS 26.3.1), warm, with 20 Harvard sentences per voice, each with its own seed:

| Voice | Mean latency | Real-time factor | WER (Qwen3-ASR-0.6B) |
|---|---:|---:|---:|
| Built-in | 1.08 s | 0.43 | 0.6% |
| Cloned, male reference | 1.02 s | 0.44 | 0.0% |
| Cloned, female reference | 1.22 s | 0.45 | 0.0% |

The latency is end to end over HTTP and includes cloning the voice from the reference on every request. Peak MLX memory is about 3.4 to 3.9 GB, and the server is ready about 1.3 s after it starts.

With the same seed and MLX build, the speech tokens of the native runtime are identical to the [mlx-audio](https://github.com/Blaizzy/mlx-audio) port on the same weights, and so is its audio in the built-in voice. A cloned voice's speech tokens are identical too. Its audio differs from mlx-audio's: the CAM++ speaker encoder follows the original model where mlx-audio does not, averaging its last, partial 100-frame segment over its own frames (ceil-mode average pooling) rather than zero padding it, and pooling the unbiased standard deviation rather than the population one. Mac CI checks five seeded requests (two in the built-in voice, one of them capped by `max_new_tokens`, two cloned voices, and 1024 fixed speech tokens voiced at the token cap), their tokens and their audio, against `sglang_omni_mlx/native/ci/model_golden/chatterbox-turbo.json`.

### Known limitations

- **About 10 seconds per request.** T3 stops after about 250 speech tokens (10 s of speech) however long the input is, so longer text should be sent one sentence or paragraph at a time.
- **References of 5 to 60 seconds.** Only the first 15 s (for T3) and 10 s (for S3Gen) are used, but loudness is measured over the whole recording, so a longer one is refused rather than cut.
- **No streaming.** The whole utterance is generated before the response is sent.
- **English.** Chatterbox-Turbo is an English model.
