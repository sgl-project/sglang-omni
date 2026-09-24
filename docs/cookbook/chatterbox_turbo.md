# Chatterbox-Turbo

[Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) is a 350M-parameter text-to-speech model from Resemble AI. It clones a speaker from a reference clip and synthesises 24 kHz speech.

Chatterbox-Turbo is a codec model. Its autoregressive backbone emits speech tokens, which a flow-matching vocoder turns into a waveform.

## Supported checkpoints

| Checkpoint | Status |
|---|---|
| [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) | Turbo. `examples/configs/chatterbox.yaml` |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then install the extra package and launch the server:

```bash
uv pip install --no-deps chatterbox-tts==0.1.7
uv pip install s3tokenizer resemble-perth pyloudnorm

sgl-omni serve \
  --config examples/configs/chatterbox.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

`--no-deps` is required: `chatterbox-tts` pins `torch==2.6.0`, `transformers==5.2.0`,
and `numpy<2.0.0`, which would downgrade the pinned sglang-omni stack.

On Apple Silicon the engine defaults to one request at a time. Torch/MPS
requires `max_running_requests=1`; MLX accepts `> 1` (via
`--tts_engine.engine.max_running_requests`) but it trades latency for almost no
throughput gain, so the default stays `1`. CUDA is not wired yet.

## Reference audio constraints

Voice cloning conditions the model on the reference clip twice: once through the
voice encoder (speaker embedding) and once through the S3 tokenizer (the
cond-prompt speech tokens). Matching upstream
`ChatterboxTurboTTS.prepare_conditionals`:

- the reference clip must be **strictly longer than 5 seconds** (upstream
  asserts `> 5.0 s`). Shorter clips are rejected.
- the clip is loudness-normalised to **-27 LUFS** before encoding.

A too-short reference produces few cond-prompt tokens and degrades cloning
quality (looping or near-silence); the guard above rejects it instead of
returning bad audio.

On the SeedTTS eval benchmark this means a large share of samples are
rejected: most reference clips are under 5 seconds, so the reported WER
covers only the long-reference subset. This matches the upstream
`ChatterboxTurboTTS` assertion and is not a sglang-omni limitation.

## Synthesising speech

Chatterbox-Turbo is a voice-clone model. Supply a reference clip and its
transcript to clone a speaker; a request without `references` falls back to the
built-in voice shipped in the checkpoint's `conds.pt`. There is no zero-shot
`voice` preset.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ResembleAI/chatterbox-turbo",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "seed": 42
  }' \
  --output output.wav
```

`ref_audio` / `ref_text` are accepted as a shorthand for
`references[0].audio_path` / `references[0].text`.

### Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | (required) | Text to synthesize |
| `references` | list | `null` | Reference audio for cloning. Each item has `audio_path` (local path, file URL, data URL, or HTTP URL) and `text` (transcript). The audio must be longer than 5 seconds. Omit to use the checkpoint's built-in voice |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `max_new_tokens` | int | `604` | Cap on generated speech tokens |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_k` | int | `1000` | Top-k filtering |
| `top_p` | float | `0.95` | Nucleus filtering |
| `repetition_penalty` | float | `1.2` | Penalty applied to already-seen speech tokens |
| `seed` | int | `null` | RNG seed for T3 token sampling on Torch/MPS. MLX sampling and the S3Gen vocoder do not consume a per-request seed, so the waveform is not byte-reproducible |
| `response_format` | string | `"wav"` | Output audio format (`wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`) |
| `stream` | bool | `false` | Streaming vocoder output is not supported |

## Performance

Seed-TTS EN benchmark (`zhaochenyang20/seed-tts-eval-arrow`) at `--seed 42`,
one request at a time (`max_running_requests=1`), measured on an Apple M4
(16 GB). WER is transcribed with `Qwen/Qwen3-ASR-1.7B`.

The reference-audio guard rejects the samples whose reference clip is under 5
seconds, so 357 of the 1,088 EN samples are evaluated and the other 731 are
skipped. The WER below is over that long-reference subset.

| Backend | Latency mean | RTF | WER | Evaluated |
|---|---:|---:|---:|---:|
| MLX | 3.09 s | 0.87 | 1.65% | 357 / 1088 |
| Torch/MPS | 10.52 s | 2.63 | 1.98% | 357 / 1088 |

- **Latency mean** — average end-to-end time per request (send to full response received).
- **RTF** — average ratio of processing time to generated audio duration per request. `<1` is faster than real time.
- **WER** — corpus word error rate over the evaluated (long-reference) samples.

To reproduce, start the server as in [Prerequisites](#prerequisites) and run the
generate then transcribe phases against it (the transcribe phase needs a
`Qwen3-ASR` server; on Apple Silicon start it with
`--asr.engine.max_running_requests 1`).
