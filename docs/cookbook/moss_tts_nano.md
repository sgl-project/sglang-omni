# MOSS-TTS-Nano

[MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) is a
0.1B-parameter multilingual speech model from MOSI.AI and the OpenMOSS team.
It uses 16 autoregressive audio codebooks and
[MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)
to produce native **48 kHz stereo** audio. The official checkpoint supports 20
languages, reference-less synthesis, voice cloning, and streaming.

SGLang-Omni serves it as a `preprocessing → tts_engine → vocoder` pipeline
through the OpenAI-compatible `/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).
The model and audio tokenizer are public and can be downloaded ahead of time:

```bash
hf download OpenMOSS-Team/MOSS-TTS-Nano
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano
```

## Launch the Server

The example config places the AR model and audio tokenizer on one GPU:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Nano \
  --config examples/configs/moss_tts_nano.yaml \
  --port 8000
```

## Reference-less Speech

No reference clip is required:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Nano",
    "voice": "default",
    "input": "SGLang-Omni makes small speech models easy to serve."
  }' \
  --output output.wav
```

## Voice Cloning

Pass one reference clip with `references`. The audio path may be a local path
readable by the server, an allowed HTTP(S) URL, or a base64 data URI.
`ref_audio` is the equivalent shorthand field. The Nano voice-cloning mode is
audio-only and does not accept a reference transcript.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Nano",
    "voice": "default",
    "input": "This sentence uses the voice from the reference clip.",
    "references": [{
      "audio_path": "/path/to/reference.wav"
    }]
  }' \
  --output cloned.wav
```

## Streaming

Set `stream=true` and `response_format=pcm` to receive raw signed 16-bit
little-endian PCM bytes. The response is not SSE or base64.

```bash
curl -sS -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Nano",
    "voice": "default",
    "input": "The first audio chunks can be played while generation continues.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm

ffmpeg -f s16le -ar 48000 -ac 1 -i output.pcm output_stream.wav
```

The codec is natively 48 kHz stereo, which is preserved for normal WAV
responses. The current HTTP streaming transport downmixes chunks to mono PCM16.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `references` | `null` | At most one reference clip with `audio_path`; transcript text is unsupported |
| `ref_audio` | `null` | Shorthand for the first reference clip |
| `response_format` | `wav` | Use `pcm` with `stream=true` |
| `stream` | `false` | Stream raw mono 48 kHz PCM16 bytes |
| `max_new_tokens` | `375` | Maximum generated audio frames |
| `temperature` | `1.0` text / `0.8` audio | One explicit value overrides both sampling channels |
| `top_p` | `1.0` text / `0.95` audio | One explicit value overrides both sampling channels |
| `top_k` | `50` text / `25` audio | One explicit value overrides both sampling channels |
| `repetition_penalty` | `1.2` audio | Audio-code repetition penalty |
| `seed` | `null` | Optional non-negative request seed |

## Initial Support Boundaries

- This integration serves the official PyTorch checkpoint on a GPU. It does not
  expose the upstream ONNX/CPU backend.
- The AR stage currently requires `tp_size: 1` and `pp_size: 1`.
- Each request accepts one text input and at most one reference clip. Automatic
  long-text chunking is not included yet.
- Style instructions and MOSS-TTS Local duration controls are not supported.
