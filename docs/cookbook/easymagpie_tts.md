# EasyMagpie TTS

EasyMagpie (Nemotron-TTS) is NVIDIA's streaming text-to-speech model. Its talker
is a Nemotron-H hybrid Mamba / attention / MoE backbone. A small autoregressive
local transformer predicts each frame's codes, and a phoneme stream runs a few
steps ahead of the audio. A causal spectral codec turns 16 stacked FSQ codebooks
into 22.05 kHz speech. SGLang-Omni serves it through the
`preprocessing → tts_engine → vocoder` pipeline and the OpenAI-compatible
`/v1/audio/speech` endpoint, with preset voices selected by name and language.

## Model Directory

The server loads a converted model directory, not a `.nemo` archive:

| Path | Contents |
|---|---|
| `config.json` | Nemotron-H backbone fields and the EasyMagpie scalars (vocabularies, codebooks, streaming delays) |
| `model.safetensors`, `model.safetensors.index.json` | Talker weights, with the text embedding baked into one lookup table |
| `tokenizer.json`, `tokenizer_config.json` | Text tokenizer for target and context text |
| `codec_native/config.json`, `codec_native/model.safetensors` | Causal codec decoder |
| `speaker_embeddings/` | Preset voices (see below) |

`speaker_embeddings/` holds the voices in one of two layouts:

- **Registry:** `contexts.json` + `contexts.safetensors`, the layout of the
  Nemotron-Speech repository builder. Each context has a tensor and a paired
  `context_text`. `voice_contexts` maps a voice and language to a context,
  `langs_supported` lists each language's voices, and `default_speaker_id`
  names the default voice.
- **Per-voice files:** `<voice>.pt`, one context per voice, with no paired text
  or languages. The voice `eng` is the default when present.

## Server Configuration

The first startup takes several minutes. The engine captures decode and
prefill CUDA graphs and compiles the talker; the vocoder captures its codec
graphs.

```bash
sgl-omni serve \
  --model-path /path/to/easymagpie_model \
  --config examples/configs/easymagpie_tts.yaml \
  --port 8000
```

The defaults are the tuned configuration:

| Knob | Default | Meaning |
|---|---|---|
| `--tts_engine.factory.max_running_requests` | `64` | Concurrent decode rows |
| `--tts_engine.factory.cuda_graph` | `true` | Decode and prefill CUDA graphs |
| `--tts_engine.factory.torch_compile` | `true` | Compiled talker decode |
| `--tts_engine.factory.mem_fraction_static` | `0.72` | Engine memory fraction |
| `--vocoder.factory.startup_chunk_frames` | `[2, 6]` | First streamed chunks, in stacked frames of 80 ms |
| `--vocoder.factory.steady_chunk_frames` | `8` | Chunk size after the startup chunks |
| `--vocoder.factory.cuda_graph` | `true` | CUDA graphs for the streaming codec |
| `--vocoder.factory.freeze_gc` | `true` | Freeze the garbage collector once serving starts |
| `--preprocessing.factory.lead_in_frame` | `true` | Stream the frame of the step before the speech delay, one decode step earlier |

With the prefill graph on and chunked prefill off, `max_prefill_tokens`
defaults to the largest captured prefill bucket + 1. A burst of arrivals then
runs as captured prefills instead of one eager prefill. An explicit
`max_prefill_tokens` overrides this.

The pipeline runs on one GPU (TP1).

## Synthesizing Speech

### Preset Voices

Each request selects one context:

| Request | Context used |
|---|---|
| `voice` and `language` | That voice's context for the language |
| `voice` only | The voice's first context |
| `language` only | The first voice listed for the language |
| Neither | The default voice |

The context's paired `context_text` conditions the model unless the request
sends its own `context_text`. A voice that has no context for the requested
language is rejected, and the error lists the languages it supports.
`language` takes the speech API names (`English`, `French`, `Spanish`,
`Chinese`, `Vietnamese`, ...) or language tags such as `fr-FR`. `Auto` is
treated as no language.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "easymagpie",
    "input": "Hola, ¿cómo estás? Hoy hace un día muy bonito para pasear por el parque.",
    "voice": "Diego",
    "language": "Spanish",
    "response_format": "wav"
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` and `"response_format": "pcm"` to receive 16-bit mono PCM
at 22.05 kHz as it is generated:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "easymagpie",
    "input": "This sentence streams as it is generated.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `voice` | registry default | Preset voice name; `default` selects the default voice |
| `language` | none | Selects the voice's context for that language |
| `context_text` | the context's paired text | Overrides the conditioning text |
| `temperature` | `0.7` | Sampling temperature |
| `top_k` | `80` | Top-k sampling; at most 80 |
| `seed` | `null` | Random seed for reproducibility |
| `max_new_frames` | `2048` | Maximum generated stacked frames (80 ms each) |
| `stream` | `false` | Stream raw PCM chunks |

## Benchmark Results

All results are from one NVIDIA GeForce RTX 4080 (16 GB), with streaming PCM
requests and default server settings. Each request synthesizes the same
sentence of moderate length (about 11 s of audio). "First audio" is the time to
the first PCM chunk. Throughput is seconds of audio generated per second of
wall time, summed over all requests.

| Concurrency | First audio P50 | First audio P99 | Chunk P50 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 31.0 ms | 32.4 ms | 55.0 ms | 11.3 |
| 8 | 52.4 ms | 57.0 ms | 66.5 ms | 70.1 |
| 16 | 65.8 ms | 70.2 ms | 73.5 ms | 124.9 |
| 32 | 98.5 ms | 115.8 ms | 91.5 ms | 197.6 |

This sweep used a checkpoint whose voice comes from a reference recording.
Later c=1 runs with the NemotronTTS GRPO Step120 checkpoint and its text-only
`Aria` voice measured 30.8 and 32.2 ms first audio (P50) over two runs, with
throughput 11.2. Whisper-large-v3-turbo transcripts of that checkpoint's
English, Spanish, French, Vietnamese and Chinese samples matched the input
text.

## Known Limitations

- **TP1 only.**
- **Pronunciation control is not wired up.** Checkpoints trained with
  `<bop>...<eop>` IPA spans in the target text record the markers in
  `config.json`, but the server tokenizes such spans as plain text.
- **Request errors return HTTP 500.** Unknown voices and unsupported
  voice/language pairs are rejected during preprocessing with a descriptive
  message, but the status code is 500 rather than 400.
- **Preset voices only.** Reference-audio cloning (`ref_audio`, `references`)
  is not supported.
