# MOSS-TTS

[MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) is a discrete
multi-codebook text-to-speech model from the OpenMOSS team. It pairs a Qwen3 language-model
backbone with 32 residual-vector-quantization (RVQ) audio codebooks scheduled in a **delay
pattern** (one text channel plus 32 audio channels, advanced one codebook per frame). It clones
a voice from a short reference clip, can synthesize without a reference, supports inline
duration control, and the vocoder reconstructs 24 kHz speech. In SGLang-Omni it runs as a
`preprocessing → tts_engine → vocoder` pipeline and is served through the OpenAI-compatible
`/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then
download the model (public, no token required):

```bash
hf download OpenMOSS-Team/MOSS-TTS-v1.5
```

The processor ships with the checkpoint, so no extra TTS package is needed. Decoding base64
(data-URI) reference audio additionally requires `soundfile` (`uv pip install soundfile`).

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`.

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --port 8000
```

## Synthesizing Speech

### Basic Speech

MOSS-TTS can synthesize speech without a reference clip:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "SGLang-Omni is a great project!"}' \
  --output output.wav
```

### Voice Cloning

Provide a reference clip when you want voice cloning. The `references` field accepts `audio_path`
(a local path, HTTP URL, or base64 data URI) and `text` (the transcript of that clip). Supplying
the transcript materially improves cloning quality.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and
`references[0].text`.

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Reference Audio Sources

`audio_path` / `ref_audio` may be a local filesystem path readable by the server, an HTTP(S)
URL, or a base64 **data URI** (`data:audio/wav;base64,<...>`, decoded with `soundfile`):

```json
{"ref_audio": "data:audio/wav;base64,UklGR.....", "ref_text": "Transcript of the clip."}
```

### Streaming

Set `"stream": true` to receive Server-Sent Events (SSE). Audio events carry base64-encoded WAV
bytes in `audio.data`; the final metadata event has `audio: null`, followed by `data: [DONE]`.

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true
  }'
```

### Duration Control

MOSS-TTS conditions on a target **duration token count** (codec frames; a larger count yields
longer audio). Set it with an inline `${token:N}` prefix on `input` (stripped before synthesis),
or with a `token_count` (alias `duration_tokens` / `tokens`) parameter. The count must be a
positive integer.

```json
{"input": "${token:150}A sentence with an explicit duration target.", "ref_audio": "..."}
```

If omitted, the model picks a duration on its own; the SeedTTS benchmark estimates one per
sample with `--token-count auto`.

### Text Markup, Style, and Language

Inline text markup that the model understands (for example `[pause Xs]`, pinyin, and IPA) is
passed through unchanged. An optional `instructions` (alias `instruct`) field carries a
free-text style directive, and an optional `language` hint biases the target language (omit it
to let the model infer from the text):

```json
{
  "input": "今天天气不错 [pause 0.5s] 就该出去晒晒太阳。",
  "ref_audio": "...", "ref_text": "...",
  "language": "Chinese",
  "instructions": "Speak slowly and warmly."
}
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize; may carry a `${token:N}` duration prefix and inline markup |
| `references` | `null` | Reference clip for cloning; each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `stream` | `false` | Enable SSE streaming |
| `language` | `null` | Optional target-language hint; omit to let the model infer |
| `instructions` / `instruct` | `null` | Optional free-text style directive |
| `token_count` / `duration_tokens` / `tokens` | `null` | Target duration in codec frames; must be `> 0` |
| `max_new_tokens` | `4096` | Maximum generated frames; an explicit value must be `> 0` |
| `temperature` | `1.5` text / `1.7` audio | Sampling temperature; a single `temperature` overrides both channels |
| `top_p` | `1.0` text / `0.8` audio | Top-p sampling; a single `top_p` overrides both channels |
| `top_k` | `50` text / `25` audio | Top-k sampling; a single `top_k` overrides both channels |
| `repetition_penalty` | `1.0` | Audio repetition penalty |
| `seed` | `null` | Non-negative integer; see [Seed Reproducibility](#seed-reproducibility) |

The per-channel fields (`text_temperature`, `audio_temperature`, `text_top_p`, `audio_top_p`,
`text_top_k`, `audio_top_k`, `audio_repetition_penalty`) are also accepted and take precedence
over the single-value aliases.

## Seed Reproducibility

MOSS-TTS samples each row, position, and codebook with `multinomial_with_seed`, deriving a
per-request seed from the public `seed` and combining it with a per-(step, channel) position. A
sampled token therefore depends only on its own seed and position — never on its batch
neighbours — so a fixed `seed` is reproducible at any concurrency, not just batch size 1.
Limitations:

- Reproducibility holds for a **fixed server configuration and hardware**. Floating-point
  non-determinism in the backbone (different batch shapes, GPU models, or kernels) can still
  change the logits, and thus the sampled tokens, across deployments.
- `seed` must be a non-negative integer; non-integer or negative values are rejected.
- Requests **without** a `seed` draw a fresh random per-request seed, so they are not
  reproducible across runs (but are still independent of batch neighbours).

## Benchmarking

MOSS-TTS clones from each prompt (`--ref-format references`) and estimates a per-sample duration
with `--token-count auto`. Run at `--max-concurrency 8`; higher concurrency regresses WER.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 \
  --ref-format references --token-count auto \
  --output-dir results/moss_tts_en --lang en --max-concurrency 8
```

Use `--lang zh` for the Chinese split. See `benchmarks/README.md` for the full workflow.

## Benchmark Results

Seed-TTS-Eval full set (EN = 1088, ZH = 2020) on 1× H200, concurrency 8, `--token-count auto`.
WER is scored with HF Whisper-large-v3 (EN) / FunASR paraformer-zh (ZH). These are the reference
numbers tabulated in `benchmarks/eval/benchmark_tts_seedtts.py` (source: PR #609) — reproducible
references, not CI thresholds.

| Lang | WER (corpus) | WER (excl. >50%) | Latency mean / p95 (s) | RTF mean | Throughput (qps) |
|---|---|---|---|---|---|
| EN | 1.68% | 1.32% (4 outliers) | 3.449 / 4.141 | 0.811 | 2.312 |
| ZH | 1.36% | 1.27% (2 outliers) | 3.608 / 4.153 | 0.635 | 2.213 |

A handful of utterances run away into a repetition loop (> 50% WER) and dominate the raw
micro-average; excluding them, corpus WER is ~1.3% in both languages, and per-sample median WER
is 0.00%.

## Known Limitations

- **Voice cloning depends on the reference.** Omit the reference for non-cloned speech; provide
  the transcript (`text` / `ref_text`) for the best speaker similarity when cloning.
- **Concurrency vs. WER.** Quality is best around `--max-concurrency 8`; higher concurrency
  regresses WER.
- **Rare runaway generation.** A small fraction of utterances can loop and generate up to
  `max_new_tokens`; setting a `token_count` (or lowering `max_new_tokens`) bounds the output.
- **Duration is a hint.** `${token:N}` / `token_count` steers length but is not an exact clip
  duration.
