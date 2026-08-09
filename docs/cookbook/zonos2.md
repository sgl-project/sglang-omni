# ZONOS2

[ZONOS2](https://huggingface.co/Zyphra) is a mixture-of-experts (MoE)
text-to-speech model from Zyphra. A MoE autoregressive decoder predicts
**9 DAC audio codebooks** scheduled in a **delay pattern**; the codes are then
decoded back to **44.1 kHz** speech by a DAC vocoder. It clones a voice from a
short reference clip and supports optional language-specific text normalization. In
SGLang-Omni it runs as a `preprocessing → speaker_encode → tts_engine →
vocoder` pipeline and is served through the OpenAI-compatible
`/v1/audio/speech` endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

ZONOS2 uses the Descript DAC codec, which is not included in the base
`sglang-omni` package. From the SGLang-Omni repository root, install its
model-specific dependencies:

```bash
uv pip install \
  "descript-audiotools==0.7.2" \
  "descript-audio-codec==1.0.0"
```

Then download the model:

```bash
hf download Zyphra/zonos2
```

The processor ships with the checkpoint. Voice cloning transcodes reference audio
(file, URL, or base64 data-URI) with **ffmpeg**, so `ffmpeg` must be on the
server's `PATH` (e.g. `apt-get install ffmpeg`).

## Server Configuration

The pipeline is `preprocessing → speaker_encode → tts_engine → vocoder`.

ZONOS2 ships a `params.json` whose `model_type` (`zonos2`) auto-selects the
`Zonos2ForCausalLM` architecture, so `serve` needs only `--model-path` — no
`--config` (mirrors Higgs).

```bash
sgl-omni serve \
  --model-path Zyphra/zonos2 \
  --port 8000
```

## Synthesizing Speech

### Voice Cloning

ZONOS2 clones a voice from a reference clip. The `references` field accepts `audio_path`
(a local path, HTTP URL, or base64 data URI) and `text` (the transcript of that clip).
The shared speech API accepts the transcript, but ZONOS2 currently conditions only on
the reference audio.

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
`references[0].text`. `ref_text` is currently unused by ZONOS2.

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
URL, or a base64 **data URI** (`data:audio/wav;base64,<...>`, transcoded via `ffmpeg`):

```json
{"ref_audio": "data:audio/wav;base64,UklGR.....", "ref_text": "Transcript of the clip."}
```

### Streaming

Set `"stream": true` with `"response_format": "pcm"` to receive raw signed 16-bit
little-endian PCM bytes. The response is not SSE or base64. ZONOS2 returns mono 44.1 kHz
audio with `Content-Type: audio/pcm` and the `X-Sample-Rate`, `X-Channels`, and
`X-Bit-Depth` headers.

```bash
curl -sS -D output.headers -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "response_format": "pcm",
    "stream": true
  }' \
  --output output.pcm

ffmpeg -f s16le -ar 44100 -ac 1 -i output.pcm output.wav
```

### Language

`language` selects the NeMo written-to-spoken normalizer before the prompt is
tokenized. `English`, `Chinese`, `Japanese`, `Korean`, `German`, `French`,
`Portuguese`, `Spanish`, and `Italian` are normalized. Omit it or use `Auto` to
keep the input unchanged; accepted languages without a NeMo package, such as
`Russian`, also pass through unchanged. The model infers the spoken language
from the resulting text.

```json
{
  "input": "今天天气不错，就该出去晒晒太阳。",
  "ref_audio": "...", "ref_text": "...",
  "language": "Chinese"
}
```

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Text to synthesize |
| `references` | `null` | Reference clip for cloning; `text` is accepted but currently unused |
| `ref_audio` / `ref_text` | `null` | Shorthand fields; `ref_text` is currently unused |
| `response_format` | `wav` | Use `pcm` when `stream=true` |
| `stream` | `false` | Stream raw mono 44.1 kHz PCM16 bytes; requires `response_format=pcm` |
| `language` | `null` | Optional NeMo text-normalization language; `Auto` preserves raw text |
| `max_new_tokens` | (model default) | Maximum generated frames; an explicit value must be `> 0` |
| `temperature` | (model default) | Sampling temperature |
| `top_p` | (model default) | Top-p sampling |
| `top_k` | (model default) | Top-k sampling |
| `min_p` | (model default) | Min-p sampling |
| `repetition_penalty` | (model default) | Audio repetition penalty |

## Benchmarking

ZONOS2 clones from each prompt (`--ref-format references`). Run the seed-tts-eval voice-clone
benchmark against a running server:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Zyphra/zonos2 --port 8000 \
  --ref-format references \
  --output-dir results/zonos2_en --lang en --max-concurrency 16
```

Use `--lang zh` for the Chinese split. See `benchmarks/README.md` for the full workflow.

## Benchmark Results

Seed-TTS-Eval **full sets** (EN 1088 / ZH 2020), 1× H100, concurrency 16, cold full clock,
`--ref-format references`, `fp8 + frame_graph + async_decode + compile_sampler`. Scored with
**Qwen3-ASR-1.7B** (the CI scorer; EN word-WER, ZH char-CER). **Median WER/CER is 0% for every
config** — the corpus numbers below are inflated by a few catastrophic-collapse samples from
unseeded sampling (EN ≤3/1088, ZH 9–16/2020), so cross-config corpus deltas are run noise, not
config differences (excluding those, corpus is EN ~1.3% / ZH ~1.4–1.7%). All EN configs clear the
<2% gate.

| Config | WER (corpus) | RTF mean | Throughput (qps) |
|---|---|---|---|
| non-streaming | 1.7% | 0.69 | 6.2 |
| streaming, `stream_emit_chunk_frames=1` | 1.4% | 0.92 | 4.6 |
| streaming, adaptive `24→32` (historical benchmark) | 1.3% | **0.77** | **5.6** |
| streaming, 2-GPU (`multi_gpu`) | 1.5% | 0.69 | 6.2 |

### Streaming throughput (`stream_emit_chunk_frames`)

In streaming mode the AR engine pushes sampled frames to the vocoder over an in-process queue.
In steady state the engine **coalesces `stream_emit_chunk_frames=32` frames into one message**
instead of one `put()` per frame; the per-frame puts run on the resolve host loop and serialize
against the next decode launch, so batching them gives **−17% streaming RTF and +21% throughput,
WER-neutral**, vs the per-frame path (inter-chunk latency also improves, 0.37 s → 0.28 s). Those
measurements used the historical `24→32` producer cadence shown above.

The current continuity-safe default sends 58 delayed-code rows in the first producer message:
the vocoder's 40-frame initial chunk plus its shared 18-row de-shear/EOS lookahead. Later
producer messages return to 32 rows. An explicit request-level `initial_codec_chunk_frames`
override recalculates the first producer boundary; request value `0` selects the vocoder's
steady 40-frame chunk. The independent `stream_emit_first_chunk_frames=0` pipeline setting
disables the adaptive producer boundary and uses the steady 32-row message size from the start.
Set `stream_emit_chunk_frames=1` for per-frame streaming. The 2-GPU `multi_gpu` pipeline
(codec + speaker encoder on `cuda:1`) stacks on top for the best measured streaming RTF (~0.69).

> ZH (`--lang zh`, full 2020) — Qwen3-ASR corpus CER 1.8–3.0% (median 0%; the corpus reflects a
> 9–16/2020 catastrophic-sample tail, not config differences — excluding those, ~1.4–1.7%).
> Non-streaming RTF ~0.62–0.64, streaming ~0.65–0.72; coalescing is audio-neutral after the
> `on_stream_done` tail fix.

## Known Limitations

- **Reference transcripts are not used.** The shared API accepts `text` / `ref_text`, but
  ZONOS2 currently conditions only on reference audio.
- **Language controls text normalization only.** It does not add a separate model-side
  language condition.
- **Per-request seeds are unsupported.** Requests with `seed` are rejected until sampling
  has an isolated request RNG.
- **Rare runaway generation.** A small fraction of utterances can loop and generate up to
  `max_new_tokens`; lowering `max_new_tokens` bounds the output.
