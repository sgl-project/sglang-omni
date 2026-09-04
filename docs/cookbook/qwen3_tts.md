# Qwen3 TTS

[Qwen3-TTS-12Hz-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) is a discrete
multi-codebook text-to-speech model from the Qwen team. It performs fast voice cloning from a
short reference clip, supports 10 languages, and streams 24 kHz speech with low latency. The
`12Hz` in the name refers to the codec **frame rate** (12 acoustic frames per second), not the
playback sample rate. SGLang-Omni serves two checkpoints — `0.6B` and `1.7B` — through the same
`preprocessing → tts_engine → vocoder` pipeline and the OpenAI-compatible `/v1/audio/speech`
endpoint.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

Qwen3-TTS Base uses the upstream `qwen-tts` package. Install it without
dependencies so the SGLang-Omni Transformers 5.12 / SGLang 0.5.18 stack remains
in place:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

`--no-deps` is required on **both** lines, for two different reasons.

`qwen-tts` pins Transformers 4.57.3, which would replace the project's 5.12.1.
And resolving `sox` normally pulls `numpy` past the ceiling `numba==0.65.1`
imposes (numba requires `numpy<=2.4`); the upgraded `numpy` then breaks
`librosa`, so `import qwen_tts` fails with `Numba needs NumPy 2.4 or less`
before the server can start.

Do not add `onnxruntime` to that line either — it is already a SGLang-Omni
dependency, and resolving it pulls `numpy` the same way.

> Do **not** install `qwen-tts` with dependencies here. Its declared dependency
> set can pull a different Transformers/Torch stack than the SGLang-Omni runtime.

Concretely, `qwen-tts` 0.1.1 pins Transformers 4.57.3, and its model code calls
APIs that Transformers 5.12 has since renamed or removed — most visibly the mask
factories (`create_causal_mask` and friends), which now spell `input_embeds` as
`inputs_embeds` and no longer accept `cache_position`. SGLang-Omni patches these
differences in
`sglang_omni/models/qwen3_tts/compat.py`, which every Qwen3-TTS entry point
applies before importing `qwen_tts`. The pinned Transformers 5.12 / SGLang 0.5.18
stack is therefore the supported configuration, not a workaround.

If you hit a `TypeError` raised from inside `qwen_tts`, do not resolve it by
installing the package's own Transformers pin — that breaks the rest of the
runtime. Report it instead, so the shim can cover it.

The Python `sox` package shells out to the system `sox` binary on some paths, so install both.

Download a checkpoint (both repositories are public, no token required):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

## Server Configuration

The pipeline is `preprocessing → tts_engine → vocoder`. First startup can take several minutes
while the `tts_engine` captures CUDA graphs.

```bash
# 0.6B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

```bash
# 1.7B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

### Deterministic Inference

Dynamic batching can change Qwen3-TTS codec and waveform outputs even when the
prompt, reference audio, and seed are unchanged. Both the 0.6B and 1.7B Base
checkpoints provide an opt-in deterministic mode:

```yaml
enable_deterministic_inference: true
```

When enabled, the same prompt, reference audio, and seed produce byte-identical
PCM across runtime batch sizes. This mode reduces throughput because it
serializes reference preprocessing and vocoder decoding and disables both the
initial and follow-up vocoder CUDA Graphs, so it is disabled by default.

### Overload / admission policy

Two SGLang generation-stage knobs bound how the server behaves past saturation:

| Knob | Meaning | Qwen3-TTS default |
|---|---|---|
| `--tts_engine.engine.max_running_requests` | Concurrent running slots | `16` |
| `--tts_engine.engine.max_queued_requests` | Waiting-queue depth before fast-reject | `16` |

Every request enters the waiting queue first, so `max_queued_requests`
must be **≥ 1**. Capacity is about `running + queued`. Extra arrivals get
HTTP **503** (`The request queue is full.`) before preprocessing, or later
if the AR waiting queue or request-build backlog is full. Qwen3-TTS
defaults to 4 request-build workers with pending depth 16.

Raising `max_running_requests` does **not** automatically raise the waiting
bound. For a ceiling-32 experiment:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --tts_engine.engine.max_running_requests 32 \
  --tts_engine.engine.max_queued_requests 16 \
  --port 8000
```

Stepped `--concurrencies` is a closed-loop client: it never holds more than
N in-flight requests, so past-ceiling load is a burst that drains. Keep
offered load above `max_running_requests + max_queued_requests` for a
duration with open-loop sustained overshoot:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server --stream \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --max-running-requests 32 \
  --max-queued-requests 16 \
  --sustained-overshoot \
  --overshoot-duration-s 10 \
  --max-samples 64
```

Arrivals default to `2 × capacity` (`--request-rate` overrides). Stats are
on successes only; artifacts land in `<output-dir>/overshoot/`.

A closed-loop `--concurrencies 16,32,48,64` sweep is still available for
comparing healthy vs past-ceiling points, but it does not hold overshoot. Each
concurrency writes inspectable artifacts under `<output-dir>/c<N>/`.

### Prefill Admission Coalescing

Under concurrent load, the `tts_engine` stage can coalesce prefill admission:
instead of admitting each prepared request into its own prefill batch, the
scheduler can briefly hold admission so that multiple ready requests are
prefilled together.

A prefill step has a largely fixed scheduler cost, so fuller batches can reduce
prefill overhead. The end-to-end benefit depends on whether that saving
outweighs the extra admission delay and any resulting reduction in decode
occupancy.

Coalescing is **off by default** and opt-in through the `tts_engine` factory
configuration:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.prefill_coalesce_requests 2 \
  --tts_engine.factory.prefill_coalesce_wait_ms 30 \
  --port 8000
```

or per-stage in YAML:

```yaml
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 2
      prefill_coalesce_wait_ms: 30.0
```

The gate engages only when `prefill_coalesce_requests >= 2`. Once engaged,
admission is released as soon as any of the following holds:

- decode is idle, so a ready request can start immediately;
- the waiting queue reaches `prefill_coalesce_requests`;
- the oldest waiting request has waited `prefill_coalesce_wait_ms`.

`prefill_coalesce_wait_ms` is therefore an upper bound on the added admission
wait. Admission may be released earlier if the target queue size is reached.

The values above are an example for the Qwen3-TTS workload and are not intended
as universal defaults. Match both `prefill_coalesce_requests` and
`prefill_coalesce_wait_ms` to the workload you actually serve. Coalescing is
most useful when natural prefill batches are small and a short hold can increase
batching without materially reducing decode occupancy. If the wait is too long,
the reduced decode occupancy can offset the prefill savings.

Leave coalescing disabled for latency-sensitive traffic or workloads where the
added wait does not produce enough additional batching.

### Process topology

By default all three stages share one process. Per-request reference
preprocessing (speech-tokenizer encode, speaker embedding, prompt embedding)
then competes with the AR scheduler and the vocoder for the same interpreter,
which caps single-replica throughput once concurrency passes ~32. Moving the
preprocessing stage to its own process removes that contention: the stage
loads a prompt-only frontend (embedding tables, text projection, predictor
codec embeddings, speaker encoder) plus the speech tokenizer, together about
2.2 GB of GPU memory for the extra process on a 1.7B checkpoint, and ships the
prepared prompt tensors to the engine through the payload. Every GPU stage must
then declare a memory fraction, and the engine's static fraction has to agree
with the one it declares:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --preprocessing.process tts_frontend \
  --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --vocoder.gpu_memory_fraction 0.12
```

`--vocoder.process vocoder` composes with it (lower `tts_engine` to 0.72 and
give the vocoder 0.15). Six SeedTTS samples at a fixed seed produced identical
PCM in both layouts; the extra process costs its CUDA context plus the frontend
weights.

## Synthesizing Speech

### Text-only Requests

Qwen3-TTS Base checkpoints require a reference clip. Text-only requests are supported by CustomVoice and VoiceDesign checkpoints; see [TTS Model Usage](../basic_usage/tts.md) for those launch commands.

### Voice Cloning

The `references` field accepts `audio_path` (a local path or HTTP URL) and `text` (the
transcript of that clip). Supplying the transcript enables in-context-learning (ICL) mode and
materially improves cloning quality; omitting it falls back to speaker-embedding (x-vector)
mode.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
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
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "voice": "default",
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

Non-streaming responses include `X-Finish-Reason: stop` after codec EOS or
`X-Finish-Reason: length` when generation reaches `max_new_tokens`. A `length`
response still contains decodable audio, but the utterance may be incomplete.
Batch responses expose the same value as each item's `finish_reason`.

### Language Hint

`language` biases the model toward a target language. It defaults to `auto` (let the model
detect). Supported languages are Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "今天天气不错，就该出去晒晒太阳。",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "language": "Chinese"
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` and `"response_format": "pcm"` to receive raw PCM audio
chunks in real time:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Streaming returns `audio/pcm` 16-bit mono PCM bytes with sample-rate metadata in
the response headers. See the [Higgs TTS cookbook](../cookbook/higgs_tts.md#streaming)
for a full Python raw PCM consumer.

All three task types (Base/reference-cloning, CustomVoice and VoiceDesign) use
true incremental codec and vocoder streaming, for both this HTTP endpoint and
`/v1/audio/speech/stream` WebSocket sessions with `stream_audio=true`. Pass
`"stream_codec_output": false` on a request, or launch with
`--preprocessing.factory.stream_codec_output false`, to restore whole-utterance
decoding.

When `initial_codec_chunk_frames` is omitted, Qwen3-TTS defaults to `8`
codec frames for the first vocoder chunk so concurrent streams stay continuous.
Pass a smaller value only when trading continuity for lower time-to-first-audio.
Utterances that finish in fewer than `8` generated codec frames never reach the
first chunk, so their audio arrives complete in a single final flush.

#### First-audio chunk ramp

For latency-sensitive deployments the whole early chunk schedule can be
configured server-side with `stream_chunk_ramp` on the vocoder stage: entry
`i` sizes streaming decode chunk `i + 1` in codec frames, and past the ramp
the steady stride takes over, so `[2, 4, 8]` yields a
`2 -> 4 -> 8 -> 8 -> ...` schedule. Set it through a pipeline config file:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config qwen3_tts_ramp.yaml
```

Smaller early chunks lower time-to-first-audio but start playback with less
buffered audio, so the continuity cost grows with concurrency: keep
`[2, 4, 8]` to low concurrency, prefer `[4, 8]` up to moderate concurrency,
and keep the default schedule for saturated serving. The ramp is mutually
exclusive with the legacy `initial_chunk_frames` /
`stream_initial_followup_stride` options, its first entry must not exceed the
steady stride, and a per-request `initial_codec_chunk_frames` still overrides
only the first chunk.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | served model | Served model identifier |
| `input` | (required) | Text to synthesize |
| `voice` | `default` | Voice identifier. For Base reference cloning, the reference clip provides the speaker conditioning |
| `references` | `null` | Reference clip for cloning. Each item has `audio_path` and `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `language` | `auto` | Target-language hint (see list above) |
| `temperature` | `0.9` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | `50` | Top-k sampling |
| `repetition_penalty` | `1.05` | Repetition penalty |
| `max_new_tokens` | `2048` | Maximum number of generated codec tokens |
| `seed` | `null` | Random seed for reproducibility |
| `stream` | `false` | Stream raw PCM audio chunks |
| `initial_codec_chunk_frames` | `8` (model default when omitted) | First streaming vocoder chunk size in codec frames. Smaller values lower TTFA but underrun more easily; `0` uses the steady stride from the start |
| `stream_codec_output` | `true` | Forward codec frames to the vocoder as they are generated. Set `false` to restore whole-utterance decoding for CustomVoice/VoiceDesign |

## Model Variants

| Checkpoint | Parameters | Config |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | `examples/configs/qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `examples/configs/qwen3_tts_1_7b.yaml` |

Both expose an identical request API. The 1.7B model has higher capacity (typically better
quality) at a larger memory and latency cost; the 0.6B model is lighter and faster.

## Benchmark Results

Qwen3-TTS-12Hz-0.6B-Base on Seed-TTS EN (1088 utterances, reference voice cloning from each
prompt), concurrency 16, WER scored with HF Whisper-large-v3. Hardware: 1× H200 SXM.

| Metric | Value |
|---|---|
| WER (corpus, excl. runaway outliers) | 1.07% |
| WER (per-sample median / p95) | 0.00% / 9.09% |
| WER (corpus micro-avg, raw) | 18.29% |
| Runaway samples (>50% WER) | 2 / 1088 (0.2%) |
| Latency mean / median (s) | 6.61 / 6.24 |
| RTF mean / median | 1.51 / 1.48 |
| Output throughput (tok/s) | 115.4 |
| Completed / failed requests | 1088 / 0 |

Typical output is clean (0.00% median WER, 9.09% p95). Two utterances (0.2%) ran away into a
repetition loop and generated ~164 s of looping audio up to `max_new_tokens`, which alone lifts
the raw micro-average to 18.29%; excluding those, corpus WER is 1.07%. RTF > 1 reflects the
0.6B codec pipeline at concurrency 16, not single-stream latency. The 1.7B checkpoint trades
latency for quality.

## Known Limitations

- **Reference audio recommended.** As a cloning model, Qwen3-TTS Base produces robotic speech
  without a reference clip.
- **Transcript improves cloning.** Providing `text` in `references` (ICL mode) yields better
  speaker similarity than speaker-embedding-only (x-vector) mode.
- **Language detection.** `language: auto` may misdetect for short or code-switched inputs;
  set `language` explicitly when you know the target language.
- **Rare runaway generation.** Roughly 0.2% of utterances (observed on the 0.6B checkpoint) can
  fall into a repetition loop and keep generating up to `max_new_tokens`. Raising
  `repetition_penalty` (default `1.05`) or lowering `max_new_tokens` mitigates it; the 1.7B
  checkpoint is less prone.
