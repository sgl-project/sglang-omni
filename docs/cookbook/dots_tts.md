# dots.tts

[dots.tts-mf](https://huggingface.co/dots-studio/dots.tts-mf) is a 2B continuous
AR TTS model (MeanFlow distillation of dots.tts-soar). The current SGLang-Omni
path supports **zero-shot continuation cloning** through `/v1/audio/speech` and
produces **48 kHz** audio.

```text
preprocessing -> reference_encode -> tts_engine -> audio_decode
```

`reference_encode` encodes the reference clip into AudioVAE prompt latents plus
a CAM++ speaker embedding and builds the generation schedule. `tts_engine` is an
**OmniScheduler** stage: SGLang owns the Qwen2 AR backbone's KV cache and
continuous batching, and the MeanFlow DiT plus patch encoder run batched across
the whole running batch after each backbone step. `audio_decode` converts the
emitted latents with the AudioVAE.

`tts_engine` is single-GPU and runs with `disable_radix_cache=true` and
`chunked_prefill_size=0`: prompt audio spans carry continuous embeddings that
token ids cannot key, and the acoustic tail seeds its flow history from one whole
prefill forward.

## Prerequisites

```bash
hf download dots-studio/dots.tts-mf --local-dir /path/to/dots.tts-mf
```

## Server

```bash
sgl-omni serve \
  --model-path /path/to/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

## Voice cloning

Provide one local reference clip and its transcript:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-tts-mf",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "file:///path/to/references/prompt.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` / `ref_text` are accepted as shorthand for the single reference.

## Generation parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | required | Non-empty text to synthesize |
| `references` | required | Exactly one local clip with non-empty `text` |
| `num_steps` | `4` | MeanFlow NFE; fixed by the `tts_engine` stage (see below) |
| `guidance_scale` | `1.2` | Ignored on mf (CFG fused into the student) |
| `speaker_scale` | `1.5` | X-vector scale |
| `seed` | `null` | Optional deterministic noise seed |
| `stream` | `false` | Streaming is not supported yet |
| `max_audio_patches` | `500` | 160 ms per patch; must not exceed the stage's capacity |

`num_steps` and `max_audio_patches` are engine-wide: the tail shares one ODE grid
across the batch and sizes its DiT / patch-encoder pools once at startup. Set
them in `factory_args` of the `tts_engine` stage; a request that asks for a
different `num_steps`, or for more patches than the stage was built for, is
rejected instead of being silently reinterpreted.

## Acceptance gate

Official mf Seed-TTS-Eval **test-en WER is 1.29%** (NFE=4). The serving path is
accepted when a single-GPU run of the full EN 1088 set at the benchmark's default
**concurrency 16** satisfies both:

- generation finishes in **under 10 minutes**, and
- corpus **WER < 3%**.

A run that takes tens of minutes means the AR stage is not really batching on the
SGLang backend; a WER far above the official number means the feedback / latent /
AudioVAE path is broken.

Bring the server up first (single GPU is enough), then generate against it and
score with ASR on a second GPU:

```bash
# Terminal A — TTS (1 GPU)
sgl-omni serve \
  --model-path /path/to/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path / \
  --port 8000

# Terminal B — generate EN 1088
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model dots-tts-mf \
  --base-url http://127.0.0.1:8000 \
  --use-existing-server \
  --generate-only \
  --lang en \
  --max-concurrency 16 \
  --ref-format references \
  --output-dir results/dots_tts_seedtts_en

# Terminal C — ASR WER (separate GPU)
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model dots-tts-mf \
  --transcribe-only \
  --lang en \
  --port 8001 \
  --output-dir results/dots_tts_seedtts_en
```

`benchmarks.eval.benchmark_tts_seedtts`'s own `managed_omni_server` does not pass
`--config`, so dots.tts must be served separately and driven with
`--use-existing-server --generate-only`.

Measured on one H100 80GB (`max_running_requests=8`, `num_steps=4`,
`disable_cuda_graph=true`), full EN 1088 at `--max-concurrency 16`:

| Metric | Value |
|---|---|
| Generation wall clock | 365 s (2.98 req/s), 0 failures |
| Corpus WER (Qwen3-ASR-1.7B) | **1.11%** |
| Samples above 50% WER | 0 |

## Known limitations

- Non-streaming only (`BatchVocoderBase`).
- Single GPU only; no tensor parallelism for the AR backbone.
- MeanFlow checkpoints only; the flow-matching / CFG solver is not wired up.
- Prefix (radix) cache and chunked prefill are unsupported, so retraction and
  prefix reuse are rejected rather than silently mishandled.
- soar checkpoint and x-vector-only / text-only modes are follow-ups.
- `torch.compile` / DiT CUDA Graph are not enabled; the backbone runs without
  CUDA graphs too (`disable_cuda_graph=true`).
