# Ming-Omni-TTS

[Ming-Omni-TTS-16.8B-A3B](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B)
is a mixture-of-experts audio generation model from inclusionAI. The current SGLang-Omni
serving path supports **text-to-speech** and **zero-shot voice cloning** through the
OpenAI-compatible `/v1/audio/speech` endpoint and produces **44.1 kHz** audio.

![Ming-Omni-TTS model architecture](https://github.com/inclusionAI/Ming-omni-tts/raw/main/figures/ming_omni_tts.png)

The serving pipeline keeps the SGLang autoregressive backbone and the Ming acoustic feedback
loop in one generation stage:

```text
preprocessing -> reference_encode -> tts_engine -> audio_decode
                                      |       ^
                                      +-------+
                                       latent feedback
```

`reference_encode` is a no-op for text-only requests. For voice cloning it extracts the speaker
embedding and prompt latents before the autoregressive loop starts. `tts_engine` runs the
SGLang backbone, FlowLoss/CFM acoustic tail, stop head, and feedback projection. `audio_decode`
converts the generated latent sequence into the final waveform with the Ming AudioVAE.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download
the checkpoint:

```bash
hf download inclusionAI/Ming-omni-tts-16.8B-A3B
```

The provided configuration is the recommended TP2 deployment and uses GPUs 0 and 1.

## Server Configuration

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --port 8000
```

## Synthesizing Speech

### Text Only

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni is a great project!",
    "response_format": "wav"
  }' \
  --output output.wav
```

### Voice Cloning

Ming-Omni-TTS currently accepts one local reference clip and requires its transcript. Start the
server with access to the directory containing the clip:

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

Then submit the reference as a `file://` URL:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "file:///path/to/references/prompt.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for the single `references` item.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `input` | (required) | Non-empty text to synthesize |
| `references` | `null` | At most one local reference clip with non-empty `text` |
| `ref_audio` / `ref_text` | `null` | Shorthand for the reference clip and transcript |
| `max_new_tokens` | `200` | Maximum acoustic generation steps; the provided config caps this at `256` |
| `temperature` | `0.0` | Non-negative SDE temperature used by the FlowLoss sampler |
| `response_format` | `wav` | Audio response format; `wav` is used by the reference benchmark |
| `voice` | `default` | Only the default voice selector is accepted |
| `speed` | `1.0` | Other speed values are not supported |

Advanced FlowLoss controls can be passed through `stage_params.tts_engine`:

```json
{
  "stage_params": {
    "tts_engine": {
      "cfg": 2.0,
      "sigma": 0.25,
      "temperature": 0.0
    }
  }
}
```

## Benchmarking

The reference serving configuration uses Seed-TTS-Eval with concurrency 8. Run generation
against the existing Ming-TTS server and save the audio for a separate ASR pass:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/ming_tts_reference_en \
  --lang en --ref-format references \
  --max-new-tokens 256 --max-concurrency 8 --warmup 0
```

Use `--no-ref-audio` for text-only synthesis. Use `--lang zh` and a different output directory
for the Chinese split. Release the TTS server GPUs before starting the ASR server, then compute
WER from the saved audio:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8100

python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --use-existing-server \
  --host 127.0.0.1 --port 8100 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/ming_tts_reference_en \
  --lang en --ref-format references \
  --max-new-tokens 256
```

## Benchmark Results

### Recommended TP2

The recommended TP2 configuration was evaluated on the complete
**Seed-TTS-Eval EN and ZH splits** with **2× H100 80 GB**, concurrency 8, AR and tail CUDA
graphs enabled, and **Qwen3-ASR-1.7B** for transcription.

| Slice | Lang | Samples | Failed | Corpus WER | RTF Mean | Latency Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 1.06% | 0.5052 | 2.287 | 3.489 | 16.480 |
| text-only | ZH | 2020 | 0 | 0.72% | 0.4802 | 2.347 | 3.404 | 17.038 |
| reference | EN | 1088 | 0 | 1.36% | 0.5707 | 2.443 | 3.268 | 14.728 |
| reference | ZH | 2020 | 0 | 0.92% | 0.5298 | 2.970 | 2.690 | 15.399 |

**Median per-sample WER is 0% for every full-set slice**; corpus WER includes a small
near-silent tail from unseeded acoustic sampling.

### Single-GPU TP1

TP1 was also verified on **1× H100 80 GB** with the first 100 samples of each slice. This
confirms that the complete pipeline fits and runs on one GPU; **TP2** remains the recommended
deployment.

| Slice | Lang | Samples | Failed | Corpus WER | RTF Mean | Latency Mean (s) | Throughput (qps) | Audio s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 100 | 0 | 0.64% | 1.5151 | 6.467 | 1.224 | 5.540 |
| text-only | ZH | 100 | 0 | 0.32% | 1.4429 | 6.329 | 1.251 | 5.606 |
| reference | EN | 100 | 0 | 0.82% | 1.4584 | 5.636 | 1.406 | 5.852 |
| reference | ZH | 100 | 0 | 0.21% | 1.4168 | 7.985 | 0.992 | 5.693 |

## Known Limitations

- **Serving optimizations.** Prefix/radix cache and streaming output are follow-up features not yet
  implemented in the current SGLang-Omni integration. `torch.compile` has not yet been validated
  and remains disabled in the provided configuration.
- **Reference inputs.** The current request adapter accepts one local reference audio file with a
  non-empty transcript; remote URLs, data URLs, precomputed prompt latents, and speaker embeddings
  are not yet exposed.
- **Generation controls.** Request-local `seed`, logits sampling fields (`top_p`, `top_k`,
  `repetition_penalty`), named voices, explicit language selection, instructions, and duration
  control are not yet exposed.
- **Checkpoint coverage.** The provided configuration targets the 16.8B-A3B checkpoint. A
  configuration for the 0.5B checkpoint has not yet been added.
